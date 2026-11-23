//! Chunked prefill: the prompt runs through each layer `chunk` tokens at a time with tensor-core GEMMs. Routed experts
//! run as grouped GEMMs over the chunk's (token, expert) pairs sorted by expert; experts not resident in the cache are
//! copied into a small staging area, a group of them at a time, each group's copy beside the previous group's GEMMs.

use super::{Ffn, GpuModel, HcAt, Mixer};
use crate::backend::{Backend, Buf, GEMM_BM, SAME_X};
use crate::expert_cache::MATS;
use crate::gguf::MappedBytes;
use crate::model::Model;
use crate::weights::{ExpertCacheReservation, Prefetch, expert_cache_ram_capacity, expert_cache_ram_set_capacity, expert_cache_ram_state};
use anyhow::Context;
use std::cell::Cell;
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Experts staged per group when a layer routes to experts outside the cache. Staging holds three groups.
const STAGE: usize = 16;


/// Actuation signal for the fault-based adaptive cap. A page fault on host upload means the cache page was
/// not resident in physical RAM (it was paged out to the file backing), so sustained faults mean the cache
/// is thrashing and its budget sits above the resident-RAM envelope.
fn page_faults() -> Option<u64> {
    #[cfg(target_os = "windows")]
    {
        use std::mem::size_of;
        #[repr(C)]
        struct ProcessMemoryCounters {
            cb: u32,
            page_fault_count: u32,
            peak_working_set_size: usize,
            working_set_size: usize,
            quota_peak_working_set_size: usize,
            quota_working_set_size: usize,
            page_fault_count_high: u32,
            page_file_usage: usize,
            peak_page_file_usage: usize,
        }
        unsafe extern "system" {
            #[link_name = "GetProcessMemoryInfo"]
            fn get_process_memory_info(hprocess: *mut std::ffi::c_void, pmic: *mut ProcessMemoryCounters, cb: u32) -> i32;
        }
        #[link(name = "psapi")]
        unsafe extern "system" {}
        unsafe {
            // GetCurrentProcess sentinel; the current process is always a valid handle.
            let hprocess = std::ptr::null_mut();
            let mut pmc = ProcessMemoryCounters {
                cb: size_of::<ProcessMemoryCounters>() as u32,
                page_fault_count: 0,
                peak_working_set_size: 0,
                working_set_size: 0,
                quota_peak_working_set_size: 0,
                quota_working_set_size: 0,
                page_fault_count_high: 0,
                page_file_usage: 0,
                peak_page_file_usage: 0,
            };
            if get_process_memory_info(hprocess, &mut pmc, pmc.cb) != 0 {
                return Some(pmc.page_fault_count as u64);
            }
            None
        }
    }
    #[cfg(not(target_os = "windows"))]
    {
        None
    }
}

fn expert_run_offsets(grp: &[(usize, [u64; 3])]) -> Vec<(usize, usize)> {
    let mut runs = Vec::new();
    let mut start = 0;
    for i in 1..=grp.len() {
        if i == grp.len() || grp[i].0 != grp[i - 1].0 + 1 {
            runs.push((start, i - start));
            start = i;
        }
    }
    runs
}

struct CachedExpert { mats: [MappedBytes; 3], bytes: usize, _reservation: Option<Arc<ExpertCacheReservation>> }

#[derive(Clone, Copy, Default)]
struct PrefillCacheCounters {
    hits: u64,
    misses: u64,
    insertions: u64,
    evictions: u64,
    cached_bytes_loaded: u64,
    uncached_bytes_read: u64,
    capacity_changes: u64,
    // Bytes of cached experts served to the caller (and thus uploaded to the GPU from host RAM).
    // The denominator for faults-per-GB in the adaptive cap: page faults accrue during this upload.
    cached_bytes_uploaded: u64,
}

#[derive(Clone)]
struct PrefillCacheSnapshot {
    counters: PrefillCacheCounters,
    capacity: usize,
    min_capacity: usize,
    max_capacity: usize,
    used: usize,
    experts: usize,
    layer_count: usize,
    layer_counters: HashMap<usize, PrefillLayerCounters>,
    layer_used: HashMap<usize, usize>,
    layer_experts: HashMap<usize, usize>,
    layer_expert_bytes: HashMap<usize, usize>,
    layer_quota: HashMap<usize, usize>,
    faults_delta: u64,
}

#[derive(Clone, Default)]
struct PrefillLayerCounters {
    hits: u64,
    misses: u64,
    unique_hits: HashSet<usize>,
    unique_misses: HashSet<usize>,
}

fn proportional_layer_quotas(budget: usize, bank_bytes: &HashMap<usize, usize>, total: usize) -> HashMap<usize, usize> {
    if total == 0 { return bank_bytes.keys().map(|&layer| (layer, 0)).collect(); }
    bank_bytes.iter()
        .map(|(&layer, &bytes)| (layer, ((budget as u128 * bytes as u128) / total as u128) as usize))
        .collect()
}

/// Tokens between working-set-based per-layer quota rebalances. Configurable via
/// `STRATA_REBALANCE_TOKENS` (default 64).
fn rebalance_every_tokens() -> usize {
    std::env::var("STRATA_REBALANCE_TOKENS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&n| n > 0)
        .unwrap_or(64)
}

struct PrefillProfile {
    started: Instant,
    cache_lookup: Cell<Duration>,
    expert_range: Cell<Duration>,
    host_upload: Cell<Duration>,
}

impl PrefillProfile {
    fn new() -> PrefillProfile {
        PrefillProfile {
            started: Instant::now(),
            cache_lookup: Cell::new(Duration::ZERO),
            expert_range: Cell::new(Duration::ZERO),
            host_upload: Cell::new(Duration::ZERO),
        }
    }
}

#[derive(Default)]
struct PrefillExpertCache {
    entries: HashMap<(usize, usize), Arc<CachedExpert>>,
    lru: VecDeque<(usize, usize)>,
    layer_used: HashMap<usize, usize>,
    used: usize,
    budget: usize,
    layer_quota: HashMap<usize, usize>,
    layer_count: usize,
    counters: PrefillCacheCounters,
    layer_counters: HashMap<usize, PrefillLayerCounters>,
    layer_expert_bytes: HashMap<usize, usize>,
    layer_bank_bytes: HashMap<usize, usize>,
    total_expert_bank_bytes: usize,
    min_capacity: usize,
    max_capacity: usize,
    // Unique experts routed per layer, for adaptive per-layer quotas: (layer, expert) -> present.
    seen: HashSet<(usize, usize)>,
    // Tokens since the last working-set rebalance, and the token period for the next one.
    tokens_since_rebalance: usize,
    rebalance_every: usize,
    // Fault-based adaptive cap. The cap is the shared `ExpertCacheRam` pool's capacity, which both
    // the GPU prefill cache and the CPU expert cache draw from. It starts at the initial budget,
    // shrinks on sustained page faults (cache above the resident-RAM envelope, thrashing), and
    // grows back toward free RAM when faults subside.
    initial_budget: usize,
    last_faults: Option<u64>,
    last_uploaded: u64,
    fault_streak: u32,
    grow_streak: u32,
    faults_delta: u64,
    // Per-layer routing consistency (mean-of-means), set by `GpuModel` before each
    // `rebalance_chunk`. Used as a multiplier on each layer's working-set demand in
    // `rebalance()`: high-consistency layers get a tighter budget.
    consistency: Option<Vec<f64>>,
    // C2 protection mask, one per layer (`n_expert/64` words each), set at the start of
    // every chunk from the live routing table. `prot` is the consistency (intersection)
    // set: experts present in *every* recorded occurrence of the tokens in this chunk —
    // tier 1, never evicted while the mask is live. Everything else is tier 3 (plain
    // LRU). The prediction/union set is deliberately NOT protected: it is far larger
    // than a layer's capacity, so pinning it starves the working set.
    // Empty tables (no seed file, first calibrate pass) give all-zero masks: C2
    // degrades to plain LRU.
    prot: Vec<Vec<u64>>,
}

impl PrefillExpertCache {
    fn new(m: &Model) -> anyhow::Result<PrefillExpertCache> {
        if m.w.g.uses_mmap() || !m.cfg.is_moe() { return Ok(PrefillExpertCache::default()); }
        let Some(budget) = expert_cache_ram_capacity() else {
            eprintln!("Prefill expert RAM cache: physical-memory availability is unknown; prefill reads uncached experts from disk");
            return Ok(PrefillExpertCache::default());
        };
        let layer_count = m.cfg.n_layer;
        let mut layer_expert_bytes = HashMap::with_capacity(layer_count);
        let mut layer_bank_bytes = HashMap::with_capacity(layer_count);
        let mut total_bank_bytes = 0usize;
        for il in 0..layer_count {
            let expert_bytes = MATS.iter().try_fold(0usize, |size, name| {
                size.checked_add(m.w.expert_shape(il, name)?.3)
                    .context("GPU prefill expert size overflow")
            })?;
            layer_expert_bytes.insert(il, expert_bytes);
            let bank_bytes = expert_bytes.checked_mul(m.cfg.n_expert)
                .context("GPU prefill expert bank size overflow")?;
            total_bank_bytes = total_bank_bytes.checked_add(bank_bytes)
                .context("GPU prefill expert bank total overflow")?;
            layer_bank_bytes.insert(il, bank_bytes);
        }
        let layer_quota = proportional_layer_quotas(budget, &layer_bank_bytes, total_bank_bytes);
        let initial_budget = budget;
        let rebalance_every = rebalance_every_tokens();
        eprintln!(
            "Prefill expert RAM cache [{} layers]: file-backed LRU enabled, initial physical-RAM budget {:.0} MB (layer shares proportional to expert-bank bytes, re-balanced from the observed working set every {rebalance_every} tokens; unused share is borrowable; commit is not reserved; adaptive cap starts at {:.0} MB, floor {:.0} MB)",
            layer_count,
            budget as f64 / 1048576.0,
            initial_budget as f64 / 1048576.0,
            (initial_budget / 4) as f64 / 1048576.0,
        );
        Ok(PrefillExpertCache {
            budget,
            layer_quota,
            layer_count,
            layer_expert_bytes,
            layer_bank_bytes,
            total_expert_bank_bytes: total_bank_bytes,
            initial_budget,
            rebalance_every,
            ..PrefillExpertCache::default()
        })
    }

    /// `count` tallies the access into the hit/miss counters. `false` for the prediction
    /// pre-load, which only warms the cache: the staging `copy` below is the real lookup,
    /// and counting the pre-load too would make every preloaded expert a spurious
    /// miss-then-hit pair (inflating both counters and masking the true hit rate).
    fn expert(&mut self, m: &Model, il: usize, e: usize, count: bool) -> anyhow::Result<Option<Arc<CachedExpert>>> {
        let mut state = self.refresh_budget();
        while !state.is_some_and(|(budget, used)| used <= budget) && !self.lru.is_empty() {
            // C2: evict the lowest-tier resident (tier 3, then 2, then 1), LRU order within
            // a tier, preferring `il`. Plain LRU when the protection masks are empty.
            self.evict_for(il);
            state = self.refresh_budget();
        }
        let key = (il, e);
        self.note_seen(il, e);
        if let Some(cached) = self.entries.get(&key) {
            if count {
                self.counters.hits += 1;
                let layer = self.layer_counters.entry(il).or_default();
                layer.hits += 1;
                layer.unique_hits.insert(e);
            }
            self.counters.cached_bytes_uploaded += cached.bytes as u64;
            let cached = cached.clone();
            self.lru.retain(|&item| item != key);
            self.lru.push_back(key);
            return Ok(Some(cached));
        }
        if count {
            self.counters.misses += 1;
            let layer = self.layer_counters.entry(il).or_default();
            layer.misses += 1;
            layer.unique_misses.insert(e);
        }

        let size = MATS.iter().try_fold(0usize, |size, name| {
            size.checked_add(m.w.expert_shape(il, name)?.3)
                .context("GPU prefill expert size overflow")
        })?;
        if size > self.budget || !self.make_room(size, il) {
            return Ok(None);
        }
        let reservation = loop {
            if let Some(reservation) = ExpertCacheReservation::new(size) { break Some(reservation); }
            if self.lru.is_empty() { break None; }
            self.evict_at(0);
        };
        let Some(reservation) = reservation else { return Ok(None) };

        let loaded = (|| -> anyhow::Result<[MappedBytes; 3]> {
            let mut mats = Vec::with_capacity(MATS.len());
            for name in MATS {
                mats.push(m.w.map_expert(il, name, e)?);
            }
            mats.try_into().map_err(|_| anyhow::anyhow!("expected three GPU prefill expert matrices"))
        })();
        let mats = match loaded {
            Ok(mats) => mats,
            Err(err) => return Err(err),
        };
        let cached = Arc::new(CachedExpert {
            mats,
            bytes: size,
            _reservation: Some(Arc::new(reservation)),
        });
        self.entries.insert(key, cached.clone());
        self.lru.push_back(key);
        self.used += size;
        *self.layer_used.entry(il).or_default() += size;
        self.counters.insertions += 1;
        self.counters.cached_bytes_loaded += size as u64;
        Ok(Some(cached))
    }

    fn refresh_budget(&mut self) -> Option<(usize, usize)> {
        let state = expert_cache_ram_state();
        let previous_budget = self.budget;
        // The shared `ExpertCacheRam` pool's capacity already includes the adaptive cap (set via
        // `expert_cache_ram_set_capacity`), so the pool's capacity is the effective budget.
        self.budget = match state {
            Some((budget, _)) => budget,
            None => 0,
        };
        if self.budget != previous_budget {
            self.counters.capacity_changes += 1;
            self.layer_quota = proportional_layer_quotas(self.budget, &self.layer_bank_bytes, self.total_expert_bank_bytes);
        }
        self.min_capacity = self.min_capacity.min(self.budget);
        self.max_capacity = self.max_capacity.max(self.budget);
        state
    }

    /// Run one chunk of `tokens`; re-balances the working-set quota once the accumulated
    /// token count reaches the configured period.
    fn begin_chunk(&mut self, tokens: usize) -> PrefillCacheSnapshot {
        if self.is_live() {
            self.adjust_cap();
            self.tokens_since_rebalance += tokens;
            if self.tokens_since_rebalance >= self.rebalance_every {
                self.rebalance();
                self.tokens_since_rebalance = 0;
            }
        }
        // Note: `unique_hits`/`unique_misses` are deliberately NOT cleared here; they accumulate
        // over the whole run so the report can show the growing per-layer working set.
        // They are reset by `gm.reset()` between runs.
        self.min_capacity = self.budget;
        self.max_capacity = self.budget;
        self.snapshot()
    }

    /// Sample page faults since the last chunk and adjust the adaptive cap. Sustained faults
    /// (cache above the resident-RAM envelope) shrink the cap; sustained low faults grow it back
    /// toward the external budget, bounded by free RAM. Asymmetric hysteresis avoids oscillation.
    fn adjust_cap(&mut self) {
        let Some(current_faults) = page_faults() else {
            // No fault counter (non-Windows, or API failure): leave the cap untouched.
            self.faults_delta = 0;
            self.last_faults = None;
            return;
        };
        let delta = current_faults.saturating_sub(self.last_faults.unwrap_or(current_faults));
        self.last_faults = Some(current_faults);
        self.faults_delta = delta;

        // Bytes of cached experts uploaded this chunk: the faults-per-GB denominator.
        let uploaded_bytes = self.counters.cached_bytes_uploaded.saturating_sub(self.last_uploaded);
        self.last_uploaded = self.counters.cached_bytes_uploaded;
        let uploaded_gb = uploaded_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
        let rate = if uploaded_gb > 0.0 { delta as f64 / uploaded_gb } else { 0.0 };

        // Thresholds (per GB uploaded): sustained high rate -> shrink; sustained low rate -> grow.
        const SHRINK_RATE: f64 = 20_000.0; // faults/GB that signals thrashing
        const GROW_RATE: f64 = 2_000.0;   // faults/GB low enough to trust headroom
        const SHRINK_STREAK: u32 = 2;     // consecutive hot chunks before shrinking
        const GROW_STREAK: u32 = 4;       // consecutive cool chunks before growing (slower)
        const SHRINK_STEP: f64 = 0.25;    // shrink cap by 25%
        const GROW_STEP: f64 = 0.125;     // grow cap by 12.5% (half the shrink step)
        const FLOOR: f64 = 0.25;          // never below 25% of the initial budget

        let floor = (self.initial_budget as f64 * FLOOR) as usize;
        let Some(cap) = expert_cache_ram_capacity() else { return };
        match rate {
            r if r > SHRINK_RATE => {
                self.fault_streak = self.fault_streak.saturating_add(1);
                self.grow_streak = 0;
                if self.fault_streak >= SHRINK_STREAK {
                    let next = ((cap as f64) * (1.0 - SHRINK_STEP)) as usize;
                    let next = next.max(floor);
                    expert_cache_ram_set_capacity(next);
                    eprintln!(
                        "expert RAM cap: {:.1} faults/GB over {:.0} MB uploaded ({} faults) -> cap {:.0} -> {:.0} MB",
                        rate,
                        uploaded_gb * 1024.0 * 1024.0,
                        delta,
                        cap as f64 / 1048576.0,
                        next as f64 / 1048576.0,
                    );
                }
            }
            r if r < GROW_RATE => {
                self.grow_streak = self.grow_streak.saturating_add(1);
                self.fault_streak = 0;
                if self.grow_streak >= GROW_STREAK {
                    let next = ((cap as f64) * (1.0 + GROW_STEP)) as usize;
                    // Grow only back toward the initial budget, never past it.
                    let next = next.min(self.initial_budget);
                    expert_cache_ram_set_capacity(next);
                    eprintln!(
                        "expert RAM cap: {:.1} faults/GB over {:.0} MB uploaded ({} faults) -> cap {:.0} -> {:.0} MB",
                        rate,
                        uploaded_gb * 1024.0 * 1024.0,
                        delta,
                        cap as f64 / 1048576.0,
                        next as f64 / 1048576.0,
                    );
                }
            }
            _ => {
                // Mid-band: no change, but don't reset a streak in progress.
            }
        }
    }

    fn snapshot(&self) -> PrefillCacheSnapshot {
        let mut layer_experts: HashMap<usize, usize> = HashMap::new();
        for &(layer, _) in self.entries.keys() {
            *layer_experts.entry(layer).or_default() += 1;
        }
        PrefillCacheSnapshot {
            counters: self.counters,
            capacity: self.budget,
            min_capacity: self.min_capacity,
            max_capacity: self.max_capacity,
            used: self.used,
            experts: self.entries.len(),
            layer_count: self.layer_count,
            layer_counters: self.layer_counters.clone(),
            layer_used: self.layer_used.clone(),
            layer_experts,
            layer_expert_bytes: self.layer_expert_bytes.clone(),
            layer_quota: self.layer_quota.clone(),
            faults_delta: self.faults_delta,
        }
    }

    fn record_uncached_bytes(&mut self, bytes: usize) {
        self.counters.uncached_bytes_read += bytes as u64;
    }

    /// Record that `(il, e)` was routed, for adaptive per-layer quotas.
    fn note_seen(&mut self, il: usize, e: usize) {
        self.seen.insert((il, e));
    }

    /// True when the cache is live (not the mmap / no-MoE default).
    fn is_live(&self) -> bool {
        self.budget > 0
    }

    /// Rebalance `layer_quota` from observed unique-expert demand, weighted by each
    /// layer's routing consistency: high-consistency layers (reliably predicted
    /// experts) get a tighter budget, low-consistency layers get headroom.
    fn rebalance(&mut self) {
        let seen = &self.seen;
        if seen.is_empty() { return; }
        let mut per_layer: HashMap<usize, usize> = HashMap::new();
        for &(il, _) in seen {
            *per_layer.entry(il).or_default() += 1;
        }
        // Weight each layer's demand by its consistency multiplier.
        let mut weighted: Vec<(usize, u128)> = Vec::new();
        for (l, &n) in &per_layer {
            let mult = self.consistency.as_ref()
                .and_then(|c| c.get(*l).copied())
                .map(consistency_multiplier)
                .unwrap_or(1.0);
            weighted.push((*l, (n as u128) * (mult * 100.0) as u128));
        }
        let total: u128 = weighted.iter().map(|&(_, w)| w).sum();
        if total == 0 { return; }
        let mut new_quota = HashMap::new();
        let mut allocated = 0usize;
        let mut layers = weighted;
        layers.sort_by_key(|&(_, w)| std::cmp::Reverse(w));
        for &(l, w) in &layers {
            let q = (self.budget as u128 * w / total) as usize;
            new_quota.insert(l, q);
            allocated += q;
        }
        // Hand out any leftover bytes to the highest-demand layers first.
        let mut leftover = self.budget.saturating_sub(allocated);
        if leftover > 0 {
            for &(l, _) in &layers {
                if leftover == 0 { break; }
                let q = new_quota.get_mut(&l).unwrap();
                *q = q.saturating_add(1);
                leftover -= 1;
            }
        }
        self.layer_quota = new_quota;
    }

    /// Set the per-layer routing-consistency scores used by the next `rebalance()`.
    /// Called by `GpuModel` from the live table before each chunk's rebalance.
    fn set_consistency(&mut self, scores: Vec<f64>) {
        self.consistency = Some(scores);
    }

    /// Set the C2 protection mask for the current chunk. `prot` is the consistency
    /// (intersection) mask per layer — tier 1, never evicted while live. Everything
    /// else is tier 3 (plain LRU); the prediction/union set is deliberately not
    /// protected. Called from `prefill_chunk` before any expert lookup in the chunk.
    fn set_protect(&mut self, prot: Vec<Vec<u64>>) {
        self.prot = prot;
    }

    /// Residency bitmask for `layer` (`n_expert/64` words): bit `e` set iff `(layer, e)`
    /// is in the RAM cache. Used as the `b` in `to_load = prediction & !residency`.
    fn residency(&self, layer: usize, n_expert: usize) -> Vec<u64> {
        let n_words = (n_expert + 63) / 64;
        let mut bits = vec![0u64; n_words];
        for &(l, e) in &self.lru {
            if l == layer && e < n_expert { bits[e / 64] |= 1u64 << (e % 64); }
        }
        bits
    }

    /// Warm the RAM cache with the predicted experts for `layer` that are not already
    /// resident (`to_load`), so the chunk's staging copies read them from RAM rather
    /// than disk. Experts the cache cannot admit (budget full) are skipped: they fall
    /// back to the disk path in the staging `copy`. Returns the number loaded.
    fn preload(&mut self, m: &Model, layer: usize, to_load: &[usize]) -> anyhow::Result<usize> {
        let mut loaded = 0usize;
        for &e in to_load {
            if self.expert(m, layer, e, false)?.is_some() { loaded += 1; }
        }
        Ok(loaded)
    }

    fn layer_limit(&self, layer: usize) -> usize {
        let borrowed = self.layer_used.iter()
            .filter(|(other, _)| **other != layer)
            .fold(0usize, |sum, (other, &used)| {
                sum.saturating_add(used.min(self.layer_quota.get(other).copied().unwrap_or(0)))
            });
        self.budget.saturating_sub(borrowed)
    }

    fn make_room(&mut self, size: usize, layer: usize) -> bool {
        loop {
            let layer_used = self.layer_used.get(&layer).copied().unwrap_or(0);
            if self.used.checked_add(size).is_some_and(|next| next <= self.budget)
                && layer_used.checked_add(size).is_some_and(|next| next <= self.layer_limit(layer))
            { return true; }
            // C2: pick the lowest-tier victim (tier 3 first, then tier 1); LRU order
            // (smallest index = oldest) breaks ties. Prefer candidates from a layer
            // over its quota (including the requesting layer itself — C2 protection
            // can grow a layer past its quota, and its tier-3 residents are safe to
            // evict); if no layer is over quota, consider the whole LRU.
            // A tier-1 (consistency-protected) expert is never evicted while a tier-3
            // candidate exists.
            let over_quota: Vec<usize> = self.lru.iter().enumerate()
                .filter(|(_, key)| {
                    self.layer_used.get(&key.0).copied().unwrap_or(0)
                        > self.layer_quota.get(&key.0).copied().unwrap_or(0)
                })
                .map(|(j, _)| j)
                .collect();
            let candidates: Vec<usize> = if !over_quota.is_empty() {
                over_quota
            } else {
                (0..self.lru.len()).collect()
            };
            // Evict tier 3 (non-protected) before tier 1 (protected); LRU order
            // (smallest index = oldest) breaks ties.
            let i = candidates.iter().copied()
                .min_by_key(|&j| (3u8 - self.tier(self.lru[j].0, self.lru[j].1), j))
                .unwrap_or(0);
            self.evict_at(i);
        }
    }

    fn evict_at(&mut self, index: usize) {
        let key = self.lru.remove(index).unwrap();
        if let Some(old) = self.entries.remove(&key) {
            self.used -= old.bytes;
            self.counters.evictions += 1;
            let layer_bytes = self.layer_used.get_mut(&key.0).unwrap();
            *layer_bytes -= old.bytes;
            if *layer_bytes == 0 { self.layer_used.remove(&key.0); }
        }
    }

    /// C2 tier of a resident `(layer, expert)`: 1 = consistency-protected (in the
    /// chunk's intersection set), 3 = everything else (plain LRU).
    ///
    /// This matches the proven C2 policy exactly (sim_cache.py, 57.5% hit rate):
    /// only the per-chunk consistency (intersection) set is protected; the
    /// prediction/union set is deliberately NOT a tier. The union set is far larger
    /// than a layer's capacity, so protecting it pins stale experts and starves the
    /// working set (observed: the last MoE layer, with the largest union set, had its
    /// hit rate stuck at ~1% while tier 2 was active).
    /// Empty masks (no seed table) put every resident in tier 3: plain LRU.
    fn tier(&self, layer: usize, expert: usize) -> u8 {
        let w = expert / 64;
        let b = 1u64 << (expert % 64);
        if self.prot.get(layer).is_some_and(|p| w < p.len() && p[w] & b != 0) { return 1; }
        3
    }

    /// C2 tiered eviction for `layer`: scan the LRU (oldest first) and evict the first
    /// resident that is NOT tier 1 (consistency-protected); if the whole LRU is tier 1,
    /// evict the oldest resident. Returns the number of bytes freed (0 when nothing
    /// could be evicted). This is the C2 replacement for the blind `evict_at(0)`:
    /// protected experts survive the thrash that a plain LRU would evict them with.
    fn evict_for(&mut self, layer: usize) -> usize {
        if self.lru.is_empty() { return 0; }
        // Prefer a victim from `layer` (the layer being loaded for); if none there,
        // any layer. Never evict a tier-1 (consistency-protected) expert while a
        // tier-3 resident exists: scan LRU order (oldest first) for the first
        // tier-3. If the whole cache is tier 1, evict the oldest resident.
        let i = self.lru.iter().position(|&(l, e)| l == layer && self.tier(l, e) >= 2)
            .or_else(|| self.lru.iter().position(|&(l, e)| self.tier(l, e) >= 2))
            .unwrap_or(0);
        let bytes = self.entries.get(&self.lru[i]).map(|c| c.bytes).unwrap_or(0);
        self.evict_at(i);
        bytes
    }
}

/// Float offsets into the scratch slab (each region holds `chunk` rows). Mixer and FFN regions alias, and so do the
/// hyper-connection read and PLE temporaries (`xn` .. `pv`), which run outside both.
struct Lay { h: usize, a: usize, b: usize, c: usize, d: usize, e: usize, rl: usize, wts: usize, ge: usize, ye: usize, shg: usize, shy: usize, sg: usize,
             xn: usize, lo: usize, hg: usize, pk: usize, pq: usize, pg: usize, pn: usize, pv: usize, inj: usize, mix: usize, mo: usize, emb: usize }

pub(super) struct Scratch {
    pub chunk: usize,
    lay: Lay,
    slab: Buf,
    /// ids | xrow | inv, `chunk * k` each.
    ints: Buf,
    tiles: Buf,
    staging: Buf,
    host_cache: Mutex<PrefillExpertCache>,
}

impl Scratch {
    pub fn new(m: &Model, gpu: &dyn Backend, chunk: usize) -> anyhow::Result<Scratch> {
        let c = &m.cfg;
        let mut stage_bytes = 0;
        for il in 0..c.n_layer {
            if !c.is_moe() { break; }
            let mut b = 0;
            for name in MATS { b = (b + m.w.expert_shape(il, name)?.3).next_multiple_of(16); }  // as ExpertCache lays out a slot
            stage_bytes = stage_bytes.max(b);
        }
        let (ne, k) = (c.n_embd, c.n_expert_used.max(1));
        let (nv, kv, channels) = (c.ssm_dt_rank, c.n_head_kv * c.head_dim, c.ssm_d_inner + 2 * c.ssm_n_group * c.ssm_d_state);
        let q = c.n_head * c.head_dim;
        let hd = c.hc_n.max(1) * ne;
        let (h, base) = (hd * chunk, (hd + ne) * chunk); // x (hc_n streams a token) at 0, then h
        let mut at = base;
        let mut take = |n: usize| { let o = at; at += n * chunk; o };
        let (a, b, c2, d, e) = (take((2 * q).max(channels).max(2 * ne)), take(q.max(c.ssm_d_inner)), take(channels.max(3 * kv).max(if c.idx_top_k > 0 { (1 + 2 * super::QSA_H) * super::QSA_D } else { 0 })), take(q.max(c.ssm_d_inner)), take(2 * nv));
        let mix_end = at;
        at = base;
        let mut take = |n: usize| { let o = at; at += n * chunk; o };
        let (rl, wts, ge) = (take(c.n_expert), take(k), take((k * c.n_ff_exp).max(c.n_ff)));
        let (ye, shg, shy, sg) = (take(k * ne), take(c.n_ff_shexp), take(ne), take(1));
        let mut total = at.max(mix_end);
        let (mut xn, mut lo, mut hg, mut pk, mut pq, mut pg, mut pn, mut pv, mut inj, mut mix, mut mo, mut emb) = Default::default();
        if c.hc_n > 0 {
            at = base;
            let mut take = |n: usize| { let o = at; at += n * chunk; o };
            (xn, lo, hg, pk, pq, pg, pn, pv) = (take(hd), take(c.hc_rank), take(hd), take(hd), take(hd), take(hd), take(hd), take(ne));
            at = at.max(total);
            let mut take = |n: usize| { let o = at; at += n * chunk; o };
            (inj, mix, mo, emb) = (take(c.hc_n), take(ne), take(ne), take(ne));
            total = at;
        }
        let n_tiles = 3 * (chunk * k / GEMM_BM + 2 * c.n_expert);
        Ok(Scratch {
            chunk, lay: Lay { h, a, b, c: c2, d, e, rl, wts, ge, ye, shg, shy, sg, xn, lo, hg, pk, pq, pg, pn, pv, inj, mix, mo, emb },
            slab: gpu.zeros::<f32>(total)?, ints: gpu.zeros::<i32>(3 * chunk * k)?, tiles: gpu.zeros::<u64>(2 * n_tiles)?,
            staging: gpu.zeros::<u8>(if c.is_moe() { 3 * STAGE * stage_bytes } else { 0 })?,
            host_cache: Mutex::new(PrefillExpertCache::new(m)?),
        })
    }

    /// Bytes of VRAM this scratch takes.
    pub fn bytes(&self) -> usize { self.slab.len() + self.ints.len() + self.tiles.len() + self.staging.len() }

    /// Number of experts currently in the host-RAM prefill cache.
    pub fn ram_cache_experts(&self) -> usize {
        self.host_cache.lock().map_or(0, |c| c.snapshot().experts)
    }
}

/// One token's observations at one layer.
/// `union_set` is the OR accumulator over all occurrences: the prediction table — an
/// over-approximation of the experts this token can need (monotonic, bounded by `n_expert`).
/// `cons_set` is the AND accumulator: the consistency table — the intersection of all
/// recorded sets, a subset of every live set the token can present.
/// `freq` is the per-expert occurrence count (runtime-only, not persisted to the seed file).
pub struct TokenLayerEntry {
    pub union_set: Vec<u64>,
    pub cons_set: Vec<u64>,
    pub freq: Vec<u32>,
}

/// Per-token MoE routing observed during prefill, for the token-bitset calibration study.
/// Keyed by token id, then by global layer index, a [`TokenLayerEntry`] per layer.
/// `union_set` is the live prediction table for cache loading; `cons_set` is the
/// consistency floor for cache sizing and C2 protection; `freq` is runtime-only.
pub struct TokenRouting {
    pub n_expert: usize,
    pub top_k: usize,
    pub map: HashMap<u32, Vec<TokenLayerEntry>>,
}

impl TokenRouting {
    pub fn new(n_expert: usize, top_k: usize) -> Self {
        Self { n_expert, top_k, map: HashMap::new() }
    }

    /// Serialize the table to a binary blob for the calibration seed file (STRRTBL2):
    /// a header (magic, version, n_expert, top_k), then per token id, per layer,
    /// n_words, union_set words, cons_set words. `freq` is NOT persisted (runtime-only).
    pub fn serialize(&self) -> Vec<u8> {
        let mut out: Vec<u8> = Vec::new();
        out.extend_from_slice(b"STRRTBL2");
        out.extend_from_slice(&(2u32).to_le_bytes());
        out.extend_from_slice(&(self.n_expert as u32).to_le_bytes());
        out.extend_from_slice(&(self.top_k as u32).to_le_bytes());
        out.extend_from_slice(&(self.map.len() as u32).to_le_bytes());
        for (&tok, layers) in &self.map {
            out.extend_from_slice(&tok.to_le_bytes());
            out.extend_from_slice(&(layers.len() as u32).to_le_bytes());
            for entry in layers {
                let n_words = entry.union_set.len() as u32;
                out.extend_from_slice(&n_words.to_le_bytes());
                for w in &entry.union_set { out.extend_from_slice(&w.to_le_bytes()); }
                for w in &entry.cons_set { out.extend_from_slice(&w.to_le_bytes()); }
            }
        }
        out
    }

    /// Inverse of `serialize` (STRRTBL2 only; old STRRTBL1 files are rejected).
    /// `freq` is initialized to zeros (runtime-only, not persisted).
    pub fn deserialize(bytes: &[u8]) -> Option<TokenRouting> {
        const MAGIC: &[u8; 8] = b"STRRTBL2";
        if bytes.len() < 24 || &bytes[0..8] != MAGIC { return None; }
        struct R<'a> { b: &'a [u8], p: usize }
        impl R<'_> {
            fn u32(&mut self) -> Option<u32> {
                let v = u32::from_le_bytes(self.b[self.p..self.p + 4].try_into().ok()?);
                self.p += 4;
                Some(v)
            }
            fn u64(&mut self) -> Option<u64> {
                let v = u64::from_le_bytes(self.b[self.p..self.p + 8].try_into().ok()?);
                self.p += 8;
                Some(v)
            }
        }
        let mut r = R { b: bytes, p: 8 };
        let _version = r.u32()?;
        let n_expert = r.u32()? as usize;
        let top_k = r.u32()? as usize;
        let n_tokens = r.u32()? as usize;
        let mut map = HashMap::new();
        for _ in 0..n_tokens {
            let tok = r.u32()?;
            let n_layers = r.u32()? as usize;
            let mut layers = Vec::new();
            for _ in 0..n_layers {
                let n_words = r.u32()? as usize;
                let mut union_set = Vec::with_capacity(n_words);
                for _ in 0..n_words { union_set.push(r.u64()?); }
                let mut cons_set = Vec::with_capacity(n_words);
                for _ in 0..n_words { cons_set.push(r.u64()?); }
                let freq = vec![0u32; n_expert];
                layers.push(TokenLayerEntry { union_set, cons_set, freq });
            }
            map.insert(tok, layers);
        }
        Some(TokenRouting { n_expert, top_k, map })
    }

    /// Popcount of a bitmask.
    fn popcount(bits: &[u64]) -> usize {
        bits.iter().map(|w| w.count_ones() as usize).sum()
    }

    /// Popcount of `a & !b`: the set difference of two bitmasks (experts in `a` missing from `b`).
    /// This is the cache load set when `a` is the prediction and `b` the residency.
    pub fn to_load(a: &[u64], b: &[u64]) -> usize {
        a.iter().zip(b.iter()).map(|(x, y)| (x & !y).count_ones() as usize).sum()
    }

    /// Record the routed experts for one chunk: `ids` is token-major `[t, k]` (element `i` is the expert for token
    /// `i / k`, slot `i % k`); `toks` maps chunk position to token id. `layer` is the global layer index.
    /// Updates the OR accumulator (`union_set`), AND accumulator (`cons_set`), and `freq` in place.
    pub fn record_routing(&mut self, layer: usize, toks: &[u32], ids: &[i32], k: usize) {
        let n_words = (self.n_expert + 63) / 64;
        for (tok, row) in toks.iter().zip(ids.chunks(k)) {
            let per = self.map.entry(*tok).or_insert_with(Vec::new);
            while per.len() <= layer { per.push(TokenLayerEntry::new(self.n_expert)); }
            let mut bits = vec![0u64; n_words];
            for &e in row {
                let e = e as usize;
                if e < self.n_expert {
                    bits[e / 64] |= 1u64 << (e % 64);
                }
            }
            let entry = &mut per[layer];
            // Prediction: OR the new set into the union.
            for (u, b) in entry.union_set.iter_mut().zip(&bits) { *u |= *b; }
            // Consistency: AND the new set into the intersection.
            for (c, b) in entry.cons_set.iter_mut().zip(&bits) { *c &= *b; }
            // Frequency: count each expert in this set.
            for (w, word) in bits.iter().enumerate() {
                for bit in 0..64 {
                    if word >> bit & 1 != 0 {
                        let e = w * 64 + bit;
                        if e < self.n_expert {
                            entry.freq[e] += 1;
                        }
                    }
                }
            }
        }
    }

    /// Fold `other`'s observations into `self`.
    /// `union_set` is OR-folded; `cons_set` is AND-folded; `freq` is summed.
    pub fn merge(&mut self, other: &Self) {
        for (tok, layers) in &other.map {
            let per = self.map.entry(*tok).or_insert_with(Vec::new);
            while per.len() < layers.len() { per.push(TokenLayerEntry::new(self.n_expert)); }
            for (l, o) in layers.iter().enumerate() {
                let e = &mut per[l];
                for (u, b) in e.union_set.iter_mut().zip(&o.union_set) { *u |= *b; }
                for (c, b) in e.cons_set.iter_mut().zip(&o.cons_set) { *c &= *b; }
                for (f, o_f) in e.freq.iter_mut().zip(&o.freq) { *f = f.saturating_add(*o_f); }
            }
        }
    }

    /// Consistency score for one token-layer entry, in `[0, 1]`:
    /// `popcount(cons_set) / top_k`. This is the consistency floor — the fraction of
    /// the token's top-k experts that are present in *every* recorded occurrence.
    pub fn consistency_score(entry: &TokenLayerEntry, top_k: usize) -> f64 {
        if top_k == 0 { return 0.0; }
        Self::popcount(&entry.cons_set) as f64 / top_k as f64
    }

    /// Consistency floor per layer, in `[0, 1]`: the average of each token-layer entry's
    /// `consistency_score` over all tokens with an entry at that layer. Layers with no
    /// tokens score 0. This is the live value the cache sizing policy reads.
    pub fn layer_consistency(&self, n_layer: usize) -> Vec<f64> {
        let mut sum = vec![0.0f64; n_layer];
        let mut count = vec![0usize; n_layer];
        for layers in self.map.values() {
            for (l, entry) in layers.iter().enumerate() {
                if l >= n_layer { break; }
                sum[l] += Self::consistency_score(entry, self.top_k);
                count[l] += 1;
            }
        }
        (0..n_layer).map(|l| if count[l] > 0 { sum[l] / count[l] as f64 } else { 0.0 }).collect()
    }

    /// The predicted expert set for `layer` over the tokens in `toks`, as a bitmask
    /// (`n_expert/64` words): the OR of each token's `union_set` at that layer (tokens
    /// without an entry contribute nothing). This is the live prediction table the cache
    /// load set is computed from (`to_load` = `prediction & !residency`).
    pub fn predicted_mask(&self, layer: usize, toks: &[u32]) -> Vec<u64> {
        let n_words = (self.n_expert + 63) / 64;
        let mut union = vec![0u64; n_words];
        for &tok in toks {
            if let Some(layers) = self.map.get(&tok) {
                if layer < layers.len() {
                    for (u, b) in union.iter_mut().zip(&layers[layer].union_set) { *u |= *b; }
                }
            }
        }
        union
    }

    /// The predicted expert ids (set bits) for `layer` over `toks` — the candidate set to pre-load.
    pub fn predicted_experts(&self, layer: usize, toks: &[u32]) -> Vec<usize> {
        let mut out = Vec::new();
        for (w, word) in self.predicted_mask(layer, toks).iter().enumerate() {
            for bit in 0..64 {
                if word >> bit & 1 != 0 && w * 64 + bit < self.n_expert { out.push(w * 64 + bit); }
            }
        }
        out
    }

    /// The consistency (intersection) mask for `layer` over the tokens in `toks`:
    /// the OR of each token's `cons_set` at that layer (tokens without an entry
    /// contribute nothing). This is C2's protection mask — any expert in this set is
    /// guaranteed to be activated by at least one token in the chunk.
    pub fn consistency_mask(&self, layer: usize, toks: &[u32]) -> Vec<u64> {
        let n_words = (self.n_expert + 63) / 64;
        let mut mask = vec![0u64; n_words];
        for &tok in toks {
            if let Some(layers) = self.map.get(&tok) {
                if layer < layers.len() {
                    for (m, b) in mask.iter_mut().zip(&layers[layer].cons_set) { *m |= *b; }
                }
            }
        }
        mask
    }

    /// Per-expert occurrence frequency for `layer` over `toks`: `freq[e]` is the total
    /// number of times expert `e` is routed by the tokens in `toks`. Each token id is
    /// counted once (deduplicated). Used to sort the pre-load candidate set by demand
    /// (highest first), so the budget fills with the most-needed experts.
    pub fn expert_frequency(&self, layer: usize, toks: &[u32]) -> Vec<u64> {
        let mut freq = vec![0u64; self.n_expert];
        let mut seen = std::collections::HashSet::new();
        for &tok in toks {
            if !seen.insert(tok) { continue; }
            if let Some(layers) = self.map.get(&tok) {
                if layer < layers.len() {
                    for (e, c) in freq.iter_mut().enumerate() {
                        *c += layers[layer].freq[e] as u64;
                    }
                }
            }
        }
        freq
    }
}

/// Multiplier applied to a layer's observed working-set demand by its routing
/// consistency: high-consistency layers need a tighter budget (their experts are
/// reliably predicted), low-consistency layers need headroom.
pub fn consistency_multiplier(score: f64) -> f64 {
    if score >= 0.8 { 0.5 } else if score >= 0.4 { 1.0 } else { 2.0 }
}

impl TokenLayerEntry {
    fn new(n_expert: usize) -> Self {
        let n_words = (n_expert + 63) / 64;
        Self {
            union_set: vec![0u64; n_words],
            // All-ones so the first AND in `record_routing` is a no-op.
            cons_set: vec![u64::MAX; n_words],
            freq: vec![0u32; n_expert],
        }
    }
}

impl GpuModel<'_> {
    /// Run `tokens` from the current position; returns the logits after the last one. Short prompts (and models
    /// loaded without a prefill chunk) go through the decode step.
    pub fn prefill(&mut self, tokens: &[u32]) -> anyhow::Result<Vec<f32>> { self.prefill_with(tokens, &mut |_| Ok(())) }

    /// `prefill`, calling `between` between two chunks (`serve --parallel` decodes the other slots there; it must
    /// leave the current slot's state alone).
    pub fn prefill_with(&mut self, tokens: &[u32], between: &mut dyn FnMut(&mut Self) -> anyhow::Result<()>) -> anyhow::Result<Vec<f32>> {
        let Some(chunk) = self.pf.as_ref().map(|p| p.chunk) else { return self.step_all(tokens) };
        if tokens.len() < self.short { return self.step_all(tokens); }
        anyhow::ensure!(self.pos + tokens.len() <= self.ctx, "context full ({} tokens; raise --ctx)", self.ctx);
        for (i, ch) in tokens.chunks(chunk).enumerate() {
            if i > 0 { between(self)?; }
            self.prefill_chunk(ch, None)?;
        }
        // The prompt's routing predicts the reply's: place its hot experts now instead of 64 per adapt during decode.
        // A fully resident cache has no hot missed expert to place: skip the scan over all its entries (issue #34 R2).
        let adapt = self.adapt_every > 0;
        let mut p = Some(&mut *self);
        while let Some(q) = p {
            if let Some(cache) = q.cache.as_mut() {
                if adapt && !cache.fully_resident() {
                    // Demoted experts become hot CPU misses: warm the no-mmap RAM cache for them before decode
                    // routes there (mmap: the RAM cache is disabled, so `promote` is a no-op).
                    let demoted = cache.adapt(q.m, &*q.gpu, usize::MAX)?;
                    if !demoted.is_empty() { if let Some(w) = q.worker.as_ref() { w.promote(&demoted); } }
                }
            }
            p = q.next.as_deref_mut();
        }
        let s = self.tail();
        let (g, ne) = (&s.gpu, s.m.cfg.n_embd);
        let h = s.bufs["h"].addr();
        // `prefill_chunk` leaves the last hidden state in row 0 of `x`, as after `step` (the MTP draft reads it).
        match (&s.output_norm, &s.output_hc) {
            (Some(on), _) => g.rms_norm(s.x.addr(), 0, on.ptr, h, 0, ne, 1, s.m.cfg.eps)?,
            (None, Some(hr)) => s.hc_read(hr, s.x.addr(), 1, &s.hc_at())?,
            _ => unreachable!(),
        }
        g.gemv(s.output.as_ref().unwrap(), h, s.logits.addr())?;
        g.get::<f32>(&s.logits, 0, s.m.cfg.n_vocab)
    }

    /// A fresh sequence: `prefill(tokens)` from position 0, reusing earlier work. The state after `tokens[..keep]` is
    /// saved; when the saved tokens are a prefix of the next call's `tokens[..keep]`, that call starts from them
    /// (KV past them is rewritten, the recurrent state is copied back). The state after `tokens[..sys]` (the system
    /// prompt, `sys <= keep`) is stashed in host memory: a call whose history does not extend the saved one but
    /// starts with the same system prompt starts from there. A saved state that a call replaces is parked in host
    /// memory (up to `cache_ram`), and a later call that continues it takes it back, so a side request with another
    /// prompt does not cost the main conversation its cache. Returns the logits and the tokens reused.
    pub fn prefill_cached(&mut self, tokens: &[u32], sys: usize, keep: usize) -> anyhow::Result<(Vec<f32>, usize)> {
        self.prefill_cached_with(tokens, sys, keep, &mut |_| Ok(()))
    }

    /// `prefill_cached` with `prefill_with`'s `between`.
    pub fn prefill_cached_with(&mut self, tokens: &[u32], sys: usize, keep: usize, between: &mut dyn FnMut(&mut Self) -> anyhow::Result<()>)
        -> anyhow::Result<(Vec<f32>, usize)> {
        anyhow::ensure!(sys <= keep && keep < tokens.len(), "prefill_cached: sys {sys}, keep {keep} of {} tokens", tokens.len());
        let (hit, own) = (self.saved_ids.len(), self.owner);
        let stashed = self.sys.as_ref().filter(|_| self.sys_owner == own).map_or(0, |s| if tokens[..keep].starts_with(&s.3) { s.3.len() } else { 0 });
        let extends = hit > 0 && hit >= stashed && self.saved_owner == own && tokens[..keep].starts_with(&self.saved_ids);
        // The longest parked state this prompt continues, if it beats the saved one and the stash.
        let best = self.parked.iter().enumerate().filter(|(_, p)| p.owner == own && tokens[..keep].starts_with(&p.ids))
            .max_by_key(|(_, p)| p.ids.len()).map(|(i, p)| (i, p.ids.len()))
            .filter(|&(_, n)| n > stashed && !(extends && n <= hit));
        let reused = if let Some((i, n)) = best {
            self.unpark(i)?;
            n
        } else if extends {
            self.restore_state()?;
            hit
        } else if stashed > 0 {
            self.park()?;
            self.unstash()?;
            stashed
        } else {
            self.park()?;
            self.reset()?;
            0
        };
        // The stash stays valid while its KV entries do: this prompt rewrites them only past `reused`. A parked state
        // brings its own KV, which may have been computed in other chunks.
        if best.is_some() || stashed == 0 && self.sys.as_ref().is_some_and(|s| reused < s.3.len()) { self.sys = None; }
        if sys > reused {
            self.prefill_with(&tokens[reused..sys], between)?;
            self.stash(&tokens[..sys])?;
        }
        let from = reused.max(sys);
        if keep > from {
            self.prefill_with(&tokens[from..keep], between)?;
            self.save_state()?;
            (self.saved_ids, self.saved_owner) = (tokens[..keep].to_vec(), own);
        }
        Ok((self.prefill_with(&tokens[keep..], between)?, reused))
    }

    fn step_all(&mut self, tokens: &[u32]) -> anyhow::Result<Vec<f32>> {
        let mut logits = Vec::new();
        // Hyper-connection models: TMAX tokens per forward (prompts too short for the chunked GEMM prefill).
        if self.m.cfg.hc_n > 0 {
            for ch in tokens.chunks(crate::cpu_worker::TMAX) { logits = self.verify(ch)?.pop().unwrap(); }
            return Ok(logits);
        }
        for &t in tokens {
            // The draft head's entry for `t` pairs it with the hidden state of the token before (row 0 of `x`).
            if self.pos > 0 && self.mtp_synced() { let r = self.row_ptr(0); self.mtp_run(&[t], r, false)?; }
            logits = self.step(t)?;
        }
        Ok(logits)
    }

    /// `input`: the hidden rows and PLE input from the part before (None: embed `toks`).
    fn prefill_chunk(&mut self, toks: &[u32], input: Option<(&[f32], &[f32])>) -> anyhow::Result<()> {
        // Missed experts are read from the mapped weights in the order the staging copies take them.
        let pre = self.cache.is_some().then(Prefetch::start);
        let owned: (Vec<f32>, Vec<f32>);
        let (emb, pe): (&[f32], &[f32]) = match input {
            Some(v) => v,
            None => {
                let hcn = self.m.cfg.hc_n.max(1);
                let pe = if self.ple_ring.is_some() { toks.iter().flat_map(|&tk| self.m.ple_embed(&mut self.ple_prev, tk)).collect() } else { vec![] };
                owned = (toks.iter().flat_map(|&tk| self.m.embed(tk).repeat(hcn)).collect(), pe);
                (&owned.0, &owned.1)
            }
        };
        let c = &self.m.cfg;
        let pf = self.pf.as_ref().unwrap();
        let profiling = std::env::var_os("STRATA_PREFILL_PROFILE").is_some();
        if profiling { self.gpu.sync()?; }
        let profile = profiling.then(PrefillProfile::new);
        // Refresh the per-layer consistency scores and C2 protection masks from the
        // live routing table before the chunk's expert lookups. High-consistency
        // layers get a tighter budget; tier-1 (cons) experts are protected from
        // eviction while the mask is live.
        let cache_before = {
            let mut host_cache = pf.host_cache.lock()
                .map_err(|_| anyhow::anyhow!("GPU prefill expert RAM cache lock poisoned"))?;
            if let Some(tr) = &self.token_routing {
                host_cache.set_consistency(tr.layer_consistency(c.n_layer));
                // C2: per-layer consistency (intersection) protection mask. The
                // prediction/union set is deliberately not protected (see `tier`).
                let prot = (0..c.n_layer).map(|l| tr.consistency_mask(l, toks)).collect();
                host_cache.set_protect(prot);
            }
            host_cache.begin_chunk(toks.len())
        };
        let (g, mut t, n, eps, ne) = (&self.gpu, toks.len(), pf.chunk, c.eps, c.n_embd);
        let d = c.hc_n.max(1) * ne;
        g.put(emb, &pf.slab, 0)?;
        if !pe.is_empty() { g.put(pe, &pf.slab, pf.lay.emb)?; }
        g.put(&[self.pos as i32, 0], &self.step_buf, 0)?;
        let mut step = self.step_buf.addr();
        let base = pf.slab.addr();
        let at = |o: usize| base + 4 * o as u64;
        let l = &pf.lay;
        let (x, h) = (at(0), at(l.h));
        let ha = HcAt { xn: at(l.xn), lo: at(l.lo), hg: at(l.hg), inj: at(l.inj), h, hq: 0, emb: at(l.emb), pk: at(l.pk), pq: at(l.pq), pg: at(l.pg),
                        pn: at(l.pn), pv: at(l.pv), gemm: true };
        // Hyper-connections: the mixer and FFN outputs go to "mix" and "mo", then into the streams.
        let (mx, mact) = if c.hc_n > 0 { (at(l.mix), 0) } else { (x, 2) };
        let (dk, nk, nv) = (c.ssm_d_state, c.ssm_n_group, c.ssm_dt_rank);
        let (dv, channels) = (c.ssm_dv(), c.ssm_d_inner + 2 * nk * dk);
        let (hd, nh, nkv) = (c.head_dim, c.n_head, c.n_head_kv);
        let kv = nkv * hd;
        // The draft head runs as one more layer on the chunk while it is in step with the main model.
        let fill = self.mtp_synced();
        let mtp = self.mtp.as_ref().filter(|_| fill);
        // STRATA_NAN_TRACE: one line per layer with the largest magnitude of each intermediate buffer, and the first
        // row of one holding a non-finite value. Each probe waits for the GPU: diagnostic runs only.
        let trace = std::env::var_os("STRATA_NAN_TRACE").is_some();
        let mut tr = String::new();
        let probe = |tr: &mut String, what: &str, b: &Buf, o: u64, w: usize, rows: usize| -> anyhow::Result<()> {
            if !trace { return Ok(()); }
            let v: Vec<f32> = g.get(b, ((o - b.addr()) / 4) as usize, rows * w)?;
            let max = v.iter().filter(|a| a.is_finite()).fold(0f32, |m, a| m.max(a.abs()));
            tr.push_str(&format!(" {what} {max:.2e}"));
            if let Some(i) = v.iter().position(|a| !a.is_finite()) { tr.push_str(&format!(" (non-finite row {})", i / w)); }
            Ok(())
        };
        for (il, ly) in self.layers.iter().enumerate().map(|(i, ly)| (self.first + i, ly)).chain(mtp.map(|mh| (c.n_layer, &mh.layer))) {
            if let Some(mh) = mtp.filter(|_| il == c.n_layer) {
                // Its entry for token i pairs the token with the hidden state before it (`mtp_run`): h gets the last
                // hidden state of the previous chunk, then rows 0..t - 1; the first token of the sequence has none.
                let skip = (self.pos == 0) as usize;
                if skip == 0 { g.dtod(&self.x, 0, &pf.slab, 4 * l.h, 4 * ne)?; }
                g.dtod(&pf.slab, 0, &pf.slab, 4 * (l.h + (1 - skip) * ne), 4 * (t - 1) * ne)?;
                g.dtod(&pf.slab, 4 * (t - 1) * ne, &self.x, 0, 4 * ne)?;
                let emb: Vec<f32> = toks[skip..].iter().flat_map(|&tk| self.m.embed(tk)).collect();
                g.put(&emb, &pf.slab, 0)?;
                t -= skip;
                g.rms_norm(x, ne, mh.enorm.ptr, at(l.a), 2 * ne, ne, t, eps)?;
                g.rms_norm(h, ne, mh.hnorm.ptr, at(l.a + ne), 2 * ne, ne, t, eps)?;
                g.gemm_dense(&mh.eh_proj, t, at(l.a), 2 * ne, x, ne, 0)?;
                g.put(&[mh.pos as i32, 0], &self.step_buf, 2)?;
                step += 8;
            }
            if let Some(pl) = &ly.ple { self.ple(pl, x, step, t, &ha)?; }
            match (&ly.attn_norm, &ly.hc) {
                (Some(an), _) => g.rms_norm(x, ne, an.ptr, h, ne, ne, t, eps)?,
                (None, Some(hr)) => self.hc_read(&hr[0], x, t, &ha)?,
                _ => unreachable!(),
            }
            match &ly.mixer {
                Mixer::Delta { qkv, gate, beta, alpha, dt_bias, a, conv_w, norm, out, conv, s, .. } => {
                    let (eb, eg) = (at(l.e), at(l.e + n * nv));
                    g.gemm_dense(qkv, t, h, ne, at(l.a), channels, 0)?;
                    // The qkv GEMM above already quantized `h`; when it did not (IQ2_XS, or a shape the int8 GEMM
                    // skips), prepare it now so beta, alpha and the gate can all reuse one copy.
                    g.prequant_gemm_x(gate, t, h, ne)?;
                    g.conv_seq(at(l.a), conv[self.cur].addr(), conv_w.ptr, at(l.c), channels, c.ssm_conv, t, 0, 0)?;
                    // The beta and alpha GEMMs (16 blocks each) run beside the norm: batch launches alternate streams.
                    g.batch(&mut || {
                        g.gemm_dense(beta, t, h, ne, eb, nv, SAME_X)?;
                        g.l2_norm(at(l.c), dk, 2 * nk, channels, t * 2 * nk, eps)?;
                        g.gemm_dense(alpha, t, h, ne, eg, nv, SAME_X)
                    })?;
                    g.ssm_gates(eb, eg, dt_bias.ptr, a.ptr, t * nv, nv)?;
                    probe(&mut tr, "qkv", &pf.slab, at(l.a), channels, t)?;
                    probe(&mut tr, "conv", &pf.slab, at(l.c), channels, t)?;
                    probe(&mut tr, "beta", &pf.slab, eb, nv, t)?;
                    probe(&mut tr, "alpha", &pf.slab, eg, nv, t)?;
                    // The z GEMM runs beside the scan, on the SMs its last wave of blocks leaves idle.
                    g.batch(&mut || {
                        g.deltanet(at(l.c), s[self.cur].addr(), eb, eg, 0, 0, at(l.d), nk, nv, c.k_interleave, dk, dv, 1.0 / (dk as f32).sqrt(), t, channels, 0, 0)?;
                        g.gemm_dense(gate, t, h, ne, at(l.b), c.ssm_d_inner, SAME_X)
                    })?;
                    probe(&mut tr, "deltanet", &pf.slab, at(l.d), c.ssm_d_inner, t)?;
                    probe(&mut tr, "state", &s[self.cur], s[self.cur].addr(), s[self.cur].len() / 4, 1)?;
                    probe(&mut tr, "z", &pf.slab, at(l.b), c.ssm_d_inner, t)?;
                    if ly.hc.is_some() { g.norm_gate_sig(at(l.d), norm.ptr, at(l.b), dv, t * nv, eps, 0)?; } else { g.norm_gate(at(l.d), norm.ptr, at(l.b), dv, t * nv, eps)?; }
                    g.gemm_dense(out, t, at(l.d), c.ssm_d_inner, mx, ne, mact)?;
                }
                Mixer::Attn { q, k, v, qn, kn, o, kc, vc, qsa } => {
                    let (kraw, kk, vv) = (at(l.c), at(l.c + n * kv), at(l.c + 2 * n * kv));
                    let qs = if c.q_gate { 2 * hd } else { hd };
                    g.gemm_dense(q, t, h, ne, at(l.a), nh * qs, 0)?;
                    g.rms_norm(at(l.a), qs, qn.ptr, at(l.b), hd, hd, t * nh, eps)?;
                    g.rope_neox(at(l.b), hd, t * nh, nh, c.n_rot, step, c.rope_base)?;
                    g.gemm_dense(k, t, h, ne, kraw, kv, SAME_X)?;
                    g.gemm_dense(v, t, h, ne, vv, kv, SAME_X)?;
                    g.rms_norm(kraw, hd, kn.ptr, kk, hd, hd, t * nkv, eps)?;
                    g.rope_neox(kk, hd, t * nkv, nkv, c.n_rot, step, c.rope_base)?;
                    g.kv_store(kk, vv, kc[self.cur].addr(), vc[self.cur].addr(), kv, t, step, hd)?;
                    // QSA: region c (free after kv_store) holds the raw indexer keys, raw queries and normed queries.
                    let (qm, mw) = match (qsa, &self.qsa) {
                        (Some(a), Some(b)) => {
                            let (ik, iq, iqn) = (at(l.c), at(l.c + n * super::QSA_D), at(l.c + n * (1 + super::QSA_H) * super::QSA_D));
                            let (dq, nq) = (super::QSA_D, super::QSA_H * super::QSA_D);
                            g.gemm_dense(&a.kp, t, h, ne, ik, dq, 0)?;
                            g.gemm_dense(&a.qp, t, h, ne, iq, nq, SAME_X)?;
                            g.qsa_pool(ik, a.ring, a.pool.addr(), a.kn.ptr, step, t, c.n_rot, c.rope_base, eps)?;
                            // Rows with at most `top` complete blocks attend densely: no mask while all of them do.
                            if self.pos + t >= super::QSA_POOL * (b.top + 1) {
                                g.rms_norm(iq, dq, a.qn.ptr, iqn, dq, dq, t * super::QSA_H, eps)?;
                                g.rope_neox(iqn, dq, t * super::QSA_H, super::QSA_H, c.n_rot, step, c.rope_base)?;
                                for r0 in (0..t).step_by(super::QSA_ROWS) {
                                    g.qsa_select(iqn + (4 * r0 * nq) as u64, a.pool.addr(), b.score.addr(), 0, b.mask.addr() + (4 * r0 * b.mw) as u64,
                                                 step, r0, super::QSA_ROWS.min(t - r0), b.nbs, b.top, 0, b.mw)?;
                                }
                                (b.mask.addr(), b.mw)
                            } else { (0, 0) }
                        }
                        _ => (0, 0),
                    };
                    g.attn_prefill_m(at(l.b), kc[self.cur].addr(), vc[self.cur].addr(), if c.q_gate { at(l.a) } else { 0 }, at(l.d), step, t, nh, nkv, hd, 1.0 / (hd as f32).sqrt(), qm, mw)?;
                    probe(&mut tr, "q", &pf.slab, at(l.a), nh * qs, t)?;
                    probe(&mut tr, "k", &pf.slab, kk, kv, t)?;
                    probe(&mut tr, "v", &pf.slab, vv, kv, t)?;
                    probe(&mut tr, "attn", &pf.slab, at(l.d), nh * hd, t)?;
                    g.gemm_dense(o, t, at(l.d), nh * hd, mx, ne, mact)?;
                }
            }

            probe(&mut tr, "| mixer", &pf.slab, x, d, t)?;
            match (&ly.post_norm, &ly.hc) {
                (Some(pn), _) => g.rms_norm(x, ne, pn.ptr, h, ne, ne, t, eps)?,
                (None, Some(hr)) => {
                    g.hc_write(x, mx, at(l.inj), ne, c.hc_n, t, false)?;
                    self.hc_read(&hr[1], x, t, &ha)?;
                }
                _ => unreachable!(),
            }
            match &ly.ffn {
                Ffn::Dense { gate, up, down } => {
                    g.gemm_dense(gate, t, h, ne, at(l.ge), c.n_ff, 0)?;
                    g.gemm_dense(up, t, h, ne, at(l.ge), c.n_ff, 1 | SAME_X)?;
                    probe(&mut tr, "gate_up", &pf.slab, at(l.ge), c.n_ff, t)?;
                    g.gemm_dense(down, t, at(l.ge), c.n_ff, x, ne, 2)?;
                }
                Ffn::Moe { router, shared } => {
                    let k = c.n_expert_used;
                    let ids_d = pf.ints.addr();
                    // The 1-row shared gate_inp GEMM runs beside the router GEMM (both float: one launch each).
                    g.batch(&mut || {
                        g.gemm_dense(router, t, h, ne, at(l.rl), c.n_expert, 0)?;
                        if let Some(s) = shared { g.gemm_dense(&s.gate_inp, t, h, ne, at(l.sg), 1, SAME_X)?; }
                        Ok(())
                    })?;
                    g.router_topk(at(l.rl), c.n_expert, k, c.expert_weights_scale, ids_d, at(l.wts), t)?;
                    let (sh, sg) = match shared {
                        Some(s) => {
                            g.gemm_dense(&s.gate, t, h, ne, at(l.shg), c.n_ff_shexp, SAME_X)?;
                            g.gemm_dense(&s.up, t, h, ne, at(l.shg), c.n_ff_shexp, 1 | SAME_X)?;
                            (at(l.shy), at(l.sg))
                        }
                        None => (0, 0),
                    };
                    probe(&mut tr, "router", &pf.slab, at(l.rl), c.n_expert, t)?;
                    probe(&mut tr, "weights", &pf.slab, at(l.wts), k, t)?;
                    let ids = g.get::<i32>(&pf.ints, 0, t * k)?;
                    let cache = self.cache.as_mut().unwrap();
                    cache.record(il, &ids);
                    if let Some(tr) = self.token_routing.as_mut() { tr.record_routing(il, &toks[..t], &ids[..t * k], k); }
                    // Sort (token, j) pairs by expert: slot s holds token xrow[s]; inv maps t * k + j to its slot.
                    let mut off = vec![0usize; c.n_expert + 1];
                    for &e in &ids { off[e as usize + 1] += 1; }
                    for e in 0..c.n_expert { off[e + 1] += off[e]; }
                    let (mut fill, mut xrow, mut inv) = (off.clone(), vec![0i32; t * k], vec![0i32; t * k]);
                    for (i, &e) in ids.iter().enumerate() {
                        let s = &mut fill[e as usize];
                        xrow[*s] = (i / k) as i32;
                        inv[i] = *s as i32;
                        *s += 1;
                    }
                    g.put(&xrow, &pf.ints, n * k)?;
                    g.put(&inv, &pf.ints, 2 * n * k)?;
                    let lx = cache.layer(il).unwrap();
                    let (pool, stride) = cache.base(lx);
                    let used = |e: &usize| off[e + 1] > off[*e];
                    let missed: Vec<usize> = (0..c.n_expert).filter(used).filter(|&e| cache.slot(il, e) < 0).collect();
                    // Pre-load the predicted experts (live OR table) that are not already in the
                    // RAM cache, so the staging copies below read them from RAM rather than disk.
                    // `to_load` = prediction & !residency (AND-NOT, not XOR).
                    if let Some(tr) = &self.token_routing {
                        let to_load_mask = tr.predicted_mask(il, &toks[..t]);
                        let freq = tr.expert_frequency(il, &toks[..t]);
                        let mut hc = pf.host_cache.lock()
                            .map_err(|_| anyhow::anyhow!("GPU prefill expert RAM cache lock poisoned"))?;
                        let residency = hc.residency(il, c.n_expert);
                        // Sort the candidate set by demand (frequency desc, id as tiebreak),
                        // so a saturated budget fills with the most-needed experts. ID order
                        // loaded low-frequency experts first and starved the hot ones (e.g.
                        // layer 47: 25.5% hit at 58 experts vs 79.5% achievable).
                        let mut to_load: Vec<usize> = to_load_mask.iter().enumerate()
                            .flat_map(move |(w, word)| (0..64).filter_map(move |b| if word >> b & 1 != 0 { Some(w * 64 + b) } else { None }))
                            .filter(|&e| e < c.n_expert)
                            .filter(|&e| {
                                let w = e / 64;
                                residency[w] >> (e % 64) & 1 == 0
                            })
                            .collect();
                        to_load.sort_by_key(|&e| std::cmp::Reverse((freq[e], std::cmp::Reverse(e))));
                        if !to_load.is_empty() { hc.preload(self.m, il, &to_load)?; }
                    }
                    // The mapped views the prefetch thread reads. They outlive the prefetch reads and the staging
                    // copies, so they are dropped only after `beside_join` below: dropping a `MappedBytes` unmaps
                    // its view, which would release physical pages the GPU upload is still reading.
                    let mut views: Vec<MappedBytes> = Vec::new();
                    if let Some(pre) = &pre {
                        let mut ranges = Vec::new();
                        for grp in missed.chunks(STAGE) {
                            for name in MATS {
                                for run in grp.chunk_by(|a, b| *b == a + 1) {
                                    let data = self.m.w.map_expert_range(il, name, run[0], run.len())?;
                                    ranges.push((data.as_ptr() as usize, data.len()));
                                    views.push(data);
                                }
                            }
                        }
                        pre.send(ranges);
                    }
                    // Staged groups of at most STAGE missed experts, group j in staging third j % 3. Staging holds
                    // each matrix in its own section (all gates, then ups, then downs), so experts with consecutive
                    // ids are contiguous there as in the weights and copy in one call.
                    let sp = pf.staging.addr();
                    let half = |j: usize| j % 3 * pf.staging.len() / 3;
                    let staged: Vec<Vec<(usize, [u64; 3])>> = missed.chunks(STAGE).enumerate()
                        .map(|(j, grp)| grp.iter().enumerate().map(|(i, &e)| (e, lx.mats.each_ref().map(|m| sp + (half(j) + STAGE * m.off + i * m.bytes) as u64))).collect())
                        .collect();
                    // The resident experts split into one chunk per staged group, of about equal positions, so the
                    // copies run beside them too: step s runs group s - 2 and chunk s beside the copy of group s
                    // (into the third group s - 3 used, done in step s - 1). A step waits only for the copies of the
                    // step two before it (`beside`), so each copy has two steps of GEMMs to hide behind.
                    // Without overlap (Vulkan without a transfer queue) one chunk: more, smaller GEMMs only cost there.
                    let ns = if g.beside_overlaps() { staged.len().max(1) } else { 1 };
                    let res: Vec<(usize, [u64; 3])> = (0..c.n_expert).filter(used)
                        .filter_map(|e| { let s = cache.slot(il, e); (s >= 0).then(|| (e, lx.mats.each_ref().map(|m| pool + (s as usize * stride + m.off) as u64))) }).collect();
                    let total = res.iter().map(|r| off[r.0 + 1] - off[r.0]).sum::<usize>().max(1);
                    let mut chunks = vec![Vec::new(); ns];
                    let mut acc = 0;
                    for &r in &res {
                        chunks[(acc * ns / total).min(ns - 1)].push(r);
                        acc += off[r.0 + 1] - off[r.0];
                    }
                    let mut chunks = chunks.into_iter();
                    let mut groups = vec![];
                    // Per step: its groups and the staged group it copies.
                    let mut steps = vec![];
                    for s in 0..staged.len() + 2 {
                        let g0 = groups.len();
                        groups.extend(s.checked_sub(2).and_then(|j| staged.get(j)).cloned());
                        groups.extend(chunks.next());
                        steps.push((g0..groups.len(), (s < staged.len()).then_some(s)));
                    }
                    let mut tiles = Vec::new();
                    let push = |tiles: &mut Vec<u64>, grp: &[(usize, [u64; 3])], mi: usize, base: usize| {
                        let start = tiles.len() / 2;
                        for &(e, w) in grp {
                            for m0 in (off[e]..off[e + 1]).step_by(GEMM_BM) {
                                tiles.extend([w[mi], (m0 - base) as u64 | (GEMM_BM.min(off[e + 1] - m0) as u64) << 32]);
                            }
                        }
                        (start, tiles.len() / 2 - start)
                    };
                    // Per step: the gate and up tiles of all its groups (one launch each: more blocks, less tail), then
                    // per group its down tiles and the slot rows lo..hi it covers. The down GEMM reads rows lo..hi of
                    // `ge` only, so its tiles count from lo and it quantizes just those rows.
                    let mut spans = Vec::new();
                    for (gs, _) in &steps {
                        let gu = [0, 1].map(|mi| { let s = tiles.len() / 2; for gi in gs.clone() { push(&mut tiles, &groups[gi], mi, 0); } (s, tiles.len() / 2 - s) });
                        let downs: Vec<_> = gs.clone().filter(|&gi| !groups[gi].is_empty()).map(|gi| {
                            let grp = &groups[gi];
                            let (lo, hi) = (off[grp[0].0], off[grp[grp.len() - 1].0 + 1]);
                            (push(&mut tiles, grp, 2, lo), lo, hi)
                        }).collect();
                        spans.push((gu, downs));
                    }
                    anyhow::ensure!(8 * tiles.len() <= pf.tiles.len(), "prefill tile list overflow");
                    g.put(&tiles, &pf.tiles, 0)?;
                    let tp = |s: usize| pf.tiles.addr() + 16 * s as u64;
                    let xrow_d = pf.ints.addr() + 4 * (n * k) as u64;
                    let [mg, mu, md] = lx.mats;
                    let copy = |j: usize| -> anyhow::Result<()> {
                        let grp = &staged[j];
                        let cached = if self.m.w.g.uses_mmap() {
                            Vec::new()
                        } else {
                            let lookup_started = profile.as_ref().map(|_| Instant::now());
                            let mut host_cache = pf.host_cache.lock()
                                .map_err(|_| anyhow::anyhow!("GPU prefill expert RAM cache lock poisoned"))?;
                            let cached = grp.iter().map(|&(e, _)| host_cache.expert(self.m, il, e, true))
                                .collect::<anyhow::Result<Vec<_>>>()?;
                            if let (Some(profile), Some(started)) = (&profile, lookup_started) {
                                profile.cache_lookup.set(profile.cache_lookup.get() + started.elapsed());
                            }
                            cached
                        };
                        let upload = |data: &[u8], dst: usize| -> anyhow::Result<()> {
                            if let Some(profile) = &profile {
                                let started = Instant::now();
                                let result = g.htod(data, &pf.staging, dst);
                                profile.host_upload.set(profile.host_upload.get() + started.elapsed());
                                result
                            } else {
                                g.htod(data, &pf.staging, dst)
                            }
                        };
                        let expert_range = |name: &str, first: usize, count: usize| -> anyhow::Result<std::borrow::Cow<'_, [u8]>> {
                            if let Some(profile) = &profile {
                                let started = Instant::now();
                                let result = self.m.w.expert_range(il, name, first, count);
                                profile.expert_range.set(profile.expert_range.get() + started.elapsed());
                                result
                            } else {
                                self.m.w.expert_range(il, name, first, count)
                            }
                        };
                        let mut uncached_bytes = 0usize;
                        for (name, mat) in MATS.iter().zip(&lx.mats) {
                            for (run_start, run_len) in expert_run_offsets(grp) {
                                let run = &grp[run_start..run_start + run_len];
                                let dst = half(j) + STAGE * mat.off + run_start * mat.bytes;
                                if self.m.w.g.uses_mmap() {
                                    let data = expert_range(name, run[0].0, run.len())?;
                                    upload(data.as_ref(), dst)?;
                                    continue;
                                }
                                let mi = MATS.iter().position(|&n| n == *name).unwrap();
                                let mut i = 0;
                                while i < run.len() {
                                    if let Some(expert) = &cached[run_start + i] {
                                        upload(expert.mats[mi].as_ref(), dst + i * mat.bytes)?;
                                        i += 1;
                                    } else {
                                        let start = i;
                                        while i < run.len() && cached[run_start + i].is_none() { i += 1; }
                                        let data = expert_range(name, run[start].0, i - start)?;
                                        uncached_bytes += data.len();
                                        upload(data.as_ref(), dst + start * mat.bytes)?;
                                    }
                                }
                            }
                        }
                        if uncached_bytes > 0 {
                            pf.host_cache.lock()
                                .map_err(|_| anyhow::anyhow!("GPU prefill expert RAM cache lock poisoned"))?
                                .record_uncached_bytes(uncached_bytes);
                        }
                        Ok(())
                    };
                    // `beside` starts a step's copies after the GEMMs before it. They queue last, so the GEMMs need not
                    // wait for the host to stage them.
                    for ((_, cp), (gu, downs)) in steps.iter().zip(&spans) {
                        g.beside(&mut || {
                            if gu[0].1 > 0 {
                                g.gemm(mg.ty, mg.rows, mg.cols, tp(gu[0].0), gu[0].1, h, ne, t, xrow_d, at(l.ge), c.n_ff_exp, SAME_X)?;
                                g.gemm(mu.ty, mu.rows, mu.cols, tp(gu[1].0), gu[1].1, h, ne, t, xrow_d, at(l.ge), c.n_ff_exp, 1 | SAME_X)?;
                            }
                            for &((s0, n0), lo, hi) in downs {
                                g.gemm(md.ty, md.rows, md.cols, tp(s0), n0, at(l.ge) + (4 * lo * c.n_ff_exp) as u64, c.n_ff_exp, hi - lo, 0, at(l.ye) + (4 * lo * ne) as u64, ne, 0)?;
                            }
                            if let Some(j) = *cp { copy(j)?; }
                            Ok(())
                        })?;
                    }
                    g.beside_join()?;
                    // The shared down runs last, so the routed gate and up above reuse the int8 h of the shared up.
                    if let Some(s) = shared { g.gemm_dense(&s.down, t, at(l.shg), c.n_ff_shexp, at(l.shy), ne, 0)?; }
                    // Expert rows are slots (token, j) sorted by expert, not tokens.
                    probe(&mut tr, "experts_gate_up", &pf.slab, at(l.ge), c.n_ff_exp, t * k)?;
                    probe(&mut tr, "experts_down", &pf.slab, at(l.ye), ne, t * k)?;
                    if shared.is_some() {
                        probe(&mut tr, "shared_gate_up", &pf.slab, at(l.shg), c.n_ff_shexp, t)?;
                        probe(&mut tr, "shared_down", &pf.slab, at(l.shy), ne, t)?;
                        probe(&mut tr, "shared_gate", &pf.slab, at(l.sg), 1, t)?;
                    }
                    // "mo" is zero: hc_write clears it after use.
                    let fx = if ly.hc.is_some() { at(l.mo) } else { x };
                    g.moe_combine_seq(fx, at(l.ye), at(l.wts), pf.ints.addr() + 4 * (2 * n * k) as u64, k, sh, sg, ne, t)?;
                    if ly.hc.is_some() { g.hc_write(x, fx, at(l.inj), ne, c.hc_n, t, true)?; }
                }
            }
            probe(&mut tr, "| ffn", &pf.slab, x, d, t)?;
            if trace { eprintln!("nan-trace pos {} layer {il}:{tr}", self.pos); tr.clear(); }
            g.kick()?;
        }
        let final_sync_wait = if profile.is_some() {
            let started = Instant::now();
            g.sync()?;
            started.elapsed()
        } else {
            Duration::ZERO
        };
        if let Some(profile) = &profile {
            eprintln!(
                "GPU prefill profile ({} tokens): GPU-synced chunk elapsed {:.1} ms, final sync wait {:.1} ms, expert-cache lookup/map {:.1} ms, expert-range access/read {:.1} ms, host upload calls {:.1} ms",
                toks.len(),
                profile.started.elapsed().as_secs_f64() * 1000.0,
                final_sync_wait.as_secs_f64() * 1000.0,
                profile.cache_lookup.get().as_secs_f64() * 1000.0,
                profile.expert_range.get().as_secs_f64() * 1000.0,
                profile.host_upload.get().as_secs_f64() * 1000.0,
            );
        }
        let cache_after = pf.host_cache.lock()
            .map_err(|_| anyhow::anyhow!("GPU prefill expert RAM cache lock poisoned"))?
            .snapshot();
        if cache_after.capacity > 0 || cache_before.capacity > 0 {
            let before = cache_before.counters;
            let after = cache_after.counters;
            eprintln!(
                "Prefill RAM cache chunk ({} tokens): hits {}, misses {}, insertions {}, evictions {}, cached GGUF {:.0} MB, uncached GGUF {:.0} MB; capacity {:.0}->{:.0} MB (range {:.0}-{:.0}, {} changes), in use {:.0} MB ({} experts), faults +{}",
                toks.len(),
                after.hits - before.hits,
                after.misses - before.misses,
                after.insertions - before.insertions,
                after.evictions - before.evictions,
                (after.cached_bytes_loaded - before.cached_bytes_loaded) as f64 / 1048576.0,
                (after.uncached_bytes_read - before.uncached_bytes_read) as f64 / 1048576.0,
                cache_before.capacity as f64 / 1048576.0,
                cache_after.capacity as f64 / 1048576.0,
                cache_after.min_capacity as f64 / 1048576.0,
                cache_after.max_capacity as f64 / 1048576.0,
                after.capacity_changes - before.capacity_changes,
                cache_after.used as f64 / 1048576.0,
                cache_after.experts,
                cache_after.faults_delta,
            );
        }
        if profile.is_some() {
            for il in 0..cache_after.layer_count.max(cache_before.layer_count) {
                let Some(&expert_bytes) = cache_after.layer_expert_bytes.get(&il)
                    .or_else(|| cache_before.layer_expert_bytes.get(&il)) else { continue };
                let before = cache_before.layer_counters.get(&il).cloned().unwrap_or_default();
                let after = cache_after.layer_counters.get(&il).cloned().unwrap_or_default();
                let used = cache_after.layer_used.get(&il).copied().unwrap_or(0);
                let experts = cache_after.layer_experts.get(&il).copied().unwrap_or(0);
                let quota = cache_after.layer_quota.get(&il).copied().unwrap_or(0);
                let total_experts = self.m.cfg.n_expert;
                let hits = after.hits - before.hits;
                let misses = after.misses - before.misses;
                let total = hits + misses;
                // Distinct experts accessed (cumulative): union of hit and miss sets (an expert
                // can appear in both: cached early, evicted mid-run, requested again).
                let mut u_total = after.unique_hits.clone();
                u_total.extend(after.unique_misses.iter().copied());
                let u_total = u_total.len();
                let hit_rate = if total > 0 { hits as f64 / total as f64 * 100.0 } else { 0.0 };
                // Working-set pressure: how many distinct experts the layer's working set needs
                // vs how many the current share can hold.
                let fits = if u_total > 0 { experts as f64 / u_total as f64 * 100.0 } else { 100.0 };
                eprintln!(
                    "Prefill RAM cache layer {il}: expert {:.2} MB, bank {:.0} MB, share {:.0} MB, cached {:.0} MB ({}/{}) ; req {} (hit {:.1}%), unique {} accessed, cache fits {:.0}%",
                    expert_bytes as f64 / 1048576.0,
                    expert_bytes as f64 * total_experts as f64 / 1048576.0,
                    quota as f64 / 1048576.0,
                    used as f64 / 1048576.0,
                    experts,
                    total_experts,
                    total,
                    hit_rate,
                    u_total,
                    fits,
                );
            }
        }
        match self.mtp.as_mut() {
            Some(mh) if fill => mh.pos += t,
            // Row 0 of `x` gets the last hidden state (with the draft head, the copy above did it).
            _ => g.dtod(&pf.slab, 4 * (t - 1) * d, &self.x, 0, 4 * d)?,
        }
        self.pos += toks.len();
        let Some(nx) = self.next.as_deref_mut() else { return Ok(()) };
        let xs = self.gpu.get::<f32>(&self.pf.as_ref().unwrap().slab, 0, toks.len() * d)?;
        nx.prefill_chunk(toks, Some((&xs, pe)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cached_expert() -> CachedExpert {
        CachedExpert {
            mats: [MappedBytes::owned(vec![0]), MappedBytes::owned(vec![0]), MappedBytes::owned(vec![0])],
            bytes: 4,
            _reservation: None,
        }
    }

    fn insert(cache: &mut PrefillExpertCache, key: (usize, usize)) {
        cache.entries.insert(key, Arc::new(cached_expert()));
        cache.lru.push_back(key);
        cache.used += 4;
        *cache.layer_used.entry(key.0).or_default() += 4;
    }

    #[test]
    fn token_routing_record_routing_accumulates_union_cons_freq() {
        // Two occurrences of the same token at layer 0 with different expert sets. Each row
        // carries exactly `k = 10` real experts, as the router outputs (no padding):
        // a = {1, 2, 3, 4, 7, 8, 9, 11, 12, 66}; b = {1, 2, 3, 4, 7, 8, 9, 5, 12, 66}.
        let mut tr = TokenRouting::new(130, 10);
        let toks = [7u32, 7];
        let ids_a = [1i32, 2, 3, 4, 7, 8, 9, 11, 12, 66];
        let ids_b = [1i32, 2, 3, 4, 7, 8, 9, 5, 12, 66];
        tr.record_routing(0, &toks, &ids_a, 10);
        tr.record_routing(0, &toks, &ids_b, 10);

        let entry = &tr.map[&7][0];
        // Union: a | b = {1,2,3,4,5,7,8,9,11,12,66}
        let union = [(1u64 << 1) | (1 << 2) | (1 << 3) | (1 << 4) | (1 << 5) | (1 << 7) | (1 << 8) | (1 << 9) | (1 << 11) | (1 << 12), 1u64 << 2, 0];
        assert_eq!(entry.union_set, union);
        // Cons: a & b = {1,2,3,4,7,8,9,12,66}
        let cons = [(1u64 << 1) | (1 << 2) | (1 << 3) | (1 << 4) | (1 << 7) | (1 << 8) | (1 << 9) | (1 << 12), 1u64 << 2, 0];
        assert_eq!(entry.cons_set, cons);
        // Freq: the 9 shared experts appear twice, {5, 11} once, everything else 0.
        for e in [1, 2, 3, 4, 7, 8, 9, 12, 66] { assert_eq!(entry.freq[e], 2); }
        assert_eq!(entry.freq[5], 1);
        assert_eq!(entry.freq[11], 1);
        assert!(entry.freq.iter().enumerate().all(|(e, &c)| matches!(e, 1 | 2 | 3 | 4 | 5 | 7 | 8 | 9 | 11 | 12 | 66) || c == 0));
        // Consistency score: popcount(cons) / top_k = 9/10
        assert!((TokenRouting::consistency_score(entry, 10) - 0.9).abs() < 1e-9);
    }

    #[test]
    fn token_routing_merge_folds_union_and_cons_and_freq() {
        // Two routing tables for the same token; merge should OR-union, AND-cons, sum freq.
        let n_expert = 130;
        let a = vec![1u64 << 3, 1u64 << 2, 0];
        let b = vec![1u64 << 3 | 1u64 << 5, 1u64 << 2, 0];

        let mut tr1 = TokenRouting::new(n_expert, 10);
        tr1.map.insert(7, vec![TokenLayerEntry {
            union_set: a.clone(),
            cons_set: a.clone(),
            freq: { let mut f = vec![0u32; n_expert]; f[3] = 2; f[66] = 2; f },
        }]);

        let mut tr2 = TokenRouting::new(n_expert, 10);
        tr2.map.insert(7, vec![TokenLayerEntry {
            union_set: b.clone(),
            cons_set: b.clone(),
            freq: { let mut f = vec![0u32; n_expert]; f[3] = 1; f[5] = 1; f[66] = 2; f },
        }]);

        tr1.merge(&tr2);
        let e = &tr1.map[&7][0];
        // Union: a | b = {3, 5, 66}
        assert_eq!(e.union_set, [a[0] | b[0], a[1] | b[1], a[2] | b[2]]);
        // Cons: a & b = {3, 66}
        assert_eq!(e.cons_set, [a[0] & b[0], a[1] & b[1], a[2] & b[2]]);
        // Freq: 3=3, 5=1, 66=4
        assert_eq!(e.freq[3], 3);
        assert_eq!(e.freq[5], 1);
        assert_eq!(e.freq[66], 4);
    }

    #[test]
    fn token_routing_consistency_mask_and_predicted_mask() {
        let n_expert = 130;
        let n_words = (n_expert + 63) / 64;
        let a = vec![1u64 << 3, 1u64 << 2, 0];
        let b = vec![1u64 << 3 | 1u64 << 5, 1u64 << 2, 0];

        let mut tr = TokenRouting::new(n_expert, 10);
        tr.map.insert(7, vec![TokenLayerEntry { union_set: a.clone(), cons_set: a.clone(), freq: vec![0; n_expert] }]);
        tr.map.insert(42, vec![TokenLayerEntry { union_set: b.clone(), cons_set: b.clone(), freq: vec![0; n_expert] }]);

        let toks = [7u32, 42];
        // predicted_mask: a | b = {3, 5, 66}
        let pm = tr.predicted_mask(0, &toks);
        assert_eq!(pm, [a[0] | b[0], a[1] | b[1], a[2] | b[2]]);
        // consistency_mask: a & b... no, OR of cons: a | b = {3, 5, 66}
        let cm = tr.consistency_mask(0, &toks);
        assert_eq!(cm, [a[0] | b[0], a[1] | b[1], a[2] | b[2]]);
        // A token not in the map contributes nothing
        let cm2 = tr.consistency_mask(0, &[99]);
        assert_eq!(cm2, vec![0u64; n_words]);
        // A layer with no entry contributes nothing
        let cm3 = tr.consistency_mask(5, &toks);
        assert_eq!(cm3, vec![0u64; n_words]);
    }

    #[test]
    fn token_routing_expert_frequency_dedups_tokens() {
        let n_expert = 130;
        let mut tr = TokenRouting::new(n_expert, 10);
        // Token 7: expert 3 appears 2 times, expert 66 appears 2 times.
        tr.map.insert(7, vec![{
            let mut f = vec![0u32; n_expert]; f[3] = 2; f[66] = 2;
            TokenLayerEntry { union_set: vec![0; 3], cons_set: vec![0; 3], freq: f }
        }]);
        // Token 42: expert 3 appears 1 time, expert 5 appears 1 time, expert 66 appears 2 times.
        tr.map.insert(42, vec![{
            let mut f = vec![0u32; n_expert]; f[3] = 1; f[5] = 1; f[66] = 2;
            TokenLayerEntry { union_set: vec![0; 3], cons_set: vec![0; 3], freq: f }
        }]);

        let freq = tr.expert_frequency(0, &[7, 42]);
        assert_eq!(freq[3], 3);
        assert_eq!(freq[5], 1);
        assert_eq!(freq[66], 4);
        // Duplicate token ids count once
        let freq = tr.expert_frequency(0, &[7, 7, 42, 99]);
        assert_eq!(freq[3], 3);
        assert_eq!(freq[5], 1);
        assert_eq!(freq[66], 4);
        // A layer with no entry yields zeros
        assert!(tr.expert_frequency(5, &[7, 42]).iter().all(|&c| c == 0));
    }

    #[test]
    fn token_routing_serialize_roundtrip_strrtbl2() {
        // Two tokens, two layers each; verify union_set and cons_set survive the
        // round trip. Freq is NOT persisted (resets to zeros).
        let n_expert = 130;
        let a = vec![1u64 << 3, 1u64 << 2, 0];
        let b = vec![1u64 << 3 | 1u64 << 5, 1u64 << 2, 0];

        let mut tr = TokenRouting::new(n_expert, 10);
        // Token 7, layer 0: union = a|b, cons = a&b, freq = {3:3, 5:1, 66:4}
        let mut f0 = vec![0u32; n_expert]; f0[3] = 3; f0[5] = 1; f0[66] = 4;
        // Token 7, layer 1: union = a, cons = a, freq = {3:1}
        let mut f1 = vec![0u32; n_expert]; f1[3] = 1;
        // Token 42, layer 0: union = b, cons = b, freq = {3:4}
        let mut f2 = vec![0u32; n_expert]; f2[3] = 4;
        tr.map.insert(7, vec![
            TokenLayerEntry { union_set: [a[0] | b[0], a[1] | b[1], a[2] | b[2]].to_vec(), cons_set: [a[0] & b[0], a[1] & b[1], a[2] & b[2]].to_vec(), freq: f0 },
            TokenLayerEntry { union_set: a.clone(), cons_set: a.clone(), freq: f1 },
        ]);
        tr.map.insert(42, vec![
            TokenLayerEntry { union_set: b.clone(), cons_set: b.clone(), freq: f2 },
        ]);

        let rt = TokenRouting::deserialize(&tr.serialize()).expect("round trip");
        assert_eq!(rt.n_expert, 130);
        assert_eq!(rt.top_k, 10);
        assert_eq!(rt.map.len(), 2);
        let (r0, r1) = (&rt.map[&7][0], &rt.map[&7][1]);
        assert_eq!(r0.union_set, [a[0] | b[0], a[1] | b[1], a[2] | b[2]]);
        assert_eq!(r0.cons_set, [a[0] & b[0], a[1] & b[1], a[2] & b[2]]);
        assert_eq!(r1.union_set, a);
        assert_eq!(r1.cons_set, a);
        assert_eq!(rt.map[&42][0].union_set, b);
        assert_eq!(rt.map[&42][0].cons_set, b);
        // Freq is NOT persisted: all zeros after round-trip
        assert!(r0.freq.iter().all(|&c| c == 0));
        assert!(r1.freq.iter().all(|&c| c == 0));
        assert!(rt.map[&42][0].freq.iter().all(|&c| c == 0));
        // A bad magic (old STRRTBL1) yields None; an empty blob yields None.
        let mut old = vec![0u8; 20];
        old[0..8].copy_from_slice(b"STRRTBL1");
        assert!(TokenRouting::deserialize(&old).is_none());
        assert!(TokenRouting::deserialize(&[]).is_none());
    }

    #[test]
    fn prefill_cache_reclaims_borrowed_share_for_other_layers() {
        let mut cache = PrefillExpertCache { budget: 16, layer_quota: HashMap::from([(0, 8), (1, 8)]), ..Default::default() };
        insert(&mut cache, (0, 0));
        insert(&mut cache, (0, 1));
        insert(&mut cache, (0, 2));

        assert!(cache.make_room(8, 1));
        assert_eq!(cache.used, 8);
        assert_eq!(cache.layer_used.get(&0), Some(&8));
        assert!(!cache.entries.contains_key(&(0, 0)));
    }

    #[test]
    fn prefill_cache_layer_at_quota_evicts_its_own_tier3_to_admit_a_miss() {
        // C2: a layer that has grown to (or past) its own quota must be allowed
        // to evict its own tier-3 residents to admit a new expert — otherwise it
        // freezes and can never serve a miss again.
        let mut cache = PrefillExpertCache { budget: 16, layer_quota: HashMap::from([(0, 8), (1, 8)]), ..Default::default() };
        insert(&mut cache, (0, 0));
        insert(&mut cache, (0, 1));
        insert(&mut cache, (1, 0));
        insert(&mut cache, (1, 1));
        // Layer 0 is at 8/8 (at its quota); admitting 4 more would exceed it.
        assert!(cache.make_room(4, 0));
        // It evicted its own LRU-oldest tier-3 entry (0,0) to make room.
        assert_eq!(cache.layer_used.get(&0), Some(&4));
        assert_eq!(cache.layer_used.get(&1), Some(&8));
        assert!(!cache.entries.contains_key(&(0, 0)));
        assert!(cache.entries.contains_key(&(0, 1)));
        assert!(cache.entries.contains_key(&(1, 0)));
        assert!(cache.entries.contains_key(&(1, 1)));
    }

    #[test]
    fn prefill_cache_protected_tier1_survives_eviction_when_layer_over_quota() {
        // C2: when a layer over its quota must evict to admit a new expert,
        // tier-1 (consistency-protected) residents survive; only tier-3 is evicted.
        let mut cache = PrefillExpertCache { budget: 16, layer_quota: HashMap::from([(0, 8), (1, 8)]), ..Default::default() };
        insert(&mut cache, (0, 0));
        insert(&mut cache, (0, 1));
        insert(&mut cache, (1, 0));
        insert(&mut cache, (1, 1));
        // Protect (0,0) as tier 1: layer 0, word 0, bit 0.
        cache.prot.push(vec![1u64]);
        // Layer 0 is at 8/8; admitting 4 more exceeds its limit.
        assert!(cache.make_room(4, 0));
        // It evicted (0,1) (tier 3), NOT (0,0) (tier 1 protected).
        assert_eq!(cache.layer_used.get(&0), Some(&4));
        assert!(cache.entries.contains_key(&(0, 0)));
        assert!(!cache.entries.contains_key(&(0, 1)));
        assert!(cache.entries.contains_key(&(1, 0)));
        assert!(cache.entries.contains_key(&(1, 1)));
    }

    #[test]
    fn prefill_cache_shares_budget_proportionally_to_layer_bank_size() {
        let bank_bytes = HashMap::from([(0, 1), (1, 3)]);
        assert_eq!(proportional_layer_quotas(16, &bank_bytes, 4), HashMap::from([(0, 4), (1, 12)]));
        assert_eq!(proportional_layer_quotas(16, &bank_bytes, 0), HashMap::from([(0, 0), (1, 0)]));
    }

    #[test]
    fn staging_runs_keep_their_group_slot_offsets() {
        let experts = [(2, [0; 3]), (5, [0; 3]), (6, [0; 3]), (11, [0; 3])];
        assert_eq!(expert_run_offsets(&experts), [(0, 1), (1, 2), (3, 1)]);
    }

}
