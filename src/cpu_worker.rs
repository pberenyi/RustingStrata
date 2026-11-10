//! CPU side of the expert split, without per-layer stream syncs. For each MoE layer the GPU writes routing ids,
//! weights, cache slots and (when some expert missed) the normed activation into host-mapped pinned memory, then
//! raises the layer's request flag (on CUDA a layer with no miss raises only its skip flag). This worker spins on the
//! flag, computes the missed experts with `cpu_moe`, writes their weighted sum back and raises the done flag, which the
//! GPU combine kernel waits on only if the layer missed.

use crate::model::Model;
use crate::backend::{Backend, Buf};
use crate::weights::{ExpertCacheReservation, Tensor, available_ram, expert_cache_ram_capacity, expert_cache_ram_state};
use anyhow::Context;
use std::borrow::Cow;
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering, fence};
use std::sync::{Arc, mpsc};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

/// Word offsets inside one layer's mailbox. Flags sit on separate cache lines.
pub const REQ: usize = 0;
/// Tokens in the request (1 for decode, up to `TMAX` for a multi-token verify).
pub const NTOK: usize = 1;
/// Raised instead of REQ, with no data, when no expert of the layer missed (CUDA only; Vulkan raises REQ).
pub const SKIP: usize = 2;
/// Vulkan: the combine's verdict on its wait (0 open, 1 the worker answered, 2 gave up; see mb_wait.glsl).
pub const VERDICT: usize = 8;
pub const DONE: usize = 16;
pub const IDS: usize = 32;

/// Most tokens one request may carry.
pub const TMAX: usize = 4;

struct CachedExpert { mats: [Box<[u8]>; 3], bytes: usize, _reservation: Option<ExpertCacheReservation> }

#[derive(Default)]
pub struct CpuExpertCache {
    entries: HashMap<(usize, usize), CachedExpert>,
    lru: VecDeque<(usize, usize)>,
    used: usize,
    budget: usize,
    layer_quota: usize,
    layer_count: usize,
    layer_used: HashMap<usize, usize>,
    reported_bucket: Option<(usize, usize)>,
}

impl CpuExpertCache {
    fn new(m: &Model, moe_layers: &[usize]) -> CpuExpertCache {
        if m.w.g.uses_mmap() {
            eprintln!("CPU expert RAM cache: disabled because GGUF memory mapping is enabled; use --no-mmap to enable it");
            return CpuExpertCache::default();
        }
        let Some(budget) = expert_cache_ram_capacity() else {
            eprintln!("CPU expert RAM cache: physical-memory availability is unknown; reading uncached experts from disk");
            return CpuExpertCache::default();
        };
        let status = available_ram();
        let available = status.map_or(0, |status| status.available);
        let commit_available = status.and_then(|status| status.commit_available);
        let layer_count = moe_layers.iter().copied().collect::<HashSet<_>>().len();
        let layer_quota = if layer_count == 0 { 0 } else { budget / layer_count };
        eprintln!(
            "CPU expert RAM cache [{} layers]: LRU enabled for CPU-routed decode misses, initial capacity {:.0} MB ({:.0} MB fair share per layer; unused share is borrowable; {:.0} MB physical, {:.0} MB commit available; reserving 5%; GPU prefill uses separate staging)",
            layer_count,
            budget as f64 / 1048576.0,
            layer_quota as f64 / 1048576.0,
            available as f64 / 1048576.0,
            commit_available.unwrap_or(0) as f64 / 1048576.0,
        );
        CpuExpertCache { budget, layer_quota, layer_count, ..CpuExpertCache::default() }
    }

    fn prepare(&mut self, m: &Model, il: usize, experts: &[usize]) -> anyhow::Result<()> {
        let active: HashSet<_> = experts.iter().map(|&e| (il, e)).collect();
        let mut state = self.refresh_budget();
        while !state.is_some_and(|(budget, used)| used <= budget) && !self.lru.is_empty() {
            self.evict_at(0);
            state = self.refresh_budget();
        }
        let mut unique = HashSet::new();
        for &e in experts {
            if !unique.insert(e) { continue; }
            let key = (il, e);
            if self.entries.contains_key(&key) {
                self.lru.retain(|&k| k != key);
                self.lru.push_back(key);
                continue;
            }
            if self.budget == 0 { continue; }
            let mut size = 0usize;
            for name in crate::expert_cache::MATS {
                size = size.checked_add(m.w.expert_shape(il, name)?.3)
                    .context("CPU expert size overflow")?;
            }
            if size > self.budget { continue; }
            if !self.make_room(size, il, &active) { continue; }
            let reservation = loop {
                if let Some(reservation) = ExpertCacheReservation::new(size) { break Some(reservation); }
                let Some(i) = self.lru.iter().position(|key| !active.contains(key)) else { break None };
                self.evict_at(i);
            };
            let Some(reservation) = reservation else { continue };

            let mats = (|| -> anyhow::Result<[Box<[u8]>; 3]> {
                let mut mats = Vec::with_capacity(crate::expert_cache::MATS.len());
                for name in crate::expert_cache::MATS {
                    mats.push(m.w.expert(il, name, e)?.data.into_owned().into_boxed_slice());
                }
                mats.try_into().map_err(|_| anyhow::anyhow!("expected three CPU expert matrices"))
            })();
            let mats = match mats {
                Ok(mats) => mats,
                Err(err) => return Err(err),
            };
            self.entries.insert(key, CachedExpert { mats, bytes: size, _reservation: Some(reservation) });
            self.lru.push_back(key);
            self.used += size;
            *self.layer_used.entry(il).or_default() += size;
        }
        let bucket = (self.used / (128 << 20), self.budget / (128 << 20));
        if self.used > 0 && self.reported_bucket != Some(bucket) {
            eprintln!(
                "CPU expert RAM cache [{} layers]: {:.0} MB in use / {:.0} MB current capacity ({} experts cached)",
                self.layer_count,
                self.used as f64 / 1048576.0,
                self.budget as f64 / 1048576.0,
                self.entries.len(),
            );
            self.reported_bucket = Some(bucket);
        }
        Ok(())
    }

    fn refresh_budget(&mut self) -> Option<(usize, usize)> {
        if let Some((budget, used)) = expert_cache_ram_state() {
            self.budget = budget;
            self.layer_quota = if self.layer_count == 0 { 0 } else { budget / self.layer_count };
            Some((budget, used))
        } else {
            self.budget = 0;
            self.layer_quota = 0;
            None
        }
    }

    fn layer_limit(&self, layer: usize) -> usize {
        let borrowed = self.layer_used.iter()
            .filter(|(other, _)| **other != layer)
            .fold(0usize, |sum, (_, &used)| sum.saturating_add(used.min(self.layer_quota)));
        self.budget.saturating_sub(borrowed)
    }

    fn evict_at(&mut self, index: usize) {
        let key = self.lru.remove(index).unwrap();
        if let Some(old) = self.entries.remove(&key) {
            self.used -= old.bytes;
            let layer_bytes = self.layer_used.get_mut(&key.0).unwrap();
            *layer_bytes -= old.bytes;
            if *layer_bytes == 0 { self.layer_used.remove(&key.0); }
        }
    }

    fn make_room(&mut self, size: usize, layer: usize, active: &HashSet<(usize, usize)>) -> bool {
        loop {
            let layer_used = self.layer_used.get(&layer).copied().unwrap_or(0);
            if layer_used.checked_add(size).is_none_or(|next| next > self.layer_limit(layer)) {
                let Some(i) = self.lru.iter().position(|key| key.0 == layer && !active.contains(key)) else { return false };
                self.evict_at(i);
                continue;
            }
            if self.used.checked_add(size).is_some_and(|next| next <= self.budget) { return true; }

            let over_quota = self.lru.iter().position(|key| {
                !active.contains(key)
                    && self.layer_used.get(&key.0).copied().unwrap_or(0) > self.layer_quota
            });
            let Some(i) = over_quota.or_else(|| self.lru.iter().position(|key| !active.contains(key))) else { return false };
            self.evict_at(i);
        }
    }

    pub fn expert<'a>(&'a self, m: &Model, il: usize, suffix: &str, e: usize) -> Option<Tensor<'a>> {
        let i = crate::expert_cache::MATS.iter().position(|&name| name == suffix)?;
        let cached = self.entries.get(&(il, e))?;
        let data = cached.mats.get(i)?;
        let (ty, rows, cols, _) = m.w.expert_shape(il, suffix).ok()?;
        Some(Tensor { ty, rows, cols, data: Cow::Borrowed(data) })
    }
}

/// Per layer: ids, weights and slots for up to TMAX tokens of `k` entries (token t at t * k; the three arrays are
/// each NTOK * k long), then TMAX activations of `n` floats at `h` and TMAX results at `y`.
pub struct Mailbox { pub k: usize, pub n: usize, pub h: usize, pub y: usize, pub stride: usize }

impl Mailbox {
    pub fn new(k: usize, n: usize) -> Mailbox {
        let h = (IDS + 3 * TMAX * k).next_multiple_of(16);
        Mailbox { k, n, h, y: h + TMAX * n, stride: (h + 2 * TMAX * n).next_multiple_of(16) }
    }
}

pub struct CpuWorker {
    _mem: Buf,
    host: usize,
    /// Device address of the mailboxes (layer `il` at `dev + 4 * il * layout.stride`).
    pub dev: u64,
    pub layout: Mailbox,
    seq: u32,
    layers: Vec<usize>,
    tx: Option<mpsc::Sender<u32>>,
    stop: Arc<AtomicBool>,
    busy_ns: Arc<AtomicU64>,
    thread: Option<JoinHandle<()>>,
}

impl CpuWorker {
    /// Mailboxes cover layers up to the largest of `moe_layers`. Safety: `m` must outlive the worker (it is joined on drop).
    pub fn new(m: &Model, gpu: &dyn Backend, moe_layers: Vec<usize>) -> anyhow::Result<CpuWorker> {
        let c = &m.cfg;
        let mut ram_cache = CpuExpertCache::new(m, &moe_layers);
        let layout = Mailbox::new(c.n_expert_used, c.n_embd);
        let len = layout.stride * moe_layers.iter().max().map_or(0, |&il| il + 1);
        let (mem, host) = gpu.host_mailbox(len)?;
        let (host, dev) = (host as usize, mem.addr());

        let (tx, rx) = mpsc::channel::<u32>();
        let (stop, busy_ns) = (Arc::new(AtomicBool::new(false)), Arc::new(AtomicU64::new(0)));
        let (st, bn) = (stop.clone(), busy_ns.clone());
        let mp = m as *const Model as usize;
        let lay = Mailbox::new(c.n_expert_used, c.n_embd);
        let layers = moe_layers.clone();
        // STRATA_TEST_WORKER_DELAY=ms: answer some requests (layer and seq picked by a hash) that much later, as a
        // worker reading experts from a cold drive would (tests the combines that give up).
        let delay = std::env::var("STRATA_TEST_WORKER_DELAY").ok().and_then(|v| v.parse::<u64>().ok()).map(Duration::from_millis);
        // The loop runs on a thread of its own pool, so the expert passes start on it at once and it takes part
        // in them instead of sleeping until a woken pool finishes (~12 us less per layer).
        let pool = rayon::ThreadPoolBuilder::new().num_threads(rayon::current_num_threads())
            .thread_name(|i| format!("strata-experts-{i}")).build()?;
        let thread = std::thread::Builder::new().name("strata-cpu-experts".into()).spawn(move || pool.install(move || {
            let m = unsafe { &*(mp as *const Model) };
            // Experts outside the RAM and VRAM caches are read from the drive: reading the next layer's misses
            // together keeps storage reads in flight instead of issuing one read at a time per compute thread.
            let pre = crate::weights::Prefetch::start();
            let w = |i: usize| host as *mut u32 as usize + 4 * i;
            while let Ok(seq) = rx.recv() {
                for &il in &moe_layers {
                    let b = il * lay.stride;
                    let at = |o: usize| (w(b + o)) as *mut u32;
                    // Skip the layer on its skip flag, or on a newer seq anywhere (the GPU already moved on to the
                    // next token, so it did not wait for this one). Waiting for the old seq would hang.
                    let ahead = |v: u32| v != 0 && (v.wrapping_sub(seq) as i32) >= 0;
                    let mut r;
                    while { r = unsafe { at(REQ).read_volatile() }; r != seq } {
                        if st.load(Ordering::Relaxed) { return; }
                        if ahead(r) || ahead(unsafe { at(SKIP).read_volatile() }) { break; }
                        std::hint::spin_loop();
                    }
                    if r != seq { continue; }
                    // mfence: loads of write-combining memory (the Vulkan BAR mailbox) are weakly ordered, so an
                    // acquire fence (no instruction on x86) would not keep them after the REQ load.
                    fence(Ordering::SeqCst);
                    let t0 = Instant::now();
                    let rd = |o: usize| unsafe { at(o).read_volatile() };
                    let nt = (rd(NTOK) as usize).clamp(1, TMAX);
                    let (k, kt, n) = (lay.k, lay.k * nt, lay.n);
                    // Bulk streaming reads, as in `routing`: plain loads of the Vulkan BAR mailbox are each an
                    // uncached read (~130 us for one activation).
                    let mut v = vec![0u32; (3 * kt).next_multiple_of(4)];
                    unsafe { read_wc(at(IDS), &mut v) };
                    let missed: Vec<usize> = (0..kt).filter(|&j| (v[2 * kt + j] as i32) < 0).map(|j| v[j] as usize).collect();
                    if let Err(e) = ram_cache.prepare(m, il, &missed) { panic!("preparing CPU expert RAM cache: {e:#}"); }
                    if !missed.is_empty() && m.w.g.uses_mmap() {
                        // Gate and up first: they are read first.
                        pre.send(["ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight"].iter()
                            .flat_map(|s| missed.iter().filter_map(move |&e| m.w.expert(il, s, e).ok())
                                .map(|t| (t.data.as_ptr() as usize, t.data.len())).collect::<Vec<_>>())
                            .collect());
                    }
                    for t in 0..nt {
                        let sel: Vec<(usize, f32)> = (t * k..(t + 1) * k).filter(|&j| (v[2 * kt + j] as i32) < 0)
                            .map(|j| (v[j] as usize, f32::from_bits(v[kt + j]))).collect();
                        if sel.is_empty() { continue; }
                        let mut h = vec![0u32; n];
                        unsafe { read_wc(at(lay.h + t * n), &mut h) };
                        let h: Vec<f32> = h.into_iter().map(f32::from_bits).collect();
                        let y = crate::cpu_moe::experts_cached(m, il, &h, &sel, &ram_cache);
                        unsafe { std::ptr::copy_nonoverlapping(y.as_ptr(), at(lay.y + t * n) as *mut f32, n) };
                    }
                    // SeqCst fences (mfence on x86) also order and flush write-combining stores (Vulkan maps the
                    // mailbox through the BAR): y before DONE, and DONE out now.
                    if let Some(d) = delay { if (seq as usize * 31 + il) % 17 == 0 { std::thread::sleep(d) } }
                    fence(Ordering::SeqCst);
                    unsafe { at(DONE).write_volatile(seq) };
                    fence(Ordering::SeqCst);
                    bn.fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
                }
            }
        }))?;
        Ok(CpuWorker { _mem: mem, host, dev, layout, seq: 0, layers, tx: Some(tx), stop, busy_ns, thread: Some(thread) })
    }

    /// Start a new token: returns the sequence number the GPU must publish with.
    pub fn begin(&mut self) -> u32 {
        self.seq = self.seq.wrapping_add(1).max(1);
        let _ = self.tx.as_ref().unwrap().send(self.seq);
        self.seq
    }

    /// Routing of layer `il` from the last finished request, all its tokens: (expert id, routed weight, slot or -1).
    pub fn routing(&self, il: usize) -> Vec<(i32, f32, i32)> {
        let b = il * self.layout.stride;
        let rd = |o: usize| unsafe { (self.host as *const u32).add(b + o).read_volatile() };
        let k = self.layout.k * (rd(NTOK) as usize).clamp(1, TMAX);
        // One bulk read: on Vulkan the mailbox is BAR memory, where each host read is a slow uncached access.
        let mut v = vec![0u32; (3 * k).next_multiple_of(4)];
        unsafe { read_wc((self.host as *const u32).add(b + IDS), &mut v) };
        (0..k).map(|j| (v[j] as i32, f32::from_bits(v[k + j]), v[2 * k + j] as i32)).collect()
    }

    /// The first layer whose combine gave up waiting (verdict 2) in the last request: the layers before it ran.
    pub fn stalled_layer(&self) -> Option<usize> {
        self.layers.iter().copied().find(|&il| unsafe { (self.host as *const u32).add(il * self.layout.stride + VERDICT).read_volatile() } == 2)
    }

    /// Wait until the worker has answered layer `il` of the current request, then reopen the layer's verdict so its
    /// combine can run again.
    pub fn wait_done(&self, il: usize) -> anyhow::Result<()> {
        let at = |o: usize| unsafe { (self.host as *mut u32).add(il * self.layout.stride + o) };
        while unsafe { at(DONE).read_volatile() } != self.seq {
            anyhow::ensure!(!self.thread.as_ref().is_some_and(|t| t.is_finished()), "the CPU expert worker stopped");
            std::thread::sleep(Duration::from_micros(50));
        }
        unsafe { at(VERDICT).write_volatile(0) };
        fence(Ordering::SeqCst);
        Ok(())
    }

    /// Total time the worker spent computing experts.
    pub fn busy(&self) -> Duration { Duration::from_nanos(self.busy_ns.load(Ordering::Relaxed)) }
}

/// Copies `out.len()` (a multiple of 4) words from 16-byte aligned `p`. On write-combining memory (the Vulkan BAR
/// mailbox) streaming loads fetch a whole 64-byte line per read instead of one uncached read per load.
unsafe fn read_wc(p: *const u32, out: &mut [u32]) {
    #[cfg(target_feature = "sse4.1")]
    for (i, c) in out.chunks_exact_mut(4).enumerate() {
        use std::arch::x86_64::*;
        unsafe { _mm_storeu_si128(c.as_mut_ptr() as *mut __m128i, _mm_stream_load_si128(p.add(4 * i) as *const __m128i)) };
    }
    #[cfg(not(target_feature = "sse4.1"))]
    unsafe { std::ptr::copy_nonoverlapping(p, out.as_mut_ptr(), out.len()) };
}

impl Drop for CpuWorker {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        self.tx = None;
        if let Some(t) = self.thread.take() { let _ = t.join(); }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn expert() -> CachedExpert {
        CachedExpert {
            mats: [vec![0].into_boxed_slice(), vec![0].into_boxed_slice(), vec![0].into_boxed_slice()],
            bytes: 4,
            _reservation: None,
        }
    }

    fn insert(cache: &mut CpuExpertCache, key: (usize, usize)) {
        cache.entries.insert(key, expert());
        cache.lru.push_back(key);
        cache.used += 4;
        *cache.layer_used.entry(key.0).or_default() += 4;
    }

    #[test]
    fn cpu_expert_cache_evicts_lru_but_keeps_current_batch() {
        let (old, active) = ((0, 1), (0, 2));
        let mut cache = CpuExpertCache { budget: 8, ..CpuExpertCache::default() };
        insert(&mut cache, old);
        insert(&mut cache, active);

        assert!(cache.make_room(4, 0, &HashSet::from([active])));
        assert!(!cache.entries.contains_key(&old));
        assert!(cache.entries.contains_key(&active));
        assert_eq!(cache.used, 4);
        assert_eq!(cache.layer_used.get(&0), Some(&4));
    }

    #[test]
    fn cpu_expert_cache_does_not_evict_experts_in_current_batch() {
        let (a, b) = ((0, 1), (0, 2));
        let mut cache = CpuExpertCache { budget: 8, ..CpuExpertCache::default() };
        insert(&mut cache, a);
        insert(&mut cache, b);

        assert!(!cache.make_room(4, 0, &HashSet::from([a, b])));
        assert!(cache.entries.contains_key(&a) && cache.entries.contains_key(&b));
        assert_eq!(cache.used, 8);
    }

    #[test]
    fn cpu_expert_cache_reclaims_borrowed_share_for_other_layers() {
        let mut cache = CpuExpertCache {
            budget: 16,
            layer_quota: 8,
            ..CpuExpertCache::default()
        };
        for expert in 0..3 {
            insert(&mut cache, (0, expert));
        }

        assert_eq!(cache.layer_limit(1), 8);
        assert!(cache.make_room(8, 1, &HashSet::new()));
        assert_eq!(cache.used, 8);
        assert_eq!(cache.layer_used.get(&0), Some(&8));
        assert!(!cache.entries.contains_key(&(0, 0)));
    }

    #[test]
    fn cpu_expert_cache_stays_within_layer_quota_when_others_are_full() {
        let mut cache = CpuExpertCache {
            budget: 16,
            layer_quota: 8,
            ..CpuExpertCache::default()
        };
        insert(&mut cache, (0, 0));
        insert(&mut cache, (0, 1));
        insert(&mut cache, (1, 0));
        insert(&mut cache, (1, 1));

        assert!(cache.make_room(4, 0, &HashSet::new()));
        assert_eq!(cache.layer_used.get(&0), Some(&4));
        assert_eq!(cache.layer_used.get(&1), Some(&8));
    }
}
