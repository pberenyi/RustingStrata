//! Named tensor views over the mmapped GGUF, page cache advice for them, and a reference parallel matvec.

use crate::config::ModelConfig;
use crate::gguf::Gguf;
use crate::quant::{GgmlType, dequant_row};
use anyhow::Context;
use std::borrow::Cow;
use rayon::prelude::*;
use std::collections::HashMap;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, OnceLock};

const PLE_ROW_CACHE_BYTES: usize = 16 << 20;

#[derive(Clone)]
pub struct Tensor<'a> {
    pub ty: GgmlType,
    /// Elements per row (GGUF dims[0]).
    pub cols: usize,
    /// Product of the remaining dims; for 3-D expert tensors this is out_rows * n_expert.
    pub rows: usize,
    pub data: Cow<'a, [u8]>,
}

impl<'a> Tensor<'a> {
    fn row_bytes(&self) -> usize { self.ty.row_bytes(self.cols as u64) }

    pub fn row(&self, r: usize, out: &mut [f32]) {
        let rb = self.row_bytes();
        dequant_row(self.ty, &self.data[r * rb..(r + 1) * rb], out);
    }

    /// y[r] = row_r · x for all rows.
    pub fn matvec(&self, x: &[f32], y: &mut [f32]) {
        assert_eq!((x.len(), y.len()), (self.cols, self.rows));
        y.par_chunks_mut(16).enumerate().for_each_init(|| vec![0f32; self.cols], |buf, (ci, ys)| {
            for (k, yv) in ys.iter_mut().enumerate() {
                self.row(ci * 16 + k, buf);
                *yv = buf.iter().zip(x).map(|(a, b)| a * b).sum();
            }
        });
    }

    /// Rows belonging to expert `e` of an [cols, out, n_expert] tensor.
    pub fn expert(&self, e: usize, n_expert: usize) -> Tensor<'_> {
        let rows = self.rows / n_expert;
        let rb = self.row_bytes();
        Tensor { rows, data: Cow::Borrowed(&self.data[e * rows * rb..(e + 1) * rows * rb]), ty: self.ty, cols: self.cols }
    }

    pub fn to_f32(&self) -> Vec<f32> {
        let mut v = vec![0f32; self.rows * self.cols];
        for (r, out) in v.chunks_exact_mut(self.cols).enumerate() { self.row(r, out); }
        v
    }
}

fn borrowed_tensor<'a>(t: &'a Tensor<'_>) -> Tensor<'a> {
    Tensor { ty: t.ty, cols: t.cols, rows: t.rows, data: Cow::Borrowed(t.data.as_ref()) }
}

pub struct Weights<'a> {
    pub g: &'a Gguf,
    aliases: &'static [(&'static str, &'static str)],
    n_expert: usize,
    /// Loader-owned overrides: synthesized `ssm_beta`/`ssm_alpha` and LoRA-baked tensors.
    pub built: HashMap<String, Tensor<'a>>,
    pub lora_baked: std::collections::HashSet<String>,
    ple_row_cache: Mutex<(HashMap<usize, Box<[u8]>>, VecDeque<usize>, usize)>,
}

impl<'a> Weights<'a> {
    pub fn new(g: &'a Gguf) -> anyhow::Result<Weights<'a>> {
        let c = ModelConfig::from_gguf(g)?;
        let mut w = Weights {
            g, aliases: c.aliases, n_expert: c.n_expert, built: HashMap::new(),
            lora_baked: std::collections::HashSet::new(), ple_row_cache: Mutex::new((HashMap::new(), VecDeque::new(), 0)),
        };
        if !g.uses_mmap() {
            w.keep_owned("token_embd.weight")?;
        }
        // llama.cpp src/models/qwen3next.cpp: per key head group, r = nv / nk beta rows, then r alpha rows.
        for il in 0..c.n_layer {
            let Ok(ba) = w.layer(il, "ssm_ba.weight") else { continue };
            let (nk, nv, rb) = (c.ssm_n_group, c.ssm_dt_rank, ba.row_bytes());
            anyhow::ensure!(nk > 0 && nv > 0 && nv % nk == 0 && ba.rows == 2 * nv && ba.data.len() == ba.rows * rb,
                "blk.{il}.ssm_ba: {} rows, {} bytes for group_count {nk}, time_step_rank {nv}", ba.rows, ba.data.len());
            let r = nv / nk;
            let (mut b, mut a) = (Vec::with_capacity(ba.data.len() / 2), Vec::with_capacity(ba.data.len() / 2));
            for grp in ba.data.chunks_exact(2 * r * rb) {
                b.extend_from_slice(&grp[..r * rb]);
                a.extend_from_slice(&grp[r * rb..]);
            }
            let half = Tensor { rows: ba.rows / 2, ty: ba.ty, cols: ba.cols, data: Cow::Owned(Vec::new()) };
            w.built.insert(format!("blk.{il}.ssm_beta.weight"), Tensor { data: Cow::Owned(b), ..half.clone() });
            w.built.insert(format!("blk.{il}.ssm_alpha.weight"), Tensor { data: Cow::Owned(a), ..half });
        }
        Ok(w)
    }

    /// `name` with the architecture's alias for a canonical suffix tried second.
    pub fn get(&self, name: &str) -> anyhow::Result<Tensor<'_>> {
        if let Some(t) = self.built.get(name) {
            return Ok(borrowed_tensor(t));
        }
        let t = self.g.tensors.get(name)
            .or_else(|| self.aliases.iter().find_map(|&(c, a)| self.g.tensors.get(&format!("{}{a}", name.strip_suffix(c)?))))
            .with_context(|| format!("missing tensor {name}"))?;
        let ty = GgmlType::from_u32(t.ty).with_context(|| format!("tensor {name}"))?;
        Ok(Tensor {
            ty,
            cols: t.dims[0] as usize,
            rows: t.dims[1..].iter().product::<u64>() as usize,
            data: self.g.bytes(t)?,
        })
    }

    pub fn layer(&self, il: usize, suffix: &str) -> anyhow::Result<Tensor<'_>> { self.get(&format!("blk.{il}.{suffix}")) }

    pub fn contains(&self, name: &str) -> bool { self.info(name).is_ok() }

    pub fn row(&self, name: &str, row: usize, out: &mut [f32]) -> anyhow::Result<()> {
        let info = self.info(name)?;
        let ty = GgmlType::from_u32(info.ty).with_context(|| format!("tensor {name}"))?;
        let cols = info.dims[0] as usize;
        let rows = info.dims[1..].iter().product::<u64>() as usize;
        anyhow::ensure!(row < rows, "row {row} out of range for tensor {name} ({rows} rows)");
        anyhow::ensure!(out.len() == cols, "row output has {} elements, tensor {name} has {cols} columns", out.len());
        let row_bytes = ty.row_bytes(cols as u64);

        if self.g.uses_mmap() {
            self.get(name)?.row(row, out);
            return Ok(());
        }

        let is_ple = name == "per_layer_token_embd.weight";
        if is_ple {
            let cache = self.ple_row_cache.lock().map_err(|_| anyhow::anyhow!("PLE row cache lock poisoned"))?;
            if let Some(data) = cache.0.get(&row) {
                crate::quant::dequant_row(ty, data, out);
                return Ok(());
            }
        }

        let data = self.g.bytes_range(info, row * row_bytes, row_bytes)?;
        crate::quant::dequant_row(ty, &data, out);
        if is_ple && row_bytes <= PLE_ROW_CACHE_BYTES {
            let mut cache = self.ple_row_cache.lock().map_err(|_| anyhow::anyhow!("PLE row cache lock poisoned"))?;
            if !cache.0.contains_key(&row) {
                while cache.2 + row_bytes > PLE_ROW_CACHE_BYTES {
                    let Some(old) = cache.1.pop_front() else { break };
                    if let Some(data) = cache.0.remove(&old) { cache.2 -= data.len(); }
                }
                cache.2 += row_bytes;
                cache.1.push_back(row);
                cache.0.insert(row, data.into_owned().into_boxed_slice());
            }
        }
        Ok(())
    }

    fn keep_owned(&mut self, name: &str) -> anyhow::Result<()> {
        let (ty, cols, rows, data) = {
            let t = self.info(name)?;
            let ty = GgmlType::from_u32(t.ty).with_context(|| format!("tensor {name}"))?;
            (ty, t.dims[0] as usize, t.dims[1..].iter().product::<u64>() as usize, self.g.bytes(t)?.into_owned())
        };
        self.built.insert(name.to_string(), Tensor { ty, cols, rows, data: Cow::Owned(data) });
        Ok(())
    }

    fn info(&self, name: &str) -> anyhow::Result<&crate::gguf::TensorInfo> {
        self.g.tensors.get(name).or_else(|| self.aliases.iter().find_map(|&(c, a)| {
            self.g.tensors.get(&format!("{}{a}", name.strip_suffix(c)?))
        })).with_context(|| format!("missing tensor {name}"))
    }

    pub fn expert_shape(&self, il: usize, suffix: &str) -> anyhow::Result<(GgmlType, usize, usize, usize)> {
        let name = format!("blk.{il}.{suffix}");
        let t = self.info(&name)?;
        let ty = GgmlType::from_u32(t.ty).with_context(|| format!("tensor {name}"))?;
        anyhow::ensure!(t.dims.len() >= 3, "tensor {name} is not an expert tensor");
        let rows = t.dims[1..].iter().product::<u64>() as usize;
        anyhow::ensure!(self.n_expert > 0 && rows % self.n_expert == 0, "invalid expert count for {name}");
        let expert_rows = rows / self.n_expert;
        let bytes = ty.row_bytes(t.dims[0]) * expert_rows;
        Ok((ty, expert_rows, t.dims[0] as usize, bytes))
    }

    pub fn expert(&self, il: usize, suffix: &str, e: usize) -> anyhow::Result<Tensor<'_>> {
        let (ty, rows, cols, bytes) = self.expert_shape(il, suffix)?;
        let name = format!("blk.{il}.{suffix}");
        anyhow::ensure!(e < self.n_expert, "expert {e} out of range for {name}");
        let t = self.info(&name)?;
        let data = self.g.bytes_range(t, e * bytes, bytes)?;
        Ok(Tensor { ty, rows, cols, data })
    }

    pub fn expert_range(&self, il: usize, suffix: &str, first: usize, count: usize) -> anyhow::Result<Cow<'_, [u8]>> {
        let (_, _, _, bytes) = self.expert_shape(il, suffix)?;
        let name = format!("blk.{il}.{suffix}");
        let t = self.info(&name)?;
        anyhow::ensure!(first <= self.n_expert && count <= self.n_expert - first, "expert range out of bounds for {name}");
        self.g.bytes_range(t, first * bytes, count * bytes)
    }

    /// Refuse files containing quant types this engine does not implement, except in the MTP (nextn) layers, blocks
    /// `n_layer` on: false if one of those has such a type, so that the model loads without MTP.
    /// Drop the LoRA-baked overrides; synthesized `ssm_*` tensors stay.
    pub fn clear_baked(&mut self) {
        for name in std::mem::take(&mut self.lora_baked) { self.built.remove(&name); }
    }

    pub fn bake_lora(&mut self, loras: &[crate::model::Lora]) {
        for lora in loras {
            for (name, entry) in &lora.entries {
                if name.ends_with("ffn_down_exps.weight") { continue; }
                if entry.n_expert != 1 {
                    eprintln!("lora: target {name} has {} experts, expected 1; skipping", entry.n_expert);
                    continue;
                }
                let base = match self.get(name) { Ok(t) => t, Err(_) => { eprintln!("lora: target {name} not found, skipping"); continue; } };
                if base.cols != entry.in_dim || base.rows != entry.out_dim {
                    eprintln!("lora: target {name} is {}x{}, adapter is {}x{}; skipping", base.rows, base.cols, entry.out_dim, entry.in_dim);
                    continue;
                }
                let (base_cols, base_rows, mut w) = (base.cols, base.rows, base.to_f32());
                drop(base);
                let (a, b) = (&entry.a[0], &entry.b[0]);
                let (rank, in_dim, out_dim) = (entry.rank, entry.in_dim, entry.out_dim);
                let mult = if lora.alpha > 0.0 { lora.scale * lora.alpha / rank as f32 } else { lora.scale };
                for o in 0..out_dim {
                    for i in 0..in_dim {
                        let mut d = 0f32;
                        for r in 0..rank { d += b[o * rank + r] * a[r * in_dim + i]; }
                        w[o * in_dim + i] += mult * d;
                    }
                }
                self.built.remove(name);
                let data = w.iter().flat_map(|f| f.to_le_bytes()).collect();
                let t = Tensor { ty: crate::quant::GgmlType::F32, cols: base_cols, rows: base_rows, data: Cow::Owned(data) };
                self.built.insert(name.clone(), t);
                self.lora_baked.insert(name.clone());
            }
        }
    }
    pub fn validate(&self, n_layer: usize) -> anyhow::Result<bool> {
        let mut nextn = true;
        for t in self.g.tensors.values() {
            if let Err(e) = GgmlType::from_u32(t.ty) {
                let il = t.name.strip_prefix("blk.").and_then(|r| r.split('.').next()).and_then(|i| i.parse::<usize>().ok());
                if il.is_some_and(|il| il >= n_layer) { nextn = false; continue; }
                return Err(e).with_context(|| format!("tensor {}", t.name));
            }
        }
        Ok(nextn)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn borrowed_tensor_does_not_clone_owned_bytes() {
        let owned = Tensor { ty: GgmlType::F32, cols: 4, rows: 1024, data: Cow::Owned(vec![7; 16 * 1024]) };
        let borrowed = borrowed_tensor(&owned);

        assert!(matches!(&borrowed.data, Cow::Borrowed(_)));
        assert_eq!(borrowed.data.as_ptr(), owned.data.as_ptr());
        assert_eq!(borrowed.data.len(), owned.data.len());
    }
}

#[cfg(target_os = "linux")]
pub(crate) fn available_ram() -> Option<(u64, u64)> {
    let text = std::fs::read_to_string("/proc/meminfo").ok()?;
    let bytes = |key: &str| {
        text.lines().find_map(|line| {
            let value = line.strip_prefix(key)?.split_whitespace().next()?.parse::<u64>().ok()?;
            value.checked_mul(1024)
        })
    };
    Some((bytes("MemTotal:")?, bytes("MemAvailable:")?))
}

#[cfg(windows)]
pub(crate) fn available_ram() -> Option<(u64, u64)> {
    #[repr(C)]
    struct MemStatus {
        len: u32,
        load: u32,
        total: u64,
        avail: u64,
        rest: [u64; 5],
    }
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GlobalMemoryStatusEx(m: *mut MemStatus) -> i32;
    }
    let mut m = MemStatus { len: std::mem::size_of::<MemStatus>() as u32, load: 0, total: 0, avail: 0, rest: [0; 5] };
    (unsafe { GlobalMemoryStatusEx(&mut m) } != 0).then_some((m.total, m.avail))
}

#[cfg(not(any(target_os = "linux", windows)))]
pub(crate) fn available_ram() -> Option<(u64, u64)> { None }

static EXPERT_CACHE_RAM: OnceLock<Option<(usize, AtomicUsize)>> = OnceLock::new();

fn expert_cache_ram() -> Option<&'static (usize, AtomicUsize)> {
    EXPERT_CACHE_RAM.get_or_init(|| {
        let (total, available) = available_ram()?;
        let budget = usize::try_from(available.saturating_sub(total / 20)).ok()?;
        Some((budget, AtomicUsize::new(0)))
    }).as_ref()
}

pub(crate) fn expert_cache_ram_capacity() -> Option<usize> {
    expert_cache_ram().map(|(budget, _)| *budget)
}

fn reserve_expert_cache_ram(bytes: usize) -> bool {
    let Some((budget, used)) = expert_cache_ram() else { return false };
    let mut current = used.load(Ordering::Relaxed);
    loop {
        let Some(next) = current.checked_add(bytes) else { return false };
        if next > *budget { return false; }
        match used.compare_exchange_weak(current, next, Ordering::AcqRel, Ordering::Relaxed) {
            Ok(_) => return true,
            Err(actual) => current = actual,
        }
    }
}

fn release_expert_cache_ram(bytes: usize) {
    if let Some((_, used)) = expert_cache_ram() {
        let previous = used.fetch_sub(bytes, Ordering::AcqRel);
        debug_assert!(previous >= bytes, "expert cache RAM reservation underflow");
    }
}

pub(crate) struct ExpertCacheReservation(usize);

impl ExpertCacheReservation {
    pub(crate) fn new(bytes: usize) -> Option<ExpertCacheReservation> {
        reserve_expert_cache_ram(bytes).then_some(ExpertCacheReservation(bytes))
    }
}

impl Drop for ExpertCacheReservation {
    fn drop(&mut self) { release_expert_cache_ram(self.0); }
}

#[cfg(target_os = "linux")]
fn madvise(p: usize, n: usize, advice: i32) {
    unsafe extern "C" { fn madvise(addr: *mut std::ffi::c_void, len: usize, advice: i32) -> i32; }
    unsafe { madvise(p as *mut _, n, advice) };
}
#[cfg(not(target_os = "linux"))]
fn madvise(_: usize, _: usize, _: i32) {}

#[cfg(windows)]
fn prefetch_virtual_memory(rs: &[(usize, usize)]) {
    use std::ffi::c_void;
    use std::sync::OnceLock;

    #[repr(C)]
    struct MemoryRangeEntry {
        address: *mut c_void,
        size: usize,
    }

    type Prefetch = unsafe extern "system" fn(isize, usize, *const MemoryRangeEntry, u32) -> i32;

    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GetCurrentProcess() -> isize;
        fn GetModuleHandleW(name: *const u16) -> *mut c_void;
        fn GetProcAddress(module: *mut c_void, name: *const u8) -> Option<Prefetch>;
    }

    static PREFETCH: OnceLock<Option<Prefetch>> = OnceLock::new();
    let prefetch = PREFETCH.get_or_init(|| unsafe {
        let name = "kernel32.dll\0".encode_utf16().collect::<Vec<_>>();
        let module = GetModuleHandleW(name.as_ptr());
        if module.is_null() {
            return None;
        }
        GetProcAddress(module, b"PrefetchVirtualMemory\0".as_ptr())
    });
    let Some(prefetch) = prefetch else { return };
    let ranges: Vec<_> = rs
        .iter()
        .filter(|(_, n)| *n > 0)
        .map(|&(p, n)| MemoryRangeEntry {
            address: p as *mut c_void,
            size: n,
        })
        .collect();
    if !ranges.is_empty() {
        unsafe { prefetch(GetCurrentProcess(), ranges.len(), ranges.as_ptr(), 0); }
    }
}

/// Starts reading mapped bytes into the system cache without waiting when the OS supports a prefetch hint.
pub fn will_need(p: usize, n: usize) {
    will_need_ranges(&[(p, n)]);
}

fn will_need_ranges(rs: &[(usize, usize)]) {
    #[cfg(target_os = "linux")]
    for &(p, n) in rs {
        // One call reads at most one readahead window (read_ahead_kb, 128-256 KB on NVMe).
        let (mut a, end) = (p & !4095, p + n);
        while a < end {
            let l = (end - a).min(256 << 10);
            madvise(a, l, 3);
            a += l;
        }
    }
    #[cfg(windows)]
    prefetch_virtual_memory(rs);
}

/// Marks the whole pages of `b` first to reclaim (MADV_COLD): bytes now on the GPU. Expert files exceed the page
/// cache, and pages read once at load would otherwise outlive the experts a prompt reads from it. Windows manages
/// mapped-file pages through its working-set policy; DiscardVirtualMemory cannot be used on these read-only views.
pub fn cold(b: &[u8]) {
    let (a, e) = ((b.as_ptr() as usize + 4095) & !4095, (b.as_ptr() as usize + b.len()) & !4095);
    if e > a { madvise(a, e - a, 20); }
}

/// Reads ranges of mapped weights into the system cache from a background thread, in the order sent, so a cold read
/// keeps many drive reads in flight instead of one page fault at a time. Dropping it joins the thread.
pub struct Prefetch { tx: Option<std::sync::mpsc::Sender<Vec<(usize, usize)>>>, th: Option<std::thread::JoinHandle<()>> }

impl Prefetch {
    pub fn start() -> Prefetch {
        let (tx, rx) = std::sync::mpsc::channel::<Vec<(usize, usize)>>();
        let th = std::thread::spawn(move || for rs in rx { will_need_ranges(&rs); });
        Prefetch { tx: Some(tx), th: Some(th) }
    }
    /// (address, bytes) ranges of mapped weights.
    pub fn send(&self, rs: Vec<(usize, usize)>) { let _ = self.tx.as_ref().unwrap().send(rs); }
}

impl Drop for Prefetch {
    fn drop(&mut self) {
        self.tx = None;
        if let Some(t) = self.th.take() { let _ = t.join(); }
    }
}
