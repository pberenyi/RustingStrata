//! GGUF v2/v3 parser over a memory map or positioned file reads.

use anyhow::{Context, bail};
use std::borrow::Cow;
use std::fs::File;
use memmap2::Mmap;
use std::collections::BTreeMap;
use std::path::Path;
use std::ops::Deref;
use std::sync::{Arc, OnceLock};
#[cfg(windows)]
use std::os::windows::io::AsRawHandle;

use crate::quant::GgmlType;

#[derive(Clone, Debug)]
pub enum Value {
    U8(u8), I8(i8), U16(u16), I16(i16), U32(u32), I32(i32), U64(u64), I64(i64),
    F32(f32), F64(f64), Bool(bool), Str(String), Arr(Vec<Value>),
}

impl Value {
    pub fn as_u64(&self) -> Option<u64> {
        Some(match *self {
            Value::U8(v) => v as u64, Value::U16(v) => v as u64, Value::U32(v) => v as u64, Value::U64(v) => v,
            Value::I8(v) if v >= 0 => v as u64, Value::I16(v) if v >= 0 => v as u64,
            Value::I32(v) if v >= 0 => v as u64, Value::I64(v) if v >= 0 => v as u64,
            Value::Bool(v) => v as u64,
            _ => return None,
        })
    }
    pub fn as_f32(&self) -> Option<f32> {
        match *self { Value::F32(v) => Some(v), Value::F64(v) => Some(v as f32), _ => self.as_u64().map(|v| v as f32) }
    }
    pub fn as_str(&self) -> Option<&str> { if let Value::Str(s) = self { Some(s) } else { None } }
    pub fn as_arr(&self) -> Option<&[Value]> { if let Value::Arr(a) = self { Some(a) } else { None } }
}

#[derive(Clone, Debug)]
pub struct TensorInfo { pub name: String, pub dims: Vec<u64>, pub ty: u32, pub offset: u64, /** Shard index. */ pub file: usize }

pub struct Gguf {
    pub meta: BTreeMap<String, Value>,
    pub tensors: BTreeMap<String, TensorInfo>,
    /// Each shard's retained whole-file map (mmap mode), shared file section, open file and data offset (one entry unless split).
    files: Vec<GgufFile>,
    mmap: bool,
}

struct GgufFile {
    mmap: Option<Arc<Mmap>>,
    file: File,
    data_offset: usize,
    file_len: usize,
    // Windows file section backing the partial views, created lazily (Unix mmaps per range, so this
    // stays empty there). Lazy because in-memory files used by tests cannot be sectioned at all.
    #[cfg(windows)]
    section: OnceLock<Result<*mut std::ffi::c_void, String>>,
}

/// A read-only view over exactly the range `[ptr, ptr + len)` of the backing file, owned by one
/// `MappedBytes`. On Windows it is a sub-range view of the file's shared section; on Unix it is an
/// independent private mapping. Dropping the view unmaps the range, so evicting a cached expert
/// releases its physical pages: the LRU's logical budget then matches physical residency, and
/// `available_ram()` stays truthful for a long-running server.
struct PartialView {
    ptr: *mut u8,
    len: usize,
    // Unix unmap anchors: the mapping may start before `ptr` (offset is rounded down to a page), so
    // the base and total length are kept separately. On Windows the view is exactly `[ptr, ptr+len)`
    // and `UnmapViewOfFile` takes `ptr`, so these are absent.
    #[cfg(not(windows))]
    base: *mut u8,
    #[cfg(not(windows))]
    total: usize,
}

impl Deref for PartialView {
    type Target = [u8];
    fn deref(&self) -> &[u8] { unsafe { std::slice::from_raw_parts(self.ptr, self.len) } }
}

impl Drop for PartialView {
    fn drop(&mut self) {
        #[cfg(windows)]
        unsafe { let _ = UnmapViewOfFile(self.ptr); }
        #[cfg(not(windows))]
        unsafe { if self.total > 0 { let _ = munmap(self.base as *mut std::ffi::c_void, self.total); } }
    }
}

// The view is a read-only mapping of a file: multiple threads may read it at once, and the pointer
// stays valid for the process's lifetime while the view is alive.
unsafe impl Send for PartialView {}
unsafe impl Sync for PartialView {}

#[cfg(windows)]
#[link(name = "kernel32")]
unsafe extern "system" {
    fn CreateFileMappingW(file: *mut std::ffi::c_void, sa: *const std::ffi::c_void, prot: u32, max_h: u32, max_l: u32, name: *const u16) -> *mut std::ffi::c_void;
    fn MapViewOfFile(s: *mut std::ffi::c_void, prot: u32, off_h: u32, off_l: u32, size: usize) -> *mut u8;
    fn UnmapViewOfFile(p: *const u8) -> i32;
    fn CloseHandle(h: *mut std::ffi::c_void) -> i32;
    fn GetLastError() -> u32;
}

#[cfg(not(windows))]
extern "C" {
    fn mmap(addr: *mut std::ffi::c_void, len: usize, prot: i32, flags: i32, fd: i32, offset: i64) -> *mut std::ffi::c_void;
    fn munmap(addr: *mut std::ffi::c_void, len: usize) -> i32;
}

enum MappedStorage {
    File(PartialView),
    #[cfg(test)]
    Owned(Arc<[u8]>),
}

pub struct MappedBytes {
    storage: MappedStorage,
    offset: usize,
    len: usize,
}

impl MappedBytes {
    #[cfg(test)]
    pub(crate) fn owned(bytes: Vec<u8>) -> MappedBytes {
        let len = bytes.len();
        MappedBytes { storage: MappedStorage::Owned(Arc::from(bytes)), offset: 0, len }
    }
}

impl Deref for MappedBytes {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        let data = match &self.storage {
            MappedStorage::File(mmap) => &mmap[..],
            #[cfg(test)]
            MappedStorage::Owned(data) => data,
        };
        &data[self.offset..self.offset + self.len]
    }
}

impl AsRef<[u8]> for MappedBytes {
    fn as_ref(&self) -> &[u8] { self }
}

impl GgufFile {
    /// A read-only view over exactly `[file_offset, file_offset + len)` of this shard's file, owned by
    /// the returned `PartialView`. Windows maps a sub-range of the shard's shared file section; Unix
    /// mmaps the range directly. Dropping the view unmaps it, so a cached expert's pages are released
    /// when the expert is evicted.
    fn map_partial(&self, file_offset: usize, len: usize) -> anyhow::Result<PartialView> {
        anyhow::ensure!(len > 0 && file_offset.checked_add(len).is_some_and(|end| end <= self.file_len),
            "GGUF range {file_offset}..{} exceeds file of {} bytes", file_offset.saturating_add(len), self.file_len);
        #[cfg(windows)]
        {
            // The view is write-copy, not page-readonly: this handle is opened GENERIC_READ, which lacks
            // the FILE_MAP_READ right, and Windows denies a PAGE_READONLY view on it (ERROR_ACCESS_DENIED)
            // even when the section is PAGE_READONLY. A write-copy view of a read-only section is the
            // standard read-only file mapping (the same one memmap2 uses); we never write it, so the
            // file is untouched and unmap still releases its physical pages. In-memory files cannot be
            // sectioned; the expert cache is disabled for those (no-mmap is required), so this is
            // unreachable.
            const PAGE_READONLY: u32 = 0x02;
            const PAGE_WRITECOPY: u32 = 0x01;
            let section = self.section.get_or_init(|| {
                let s = unsafe { CreateFileMappingW(self.file.as_raw_handle(), std::ptr::null(), PAGE_READONLY, 0, 0, std::ptr::null()) };
                if s.is_null() { Err(format!("create file section: {}", unsafe { GetLastError() })) } else { Ok(s) }
            });
            let section = *section.as_ref().map_err(|err| anyhow::anyhow!("{err}"))?;
            // MapViewOfFile requires the view offset to be a multiple of the 64 KB allocation
            // granularity (ERROR_DIRECTORY otherwise); GGUF offsets are 32-byte aligned, so round the
            // base down and advance the pointer, as the Unix branch does for pages.
            let base_off = file_offset & !(64 * 1024 - 1);
            let total = len + (file_offset - base_off);
            let base = unsafe { MapViewOfFile(section, PAGE_WRITECOPY, (base_off >> 32) as u32, base_off as u32, total) };
            anyhow::ensure!(!base.is_null(), "map GGUF expert range: {}", unsafe { GetLastError() });
            let ptr = unsafe { base.add(file_offset - base_off) };
            Ok(PartialView { ptr, len })
        }
        #[cfg(not(windows))]
        {
            const PROT_READ: i32 = 1;
            const MAP_PRIVATE: i32 = 2;
            let page = 4096usize;
            let base_off = file_offset & !(page - 1);
            let total = len + (file_offset - base_off);
            let fd = std::os::unix::fs::FileExt::as_raw_fd(&self.file);
            let base = unsafe { mmap(std::ptr::null_mut(), total, PROT_READ, MAP_PRIVATE, fd, base_off as i64) };
            if base == std::ptr::null_mut() { return Err(std::io::Error::last_os_error().into()); }
            Ok(PartialView { ptr: base as *mut u8 + (file_offset - base_off), len, base: base as *mut u8, total })
        }
    }
}

impl Drop for GgufFile {
    fn drop(&mut self) {
        #[cfg(windows)]
        if let Some(Ok(section)) = self.section.get() {
            let section = *section;
            if !section.is_null() { unsafe { let _ = CloseHandle(section); } }
        }
    }
}

struct Cur<'a> { b: &'a [u8], p: usize }

impl<'a> Cur<'a> {
    fn take(&mut self, n: usize) -> anyhow::Result<&'a [u8]> {
        let end = self.p.checked_add(n).context("GGUF truncated")?;
        let s = self.b.get(self.p..end).context("GGUF truncated")?;
        self.p = end;
        Ok(s)
    }
    fn u32(&mut self) -> anyhow::Result<u32> { Ok(u32::from_le_bytes(self.take(4)?.try_into()?)) }
    fn u64(&mut self) -> anyhow::Result<u64> { Ok(u64::from_le_bytes(self.take(8)?.try_into()?)) }
    fn str(&mut self) -> anyhow::Result<String> {
        let n = usize::try_from(self.u64()?).context("GGUF string length out of range")?;
        Ok(String::from_utf8_lossy(self.take(n)?).into_owned())
    }
    fn val(&mut self, t: u32) -> anyhow::Result<Value> {
        Ok(match t {
            0 => Value::U8(self.take(1)?[0]),
            1 => Value::I8(self.take(1)?[0] as i8),
            2 => Value::U16(u16::from_le_bytes(self.take(2)?.try_into()?)),
            3 => Value::I16(i16::from_le_bytes(self.take(2)?.try_into()?)),
            4 => Value::U32(self.u32()?),
            5 => Value::I32(self.u32()? as i32),
            6 => Value::F32(f32::from_bits(self.u32()?)),
            7 => Value::Bool(self.take(1)?[0] != 0),
            8 => Value::Str(self.str()?),
            9 => {
                let (et, n) = (self.u32()?, usize::try_from(self.u64()?).context("GGUF array length out of range")?);
                if n > self.b.len().saturating_sub(self.p) { bail!("GGUF array length exceeds remaining data"); }
                let mut a = Vec::with_capacity(n.min(1024));
                for _ in 0..n { a.push(self.val(et)?); }
                Value::Arr(a)
            }
            10 => Value::U64(self.u64()?),
            11 => Value::I64(self.u64()? as i64),
            12 => Value::F64(f64::from_bits(self.u64()?)),
            _ => bail!("unknown GGUF value type {t}"),
        })
    }
}

impl Gguf {
    /// Open a GGUF; for a split model (`split.count` > 1), pass the `-00001-of-0000N` shard and the others are
    /// opened from the same directory.
    pub fn open(path: &Path) -> anyhow::Result<Gguf> {
        Self::open_with_mmap(path, true)
    }

    /// Open a GGUF, optionally retaining its mapping for normal tensor reads. With mapping disabled, expert cache
    /// ranges can still create a shared lazy mapping when requested.
    pub fn open_with_mmap(path: &Path, mmap: bool) -> anyhow::Result<Gguf> {
        let mut g = Self::open_one(path, 0, mmap)?;
        g.mmap = mmap;
        let n = g.meta.get("split.count").and_then(Value::as_u64).unwrap_or(1) as usize;
        if n > 1 {
            let name = path.file_name().and_then(|s| s.to_str()).unwrap_or("");
            let tag = format!("-00001-of-{n:05}.gguf");
            let stem = name.strip_suffix(&tag).with_context(|| format!("{}: split model; open its first shard (*{tag})", path.display()))?;
            for i in 1..n {
                let s = Self::open_one(&path.with_file_name(format!("{stem}-{:05}-of-{n:05}.gguf", i + 1)), i, mmap)?;
                g.files.extend(s.files);
                for (k, t) in s.tensors { anyhow::ensure!(g.tensors.insert(k.clone(), t).is_none(), "tensor {k} in two shards"); }
            }
        }
        Ok(g)
    }

    fn open_one(path: &Path, file: usize, keep_mmap: bool) -> anyhow::Result<Gguf> {
        let f = File::open(path).with_context(|| format!("open {}", path.display()))?;
        let mmap = unsafe { Mmap::map(&f)? };
        let file_len = mmap.len();
        let mut c = Cur { b: &mmap, p: 0 };
        if c.take(4)? != b"GGUF" { bail!("{}: not a GGUF file", path.display()); }
        let ver = c.u32()?;
        if !(2..=3).contains(&ver) { bail!("unsupported GGUF version {ver}"); }
        let (nt, nkv) = (c.u64()?, c.u64()?);
        let mut meta = BTreeMap::new();
        for _ in 0..nkv {
            let k = c.str()?;
            let t = c.u32()?;
            meta.insert(k, c.val(t)?);
        }
        let mut tensors = BTreeMap::new();
        for _ in 0..nt {
            let name = c.str()?;
            let nd = c.u32()?;
            let dims = (0..nd).map(|_| c.u64()).collect::<anyhow::Result<Vec<_>>>()?;
            let (ty, offset) = (c.u32()?, c.u64()?);
            tensors.insert(name.clone(), TensorInfo { name, dims, ty, offset, file });
        }
        let align = usize::try_from(meta.get("general.alignment").and_then(Value::as_u64).unwrap_or(32))
            .context("GGUF alignment out of range")?;
        let data_offset = c.p.checked_next_multiple_of(align).context("invalid GGUF alignment")?;
        // A cut-off download keeps a valid header; catch it here, not as a slice panic in `bytes`.
        for t in tensors.values() {
            let Ok(ty) = GgmlType::from_u32(t.ty) else { continue };  // Weights::validate reports it
            let n = t.dims.iter().skip(1).try_fold(ty.row_bytes(t.dims.first().copied().unwrap_or(0)) as u64, |a, &d| a.checked_mul(d));
            if n.and_then(|n| n.checked_add(t.offset)).is_none_or(|e| e > (mmap.len() as u64).saturating_sub(data_offset as u64)) {
                bail!("{}: GGUF truncated (tensor {} lies past the end of the file; re-download it)", path.display(), t.name);
            }
        }
        let map = keep_mmap.then(|| Arc::new(mmap));
        let shard = GgufFile {
            mmap: map,
            file: f,
            data_offset,
            file_len,
            #[cfg(windows)]
            section: OnceLock::new(),
        };
        Ok(Gguf { meta, tensors, files: vec![shard], mmap: keep_mmap })
    }

    pub fn get(&self, key: &str) -> anyhow::Result<&Value> {
        self.meta.get(key).with_context(|| format!("missing GGUF key {key}"))
    }

    pub fn uses_mmap(&self) -> bool { self.mmap }

    /// Raw bytes of a tensor. Mapped models borrow the mapping; no-mmap models read the bytes into an owned buffer.
    pub fn bytes(&self, t: &TensorInfo) -> anyhow::Result<Cow<'_, [u8]>> {
        let ty = GgmlType::from_u32(t.ty).unwrap();
        let n = ty.row_bytes(t.dims[0]) * t.dims[1..].iter().product::<u64>() as usize;
        self.bytes_range(t, 0, n)
    }

    /// Read a range within a tensor, using the mapping when enabled and positioned file I/O otherwise.
    pub fn bytes_range(&self, t: &TensorInfo, offset: usize, len: usize) -> anyhow::Result<Cow<'_, [u8]>> {
        let ty = GgmlType::from_u32(t.ty).unwrap();
        let total = ty.row_bytes(t.dims[0]) * t.dims[1..].iter().product::<u64>() as usize;
        let end = offset.checked_add(len).context("tensor range overflow")?;
        anyhow::ensure!(end <= total, "tensor {} range {offset}..{end} exceeds {total} bytes", t.name);
        let shard = &self.files[t.file];
        let o = shard.data_offset.checked_add(t.offset as usize).and_then(|x| x.checked_add(offset))
            .context("GGUF tensor offset overflow")?;
        if let Some(mmap) = &shard.mmap {
            return Ok(Cow::Borrowed(&mmap[o..o + len]));
        }
        let mut data = vec![0u8; len];
        read_at(&shard.file, &mut data, o as u64)?;
        Ok(Cow::Owned(data))
    }

    /// Map a tensor byte range from its backing file as a view owned by the returned `MappedBytes`. The view
    /// covers exactly this range and is unmapped when the `MappedBytes` drops, so evicting a cached expert
    /// releases its physical pages.
    pub fn map_bytes_range(&self, t: &TensorInfo, offset: usize, len: usize) -> anyhow::Result<MappedBytes> {
        let ty = GgmlType::from_u32(t.ty).unwrap();
        let total = ty.row_bytes(t.dims[0]) * t.dims[1..].iter().product::<u64>() as usize;
        let end = offset.checked_add(len).context("tensor range overflow")?;
        anyhow::ensure!(end <= total, "tensor {} range {offset}..{end} exceeds {total} bytes", t.name);
        let shard = &self.files[t.file];
        let file_offset = shard.data_offset.checked_add(t.offset as usize).and_then(|x| x.checked_add(offset))
            .context("GGUF tensor offset overflow")?;
        let view = shard.map_partial(file_offset, len)?;
        Ok(MappedBytes { storage: MappedStorage::File(view), offset: 0, len })
    }
}

#[cfg(windows)]
fn read_at(file: &File, buf: &mut [u8], mut offset: u64) -> std::io::Result<()> {
    use std::os::windows::fs::FileExt;
    let mut read = 0;
    while read < buf.len() {
        let n = file.seek_read(&mut buf[read..], offset)?;
        if n == 0 { return Err(std::io::ErrorKind::UnexpectedEof.into()); }
        read += n;
        offset += n as u64;
    }
    Ok(())
}

#[cfg(unix)]
fn read_at(file: &File, buf: &mut [u8], mut offset: u64) -> std::io::Result<()> {
    use std::os::unix::fs::FileExt;
    let mut read = 0;
    while read < buf.len() {
        let n = file.read_at(&mut buf[read..], offset)?;
        if n == 0 { return Err(std::io::ErrorKind::UnexpectedEof.into()); }
        read += n;
        offset += n as u64;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    #[cfg(windows)]
    #[test]
    fn probe_file_section_mapping() {
        // Diagnostic: which (open access, section protection, view protection) combinations does this
        // Windows accept for a read-only on-disk file?
        use std::ffi::c_void;
        use std::io::Write;
        use std::os::windows::ffi::OsStrExt;
        unsafe extern "system" {
            fn CreateFileMappingW(file: *mut c_void, sa: *const c_void, prot: u32, max_h: u32, max_l: u32, name: *const u16) -> *mut c_void;
            fn MapViewOfFile(s: *mut c_void, prot: u32, off_h: u32, off_l: u32, size: usize) -> *mut u8;
            fn UnmapViewOfFile(p: *const u8) -> i32;
            fn CloseHandle(h: *mut c_void) -> i32;
            fn GetLastError() -> u32;
            fn CreateFileW(name: *const u16, access: u32, share: u32, sa: *const c_void, disp: u32, flags: u32, template: *mut c_void) -> isize;
        }
        const PAGE_READONLY: u32 = 0x02;
        const PAGE_WRITECOPY: u32 = 0x01;
        const FILE_MAP_READ: u32 = 0x04;
        const FILE_MAP_READWRITE: u32 = 0x04 | 0x02;
        const GENERIC_READ: u32 = 0x8000_0000;
        const FILE_READ_DATA: u32 = 0x0001;
        const FILE_READ_ATTRIBUTES: u32 = 0x0008;
        const READ_CONTROL: u32 = 0x0000_8000;
        const FILE_MAP_READ_ACCESS: u32 = FILE_READ_DATA | FILE_READ_ATTRIBUTES | READ_CONTROL | 0x0010_0000; // + FILE_MAP_READ
        const FILE_SHARE: u32 = 0x07;
        const OPEN_EXISTING: u32 = 3;
        const FILE_ATTRIBUTE_NORMAL: u32 = 0x80;

        let dir = std::env::temp_dir();
        let path = dir.join(format!("strata-probe-{}.bin", std::process::id()));
        {
            let mut f = std::fs::File::create(&path).unwrap();
            f.write_all(&[0u8; 262144]).unwrap();
        }
        let name: Vec<u16> = path.as_os_str().encode_wide().chain(std::iter::once(0)).collect();

        let open = |access: u32| -> isize {
            unsafe { CreateFileW(name.as_ptr(), access, FILE_SHARE, std::ptr::null(), OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, std::ptr::null_mut()) }
        };
        let h_read = open(GENERIC_READ);
        let h_readdata = open(FILE_READ_DATA | FILE_READ_ATTRIBUTES | READ_CONTROL);
        let h_mapread = open(FILE_MAP_READ_ACCESS);
        let h_writedata = open(GENERIC_READ | 0x4000_0000); // + GENERIC_WRITE
        let mut out = String::new();
        for (label, h) in [("GENERIC_READ", h_read), ("READ_DATA|ATTR|RC", h_readdata), ("+FILE_MAP_READ", h_mapread), ("GENERIC_READ|WRITE", h_writedata)] {
            if h == -1 { out.push_str(&format!("{label}: open failed {}\n", unsafe { GetLastError() })); continue; }
            for (sp, prot) in [("PAGE_READONLY", PAGE_READONLY), ("FILE_MAP_READ(0x4)", FILE_MAP_READ), ("FILE_MAP_READWRITE(0x6)", FILE_MAP_READWRITE), ("0", 0u32)] {
                let s = unsafe { CreateFileMappingW(h as *mut c_void, std::ptr::null(), prot, 0, 0, std::ptr::null()) };
                if s.is_null() {
                    out.push_str(&format!("{label} x {sp}: section {}\n", unsafe { GetLastError() }));
                    continue;
                }
                for (vp, vprot) in [("PAGE_READONLY", PAGE_READONLY), ("PAGE_WRITECOPY", 0x01u32), ("PAGE_READWRITE", 0x04u32)] {
                    let v = unsafe { MapViewOfFile(s, vprot, 0, 0, 8192) };
                    let r = if v.is_null() { format!("view {}", unsafe { GetLastError() }) } else {
                        let b = unsafe { *v };
                        let _ = b;
                        let _ = unsafe { UnmapViewOfFile(v) };
                        "ok".to_string()
                    };
                    out.push_str(&format!("{label} x {sp} x {vp}: {r}\n"));
                }
                let _ = unsafe { CloseHandle(s) };
            }
            let _ = unsafe { CloseHandle(h as *mut c_void) };
        }
        // View offsets must be multiples of the 64 KB allocation granularity.
        let h = open(GENERIC_READ);
        let s = unsafe { CreateFileMappingW(h as *mut c_void, std::ptr::null(), PAGE_READONLY, 0, 0, std::ptr::null()) };
        for off in [0usize, 4096, 65536, 131072] {
            let v = unsafe { MapViewOfFile(s, PAGE_WRITECOPY, (off >> 32) as u32, off as u32, 4096) };
            let r = if v.is_null() { format!("view {}", unsafe { GetLastError() }) } else {
                let _ = unsafe { UnmapViewOfFile(v) };
                "ok".to_string()
            };
            out.push_str(&format!("alignment offset {off}: {r}\n"));
        }
        // The fix: round the offset down to 64 KB, map the padded range, advance the pointer.
        let off = 4096usize;
        let base_off = off & !(65536 - 1);
        let len = 8192usize;
        let v = unsafe { MapViewOfFile(s, PAGE_WRITECOPY, (base_off >> 32) as u32, base_off as u32, len + (off - base_off)) };
        let r = if v.is_null() { format!("view {}", unsafe { GetLastError() }) } else {
            let p = unsafe { v.add(off - base_off) };
            let first = unsafe { *p };
            let last = unsafe { *p.add(len - 1) };
            let _ = unsafe { UnmapViewOfFile(v) };
            format!("ok (read {first}..{last})")
        };
        out.push_str(&format!("rounded offset {off} (base {base_off}, len {len}): {r}\n"));
        let _ = unsafe { CloseHandle(s) };
        let _ = unsafe { CloseHandle(h as *mut c_void) };
        let _ = std::fs::remove_file(&path);
        eprintln!("{out}");
    }
}
