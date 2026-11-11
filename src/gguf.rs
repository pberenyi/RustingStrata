//! GGUF v2/v3 parser over a memory map or positioned file reads.

use anyhow::{Context, bail};
use std::borrow::Cow;
use std::fs::File;
use memmap2::Mmap;
use std::collections::BTreeMap;
use std::path::Path;
use std::ops::Deref;
use std::sync::{Arc, OnceLock};

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
    /// Each shard's retained or lazy map, open file and data offset (one entry unless split).
    files: Vec<GgufFile>,
    mmap: bool,
}

struct GgufFile { mmap: Option<Arc<Mmap>>, lazy_mmap: OnceLock<Result<Arc<Mmap>, String>>, file: File, data_offset: usize }

enum MappedStorage {
    File(Arc<Mmap>),
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
    fn mapping(&self) -> anyhow::Result<Arc<Mmap>> {
        if let Some(mmap) = &self.mmap { return Ok(mmap.clone()); }
        self.lazy_mmap.get_or_init(|| {
            unsafe { Mmap::map(&self.file) }
                .map(Arc::new)
                .map_err(|err| err.to_string())
        }).as_ref().map(Arc::clone).map_err(|err| anyhow::anyhow!("map GGUF file for expert cache: {err}"))
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
        Ok(Gguf { meta, tensors, files: vec![GgufFile { mmap: map, lazy_mmap: OnceLock::new(), file: f, data_offset }], mmap: keep_mmap })
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

    /// Map a tensor byte range from its backing file, retaining a shared whole-file view for reuse by cached experts.
    pub fn map_bytes_range(&self, t: &TensorInfo, offset: usize, len: usize) -> anyhow::Result<MappedBytes> {
        let ty = GgmlType::from_u32(t.ty).unwrap();
        let total = ty.row_bytes(t.dims[0]) * t.dims[1..].iter().product::<u64>() as usize;
        let end = offset.checked_add(len).context("tensor range overflow")?;
        anyhow::ensure!(end <= total, "tensor {} range {offset}..{end} exceeds {total} bytes", t.name);
        let shard = &self.files[t.file];
        let file_offset = shard.data_offset.checked_add(t.offset as usize).and_then(|x| x.checked_add(offset))
            .context("GGUF tensor offset overflow")?;
        let mmap = shard.mapping()?;
        anyhow::ensure!(file_offset.checked_add(len).is_some_and(|end| end <= mmap.len()), "mapped tensor range exceeds GGUF file");
        Ok(MappedBytes { storage: MappedStorage::File(mmap), offset: file_offset, len })
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
