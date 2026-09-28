//! Blockwise normalized Sylvester Walsh-Hadamard transform used by Prism ML's
//! `prism_hadamard_qwen35` checkpoints (Ternary-Bonsai-2-27B).
//!
//! Every packed linear/embedding weight in these checkpoints is quantized in a
//! *rotated* basis: activations must be passed through a sign flip followed by
//! a blockwise Hadamard rotation (or the inverse, sign flip after rotation)
//! before/after they meet the quantized matmul. Reference: `runtime/runtime.py`
//! `fwht()` in the Prism ML checkpoint's bundled runtime.
//!
//! ```text
//! forward (linear input):    y = (x * signs) reshaped to (-1, block) @ H
//! inverse (embedding output): y = (x reshaped to (-1, block) @ H) * signs
//! ```
//! where `H` is the natural-order Sylvester Hadamard matrix of size `block`,
//! scaled by `1 / sqrt(block)` (`H_1024 / 32` for the shipped manifest).
//! `H` is never materialized: both directions run a fused sign flip +
//! O(n log n) butterfly FWHT (`BlockFwht`) on CPU and Metal.
use crate::qwen3::SafeTensorsSource;
use candle::{DType, Device, Result, Tensor};
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

/// Flat JSON shape of `hadamard.json`: top-level keys are dotted strings
/// (`"prism.hadamard.version"`, ...), not nested objects.
#[derive(Debug, Clone, serde::Deserialize)]
struct RawManifest {
    #[serde(rename = "prism.hadamard.version")]
    version: u32,
    #[serde(rename = "prism.hadamard.block_size")]
    block_size: usize,
    #[serde(rename = "prism.hadamard.transform")]
    transform: String,
    #[serde(rename = "prism.hadamard.axis")]
    axis: String,
    #[serde(rename = "prism.hadamard.sign_mode")]
    sign_mode: String,
    #[serde(rename = "prism.hadamard.weight_names")]
    weight_names: Vec<String>,
    #[serde(rename = "prism.hadamard.inverse_weight_names")]
    inverse_weight_names: Vec<String>,
    #[serde(rename = "prism.hadamard.sign_widths")]
    sign_widths: Vec<usize>,
    #[serde(rename = "prism.hadamard.sign_values")]
    sign_values: Vec<f32>,
    /// Whether the folded GDN `out_proj`/`in_proj_*` v-head layout is already
    /// grouped (contiguous per k-head) in the saved tensors, matching what
    /// this loader assumes. Absent in some manifests, in which case we
    /// conservatively assume `true` (the common case for MLX-native
    /// checkpoints); see `HadamardManifest::assert_gdn_layout_supported`.
    #[serde(rename = "prism.hadamard.gdn_v_grouped", default)]
    gdn_v_grouped: Option<bool>,
}

/// Parsed and validated `hadamard.json` (Prism contract v1).
#[derive(Debug, Clone)]
pub struct HadamardManifest {
    pub block_size: usize,
    pub weight_names: HashSet<String>,
    pub inverse_weight_names: HashSet<String>,
    pub signs_by_width: HashMap<usize, Vec<f32>>,
    pub gdn_v_grouped: bool,
}

impl HadamardManifest {
    pub fn from_json(s: &str) -> anyhow::Result<Self> {
        let raw: RawManifest = serde_json::from_str(s)
            .map_err(|e| anyhow::anyhow!("invalid hadamard manifest: {e}"))?;
        if raw.version != 1 {
            anyhow::bail!("unsupported hadamard manifest version {}", raw.version);
        }
        if raw.transform != "normalized-sylvester-walsh-hadamard" {
            anyhow::bail!("unsupported hadamard transform {:?}", raw.transform);
        }
        if raw.axis != "input-last-dimension" {
            anyhow::bail!("unsupported hadamard axis {:?}", raw.axis);
        }
        if raw.sign_mode != "explicit" {
            anyhow::bail!("unsupported hadamard sign_mode {:?}", raw.sign_mode);
        }
        if raw.block_size < 2 || !raw.block_size.is_power_of_two() {
            anyhow::bail!(
                "hadamard block_size {} must be a power of two >= 2",
                raw.block_size
            );
        }

        let weight_names: HashSet<String> = raw.weight_names.into_iter().collect();
        let inverse_weight_names: HashSet<String> = raw.inverse_weight_names.into_iter().collect();
        if !weight_names.is_disjoint(&inverse_weight_names) {
            anyhow::bail!("hadamard weight_names and inverse_weight_names overlap");
        }

        let sum_widths: usize = raw.sign_widths.iter().sum();
        if sum_widths != raw.sign_values.len() {
            anyhow::bail!(
                "hadamard sign_widths sum {sum_widths} != sign_values length {}",
                raw.sign_values.len()
            );
        }
        for &v in &raw.sign_values {
            if v != 1.0 && v != -1.0 {
                anyhow::bail!("hadamard sign_values must be exactly +-1.0, found {v}");
            }
        }

        let mut signs_by_width = HashMap::new();
        let mut offset = 0;
        for &w in &raw.sign_widths {
            if !w.is_multiple_of(raw.block_size) {
                anyhow::bail!(
                    "hadamard sign width {w} is not a multiple of block_size {}",
                    raw.block_size
                );
            }
            signs_by_width.insert(w, raw.sign_values[offset..offset + w].to_vec());
            offset += w;
        }

        Ok(Self {
            block_size: raw.block_size,
            weight_names,
            inverse_weight_names,
            signs_by_width,
            gdn_v_grouped: raw.gdn_v_grouped.unwrap_or(true),
        })
    }

    /// Mirrors the reference GGUF loader's defensive check (`runtime/runtime.py`
    /// bails with "Unimplemented ungrouped folded GDN output" in this case):
    /// an ungrouped v-head layout (`gdn_v_grouped == false`) combined with
    /// `linear_num_value_heads != linear_num_key_heads` needs a head
    /// permutation this loader does not implement, so loading such a
    /// checkpoint would silently misinterpret the v-head layout instead of
    /// failing. `n_v_heads`/`n_k_heads` come from `text_config`, not the
    /// manifest, so the caller passes them in.
    pub fn assert_gdn_layout_supported(
        &self,
        n_v_heads: usize,
        n_k_heads: usize,
    ) -> anyhow::Result<()> {
        if !self.gdn_v_grouped && n_v_heads != n_k_heads {
            anyhow::bail!(
                "hadamard.json gdn_v_grouped=false with linear_num_value_heads ({n_v_heads}) != \
                 linear_num_key_heads ({n_k_heads}) is unsupported (ungrouped folded GDN output)"
            );
        }
        Ok(())
    }
}

/// Unnormalized natural-order (Sylvester / Hadamard-ordered) fast
/// Walsh-Hadamard transform of one block, in place: equivalent to multiplying
/// the row vector `a` by the unscaled Sylvester matrix `H[i][j] =
/// (-1)^popcount(i & j)`. Butterfly strides go `1, 2, 4, ...`; a
/// sequency-ordered variant would be a different matrix.
fn fwht_in_place(a: &mut [f32]) {
    let n = a.len();
    let mut h = 1;
    while h < n {
        for pair in a.chunks_exact_mut(2 * h) {
            let (lo, hi) = pair.split_at_mut(h);
            for (x, y) in lo.iter_mut().zip(hi.iter_mut()) {
                let (a, b) = (*x, *y);
                *x = a + b;
                *y = a - b;
            }
        }
        h *= 2;
    }
}

/// Fused sign flip + blockwise normalized FWHT over the last dimension, as a
/// candle custom op on `(x, signs)`. `signs_first` selects the convention:
/// `true` = forward (`(x * signs) @ H`), `false` = inverse (`(x @ H) * signs`).
/// Both operands must be contiguous F32; `signs` has `width` elements and is
/// indexed by position within the last dimension of `x`.
struct BlockFwht {
    block: usize,
    width: usize,
    signs_first: bool,
}

impl BlockFwht {
    fn check(&self, x_layout: &candle::Layout, signs_layout: &candle::Layout) -> Result<()> {
        if !x_layout.is_contiguous() || !signs_layout.is_contiguous() {
            candle::bail!("hadamard: block_fwht requires contiguous inputs");
        }
        if signs_layout.shape().elem_count() != self.width {
            candle::bail!(
                "hadamard: signs length {} != width {}",
                signs_layout.shape().elem_count(),
                self.width
            );
        }
        if !x_layout.shape().elem_count().is_multiple_of(self.width) {
            candle::bail!(
                "hadamard: input of {} elements is not a whole number of width-{} rows",
                x_layout.shape().elem_count(),
                self.width
            );
        }
        Ok(())
    }
}

impl candle::CustomOp2 for BlockFwht {
    fn name(&self) -> &'static str {
        "hadamard-block-fwht"
    }

    fn cpu_fwd(
        &self,
        s1: &candle::CpuStorage,
        l1: &candle::Layout,
        s2: &candle::CpuStorage,
        l2: &candle::Layout,
    ) -> Result<(candle::CpuStorage, candle::Shape)> {
        self.check(l1, l2)?;
        let (candle::CpuStorage::F32(x), candle::CpuStorage::F32(signs)) = (s1, s2) else {
            candle::bail!("hadamard: block_fwht only supports f32");
        };
        let x = &x[l1.start_offset()..l1.start_offset() + l1.shape().elem_count()];
        let signs = &signs[l2.start_offset()..l2.start_offset() + self.width];
        let scale = 1f32 / (self.block as f32).sqrt();

        let blocks_per_row = self.width / self.block;
        let mut out = vec![0f32; x.len()];
        // Blocks are independent; small inputs (decode) stay on this thread
        // because rayon does not split below `with_min_len`.
        out.par_chunks_exact_mut(self.block)
            .zip(x.par_chunks_exact(self.block))
            .enumerate()
            .with_min_len(16)
            .for_each(|(i, (dst, src))| {
                let s = &signs[(i % blocks_per_row) * self.block..][..self.block];
                if self.signs_first {
                    for ((d, &v), &s) in dst.iter_mut().zip(src).zip(s) {
                        *d = v * s;
                    }
                    fwht_in_place(dst);
                    for d in dst.iter_mut() {
                        *d *= scale;
                    }
                } else {
                    dst.copy_from_slice(src);
                    fwht_in_place(dst);
                    for (d, &s) in dst.iter_mut().zip(s) {
                        *d *= scale * s;
                    }
                }
            });
        Ok((candle::CpuStorage::F32(out), l1.shape().clone()))
    }

    #[cfg(target_os = "macos")]
    fn metal_fwd(
        &self,
        s1: &candle::MetalStorage,
        l1: &candle::Layout,
        s2: &candle::MetalStorage,
        l2: &candle::Layout,
    ) -> Result<(candle::MetalStorage, candle::Shape)> {
        use candle::backend::BackendStorage;
        use candle::metal_backend::buffer_o;

        self.check(l1, l2)?;
        if s1.dtype() != DType::F32 || s2.dtype() != DType::F32 {
            candle::bail!("hadamard: block_fwht only supports f32");
        }
        if self.block > candle_metal_kernels::HADAMARD_MAX_BLOCK {
            candle::bail!(
                "hadamard: block {} exceeds the Metal kernel limit {}",
                self.block,
                candle_metal_kernels::HADAMARD_MAX_BLOCK
            );
        }
        let device = s1.device();
        let el = l1.shape().elem_count();
        // Reads `x` in place (offset honored) and writes one fresh buffer: no
        // staging copy, no separate sign-multiply pass.
        let dst = device.new_buffer(el, DType::F32, "hadamard-block-fwht")?;
        let encoder = device.command_encoder()?;
        candle_metal_kernels::call_hadamard_block_fwht(
            device.metal_device(),
            &encoder,
            device.kernels(),
            el / self.block,
            self.block,
            self.width / self.block,
            self.signs_first,
            buffer_o(s1.buffer(), l1, DType::F32),
            buffer_o(s2.buffer(), l2, DType::F32),
            &dst,
        )
        .map_err(candle::Error::wrap)?;
        Ok((
            candle::MetalStorage::new(dst, device.clone(), el, DType::F32),
            l1.shape().clone(),
        ))
    }
}

/// A ready-to-apply Hadamard rotation for one input width: the block size plus
/// this width's sign vector. The rotation itself is an O(n log n) butterfly,
/// never a dense `H_block` matmul.
#[derive(Debug)]
pub struct HadamardTransform {
    block: usize,
    signs: Tensor,
}

impl HadamardTransform {
    pub fn new(block: usize, signs: &[f32], device: &Device) -> Result<Self> {
        if block < 2 || !block.is_power_of_two() {
            candle::bail!("hadamard: block {block} must be a power of two >= 2");
        }
        if signs.is_empty() || !signs.len().is_multiple_of(block) {
            candle::bail!(
                "hadamard: sign vector length {} is not a positive multiple of block {block}",
                signs.len()
            );
        }
        let signs = Tensor::from_slice(signs, signs.len(), device)?;
        Ok(Self { block, signs })
    }

    /// `(x * signs) reshaped to (-1, block) @ H`, reshaped back to `x`'s shape.
    /// `x`: `(..., width)` F32 with `width` equal to the sign vector length.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.run(x, true, "forward")
    }

    /// `(x reshaped to (-1, block) @ H) * signs`, reshaped back to `x`'s shape.
    pub fn inverse(&self, x: &Tensor) -> Result<Tensor> {
        self.run(x, false, "inverse")
    }

    fn run(&self, x: &Tensor, signs_first: bool, op: &str) -> Result<Tensor> {
        let width = self.signs.elem_count();
        let last = *x
            .dims()
            .last()
            .ok_or_else(|| candle::Error::msg(format!("hadamard {op}: input has no dims")))?;
        if last != width {
            candle::bail!("hadamard {op}: last dim {last} != sign vector width {width}");
        }
        x.contiguous()?.apply_op2_no_bwd(
            &self.signs,
            &BlockFwht {
                block: self.block,
                width,
                signs_first,
            },
        )
    }
}

/// Applies an optional Hadamard transform, passing `x` through unchanged
/// (a cheap `Arc`-backed clone) when `tf` is `None` — the common case for
/// checkpoints with no `hadamard_config` (plain `qwen3_5`).
pub fn apply(tf: &Option<Arc<HadamardTransform>>, x: &Tensor) -> Result<Tensor> {
    match tf {
        Some(t) => t.forward(x),
        None => Ok(x.clone()),
    }
}

/// Drives Hadamard-transform construction while loading a `prism_hadamard_qwen35`
/// checkpoint: resolves which tensors the manifest says need a transform, builds
/// (and caches, by width) the shared per-width `HadamardTransform`, cross-checks
/// each tensor's `.signs` against the manifest, and tracks which manifest names
/// were actually claimed by the loader so a silently un-rotated projection (e.g.
/// a future checkpoint variant) is caught rather than silently mis-inferred.
pub struct HadamardLoader {
    manifest: HadamardManifest,
    cache: HashMap<usize, Arc<HadamardTransform>>,
    claimed: HashSet<String>,
    device: Device,
}

impl HadamardLoader {
    pub fn new(manifest: HadamardManifest, device: &Device) -> Result<Self> {
        Ok(Self {
            manifest,
            cache: HashMap::new(),
            claimed: HashSet::new(),
            device: device.clone(),
        })
    }

    /// If `"{base}.weight"` is in the manifest's `weight_names`, returns the
    /// (cached, shared) forward transform for `width`, verifying `{base}.signs`
    /// against the manifest along the way. Returns `None` for tensors the
    /// manifest doesn't rotate (e.g. the dense `in_proj_a/b` projections).
    pub fn transform_for(
        &mut self,
        base: &str,
        width: usize,
        src: &SafeTensorsSource<'_>,
    ) -> Result<Option<Arc<HadamardTransform>>> {
        let weight_name = format!("{base}.weight");
        if !self.manifest.weight_names.contains(&weight_name) {
            return Ok(None);
        }
        self.verify_signs(base, width, src)?;
        self.claimed.insert(weight_name);
        Ok(Some(self.transform_for_width(width)?))
    }

    /// Same as `transform_for`, but against `inverse_weight_names` (the
    /// embedding table).
    pub fn inverse_for(
        &mut self,
        base: &str,
        width: usize,
        src: &SafeTensorsSource<'_>,
    ) -> Result<Option<Arc<HadamardTransform>>> {
        let weight_name = format!("{base}.weight");
        if !self.manifest.inverse_weight_names.contains(&weight_name) {
            return Ok(None);
        }
        self.verify_signs(base, width, src)?;
        self.claimed.insert(weight_name);
        Ok(Some(self.transform_for_width(width)?))
    }

    /// Errors if any manifest-listed weight/inverse-weight name was never
    /// claimed by `transform_for`/`inverse_for` during loading.
    pub fn finish(self) -> Result<()> {
        let unclaimed: Vec<&String> = self
            .manifest
            .weight_names
            .iter()
            .chain(self.manifest.inverse_weight_names.iter())
            .filter(|n| !self.claimed.contains(n.as_str()))
            .collect();
        if !unclaimed.is_empty() {
            candle::bail!(
                "hadamard manifest has {} unclaimed weight name(s), e.g. {}",
                unclaimed.len(),
                unclaimed[0]
            );
        }
        Ok(())
    }

    fn transform_for_width(&mut self, width: usize) -> Result<Arc<HadamardTransform>> {
        if let Some(t) = self.cache.get(&width) {
            return Ok(t.clone());
        }
        let signs = self.manifest.signs_by_width.get(&width).ok_or_else(|| {
            candle::Error::msg(format!("hadamard manifest has no signs for width {width}"))
        })?;
        let t = Arc::new(HadamardTransform::new(
            self.manifest.block_size,
            signs,
            &self.device,
        )?);
        self.cache.insert(width, t.clone());
        Ok(t)
    }

    fn verify_signs(&self, base: &str, width: usize, src: &SafeTensorsSource<'_>) -> Result<()> {
        let signs_name = format!("{base}.signs");
        if !src.has_tensor(&signs_name) {
            candle::bail!(
                "hadamard: missing {signs_name} for manifest-claimed weight {base}.weight"
            );
        }
        let actual = src
            .plain_tensor(&signs_name)?
            .to_dtype(DType::F32)?
            .to_vec1::<f32>()?;
        let expected = self.manifest.signs_by_width.get(&width).ok_or_else(|| {
            candle::Error::msg(format!(
                "hadamard manifest has no signs for width {width} (needed by {base})"
            ))
        })?;
        if actual.len() != expected.len() {
            candle::bail!(
                "hadamard: {signs_name} length {} != manifest width {}",
                actual.len(),
                expected.len()
            );
        }
        if actual != *expected {
            candle::bail!(
                "hadamard: {signs_name} does not match manifest sign vector for width {width}"
            );
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle::DType;
    use rand::Rng;

    /// Dense reference: the natural-order Sylvester Hadamard matrix of size
    /// `block`, scaled by `1 / sqrt(block)`: `H[i][j] = (-1)^popcount(i & j) /
    /// sqrt(block)`. This is what the transform used to matmul against.
    fn sylvester_matrix(block: usize, device: &Device) -> Result<Tensor> {
        let scale = 1f32 / (block as f32).sqrt();
        let mut data = vec![0f32; block * block];
        for i in 0..block {
            for j in 0..block {
                let sign = if (i & j).count_ones() % 2 == 0 {
                    scale
                } else {
                    -scale
                };
                data[i * block + j] = sign;
            }
        }
        Tensor::from_vec(data, (block, block), device)
    }

    /// The previous dense implementation of `HadamardTransform::forward`,
    /// kept as the convention oracle: signs first, then rotate.
    fn dense_forward(x: &Tensor, signs: &Tensor, h: &Tensor, block: usize) -> Result<Tensor> {
        let dims = x.dims().to_vec();
        x.broadcast_mul(signs)?
            .reshape(((), block))?
            .matmul(h)?
            .reshape(dims)
    }

    /// The previous dense implementation of `HadamardTransform::inverse`:
    /// rotate first, then signs.
    fn dense_inverse(x: &Tensor, signs: &Tensor, h: &Tensor, block: usize) -> Result<Tensor> {
        let dims = x.dims().to_vec();
        x.reshape(((), block))?
            .matmul(h)?
            .reshape(dims)?
            .broadcast_mul(signs)
    }

    /// Reference Sylvester construction: H_1 = [1]; H_{2n} = [[H, H], [H, -H]].
    /// Not normalized (caller divides by sqrt(block)).
    fn recursive_sylvester(block: usize) -> Vec<f32> {
        let mut h = vec![1f32];
        let mut n = 1;
        while n < block {
            let mut next = vec![0f32; (n * 2) * (n * 2)];
            for i in 0..n {
                for j in 0..n {
                    let v = h[i * n + j];
                    next[i * (2 * n) + j] = v;
                    next[i * (2 * n) + n + j] = v;
                    next[(n + i) * (2 * n) + j] = v;
                    next[(n + i) * (2 * n) + n + j] = -v;
                }
            }
            h = next;
            n *= 2;
        }
        h
    }

    /// Natural-order (Hadamard-ordered) unnormalized butterfly FWHT, in place.
    fn butterfly_fwht(a: &mut [f32]) {
        let n = a.len();
        let mut h = 1;
        while h < n {
            let mut i = 0;
            while i < n {
                for j in i..i + h {
                    let x = a[j];
                    let y = a[j + h];
                    a[j] = x + y;
                    a[j + h] = x - y;
                }
                i += 2 * h;
            }
            h *= 2;
        }
    }

    fn random_vec(n: usize) -> Vec<f32> {
        let mut rng = rand::rng();
        (0..n).map(|_| rng.random_range(-1.0..1.0)).collect()
    }

    fn random_signs(n: usize) -> Vec<f32> {
        let mut rng = rand::rng();
        (0..n)
            .map(|_| if rng.random_bool(0.5) { 1.0 } else { -1.0 })
            .collect()
    }

    fn assert_close(actual: &Tensor, expected: &Tensor, tol: f32) {
        assert_eq!(actual.dims(), expected.dims());
        let a = actual.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let e = expected.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for (i, (a, e)) in a.iter().zip(e.iter()).enumerate() {
            assert!((a - e).abs() <= tol, "index {i}: {a} != {e}");
        }
    }

    #[test]
    fn sylvester_matrix_matches_recursive_construction() {
        let device = Device::Cpu;
        let block = 8;
        let scale = 1f32 / (block as f32).sqrt();
        let expected: Vec<f32> = recursive_sylvester(block)
            .iter()
            .map(|v| v * scale)
            .collect();
        let actual = sylvester_matrix(block, &device)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        for (a, e) in actual.iter().zip(expected.iter()) {
            assert!((a - e).abs() < 1e-6, "{a} != {e}");
        }
    }

    #[test]
    fn sylvester_matrix_is_orthogonal() {
        let device = Device::Cpu;
        let block = 1024;
        let h = sylvester_matrix(block, &device).unwrap();
        let product = h.matmul(&h.t().unwrap()).unwrap();
        let data = product.to_vec2::<f32>().unwrap();
        for (i, row) in data.iter().enumerate() {
            for (j, &v) in row.iter().enumerate() {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((v - expected).abs() < 1e-4, "H*H^T[{i}][{j}] = {v}");
            }
        }
    }

    #[test]
    fn fwht_in_place_equals_dense_sylvester_rows() {
        // Transforming the unit vector e_i must yield row i of the unscaled
        // Sylvester matrix (x @ H with H symmetric), for every block size.
        for log in 1..=10 {
            let block = 1usize << log;
            let h = recursive_sylvester(block);
            for i in [0, 1, block / 2, block - 1] {
                let mut e = vec![0f32; block];
                e[i] = 1.0;
                fwht_in_place(&mut e);
                assert_eq!(e, h[i * block..(i + 1) * block], "block {block} row {i}");
            }
        }
    }

    #[test]
    fn forward_matches_butterfly_fwht_per_block() {
        let device = Device::Cpu;
        let block = 8;
        let width = 16; // two blocks
        let input = random_vec(width);

        let signs = vec![1f32; width]; // isolate the Hadamard rotation from the sign flip
        let transform = HadamardTransform::new(block, &signs, &device).unwrap();
        let x = Tensor::from_vec(input.clone(), width, &device).unwrap();
        let actual = transform.forward(&x).unwrap().to_vec1::<f32>().unwrap();

        let scale = 1f32 / (block as f32).sqrt();
        let mut expected = input.clone();
        for chunk in expected.chunks_mut(block) {
            butterfly_fwht(chunk);
            for v in chunk.iter_mut() {
                *v *= scale;
            }
        }
        for (a, e) in actual.iter().zip(expected.iter()) {
            assert!((a - e).abs() < 1e-4, "{a} != {e}");
        }
    }

    /// Locks the sign conventions and the per-row sign indexing against the
    /// previous dense implementation: several blocks per row (width > block)
    /// and several rows (so a sign index taken modulo the whole tensor instead
    /// of the row would diverge), with random +-1 signs.
    #[test]
    fn forward_and_inverse_match_dense_reference_multi_block() {
        let device = Device::Cpu;
        let block = 1024;
        let width = 5 * block; // 5120, the hidden-state width
        let signs = random_signs(width);
        let transform = HadamardTransform::new(block, &signs, &device).unwrap();
        let signs_t = Tensor::from_slice(&signs, width, &device).unwrap();
        let h = sylvester_matrix(block, &device).unwrap();
        let x = Tensor::from_vec(random_vec(2 * 3 * width), (2, 3, width), &device).unwrap();

        let fwd = transform.forward(&x).unwrap();
        assert_close(&fwd, &dense_forward(&x, &signs_t, &h, block).unwrap(), 1e-4);
        let inv = transform.inverse(&x).unwrap();
        assert_close(&inv, &dense_inverse(&x, &signs_t, &h, block).unwrap(), 1e-4);
    }

    #[test]
    fn forward_and_inverse_match_dense_reference_small_blocks() {
        let device = Device::Cpu;
        for block in [2, 4, 8, 16, 32, 64, 128, 256, 512] {
            let width = 3 * block;
            let signs = random_signs(width);
            let transform = HadamardTransform::new(block, &signs, &device).unwrap();
            let signs_t = Tensor::from_slice(&signs, width, &device).unwrap();
            let h = sylvester_matrix(block, &device).unwrap();
            let x = Tensor::from_vec(random_vec(4 * width), (4, width), &device).unwrap();
            assert_close(
                &transform.forward(&x).unwrap(),
                &dense_forward(&x, &signs_t, &h, block).unwrap(),
                1e-5,
            );
            assert_close(
                &transform.inverse(&x).unwrap(),
                &dense_inverse(&x, &signs_t, &h, block).unwrap(),
                1e-5,
            );
        }
    }

    #[test]
    fn inverse_of_forward_recovers_input() {
        let device = Device::Cpu;
        let block = 1024;
        let width = 1024;
        let input = random_vec(width);
        let signs = random_signs(width);

        let transform = HadamardTransform::new(block, &signs, &device).unwrap();
        let x = Tensor::from_vec(input.clone(), width, &device).unwrap();
        let rotated = transform.forward(&x).unwrap();
        let recovered = transform
            .inverse(&rotated)
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();

        for (a, e) in recovered.iter().zip(input.iter()) {
            assert!((a - e).abs() < 1e-3, "{a} != {e}");
        }
    }

    #[test]
    fn forward_preserves_leading_dims() {
        let device = Device::Cpu;
        let block = 8;
        let signs = vec![1f32; block];
        let transform = HadamardTransform::new(block, &signs, &device).unwrap();
        let x = Tensor::zeros((2, 3, block), DType::F32, &device).unwrap();
        let y = transform.forward(&x).unwrap();
        assert_eq!(y.dims(), &[2, 3, block]);
    }

    /// `last_hidden = h.narrow(1, l - 1, 1)` in `Qwen35Weights::forward` is a
    /// strided view for batch > 1 and an offset view for batch == 1; both must
    /// be read correctly, not from the start of the underlying storage.
    #[test]
    fn forward_handles_narrowed_inputs() {
        let device = Device::Cpu;
        let block = 8;
        let width = 16;
        let signs = random_signs(width);
        let transform = HadamardTransform::new(block, &signs, &device).unwrap();
        for batch in [1, 2] {
            let x = Tensor::from_vec(random_vec(batch * 4 * width), (batch, 4, width), &device)
                .unwrap();
            let last = x.narrow(1, 3, 1).unwrap();
            let expected = transform.forward(&last.contiguous().unwrap()).unwrap();
            assert_close(&transform.forward(&last).unwrap(), &expected, 0.0);
        }
    }

    #[test]
    fn transform_rejects_mismatched_width_and_dtype() {
        let device = Device::Cpu;
        let transform = HadamardTransform::new(8, &[1f32; 16], &device).unwrap();
        // A multiple of the block, but not the sign vector's width.
        let x = Tensor::zeros((1, 8), DType::F32, &device).unwrap();
        assert!(transform.forward(&x).is_err());
        let x = Tensor::zeros((1, 16), DType::F16, &device).unwrap();
        assert!(transform.forward(&x).is_err());
    }

    #[test]
    fn transform_new_rejects_bad_block_or_signs() {
        let device = Device::Cpu;
        assert!(HadamardTransform::new(6, &[1f32; 12], &device).is_err());
        assert!(HadamardTransform::new(8, &[1f32; 12], &device).is_err());
        assert!(HadamardTransform::new(8, &[], &device).is_err());
    }

    /// Metal kernel vs the CPU butterfly (itself locked to the dense
    /// reference above) for every block size and the model's real widths,
    /// including several rows so per-row sign indexing is exercised.
    #[cfg(target_os = "macos")]
    #[test]
    fn metal_matches_cpu() {
        let Ok(metal) = Device::new_metal(0) else {
            eprintln!("skipping metal_matches_cpu: no Metal device");
            return;
        };
        let mut cases: Vec<(usize, usize)> = (1..=10).map(|l| (1usize << l, 3 << l)).collect();
        cases.extend([(1024, 5120), (1024, 6144), (1024, 17408)]);
        for (block, width) in cases {
            let signs = random_signs(width);
            let cpu_tf = HadamardTransform::new(block, &signs, &Device::Cpu).unwrap();
            let gpu_tf = HadamardTransform::new(block, &signs, &metal).unwrap();
            let x = Tensor::from_vec(random_vec(3 * width), (1, 3, width), &Device::Cpu).unwrap();
            let xg = x.to_device(&metal).unwrap();
            for signs_first in [true, false] {
                let (c, g) = if signs_first {
                    (cpu_tf.forward(&x), gpu_tf.forward(&xg))
                } else {
                    (cpu_tf.inverse(&x), gpu_tf.inverse(&xg))
                };
                let g = g.unwrap().to_device(&Device::Cpu).unwrap();
                assert_close(&g, &c.unwrap(), 1e-5);
            }
        }
    }

    /// Offset (batch 1) and strided (batch 2) views on Metal: the kernel reads
    /// `x` in place from its layout offset, so this guards the `buffer_o`
    /// offset and the `contiguous()` fallback.
    #[cfg(target_os = "macos")]
    #[test]
    fn metal_handles_narrowed_inputs() {
        let Ok(metal) = Device::new_metal(0) else {
            eprintln!("skipping metal_handles_narrowed_inputs: no Metal device");
            return;
        };
        let (block, width) = (1024, 5120);
        let signs = random_signs(width);
        let cpu_tf = HadamardTransform::new(block, &signs, &Device::Cpu).unwrap();
        let gpu_tf = HadamardTransform::new(block, &signs, &metal).unwrap();
        for batch in [1, 2] {
            let x = Tensor::from_vec(
                random_vec(batch * 4 * width),
                (batch, 4, width),
                &Device::Cpu,
            )
            .unwrap();
            let expected = cpu_tf.forward(&x.narrow(1, 3, 1).unwrap()).unwrap();
            let xg = x.to_device(&metal).unwrap();
            let actual = gpu_tf
                .forward(&xg.narrow(1, 3, 1).unwrap())
                .unwrap()
                .to_device(&Device::Cpu)
                .unwrap();
            assert_close(&actual, &expected, 1e-5);
        }
    }

    /// Median per-call wall time of `f` in microseconds. Each sample enqueues
    /// `batch` calls and then blocks once on `sync`, so Metal timings measure
    /// in-stream execution rather than a per-call commit/wait round trip.
    fn median_us(
        samples: usize,
        batch: usize,
        sync: &dyn Fn(),
        f: &mut dyn FnMut() -> Tensor,
    ) -> f64 {
        for _ in 0..3 {
            f();
        }
        sync();
        let mut times: Vec<f64> = (0..samples)
            .map(|_| {
                let t = std::time::Instant::now();
                let outs: Vec<Tensor> = (0..batch).map(|_| f()).collect();
                sync();
                let us = t.elapsed().as_secs_f64() * 1e6 / batch as f64;
                drop(outs);
                us
            })
            .collect();
        times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        times[times.len() / 2]
    }

    fn bench_devices() -> Vec<(&'static str, Device)> {
        let mut devices = vec![("cpu", Device::Cpu)];
        if let Ok(metal) = Device::new_metal(0) {
            devices.push(("metal", metal));
        }
        devices
    }

    /// Dense-matmul (previous implementation) vs butterfly FWHT, forward
    /// direction, at the model's real widths. Run with:
    /// `cargo test --release -p bonsai-candle hadamard_bench -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn hadamard_bench() {
        let block = 1024;
        let shapes: [(usize, usize); 6] = [
            (1, 5120),
            (1, 6144),
            (1, 17408),
            (128, 5120),
            (512, 5120),
            (512, 17408),
        ];
        println!(
            "{:<6} {:>5} {:>6} {:>12} {:>12} {:>8} {:>10}",
            "device", "rows", "width", "dense_us", "fwht_us", "speedup", "max_diff"
        );
        for (name, device) in bench_devices() {
            let sync = || device.synchronize().unwrap();
            let h = sylvester_matrix(block, &device).unwrap();
            for (rows, width) in shapes {
                let (samples, batch) = if rows * width > 1 << 20 {
                    (7, 4)
                } else {
                    (21, 32)
                };
                let signs = random_signs(width);
                let signs_t = Tensor::from_slice(&signs, width, &device).unwrap();
                let transform = HadamardTransform::new(block, &signs, &device).unwrap();
                let x =
                    Tensor::from_vec(random_vec(rows * width), (1, rows, width), &device).unwrap();

                let dense_us = median_us(samples, batch, &sync, &mut || {
                    dense_forward(&x, &signs_t, &h, block).unwrap()
                });
                let fwht_us = median_us(samples, batch, &sync, &mut || {
                    transform.forward(&x).unwrap()
                });

                let diff = (dense_forward(&x, &signs_t, &h, block).unwrap()
                    - transform.forward(&x).unwrap())
                .unwrap()
                .abs()
                .unwrap()
                .flatten_all()
                .unwrap()
                .max(0)
                .unwrap()
                .to_scalar::<f32>()
                .unwrap();
                println!(
                    "{name:<6} {rows:>5} {width:>6} {dense_us:>12.1} {fwht_us:>12.1} {:>7.1}x {diff:>10.2e}",
                    dense_us / fwht_us
                );
            }
        }
    }

    fn minimal_manifest_json() -> String {
        r#"{
            "prism.hadamard.version": 1,
            "prism.hadamard.block_size": 8,
            "prism.hadamard.transform": "normalized-sylvester-walsh-hadamard",
            "prism.hadamard.axis": "input-last-dimension",
            "prism.hadamard.sign_mode": "explicit",
            "prism.hadamard.weight_names": ["a.weight"],
            "prism.hadamard.inverse_weight_names": ["embed.weight"],
            "prism.hadamard.sign_widths": [8],
            "prism.hadamard.sign_values": [1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0]
        }"#
        .to_owned()
    }

    #[test]
    fn valid_manifest_parses() {
        let m = HadamardManifest::from_json(&minimal_manifest_json()).unwrap();
        assert_eq!(m.block_size, 8);
        assert!(m.weight_names.contains("a.weight"));
        assert!(m.inverse_weight_names.contains("embed.weight"));
        assert_eq!(
            m.signs_by_width[&8],
            vec![1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0]
        );
    }

    #[test]
    fn gdn_v_grouped_defaults_to_true_when_absent() {
        let m = HadamardManifest::from_json(&minimal_manifest_json()).unwrap();
        assert!(m.gdn_v_grouped);
    }

    #[test]
    fn gdn_v_grouped_reads_explicit_false() {
        let json = minimal_manifest_json().replace(
            "\"prism.hadamard.sign_values\": [1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0]",
            "\"prism.hadamard.sign_values\": [1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0], \"prism.hadamard.gdn_v_grouped\": false",
        );
        let m = HadamardManifest::from_json(&json).unwrap();
        assert!(!m.gdn_v_grouped);
    }

    #[test]
    fn assert_gdn_layout_supported_ok_when_grouped() {
        let mut m = HadamardManifest::from_json(&minimal_manifest_json()).unwrap();
        m.gdn_v_grouped = false;
        // Equal head counts: no permutation needed regardless of the flag.
        assert!(m.assert_gdn_layout_supported(16, 16).is_ok());
        m.gdn_v_grouped = true;
        assert!(m.assert_gdn_layout_supported(48, 16).is_ok());
    }

    #[test]
    fn assert_gdn_layout_supported_bails_on_ungrouped_mismatched_heads() {
        let mut m = HadamardManifest::from_json(&minimal_manifest_json()).unwrap();
        m.gdn_v_grouped = false;
        assert!(m.assert_gdn_layout_supported(48, 16).is_err());
    }

    #[test]
    fn manifest_rejects_wrong_transform() {
        let json = minimal_manifest_json().replace(
            "normalized-sylvester-walsh-hadamard",
            "some-other-transform",
        );
        assert!(HadamardManifest::from_json(&json).is_err());
    }

    #[test]
    fn manifest_rejects_non_pm1_sign_value() {
        let json = minimal_manifest_json().replace(
            "[1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0]",
            "[1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 0.5]",
        );
        assert!(HadamardManifest::from_json(&json).is_err());
    }

    #[test]
    fn manifest_rejects_width_value_count_mismatch() {
        let json = minimal_manifest_json().replace(
            "\"prism.hadamard.sign_widths\": [8]",
            "\"prism.hadamard.sign_widths\": [16]",
        );
        assert!(HadamardManifest::from_json(&json).is_err());
    }

    #[test]
    fn manifest_rejects_width_not_multiple_of_block() {
        // block_size stays a valid power of two (8); the width (12) just doesn't divide it.
        let json = minimal_manifest_json()
            .replace("\"prism.hadamard.sign_widths\": [8]", "\"prism.hadamard.sign_widths\": [12]")
            .replace(
                "\"prism.hadamard.sign_values\": [1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0]",
                "\"prism.hadamard.sign_values\": [1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0]",
            );
        assert!(HadamardManifest::from_json(&json).is_err());
    }

    #[test]
    fn manifest_rejects_overlapping_weight_names() {
        let json = minimal_manifest_json().replace(
            "\"prism.hadamard.inverse_weight_names\": [\"embed.weight\"]",
            "\"prism.hadamard.inverse_weight_names\": [\"a.weight\"]",
        );
        assert!(HadamardManifest::from_json(&json).is_err());
    }
}
