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
use crate::qwen3::SafeTensorsSource;
use candle::{DType, Device, Result, Tensor};
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

/// The natural-order Sylvester Hadamard matrix of size `block`, scaled by
/// `1 / sqrt(block)`: `H[i][j] = (-1)^popcount(i & j) / sqrt(block)`.
pub fn sylvester_matrix(block: usize, device: &Device) -> Result<Tensor> {
    if block < 2 || !block.is_power_of_two() {
        candle::bail!("sylvester_matrix: block {block} must be a power of two >= 2");
    }
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

/// A ready-to-apply Hadamard rotation for one input width: the shared
/// `H_block / sqrt(block)` matrix plus this width's sign vector.
#[derive(Debug)]
pub struct HadamardTransform {
    block: usize,
    matrix: Arc<Tensor>,
    signs: Tensor,
}

impl HadamardTransform {
    pub fn new(block: usize, matrix: Arc<Tensor>, signs: &[f32], device: &Device) -> Result<Self> {
        let signs = Tensor::from_slice(signs, signs.len(), device)?;
        Ok(Self {
            block,
            matrix,
            signs,
        })
    }

    /// `(x * signs) reshaped to (-1, block) @ H`, reshaped back to `x`'s shape.
    /// `x`: `(..., width)` with `width` a multiple of `block`.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let dims = x.dims().to_vec();
        self.check_width(&dims, "forward")?;
        let xs = x.broadcast_mul(&self.signs)?;
        let flat = xs.reshape(((), self.block))?;
        let rotated = flat.matmul(self.matrix.as_ref())?;
        rotated.reshape(dims)
    }

    /// `(x reshaped to (-1, block) @ H) * signs`, reshaped back to `x`'s shape.
    pub fn inverse(&self, x: &Tensor) -> Result<Tensor> {
        let dims = x.dims().to_vec();
        self.check_width(&dims, "inverse")?;
        let flat = x.reshape(((), self.block))?;
        let rotated = flat.matmul(self.matrix.as_ref())?;
        let rotated = rotated.reshape(dims)?;
        rotated.broadcast_mul(&self.signs)
    }

    fn check_width(&self, dims: &[usize], op: &str) -> Result<()> {
        let width = *dims
            .last()
            .ok_or_else(|| candle::Error::msg(format!("hadamard {op}: input has no dims")))?;
        if !width.is_multiple_of(self.block) {
            candle::bail!(
                "hadamard {op}: width {width} is not a multiple of block {}",
                self.block
            );
        }
        Ok(())
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
    matrix: Arc<Tensor>,
    cache: HashMap<usize, Arc<HadamardTransform>>,
    claimed: HashSet<String>,
    device: Device,
}

impl HadamardLoader {
    pub fn new(manifest: HadamardManifest, device: &Device) -> Result<Self> {
        let matrix = Arc::new(sylvester_matrix(manifest.block_size, device)?);
        Ok(Self {
            manifest,
            matrix,
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
            self.matrix.clone(),
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
    fn forward_matches_butterfly_fwht_per_block() {
        let device = Device::Cpu;
        let block = 8;
        let width = 16; // two blocks
        let mut rng = rand::rng();
        let input: Vec<f32> = (0..width).map(|_| rng.random_range(-1.0..1.0)).collect();

        let matrix = Arc::new(sylvester_matrix(block, &device).unwrap());
        let signs = vec![1f32; width]; // isolate the Hadamard rotation from the sign flip
        let transform = HadamardTransform::new(block, matrix, &signs, &device).unwrap();
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

    #[test]
    fn inverse_of_forward_recovers_input() {
        let device = Device::Cpu;
        let block = 1024;
        let width = 1024;
        let mut rng = rand::rng();
        let input: Vec<f32> = (0..width).map(|_| rng.random_range(-1.0..1.0)).collect();
        let signs: Vec<f32> = (0..width)
            .map(|_| if rng.random_bool(0.5) { 1.0 } else { -1.0 })
            .collect();

        let matrix = Arc::new(sylvester_matrix(block, &device).unwrap());
        let transform = HadamardTransform::new(block, matrix, &signs, &device).unwrap();
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
        let matrix = Arc::new(sylvester_matrix(block, &device).unwrap());
        let signs = vec![1f32; block];
        let transform = HadamardTransform::new(block, matrix, &signs, &device).unwrap();
        let x = Tensor::zeros((2, 3, block), DType::F32, &device).unwrap();
        let y = transform.forward(&x).unwrap();
        assert_eq!(y.dims(), &[2, 3, block]);
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
