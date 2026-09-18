//! Qwen3 implementation with quantization support and YaRN rope scaling.
//!
//! Adapted from candle-transformers with added YaRN (Yet Another RoPE extensioN)
//! rope scaling for extended context support required by Bonsai models.
use crate::kv_cache::ConcatKvCache;
use anyhow::anyhow;
use candle::quantized::{gguf_file, GgmlDType, QStorage, QTensor};
use candle::{DType, Device, Result, Tensor};
use candle_nn::{Activation, Embedding, Module};
use candle_transformers::models::with_tracing::QMatMul;
use candle_transformers::{quantized_nn::RmsNorm, utils::repeat_kv};
use half::f16;
use safetensors::SafeTensors;
use std::borrow::Cow;
use std::f32::consts::PI;
use std::io::{Read, Seek};
use std::sync::Arc;

/// Abstraction over a weight source, letting us share the layer-construction
/// logic between the GGUF loader and the MLX safetensors loader.
///
/// Names passed in use the GGUF naming convention (e.g. `blk.0.attn_q.weight`);
/// the safetensors implementation translates them to HuggingFace names.
pub trait WeightSource {
    fn qmatmul(&mut self, name: &str) -> Result<QMatMul>;
    fn rms_norm(&mut self, name: &str, eps: f64) -> Result<RmsNorm>;
    fn tensor(&mut self, name: &str) -> Result<QTensor>;
}

pub struct Gguf<R: Read + Seek> {
    ct: gguf_file::Content,
    reader: R,
    device: Device,
}

impl<R: Read + Seek> Gguf<R> {
    pub fn new(ct: gguf_file::Content, reader: R, device: Device) -> Self {
        Self { ct, reader, device }
    }

    pub fn metadata(&self) -> &std::collections::HashMap<String, gguf_file::Value> {
        &self.ct.metadata
    }
}

impl<R: Read + Seek> WeightSource for Gguf<R> {
    fn qmatmul(&mut self, name: &str) -> Result<QMatMul> {
        let ws = self.ct.tensor(&mut self.reader, name, &self.device)?;
        QMatMul::from_weights(ws.into())
    }

    fn rms_norm(&mut self, name: &str, eps: f64) -> Result<RmsNorm> {
        let ws = self.ct.tensor(&mut self.reader, name, &self.device)?;
        RmsNorm::from_qtensor(ws, eps)
    }

    fn tensor(&mut self, name: &str) -> Result<QTensor> {
        self.ct.tensor(&mut self.reader, name, &self.device)
    }
}

/// MLX-format safetensors weight source.
///
/// Quantized projections are stored as three tensors per layer (`<base>.weight`,
/// `<base>.scales`, `<base>.biases`). Packing: `weight` is `[rows, cols/16]` in
/// little-endian uint32 (2 bits per weight, 16 weights per uint32); `scales`
/// and `biases` are `[rows, cols/128]` in f16.
pub struct SafeTensorsSource<'a> {
    st: SafeTensors<'a>,
    device: Device,
}

impl<'a> SafeTensorsSource<'a> {
    pub fn new(st: SafeTensors<'a>, device: Device) -> Self {
        Self { st, device }
    }

    fn tensor_bytes(&self, name: &str) -> Result<&'a [u8]> {
        let view = self
            .st
            .tensor(name)
            .map_err(|e| candle::Error::msg(anyhow!("missing tensor {name}: {e}")))?;
        Ok(view.data())
    }

    fn tensor_shape(&self, name: &str) -> Result<Vec<usize>> {
        let view = self
            .st
            .tensor(name)
            .map_err(|e| candle::Error::msg(anyhow!("missing tensor {name}: {e}")))?;
        Ok(view.shape().to_vec())
    }

    /// Whether the underlying safetensors file has a tensor with this exact
    /// name (no GGUF-name translation).
    pub(crate) fn has_tensor(&self, name: &str) -> bool {
        self.st.tensor(name).is_ok()
    }

    /// Load a plain (non-quantized) tensor by its exact safetensors name,
    /// bypassing the GGUF-name `translate` map. Used by the qwen3_5 loader for
    /// tensors that have no GGUF-name equivalent (conv1d weight, A_log, dt_bias,
    /// gated-norm weight, RMSNorm weights, and — for `prism_hadamard_qwen35`
    /// checkpoints — the per-width `.signs` vectors and the dense `in_proj_a/b`
    /// weights). Note the MLX conversion folds the `+1` of the reference
    /// `(1 + weight)` RMSNorm into the stored weights, so callers apply these
    /// weights as-is (do not add `1` again).
    pub(crate) fn plain_tensor(&self, name: &str) -> Result<Tensor> {
        let view = self
            .st
            .tensor(name)
            .map_err(|e| candle::Error::msg(anyhow!("missing tensor {name}: {e}")))?;
        let dtype = match view.dtype() {
            safetensors::Dtype::F16 => DType::F16,
            safetensors::Dtype::F32 => DType::F32,
            safetensors::Dtype::BF16 => DType::BF16,
            other => candle::bail!("{name}: unsupported plain tensor dtype {other:?}"),
        };
        Tensor::from_raw_buffer(view.data(), dtype, view.shape(), &self.device)
    }

    /// Load a quantized projection encoded as `<base>.{weight,scales,biases}`
    /// into a Q2MLX-backed QTensor.
    pub(crate) fn load_q2mlx(&self, base: &str) -> Result<QTensor> {
        let weight_bytes = self.tensor_bytes(&format!("{base}.weight"))?;
        let weight_shape = self.tensor_shape(&format!("{base}.weight"))?;
        let scales_bytes = self.tensor_bytes(&format!("{base}.scales"))?;
        let scales_shape = self.tensor_shape(&format!("{base}.scales"))?;
        let biases_bytes = self.tensor_bytes(&format!("{base}.biases"))?;
        let biases_shape = self.tensor_shape(&format!("{base}.biases"))?;

        if weight_shape.len() != 2 || scales_shape.len() != 2 || biases_shape.len() != 2 {
            candle::bail!("{base}: expected 2D tensors for weight/scales/biases");
        }
        if scales_shape != biases_shape {
            candle::bail!("{base}: scales shape {scales_shape:?} != biases shape {biases_shape:?}");
        }

        let rows = weight_shape[0];
        let weight_cols_u32 = weight_shape[1]; // packed: 16 weights per uint32
        let cols = weight_cols_u32 * 16;
        let groups_per_row = scales_shape[1];
        if scales_shape[0] != rows {
            candle::bail!(
                "{base}: scales rows {} != weight rows {rows}",
                scales_shape[0]
            );
        }
        if groups_per_row * 128 != cols {
            candle::bail!(
                "{base}: group count mismatch (groups_per_row={groups_per_row}, cols={cols})"
            );
        }

        // Reinterpret byte slices. MLX writes little-endian uint32, so the
        // packed weight bytes map directly onto BlockQ2MLX::qs as [u8; 32] per
        // 128-weight group. scales/biases are f16 pairs per group.
        let scales_f16: &[f16] = unsafe {
            std::slice::from_raw_parts(scales_bytes.as_ptr() as *const f16, scales_bytes.len() / 2)
        };
        let biases_f16: &[f16] = unsafe {
            std::slice::from_raw_parts(biases_bytes.as_ptr() as *const f16, biases_bytes.len() / 2)
        };

        let total_blocks = rows * groups_per_row;
        if scales_f16.len() != total_blocks {
            candle::bail!(
                "{base}: scales length {} != expected {total_blocks}",
                scales_f16.len()
            );
        }
        if weight_bytes.len() != total_blocks * 32 {
            candle::bail!(
                "{base}: weight length {} != expected {}",
                weight_bytes.len(),
                total_blocks * 32
            );
        }

        // Build the packed block bytes directly. Layout per block:
        //   scale: 2 bytes (f16, little-endian)
        //   bias:  2 bytes (f16, little-endian)
        //   qs:    32 bytes
        // Total: 36 bytes per block.
        let mut bytes = Vec::with_capacity(total_blocks * 36);
        for row in 0..rows {
            for g in 0..groups_per_row {
                let block_idx = row * groups_per_row + g;
                let scale = scales_f16[block_idx];
                let bias = biases_f16[block_idx];
                bytes.extend_from_slice(&scale.to_le_bytes());
                bytes.extend_from_slice(&bias.to_le_bytes());
                bytes.extend_from_slice(&weight_bytes[block_idx * 32..(block_idx + 1) * 32]);
            }
        }
        let storage = QStorage::from_data(Cow::Owned(bytes), &self.device, GgmlDType::Q2MLX)?;
        QTensor::new(storage, (rows, cols))
    }

    /// Translate a GGUF tensor name to the matching MLX tensor base name
    /// (without the `.weight` suffix for quantized projections, or with the
    /// full suffix for plain tensors like layer norms).
    fn translate(gguf_name: &str) -> Result<TranslatedName> {
        if gguf_name == "token_embd.weight" {
            return Ok(TranslatedName::Quantized("model.embed_tokens".into()));
        }
        if gguf_name == "output.weight" {
            return Ok(TranslatedName::Quantized("lm_head".into()));
        }
        if gguf_name == "output_norm.weight" {
            return Ok(TranslatedName::Plain("model.norm.weight".into()));
        }
        if let Some(rest) = gguf_name.strip_prefix("blk.") {
            let (layer, suffix) = rest
                .split_once('.')
                .ok_or_else(|| candle::Error::msg(anyhow!("bad blk name: {gguf_name}")))?;
            let base = format!("model.layers.{layer}");
            let translated = match suffix {
                "attn_q.weight" => TranslatedName::Quantized(format!("{base}.self_attn.q_proj")),
                "attn_k.weight" => TranslatedName::Quantized(format!("{base}.self_attn.k_proj")),
                "attn_v.weight" => TranslatedName::Quantized(format!("{base}.self_attn.v_proj")),
                "attn_output.weight" => {
                    TranslatedName::Quantized(format!("{base}.self_attn.o_proj"))
                }
                "ffn_gate.weight" => TranslatedName::Quantized(format!("{base}.mlp.gate_proj")),
                "ffn_up.weight" => TranslatedName::Quantized(format!("{base}.mlp.up_proj")),
                "ffn_down.weight" => TranslatedName::Quantized(format!("{base}.mlp.down_proj")),
                "attn_q_norm.weight" => {
                    TranslatedName::Plain(format!("{base}.self_attn.q_norm.weight"))
                }
                "attn_k_norm.weight" => {
                    TranslatedName::Plain(format!("{base}.self_attn.k_norm.weight"))
                }
                "attn_norm.weight" => {
                    TranslatedName::Plain(format!("{base}.input_layernorm.weight"))
                }
                "ffn_norm.weight" => {
                    TranslatedName::Plain(format!("{base}.post_attention_layernorm.weight"))
                }
                other => candle::bail!("unknown qwen3 tensor suffix: {other}"),
            };
            return Ok(translated);
        }
        candle::bail!("cannot translate tensor name {gguf_name}")
    }
}

enum TranslatedName {
    Quantized(String), // base (no .weight suffix) - has .weight/.scales/.biases
    Plain(String),     // full tensor name (f16 weight only)
}

impl WeightSource for SafeTensorsSource<'_> {
    fn qmatmul(&mut self, name: &str) -> Result<QMatMul> {
        match Self::translate(name)? {
            TranslatedName::Quantized(base) => {
                let qt = self.load_q2mlx(&base)?;
                QMatMul::from_weights(Arc::new(qt))
            }
            TranslatedName::Plain(_) => {
                candle::bail!("qmatmul called on a plain tensor name: {name}")
            }
        }
    }

    fn rms_norm(&mut self, name: &str, eps: f64) -> Result<RmsNorm> {
        match Self::translate(name)? {
            TranslatedName::Plain(full) => {
                let bytes = self.tensor_bytes(&full)?;
                let shape = self.tensor_shape(&full)?;
                let storage =
                    QStorage::from_data(Cow::Borrowed(bytes), &self.device, GgmlDType::F16)?;
                let qt = QTensor::new(storage, shape)?;
                RmsNorm::from_qtensor(qt, eps)
            }
            TranslatedName::Quantized(_) => {
                candle::bail!("rms_norm called on a quantized tensor name: {name}")
            }
        }
    }

    fn tensor(&mut self, name: &str) -> Result<QTensor> {
        match Self::translate(name)? {
            TranslatedName::Quantized(base) => self.load_q2mlx(&base),
            TranslatedName::Plain(_) => {
                candle::bail!("tensor() called on a plain tensor name: {name}")
            }
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct MlpWeights {
    gate_proj: QMatMul,
    up_proj: QMatMul,
    down_proj: QMatMul,
    act_fn: Activation,
    /// `prism_hadamard_qwen35` only: Hadamard-rotate the hidden-state input
    /// before `gate_proj`/`up_proj`; `None` for every other checkpoint.
    in_tf: Option<Arc<crate::hadamard::HadamardTransform>>,
    /// `prism_hadamard_qwen35` only: Hadamard-rotate `silu(gate) * up` before
    /// `down_proj`; `None` for every other checkpoint.
    down_tf: Option<Arc<crate::hadamard::HadamardTransform>>,
    span: tracing::Span,
}

impl MlpWeights {
    /// Build a SwiGLU MLP from three already-loaded quantized projections.
    /// Shared with the qwen3_5 loader, which reads HuggingFace tensor names
    /// directly instead of going through the GGUF-name `WeightSource` map.
    pub(crate) fn from_qmatmuls(gate_proj: QMatMul, up_proj: QMatMul, down_proj: QMatMul) -> Self {
        let span = tracing::span!(tracing::Level::TRACE, "mlp");
        Self {
            gate_proj,
            up_proj,
            down_proj,
            act_fn: Activation::Silu,
            in_tf: None,
            down_tf: None,
            span,
        }
    }

    /// Attach the Hadamard transforms used by `prism_hadamard_qwen35`
    /// checkpoints. No-op (transforms stay `None`) for plain `qwen3_5`.
    pub(crate) fn with_hadamard(
        mut self,
        in_tf: Option<Arc<crate::hadamard::HadamardTransform>>,
        down_tf: Option<Arc<crate::hadamard::HadamardTransform>>,
    ) -> Self {
        self.in_tf = in_tf;
        self.down_tf = down_tf;
        self
    }

    fn new(src: &mut dyn WeightSource, prefix: &str) -> Result<Self> {
        let gate_proj = src.qmatmul(&format!("{prefix}.ffn_gate.weight"))?;
        let up_proj = src.qmatmul(&format!("{prefix}.ffn_up.weight"))?;
        let down_proj = src.qmatmul(&format!("{prefix}.ffn_down.weight"))?;
        let act_fn = Activation::Silu;
        let span = tracing::span!(tracing::Level::TRACE, "mlp");
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
            act_fn,
            in_tf: None,
            down_tf: None,
            span,
        })
    }
}

impl Module for MlpWeights {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let xh = crate::hadamard::apply(&self.in_tf, x)?;
        let gate = self.gate_proj.forward(&xh)?.apply(&self.act_fn)?;
        let up = self.up_proj.forward(&xh)?;
        let gated = (gate * up)?;
        let gated = crate::hadamard::apply(&self.down_tf, &gated)?;
        self.down_proj.forward(&gated)
    }
}

#[derive(Debug, Clone)]
pub struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
}

#[derive(Debug, Clone)]
enum RopeScaling {
    Yarn {
        factor: f32,
        original_max_position_embeddings: usize,
        beta_fast: f32,
        beta_slow: f32,
        mscale: f32,
        mscale_all_dim: f32,
    },
}

impl RotaryEmbedding {
    pub(crate) fn new_unscaled(
        dtype: DType,
        head_dim: usize,
        max_position_embeddings: usize,
        rope_theta: f64,
        dev: &Device,
    ) -> Result<Self> {
        let dim = head_dim;
        let max_seq_len = max_position_embeddings;
        let inv_freq: Vec<_> = (0..dim)
            .step_by(2)
            .map(|i| 1f32 / rope_theta.powf(i as f64 / dim as f64) as f32)
            .collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), dev)?.to_dtype(dtype)?;
        let t = Tensor::arange(0u32, max_seq_len as u32, dev)?
            .to_dtype(dtype)?
            .reshape((max_seq_len, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        Ok(Self {
            sin: freqs.sin()?,
            cos: freqs.cos()?,
        })
    }

    fn yarn_find_correction_dim(
        num_rot: f32,
        dim: usize,
        base: f32,
        max_position_embeddings: usize,
    ) -> f32 {
        (dim as f32 * (max_position_embeddings as f32 / (num_rot * 2. * PI)).ln())
            / (2. * base.ln())
    }

    fn yarn_find_correction_range(
        low_rot: f32,
        high_rot: f32,
        dim: usize,
        base: f32,
        max_position_embeddings: usize,
    ) -> (f32, f32) {
        let low =
            Self::yarn_find_correction_dim(low_rot, dim, base, max_position_embeddings).floor();
        let high =
            Self::yarn_find_correction_dim(high_rot, dim, base, max_position_embeddings).ceil();
        (low.max(0.), high.min(dim as f32 - 1.))
    }

    fn yarn_linear_ramp_mask(min: f32, mut max: f32, dim: usize, dev: &Device) -> Result<Tensor> {
        if min == max {
            max += 0.001;
        }
        let linear_func =
            ((Tensor::arange(0f32, dim as f32, dev)? - min as f64)? / (max as f64 - min as f64))?;
        linear_func.clamp(0., 1.)
    }

    fn yarn_get_mscale(scale: f32, mscale: f32) -> f32 {
        if scale <= 1. {
            return 1.;
        }
        0.1 * mscale * scale.ln() + 1.
    }

    #[allow(clippy::too_many_arguments)]
    fn new_yarn(
        dtype: DType,
        head_dim: usize,
        max_position_embeddings: usize,
        rope_theta: f64,
        dev: &Device,
        original_max_position_embeddings: usize,
        beta_fast: f32,
        beta_slow: f32,
        factor: f32,
        mscale: f32,
        mscale_all_dim: f32,
    ) -> Result<Self> {
        let rope_theta = rope_theta as f32;
        let freq_extra: Vec<_> = (0..head_dim)
            .step_by(2)
            .map(|i| 1f32 / rope_theta.powf(i as f32 / head_dim as f32))
            .collect();
        let freq_extra_len = freq_extra.len();
        let freq_extra = Tensor::from_vec(freq_extra, freq_extra_len, dev)?;
        let freq_inter: Vec<_> = (0..head_dim)
            .step_by(2)
            .map(|i| 1f32 / (factor * rope_theta.powf(i as f32 / head_dim as f32)))
            .collect();
        let freq_inter_len = freq_inter.len();
        let freq_inter = Tensor::from_vec(freq_inter, (1, freq_inter_len), dev)?;

        let (low, high) = Self::yarn_find_correction_range(
            beta_fast,
            beta_slow,
            head_dim,
            rope_theta,
            original_max_position_embeddings,
        );
        let inv_freq_mask = (1. - Self::yarn_linear_ramp_mask(low, high, head_dim / 2, dev)?)?;
        let inv_freq = freq_inter
            .broadcast_mul(&(1. - &inv_freq_mask)?)?
            .broadcast_add(&freq_extra.broadcast_mul(&inv_freq_mask)?)?;

        let t = Tensor::arange(0u32, max_position_embeddings as u32, dev)?
            .to_dtype(DType::F32)?
            .reshape((max_position_embeddings, 1))?;
        let freqs = t.matmul(&inv_freq)?;

        let mscale =
            Self::yarn_get_mscale(factor, mscale) / Self::yarn_get_mscale(factor, mscale_all_dim);
        Ok(Self {
            sin: (freqs.sin()? * mscale as f64)?.to_dtype(dtype)?,
            cos: (freqs.cos()? * mscale as f64)?.to_dtype(dtype)?,
        })
    }

    fn new_with_scaling(
        dtype: DType,
        head_dim: usize,
        max_position_embeddings: usize,
        rope_theta: f64,
        rope_scaling: Option<RopeScaling>,
        dev: &Device,
    ) -> Result<Self> {
        match rope_scaling {
            Some(RopeScaling::Yarn {
                factor,
                original_max_position_embeddings,
                beta_fast,
                beta_slow,
                mscale,
                mscale_all_dim,
            }) => Self::new_yarn(
                dtype,
                head_dim,
                max_position_embeddings,
                rope_theta,
                dev,
                original_max_position_embeddings,
                beta_fast,
                beta_slow,
                factor,
                mscale,
                mscale_all_dim,
            ),
            None => Self::new_unscaled(dtype, head_dim, max_position_embeddings, rope_theta, dev),
        }
    }

    /// Apply RoPE (q, k shape: B x H x L x D)
    pub fn apply(&self, q: &Tensor, k: &Tensor, offset: usize) -> Result<(Tensor, Tensor)> {
        let (_, _, seq_len, _) = q.dims4()?;
        let cos = self.cos.narrow(0, offset, seq_len)?.to_dtype(q.dtype())?;
        let sin = self.sin.narrow(0, offset, seq_len)?.to_dtype(q.dtype())?;
        let q_embed = candle_nn::rotary_emb::rope(&q.contiguous()?, &cos, &sin)?;
        let k_embed = candle_nn::rotary_emb::rope(&k.contiguous()?, &cos, &sin)?;
        Ok((q_embed, k_embed))
    }
}

#[derive(Debug, Clone)]
struct AttentionWeights {
    q_proj: QMatMul,
    k_proj: QMatMul,
    v_proj: QMatMul,
    o_proj: QMatMul,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    rotary_emb: Arc<RotaryEmbedding>,
    kv_cache: ConcatKvCache,
    span_attn: tracing::Span,
}

impl AttentionWeights {
    fn new(
        src: &mut dyn WeightSource,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        rms_norm_eps: f64,
        rotary_emb: Arc<RotaryEmbedding>,
        prefix: &str,
    ) -> Result<Self> {
        let num_kv_groups = num_heads / num_kv_heads;

        let q_proj = src.qmatmul(&format!("{prefix}.attn_q.weight"))?;
        let k_proj = src.qmatmul(&format!("{prefix}.attn_k.weight"))?;
        let v_proj = src.qmatmul(&format!("{prefix}.attn_v.weight"))?;
        let o_proj = src.qmatmul(&format!("{prefix}.attn_output.weight"))?;

        let q_norm = src.rms_norm(&format!("{prefix}.attn_q_norm.weight"), rms_norm_eps)?;
        let k_norm = src.rms_norm(&format!("{prefix}.attn_k_norm.weight"), rms_norm_eps)?;

        let kv_cache = ConcatKvCache::new(2);

        let span_attn = tracing::span!(tracing::Level::TRACE, "attn");

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            num_heads,
            num_kv_heads,
            num_kv_groups,
            head_dim,
            rotary_emb,
            kv_cache,
            span_attn,
        })
    }

    fn forward(&mut self, x: &Tensor, attn_mask: Option<&Tensor>, offset: usize) -> Result<Tensor> {
        let _enter = self.span_attn.enter();
        let (b, l, _) = x.dims3()?;

        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        let q = q
            .reshape((b, l, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((b, l, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b, l, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        let q_flat = q.flatten(0, 2)?;
        let k_flat = k.flatten(0, 2)?;

        let q_flat = self.q_norm.forward(&q_flat)?;
        let k_flat = self.k_norm.forward(&k_flat)?;
        let q = q_flat.reshape((b, self.num_heads, l, self.head_dim))?;
        let k = k_flat.reshape((b, self.num_kv_heads, l, self.head_dim))?;

        let (q, k) = self.rotary_emb.apply(&q, &k, offset)?;

        let (k, v) = self.kv_cache.append(&k, &v)?;

        let k = repeat_kv(k, self.num_kv_groups)?.contiguous()?;
        let v = repeat_kv(v, self.num_kv_groups)?.contiguous()?;

        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let mut scores = (q.matmul(&k.transpose(2, 3)?)? * scale)?;
        if let Some(m) = attn_mask {
            let m_dtype = m.dtype();
            let scores_dtype = scores.dtype();
            let mask = if m_dtype != scores_dtype {
                m.to_dtype(scores_dtype)?
            } else {
                m.clone()
            };
            scores = scores.broadcast_add(&mask)?;
        }
        let probs = candle_nn::ops::softmax_last_dim(&scores)?;
        let ctx = probs.matmul(&v)?;
        let reshaped_ctx = ctx
            .transpose(1, 2)?
            .reshape((b, l, self.num_heads * self.head_dim))?;
        self.o_proj.forward(&reshaped_ctx)
    }

    fn clear_kv_cache(&mut self) {
        self.kv_cache.reset();
    }
}

#[derive(Debug, Clone)]
struct LayerWeights {
    self_attn: AttentionWeights,
    mlp: MlpWeights,
    ln1: RmsNorm,
    ln2: RmsNorm,
}

impl LayerWeights {
    fn new(
        src: &mut dyn WeightSource,
        num_attention_heads: usize,
        num_key_value_heads: usize,
        head_dim: usize,
        rms_norm_eps: f64,
        rotary: Arc<RotaryEmbedding>,
        layer_idx: usize,
    ) -> Result<Self> {
        let prefix = format!("blk.{layer_idx}");

        let ln1 = src.rms_norm(&format!("{prefix}.attn_norm.weight"), rms_norm_eps)?;
        let ln2 = src.rms_norm(&format!("{prefix}.ffn_norm.weight"), rms_norm_eps)?;
        let self_attn = AttentionWeights::new(
            src,
            num_attention_heads,
            num_key_value_heads,
            head_dim,
            rms_norm_eps,
            rotary,
            &prefix,
        )?;
        let mlp = MlpWeights::new(src, &prefix)?;
        Ok(Self {
            self_attn,
            mlp,
            ln1,
            ln2,
        })
    }

    fn forward(&mut self, x: &Tensor, mask: Option<&Tensor>, offset: usize) -> Result<Tensor> {
        let h = self.ln1.forward(x)?;
        let h = self.self_attn.forward(&h, mask, offset)?;
        let x = (x + h)?;
        let h2 = self.ln2.forward(&x)?;
        let h2 = h2.apply(&self.mlp)?;
        x + h2
    }

    fn clear_kv_cache(&mut self) {
        self.self_attn.clear_kv_cache();
    }
}

/// Subset of HuggingFace `config.json` fields we need for Qwen3 models loaded
/// from MLX safetensors. Fields are named to match the JSON keys exactly.
#[derive(Debug, Clone, serde::Deserialize)]
#[allow(dead_code)]
pub struct Qwen3Config {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    #[serde(default)]
    pub rope_scaling: Option<Qwen3RopeScaling>,
    #[serde(default)]
    pub eos_token_id: Option<u32>,
    #[serde(default)]
    pub vocab_size: Option<usize>,
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct Qwen3RopeScaling {
    pub rope_type: String,
    pub factor: f32,
    pub original_max_position_embeddings: usize,
}

#[derive(Debug, Clone)]
pub struct ModelWeights {
    embed_tokens: Embedding,
    layers: Vec<LayerWeights>,
    norm: RmsNorm,
    lm_head: QMatMul,
    device: Device,
    dtype: DType,
    span: tracing::Span,
    span_output: tracing::Span,
}

impl ModelWeights {
    pub fn from_gguf<R: Read + Seek>(
        ct: gguf_file::Content,
        reader: &mut R,
        device: &Device,
    ) -> Result<Self> {
        let mut gg = Gguf::new(ct, reader, device.clone());
        let md_get = |s: &str| match gg.metadata().get(s) {
            None => candle::bail!("cannot find {s} in metadata"),
            Some(v) => Ok(v),
        };

        let num_attention_heads = md_get("qwen3.attention.head_count")?.to_u32()? as usize;
        let num_kv_heads = md_get("qwen3.attention.head_count_kv")?.to_u32()? as usize;
        let head_dim = md_get("qwen3.attention.key_length")?.to_u32()? as usize;
        let num_layers = md_get("qwen3.block_count")?.to_u32()? as usize;
        let hidden_size = md_get("qwen3.embedding_length")?.to_u32()? as usize;
        let max_position_embeddings = md_get("qwen3.context_length")?.to_u32()? as usize;
        let rms_norm_eps = md_get("qwen3.attention.layer_norm_rms_epsilon")?.to_f32()? as f64;
        let rope_freq_base = md_get("qwen3.rope.freq_base")?.to_f32()? as f64;
        let rope_scaling = match gg.metadata().get("qwen3.rope.scaling.type") {
            Some(v) if v.to_string()?.eq_ignore_ascii_case("yarn") => Some(RopeScaling::Yarn {
                factor: gg
                    .metadata()
                    .get("qwen3.rope.scaling.factor")
                    .map(|v| v.to_f32())
                    .transpose()?
                    .unwrap_or(1.0),
                original_max_position_embeddings: gg
                    .metadata()
                    .get("qwen3.rope.scaling.original_context_length")
                    .map(|v| v.to_u32().map(|v| v as usize))
                    .transpose()?
                    .unwrap_or(max_position_embeddings),
                beta_fast: gg
                    .metadata()
                    .get("qwen3.rope.scaling.yarn_beta_fast")
                    .map(|v| v.to_f32())
                    .transpose()?
                    .unwrap_or(32.0),
                beta_slow: gg
                    .metadata()
                    .get("qwen3.rope.scaling.yarn_beta_slow")
                    .map(|v| v.to_f32())
                    .transpose()?
                    .unwrap_or(1.0),
                mscale: gg
                    .metadata()
                    .get("qwen3.rope.scaling.yarn_attn_factor")
                    .map(|v| v.to_f32())
                    .transpose()?
                    .unwrap_or(1.0),
                mscale_all_dim: gg
                    .metadata()
                    .get("qwen3.rope.scaling.yarn_log_multiplier")
                    .map(|v| v.to_f32().map(|v| v / 0.1))
                    .transpose()?
                    .unwrap_or(1.0),
            }),
            _ => None,
        };

        let dtype = match gg.metadata().get("general.dtype") {
            Some(v) => match v.to_u32() {
                Ok(0) => DType::F32,
                Ok(1) => DType::F16,
                _ => DType::F16,
            },
            None => DType::F16,
        };

        let embed_tensor = gg.tensor("token_embd.weight")?;
        let embed_tokens = Embedding::new(embed_tensor.dequantize(device)?, hidden_size);

        let rotary = Arc::new(RotaryEmbedding::new_with_scaling(
            dtype,
            head_dim,
            max_position_embeddings,
            rope_freq_base,
            rope_scaling,
            device,
        )?);

        let mut layers = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            layers.push(LayerWeights::new(
                &mut gg,
                num_attention_heads,
                num_kv_heads,
                head_dim,
                rms_norm_eps,
                rotary.clone(),
                i,
            )?);
        }

        let norm = gg.rms_norm("output_norm.weight", rms_norm_eps)?;
        let lm_head_tensor = match gg.tensor("output.weight") {
            Ok(tensor) => tensor,
            Err(_) => gg.tensor("token_embd.weight")?,
        };
        let lm_head = QMatMul::from_weights(lm_head_tensor.into())?;
        let span = tracing::span!(tracing::Level::TRACE, "model");
        let span_output = tracing::span!(tracing::Level::TRACE, "output");
        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            device: device.clone(),
            dtype,
            span,
            span_output,
        })
    }

    pub fn from_safetensors(
        src: &mut SafeTensorsSource<'_>,
        cfg: &Qwen3Config,
        device: &Device,
    ) -> Result<Self> {
        let dtype = DType::F16;

        let rope_scaling = cfg
            .rope_scaling
            .as_ref()
            .and_then(|s| {
                (s.rope_type.eq_ignore_ascii_case("yarn"))
                    .then_some(())
                    .map(|_| s)
            })
            .map(|s| RopeScaling::Yarn {
                factor: s.factor,
                original_max_position_embeddings: s.original_max_position_embeddings,
                beta_fast: 32.0,
                beta_slow: 1.0,
                mscale: 1.0,
                mscale_all_dim: 1.0,
            });

        let embed_qt = src.load_q2mlx("model.embed_tokens")?;
        let embed_tokens = Embedding::new(embed_qt.dequantize(device)?, cfg.hidden_size);

        let rotary = Arc::new(RotaryEmbedding::new_with_scaling(
            dtype,
            cfg.head_dim,
            cfg.max_position_embeddings,
            cfg.rope_theta,
            rope_scaling,
            device,
        )?);

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            layers.push(LayerWeights::new(
                src,
                cfg.num_attention_heads,
                cfg.num_key_value_heads,
                cfg.head_dim,
                cfg.rms_norm_eps,
                rotary.clone(),
                i,
            )?);
        }

        let norm = src.rms_norm("output_norm.weight", cfg.rms_norm_eps)?;
        let lm_head_qt = src.load_q2mlx("lm_head")?;
        let lm_head = QMatMul::from_weights(Arc::new(lm_head_qt))?;
        let span = tracing::span!(tracing::Level::TRACE, "model");
        let span_output = tracing::span!(tracing::Level::TRACE, "output");
        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            device: device.clone(),
            dtype,
            span,
            span_output,
        })
    }

    fn causal_mask(
        &self,
        b: usize,
        tgt: usize,
        offset: usize,
        sw: Option<usize>,
    ) -> Result<Tensor> {
        let minf = f32::NEG_INFINITY;
        let mask: Vec<_> = (0..tgt)
            .flat_map(|i| {
                (0..(tgt + offset)).map(move |j| {
                    let past_ok = j <= i + offset;
                    let sw_ok = match sw {
                        Some(w) => (i + offset) as i64 - j as i64 <= w as i64,
                        None => true,
                    };
                    if past_ok && sw_ok {
                        0.
                    } else {
                        minf
                    }
                })
            })
            .collect();
        Tensor::from_slice(&mask, (b, 1, tgt, tgt + offset), &self.device)?.to_dtype(self.dtype)
    }

    pub fn forward(&mut self, input: &Tensor, offset: usize) -> Result<Tensor> {
        let _enter = self.span.enter();
        let (b, l) = input.dims2()?;
        let mut h = self.embed_tokens.forward(input)?;
        let causal_mask = if l == 1 {
            None
        } else {
            Some(self.causal_mask(b, l, offset, None)?)
        };
        for layer in &mut self.layers {
            h = layer.forward(&h, causal_mask.as_ref(), offset)?;
        }
        let h = self.norm.forward(&h)?;
        let _enter = self.span_output.enter();
        let last_hidden = h.narrow(1, l - 1, 1)?;
        self.lm_head.forward(&last_hidden)?.squeeze(1)
    }

    pub fn clear_kv_cache(&mut self) {
        for layer in &mut self.layers {
            layer.clear_kv_cache();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use safetensors::tensor::TensorView;
    use safetensors::Dtype;

    fn build_safetensors(entries: &[(&str, Dtype, Vec<usize>, Vec<u8>)]) -> Vec<u8> {
        let views: Vec<(String, TensorView<'_>)> = entries
            .iter()
            .map(|(name, dtype, shape, data)| {
                (
                    (*name).to_owned(),
                    TensorView::new(*dtype, shape.clone(), data).unwrap(),
                )
            })
            .collect();
        safetensors::serialize(views, None).unwrap()
    }

    #[test]
    fn plain_tensor_reads_f16_and_f32_by_exact_dtype() {
        let f16_bytes = f16::from_f32(1.5).to_le_bytes().to_vec();
        let f32_bytes = 2.5f32.to_le_bytes().to_vec();
        let buf = build_safetensors(&[
            ("a", Dtype::F16, vec![1], f16_bytes),
            ("b", Dtype::F32, vec![1], f32_bytes),
        ]);
        let st = SafeTensors::deserialize(&buf).unwrap();
        let src = SafeTensorsSource::new(st, Device::Cpu);

        assert!(src.has_tensor("a"));
        assert!(src.has_tensor("b"));
        assert!(!src.has_tensor("missing"));

        let a = src.plain_tensor("a").unwrap();
        assert_eq!(a.dtype(), DType::F16);
        assert_eq!(
            a.to_dtype(DType::F32).unwrap().to_vec1::<f32>().unwrap(),
            vec![1.5]
        );

        let b = src.plain_tensor("b").unwrap();
        assert_eq!(b.dtype(), DType::F32);
        assert_eq!(b.to_vec1::<f32>().unwrap(), vec![2.5]);
    }

    #[test]
    fn plain_tensor_errors_on_missing_name() {
        let buf = build_safetensors(&[("a", Dtype::F32, vec![1], 1.0f32.to_le_bytes().to_vec())]);
        let st = SafeTensors::deserialize(&buf).unwrap();
        let src = SafeTensorsSource::new(st, Device::Cpu);
        assert!(src.plain_tensor("missing").is_err());
    }
}
