//! Qwen3.5 (`model_type = qwen3_5`) hybrid architecture — text-only inference.
//!
//! Ternary-Bonsai-27B interleaves two token-mixing blocks: 48 Gated DeltaNet
//! linear-attention layers and 16 gated full-attention layers (one full layer
//! every `full_attention_interval`). This module loads the MLX 2-bit (Q2MLX)
//! checkpoint and runs the language model only (vision tower, MTP head and the
//! DSpark drafter are ignored).
//!
//! The recurrence and the depthwise causal convolution are computed with an
//! explicit per-token scan shared by prefill and decode — the simplest formulation
//! that stays correct; chunked prefill is a later optimization.
//!
//! Cross-checked line-by-line against `transformers` `models/qwen3_5`
//! (`Qwen3_5Attention`, `Qwen3_5GatedDeltaNet`, `torch_recurrent_gated_delta_rule`,
//! `Qwen3_5RMSNorm`, `Qwen3_5RMSNormGated`). Notable, non-obvious details:
//!   * `Qwen3_5RMSNorm` scales by `(1 + weight)` in the reference, but the MLX
//!     checkpoint folds the `+1` into the stored weights, so we apply plain
//!     `weight` here (see `Qwen35RmsNorm`).
//!   * The full-attention output gate is `sigmoid(gate)` (the `output_gate_type`
//!     config field is not read by the reference model).
//!   * The DeltaNet query is scaled by `1/sqrt(key_head_dim)` after L2-norm.
use crate::hadamard::{self, HadamardLoader, HadamardManifest, HadamardTransform};
use crate::kv_cache::ConcatKvCache;
use crate::qwen3::{MlpWeights, RotaryEmbedding, SafeTensorsSource};
use candle::{DType, Device, Result, Tensor, D};
use candle_nn::{Embedding, Module};
use candle_transformers::models::with_tracing::QMatMul;
use candle_transformers::utils::repeat_kv;
use std::sync::Arc;

/// Hidden-state dtype for the whole text stack. Q2MLX projections dequantize to
/// F32, so the activation path runs in F32 (matching the existing Qwen3 loader);
/// the DeltaNet recurrent state is F32 regardless (`mamba_ssm_dtype = float32`).
const DTYPE: DType = DType::F32;

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// Top-level `config.json` for a `qwen3_5` VLM checkpoint. Only the nested
/// `text_config` and a couple of shared fields are needed for text inference.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct Qwen35Config {
    pub text_config: Qwen35TextConfig,
    #[serde(default)]
    pub eos_token_id: Option<u32>,
    /// Set for `prism_hadamard_qwen35` checkpoints: relative path (from the
    /// model directory) to the `hadamard.json` manifest, e.g. `"hadamard.json"`.
    #[serde(default)]
    pub hadamard_config: Option<String>,
}

#[derive(Debug, Clone, serde::Deserialize)]
#[allow(dead_code)]
pub struct Qwen35TextConfig {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub full_attention_interval: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub partial_rotary_factor: f64,
    pub linear_num_key_heads: usize,
    pub linear_num_value_heads: usize,
    pub linear_key_head_dim: usize,
    pub linear_value_head_dim: usize,
    pub linear_conv_kernel_dim: usize,
    pub intermediate_size: usize,
    pub rms_norm_eps: f64,
    pub max_position_embeddings: usize,
    #[serde(default)]
    pub eos_token_id: Option<u32>,
    #[serde(default)]
    pub vocab_size: Option<usize>,
    #[serde(default)]
    pub rope_parameters: Option<RopeParameters>,
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct RopeParameters {
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,
}

fn default_rope_theta() -> f64 {
    10_000_000.0
}

impl Qwen35TextConfig {
    fn rope_theta(&self) -> f64 {
        self.rope_parameters
            .as_ref()
            .map(|r| r.rope_theta)
            .unwrap_or_else(default_rope_theta)
    }

    fn rotary_dim(&self) -> usize {
        (self.head_dim as f64 * self.partial_rotary_factor) as usize
    }
}

impl Qwen35Config {
    pub fn max_position_embeddings(&self) -> usize {
        self.text_config.max_position_embeddings
    }

    pub fn eos_token_id(&self) -> Option<u32> {
        self.eos_token_id.or(self.text_config.eos_token_id)
    }
}

// ---------------------------------------------------------------------------
// Math helpers
// ---------------------------------------------------------------------------

/// Numerically stable softplus: `relu(x) + ln(1 + exp(-|x|))`.
fn softplus(x: &Tensor) -> Result<Tensor> {
    let relu = x.relu()?;
    let softened = ((x.abs()?.neg()?.exp()? + 1.0)?).log()?;
    relu + softened
}

/// L2-normalize over the last dim: `x / sqrt(sum(x^2) + eps)` (eps = 1e-6, as in
/// the FLA `l2norm` used by the reference kernel).
fn l2_normalize(x: &Tensor) -> Result<Tensor> {
    let denom = (x.sqr()?.sum_keepdim(D::Minus1)? + 1e-6)?.sqrt()?;
    x.broadcast_div(&denom)
}

/// `torch.repeat_interleave(x, rep, dim=2)` for a `(b, l, heads, d)` tensor:
/// each head is duplicated `rep` times consecutively.
fn repeat_interleave_heads(x: &Tensor, rep: usize) -> Result<Tensor> {
    if rep == 1 {
        return Ok(x.clone());
    }
    let (b, l, heads, d) = x.dims4()?;
    x.reshape((b, l, heads, 1, d))?
        .broadcast_as((b, l, heads, rep, d))?
        .contiguous()?
        .reshape((b, l, heads * rep, d))
}

// ---------------------------------------------------------------------------
// Norms
// ---------------------------------------------------------------------------

/// `Qwen3_5RMSNorm`: `x_normed * weight`, computed in F32.
///
/// The reference model applies `x_normed * (1 + weight)` with weights stored
/// centered at 0, but the MLX conversion folds the `+1` into the stored weights
/// (verified: e.g. `input_layernorm` weights are centered at ~1, not ~0), so we
/// use the weight directly as a plain multiplier.
#[derive(Debug, Clone)]
struct Qwen35RmsNorm {
    weight: Tensor,
    eps: f64,
}

impl Qwen35RmsNorm {
    fn load(src: &SafeTensorsSource<'_>, name: &str, eps: f64) -> Result<Self> {
        let weight = src.plain_tensor(name)?.to_dtype(DType::F32)?;
        Ok(Self { weight, eps })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let in_dtype = x.dtype();
        let x = x.to_dtype(DType::F32)?;
        let variance = x.sqr()?.mean_keepdim(D::Minus1)?;
        let x_normed = x.broadcast_div(&(variance + self.eps)?.sqrt()?)?;
        x_normed.broadcast_mul(&self.weight)?.to_dtype(in_dtype)
    }
}

/// `Qwen3_5RMSNormGated`: `(x_normed * weight) * silu(gate)`, computed in F32.
/// Unlike `Qwen3_5RMSNorm`, the weight here is a plain multiplier (centered at 1).
#[derive(Debug, Clone)]
struct Qwen35RmsNormGated {
    weight: Tensor,
    eps: f64,
}

impl Qwen35RmsNormGated {
    fn load(src: &SafeTensorsSource<'_>, name: &str, eps: f64) -> Result<Self> {
        let weight = src.plain_tensor(name)?.to_dtype(DType::F32)?;
        Ok(Self { weight, eps })
    }

    fn forward(&self, x: &Tensor, gate: &Tensor) -> Result<Tensor> {
        let in_dtype = x.dtype();
        let x = x.to_dtype(DType::F32)?;
        let variance = x.sqr()?.mean_keepdim(D::Minus1)?;
        let x_normed = x.broadcast_div(&(variance + self.eps)?.sqrt()?)?;
        let x_normed = x_normed.broadcast_mul(&self.weight)?;
        let gate = candle_nn::ops::silu(&gate.to_dtype(DType::F32)?)?;
        (x_normed * gate)?.to_dtype(in_dtype)
    }
}

/// Registers a group of manifest weight names that all consume the same
/// (Hadamard-rotated) input — e.g. q/k/v sharing `ln1(x)`, or `in_proj_qkv`/
/// `in_proj_z` sharing the same DeltaNet input — and returns the one shared
/// transform. `None` when there is no active manifest (plain `qwen3_5`).
/// Each name is still registered individually with `HadamardLoader`, so its
/// `.signs` tensor is verified and it counts as claimed for `finish()`.
fn shared_in_transform(
    hadamard: &mut Option<HadamardLoader>,
    src: &SafeTensorsSource<'_>,
    bases: &[String],
    width: usize,
) -> Result<Option<Arc<HadamardTransform>>> {
    let Some(loader) = hadamard.as_mut() else {
        return Ok(None);
    };
    let mut results = Vec::with_capacity(bases.len());
    for base in bases {
        results.push((base.as_str(), loader.transform_for(base, width, src)?));
    }
    let rotated_count = results.iter().filter(|(_, t)| t.is_some()).count();
    if rotated_count != 0 && rotated_count != results.len() {
        let rotated: Vec<&str> = results
            .iter()
            .filter(|(_, t)| t.is_some())
            .map(|(b, _)| *b)
            .collect();
        let not_rotated: Vec<&str> = results
            .iter()
            .filter(|(_, t)| t.is_none())
            .map(|(b, _)| *b)
            .collect();
        candle::bail!(
            "hadamard manifest partially rotates a shared-input group: {rotated:?} are \
             rotated but {not_rotated:?} are not — a group that shares one input must be \
             all-or-nothing"
        );
    }
    Ok(results.into_iter().find_map(|(_, t)| t))
}

// ---------------------------------------------------------------------------
// Gated full-attention
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct GatedAttention {
    q_proj: QMatMul,
    k_proj: QMatMul,
    v_proj: QMatMul,
    o_proj: QMatMul,
    q_norm: Qwen35RmsNorm,
    k_norm: Qwen35RmsNorm,
    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    rotary_dim: usize,
    rotary: Arc<RotaryEmbedding>,
    kv_cache: ConcatKvCache,
    /// `prism_hadamard_qwen35` only: rotates the hidden-state input shared by
    /// q/k/v; `None` for plain `qwen3_5`.
    in_tf: Option<Arc<HadamardTransform>>,
    /// `prism_hadamard_qwen35` only: rotates the attention context before
    /// `o_proj`; `None` for plain `qwen3_5`.
    out_tf: Option<Arc<HadamardTransform>>,
}

impl GatedAttention {
    fn load(
        src: &mut SafeTensorsSource<'_>,
        cfg: &Qwen35TextConfig,
        rotary: Arc<RotaryEmbedding>,
        prefix: &str,
        hadamard: &mut Option<HadamardLoader>,
    ) -> Result<Self> {
        if !cfg
            .num_attention_heads
            .is_multiple_of(cfg.num_key_value_heads)
        {
            candle::bail!(
                "num_attention_heads ({}) must be divisible by num_key_value_heads ({})",
                cfg.num_attention_heads,
                cfg.num_key_value_heads
            );
        }
        let q = |n: &str| format!("{prefix}.self_attn.{n}");
        let load = |src: &mut SafeTensorsSource<'_>, n: &str| -> Result<QMatMul> {
            QMatMul::from_weights(Arc::new(src.load_q2mlx(&q(n))?))
        };
        let mixer_out_dim = cfg.num_attention_heads * cfg.head_dim;
        let in_tf = shared_in_transform(
            hadamard,
            src,
            &[q("q_proj"), q("k_proj"), q("v_proj")],
            cfg.hidden_size,
        )?;
        let out_tf = shared_in_transform(hadamard, src, &[q("o_proj")], mixer_out_dim)?;
        let q_proj = load(src, "q_proj")?;
        let k_proj = load(src, "k_proj")?;
        let v_proj = load(src, "v_proj")?;
        let o_proj = load(src, "o_proj")?;
        let q_norm = Qwen35RmsNorm::load(src, &q("q_norm.weight"), cfg.rms_norm_eps)?;
        let k_norm = Qwen35RmsNorm::load(src, &q("k_norm.weight"), cfg.rms_norm_eps)?;
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            num_heads: cfg.num_attention_heads,
            num_kv_heads: cfg.num_key_value_heads,
            num_kv_groups: cfg.num_attention_heads / cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
            rotary_dim: cfg.rotary_dim(),
            rotary,
            kv_cache: ConcatKvCache::new(2),
            in_tf,
            out_tf,
        })
    }

    /// Partial RoPE: rotate the first `rotary_dim` dims of each head, pass the
    /// rest through unchanged.
    fn apply_partial_rope(
        &self,
        q: &Tensor,
        k: &Tensor,
        offset: usize,
    ) -> Result<(Tensor, Tensor)> {
        let rd = self.rotary_dim;
        let rest = self.head_dim - rd;
        // Full rotary (rotary_dim == head_dim): rotate the whole head, no split.
        if rest == 0 {
            return self
                .rotary
                .apply(&q.contiguous()?, &k.contiguous()?, offset);
        }
        let q_rot = q.narrow(D::Minus1, 0, rd)?.contiguous()?;
        let q_pass = q.narrow(D::Minus1, rd, rest)?;
        let k_rot = k.narrow(D::Minus1, 0, rd)?.contiguous()?;
        let k_pass = k.narrow(D::Minus1, rd, rest)?;
        let (q_rot, k_rot) = self.rotary.apply(&q_rot, &k_rot, offset)?;
        // `cat` along the last dim yields a strided view the Metal matmul kernel
        // rejects; force a C-contiguous layout before it reaches attention.
        let q = Tensor::cat(&[&q_rot, &q_pass], D::Minus1)?.contiguous()?;
        let k = Tensor::cat(&[&k_rot, &k_pass], D::Minus1)?.contiguous()?;
        Ok((q, k))
    }

    fn forward(&mut self, x: &Tensor, mask: Option<&Tensor>, offset: usize) -> Result<Tensor> {
        let (b, l, _) = x.dims3()?;
        let hd = self.head_dim;
        let nh = self.num_heads;
        let xh = hadamard::apply(&self.in_tf, x)?;

        // q_proj packs [query | gate] interleaved per head: view as
        // (b, l, nh, 2*hd) and split the last axis in half.
        let qg = self.q_proj.forward(&xh)?.reshape((b, l, nh, 2 * hd))?;
        let query = qg.narrow(3, 0, hd)?.contiguous()?;
        let gate = qg
            .narrow(3, hd, hd)?
            .contiguous()?
            .reshape((b, l, nh * hd))?;

        let query = self.q_norm.forward(&query)?.transpose(1, 2)?.contiguous()?;
        let key = self
            .k_proj
            .forward(&xh)?
            .reshape((b, l, self.num_kv_heads, hd))?;
        let key = self.k_norm.forward(&key)?.transpose(1, 2)?.contiguous()?;
        let value = self
            .v_proj
            .forward(&xh)?
            .reshape((b, l, self.num_kv_heads, hd))?
            .transpose(1, 2)?
            .contiguous()?;

        let (query, key) = self.apply_partial_rope(&query, &key, offset)?;
        let (key, value) = self.kv_cache.append(&key, &value)?;
        let key = repeat_kv(key, self.num_kv_groups)?.contiguous()?;
        let value = repeat_kv(value, self.num_kv_groups)?.contiguous()?;

        let scale = 1.0 / (hd as f64).sqrt();
        let mut scores = (query.matmul(&key.transpose(2, 3)?)? * scale)?;
        if let Some(m) = mask {
            scores = scores.broadcast_add(&m.to_dtype(scores.dtype())?)?;
        }
        let probs = candle_nn::ops::softmax_last_dim(&scores)?;
        let ctx = probs
            .matmul(&value)?
            .transpose(1, 2)?
            .reshape((b, l, nh * hd))?;

        // Output gate: sigmoid(gate) (config's output_gate_type is not applied).
        let gated = (ctx * candle_nn::ops::sigmoid(&gate)?)?;
        let gated = hadamard::apply(&self.out_tf, &gated)?;
        self.o_proj.forward(&gated)
    }

    fn clear_state(&mut self) {
        self.kv_cache.reset();
    }
}

/// A linear projection that is either Q2MLX-quantized (plain `qwen3_5`) or a
/// dense F32 weight (`prism_hadamard_qwen35`'s `in_proj_a`/`in_proj_b`, which
/// are not Hadamard-rotated and are small enough that MLX ships them
/// unquantized). Detected by the presence of a `.scales` tensor, not by
/// checkpoint/model type, so either variant loads correctly either way.
#[derive(Debug, Clone)]
enum Projection {
    Quantized(QMatMul),
    /// Weight already transposed to `[in, out]` and made contiguous at load
    /// time, so `forward` is a plain `broadcast_matmul`.
    Dense(Tensor),
}

impl Projection {
    fn load(src: &mut SafeTensorsSource<'_>, base: &str) -> Result<Self> {
        if src.has_tensor(&format!("{base}.scales")) {
            Ok(Self::Quantized(QMatMul::from_weights(Arc::new(
                src.load_q2mlx(base)?,
            ))?))
        } else {
            let w = src
                .plain_tensor(&format!("{base}.weight"))?
                .to_dtype(DType::F32)?
                .t()?
                .contiguous()?;
            Ok(Self::Dense(w))
        }
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            Self::Quantized(q) => q.forward(x),
            Self::Dense(w) => x.broadcast_matmul(w),
        }
    }
}

// ---------------------------------------------------------------------------
// Gated DeltaNet (linear attention)
// ---------------------------------------------------------------------------

/// Per-layer streaming state carried across prefill/decode calls.
#[derive(Debug, Clone, Default)]
struct DeltaNetState {
    /// Last `conv_kernel - 1` conv inputs, `(b, conv_dim, conv_kernel-1)` F32.
    conv_state: Option<Tensor>,
    /// Delta-rule recurrent memory, `(b, n_v_heads, k_head_dim, v_head_dim)` F32.
    recurrent_state: Option<Tensor>,
}

impl DeltaNetState {
    fn reset(&mut self) {
        self.conv_state = None;
        self.recurrent_state = None;
    }
}

#[derive(Debug, Clone)]
struct GatedDeltaNet {
    in_proj_qkv: QMatMul,
    in_proj_z: QMatMul,
    in_proj_a: Projection,
    in_proj_b: Projection,
    out_proj: QMatMul,
    /// Depthwise causal conv taps, `conv_kernel` tensors of shape `(1, conv_dim, 1)`.
    conv_taps: Vec<Tensor>,
    /// `-exp(A_log)`, F32 `[n_v_heads]`.
    a_log_neg_exp: Tensor,
    /// `dt_bias`, F32 `[n_v_heads]`.
    dt_bias: Tensor,
    norm: Qwen35RmsNormGated,
    n_k_heads: usize,
    n_v_heads: usize,
    k_head_dim: usize,
    v_head_dim: usize,
    key_dim: usize,
    value_dim: usize,
    conv_dim: usize,
    conv_kernel: usize,
    state: DeltaNetState,
    /// `prism_hadamard_qwen35` only: rotates the hidden-state input shared by
    /// `in_proj_qkv`/`in_proj_z` (NOT `in_proj_a`/`in_proj_b`, which consume
    /// the un-rotated input); `None` for plain `qwen3_5`.
    in_tf: Option<Arc<HadamardTransform>>,
    /// `prism_hadamard_qwen35` only: rotates the gated-norm output before
    /// `out_proj`; `None` for plain `qwen3_5`.
    out_tf: Option<Arc<HadamardTransform>>,
}

impl GatedDeltaNet {
    fn load(
        src: &mut SafeTensorsSource<'_>,
        cfg: &Qwen35TextConfig,
        prefix: &str,
        hadamard: &mut Option<HadamardLoader>,
    ) -> Result<Self> {
        let n_k_heads = cfg.linear_num_key_heads;
        let n_v_heads = cfg.linear_num_value_heads;
        if !n_v_heads.is_multiple_of(n_k_heads) {
            candle::bail!(
                "linear_num_value_heads ({n_v_heads}) must be divisible by linear_num_key_heads ({n_k_heads})"
            );
        }
        let k_head_dim = cfg.linear_key_head_dim;
        let v_head_dim = cfg.linear_value_head_dim;
        let key_dim = n_k_heads * k_head_dim;
        let value_dim = n_v_heads * v_head_dim;
        let conv_dim = key_dim * 2 + value_dim;
        let conv_kernel = cfg.linear_conv_kernel_dim;

        let p = |n: &str| format!("{prefix}.linear_attn.{n}");
        let load = |src: &mut SafeTensorsSource<'_>, n: &str| -> Result<QMatMul> {
            QMatMul::from_weights(Arc::new(src.load_q2mlx(&p(n))?))
        };
        let in_tf = shared_in_transform(
            hadamard,
            src,
            &[p("in_proj_qkv"), p("in_proj_z")],
            cfg.hidden_size,
        )?;
        let out_tf = shared_in_transform(hadamard, src, &[p("out_proj")], value_dim)?;
        let in_proj_qkv = load(src, "in_proj_qkv")?;
        let in_proj_z = load(src, "in_proj_z")?;
        let in_proj_a = Projection::load(src, &p("in_proj_a"))?;
        let in_proj_b = Projection::load(src, &p("in_proj_b"))?;
        let out_proj = load(src, "out_proj")?;

        // conv1d.weight is F16 [conv_dim, conv_kernel, 1] (MLX out,kernel,in).
        // Drop the trailing singleton and split into per-tap (1, conv_dim, 1) F32.
        let conv_w = src
            .plain_tensor(&p("conv1d.weight"))?
            .to_dtype(DType::F32)?
            .reshape((conv_dim, conv_kernel))?;
        let mut conv_taps = Vec::with_capacity(conv_kernel);
        for k in 0..conv_kernel {
            conv_taps.push(
                conv_w
                    .narrow(1, k, 1)?
                    .reshape((1, conv_dim, 1))?
                    .contiguous()?,
            );
        }

        let a_log = src.plain_tensor(&p("A_log"))?.to_dtype(DType::F32)?;
        let a_log_neg_exp = a_log.exp()?.neg()?;
        let dt_bias = src.plain_tensor(&p("dt_bias"))?.to_dtype(DType::F32)?;
        let norm = Qwen35RmsNormGated::load(src, &p("norm.weight"), cfg.rms_norm_eps)?;

        Ok(Self {
            in_proj_qkv,
            in_proj_z,
            in_proj_a,
            in_proj_b,
            out_proj,
            conv_taps,
            a_log_neg_exp,
            dt_bias,
            norm,
            n_k_heads,
            n_v_heads,
            k_head_dim,
            v_head_dim,
            key_dim,
            value_dim,
            conv_dim,
            conv_kernel,
            state: DeltaNetState::default(),
            in_tf,
            out_tf,
        })
    }

    /// Depthwise causal conv over `(b, conv_dim, l)` via `conv_kernel` shifted
    /// multiply-accumulates, prepending the cached left-context. Returns the conv
    /// output `(b, conv_dim, l)` and updates `conv_state`.
    fn causal_conv(&mut self, mixed_t: &Tensor, b: usize, l: usize) -> Result<Tensor> {
        let pad = self.conv_kernel - 1;
        let prev = match &self.state.conv_state {
            Some(cs) => cs.clone(),
            None => Tensor::zeros((b, self.conv_dim, pad), DTYPE, mixed_t.device())?,
        };
        let full = Tensor::cat(&[&prev, mixed_t], 2)?; // (b, conv_dim, pad + l)
        let mut out = full.narrow(2, 0, l)?.broadcast_mul(&self.conv_taps[0])?;
        for k in 1..self.conv_kernel {
            out = (out + full.narrow(2, k, l)?.broadcast_mul(&self.conv_taps[k])?)?;
        }
        // conv_state = last `pad` real input columns.
        let full_len = pad + l;
        self.state.conv_state = Some(full.narrow(2, full_len - pad, pad)?.contiguous()?);
        Ok(out)
    }

    /// Explicit per-token delta-rule recurrence (F32 state), mirroring
    /// `torch_recurrent_gated_delta_rule`.
    fn recurrent_scan(
        &mut self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        g_exp: &Tensor,
        beta: &Tensor,
    ) -> Result<Tensor> {
        let (b, l, _, _) = q.dims4()?;
        let nv = self.n_v_heads;
        let hk = self.k_head_dim;
        let hv = self.v_head_dim;
        let mut state = match &self.state.recurrent_state {
            Some(s) => s.clone(),
            None => Tensor::zeros((b, nv, hk, hv), DTYPE, q.device())?,
        };
        let mut outs = Vec::with_capacity(l);
        for t in 0..l {
            let q_col = q.narrow(1, t, 1)?.contiguous()?.reshape((b, nv, hk, 1))?;
            let k_col = k.narrow(1, t, 1)?.contiguous()?.reshape((b, nv, hk, 1))?;
            let v_t = v.narrow(1, t, 1)?.contiguous()?.reshape((b, nv, hv))?;
            let g_t = g_exp
                .narrow(1, t, 1)?
                .contiguous()?
                .reshape((b, nv, 1, 1))?;
            let beta_t = beta.narrow(1, t, 1)?.contiguous()?.reshape((b, nv, 1))?;

            state = state.broadcast_mul(&g_t)?;
            let kv_mem = state.broadcast_mul(&k_col)?.sum(2)?; // (b, nv, hv)
            let delta = v_t.broadcast_sub(&kv_mem)?.broadcast_mul(&beta_t)?;
            let delta_row = delta.reshape((b, nv, 1, hv))?;
            state = (state + k_col.broadcast_mul(&delta_row)?)?;
            let out_t = state.broadcast_mul(&q_col)?.sum(2)?; // (b, nv, hv)
            outs.push(out_t.reshape((b, 1, nv, hv))?);
        }
        self.state.recurrent_state = Some(state);
        Tensor::cat(&outs, 1) // (b, l, nv, hv)
    }

    fn forward(&mut self, x: &Tensor) -> Result<Tensor> {
        let (b, l, _) = x.dims3()?;
        let nk = self.n_k_heads;
        let nv = self.n_v_heads;
        let hk = self.k_head_dim;
        let hv = self.v_head_dim;

        let xh = hadamard::apply(&self.in_tf, x)?;
        let mixed = self.in_proj_qkv.forward(&xh)?; // (b, l, conv_dim)
        let z = self.in_proj_z.forward(&xh)?.reshape((b, l, nv, hv))?;
        // in_proj_a/b are not Hadamard-rotated: they consume the original x.
        let b_proj = self.in_proj_b.forward(x)?; // (b, l, nv)
        let a_proj = self.in_proj_a.forward(x)?; // (b, l, nv)

        // Depthwise causal conv + silu.
        let mixed_t = mixed.transpose(1, 2)?.to_dtype(DTYPE)?.contiguous()?; // (b, conv_dim, l)
        let conv_out = self.causal_conv(&mixed_t, b, l)?;
        let mixed = candle_nn::ops::silu(&conv_out)?
            .transpose(1, 2)?
            .contiguous()?; // (b, l, conv_dim)

        let q = mixed.narrow(2, 0, self.key_dim)?.reshape((b, l, nk, hk))?;
        let k = mixed
            .narrow(2, self.key_dim, self.key_dim)?
            .reshape((b, l, nk, hk))?;
        let v = mixed
            .narrow(2, 2 * self.key_dim, self.value_dim)?
            .reshape((b, l, nv, hv))?
            .contiguous()?;

        let beta = candle_nn::ops::sigmoid(&b_proj.to_dtype(DTYPE)?)?; // (b, l, nv)
        let a = a_proj.to_dtype(DTYPE)?.broadcast_add(&self.dt_bias)?;
        let g = softplus(&a)?.broadcast_mul(&self.a_log_neg_exp)?; // (b, l, nv)
        let g_exp = g.exp()?;

        let rep = nv / nk;
        let q = repeat_interleave_heads(&q, rep)?;
        let k = repeat_interleave_heads(&k, rep)?;
        let q = l2_normalize(&q)?;
        let k = l2_normalize(&k)?;
        let q = (q * (1.0 / (hk as f64).sqrt()))?;

        let core = self.recurrent_scan(&q, &k, &v, &g_exp, &beta)?; // (b, l, nv, hv)

        let core = core.reshape((b * l * nv, hv))?;
        let z = z.to_dtype(DTYPE)?.reshape((b * l * nv, hv))?;
        let normed = self
            .norm
            .forward(&core, &z)?
            .reshape((b, l, nv * hv))?
            .to_dtype(x.dtype())?;
        let normed = hadamard::apply(&self.out_tf, &normed)?;
        self.out_proj.forward(&normed)
    }
}

// ---------------------------------------------------------------------------
// Hybrid decoder layer
// ---------------------------------------------------------------------------

enum HybridLayer {
    Full {
        attn: GatedAttention,
        mlp: MlpWeights,
        ln1: Qwen35RmsNorm,
        ln2: Qwen35RmsNorm,
    },
    Linear {
        deltanet: GatedDeltaNet,
        mlp: MlpWeights,
        ln1: Qwen35RmsNorm,
        ln2: Qwen35RmsNorm,
    },
}

impl HybridLayer {
    fn forward(&mut self, x: &Tensor, mask: Option<&Tensor>, offset: usize) -> Result<Tensor> {
        match self {
            HybridLayer::Full {
                attn,
                mlp,
                ln1,
                ln2,
            } => {
                let h = attn.forward(&ln1.forward(x)?, mask, offset)?;
                let x = (x + h)?;
                let h2 = mlp.forward(&ln2.forward(&x)?)?;
                x + h2
            }
            HybridLayer::Linear {
                deltanet,
                mlp,
                ln1,
                ln2,
            } => {
                let h = deltanet.forward(&ln1.forward(x)?)?;
                let x = (x + h)?;
                let h2 = mlp.forward(&ln2.forward(&x)?)?;
                x + h2
            }
        }
    }

    fn clear_state(&mut self) {
        match self {
            HybridLayer::Full { attn, .. } => attn.clear_state(),
            HybridLayer::Linear { deltanet, .. } => deltanet.state.reset(),
        }
    }
}

// ---------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------

pub struct Qwen35Weights {
    embed_tokens: Embedding,
    layers: Vec<HybridLayer>,
    norm: Qwen35RmsNorm,
    lm_head: QMatMul,
    device: Device,
    /// `prism_hadamard_qwen35` only: un-rotates the gathered embedding rows;
    /// `None` for plain `qwen3_5`.
    embed_inv_tf: Option<Arc<HadamardTransform>>,
    /// `prism_hadamard_qwen35` only: rotates the final hidden state before
    /// `lm_head`; `None` for plain `qwen3_5`.
    head_tf: Option<Arc<HadamardTransform>>,
}

impl Qwen35Weights {
    /// `hadamard` is `Some` iff `config.json` set `hadamard_config`
    /// (`prism_hadamard_qwen35`); it is consumed and `HadamardLoader::finish`
    /// checked once every layer is loaded, so an un-rotated projection the
    /// manifest expected to be rotated fails loudly instead of silently
    /// producing wrong activations.
    pub fn from_safetensors(
        src: &mut SafeTensorsSource<'_>,
        cfg: &Qwen35Config,
        device: &Device,
        hadamard: Option<HadamardManifest>,
    ) -> Result<Self> {
        let tc = &cfg.text_config;
        let eps = tc.rms_norm_eps;
        let mut hadamard_loader = hadamard
            .map(|m| HadamardLoader::new(m, device))
            .transpose()?;

        let rotary = Arc::new(RotaryEmbedding::new_unscaled(
            DTYPE,
            tc.rotary_dim(),
            tc.max_position_embeddings,
            tc.rope_theta(),
            device,
        )?);

        // The (large) embedding table is materialized as a dense lookup tensor.
        // Keep it in F16 (~2.5 GB) rather than F32 (~5 GB) — the gathered rows are
        // cast to the F32 compute dtype at use, so this only halves resident memory.
        let embed_qt = src.load_q2mlx("language_model.model.embed_tokens")?;
        let embed_tokens = Embedding::new(embed_qt.dequantize_f16(device)?, tc.hidden_size);
        let embed_inv_tf = match hadamard_loader.as_mut() {
            Some(loader) => {
                loader.inverse_for("language_model.model.embed_tokens", tc.hidden_size, src)?
            }
            None => None,
        };

        let mut layers = Vec::with_capacity(tc.num_hidden_layers);
        for i in 0..tc.num_hidden_layers {
            let prefix = format!("language_model.model.layers.{i}");
            let ln1 = Qwen35RmsNorm::load(src, &format!("{prefix}.input_layernorm.weight"), eps)?;
            let ln2 = Qwen35RmsNorm::load(
                src,
                &format!("{prefix}.post_attention_layernorm.weight"),
                eps,
            )?;
            let mlp = load_mlp(src, &prefix, tc, &mut hadamard_loader)?;
            let layer = if (i + 1) % tc.full_attention_interval == 0 {
                HybridLayer::Full {
                    attn: GatedAttention::load(
                        src,
                        tc,
                        rotary.clone(),
                        &prefix,
                        &mut hadamard_loader,
                    )?,
                    mlp,
                    ln1,
                    ln2,
                }
            } else {
                HybridLayer::Linear {
                    deltanet: GatedDeltaNet::load(src, tc, &prefix, &mut hadamard_loader)?,
                    mlp,
                    ln1,
                    ln2,
                }
            };
            layers.push(layer);
        }

        let norm = Qwen35RmsNorm::load(src, "language_model.model.norm.weight", eps)?;
        let head_tf = match hadamard_loader.as_mut() {
            Some(loader) => loader.transform_for("language_model.lm_head", tc.hidden_size, src)?,
            None => None,
        };
        let lm_head = QMatMul::from_weights(Arc::new(src.load_q2mlx("language_model.lm_head")?))?;

        if let Some(loader) = hadamard_loader {
            loader.finish()?;
        }

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            device: device.clone(),
            embed_inv_tf,
            head_tf,
        })
    }

    /// Additive causal mask of shape `(1, 1, tgt, tgt + offset)`. The batch and
    /// head dims broadcast in the attention `broadcast_add`, so it is valid for
    /// any batch size.
    fn causal_mask(&self, tgt: usize, offset: usize) -> Result<Tensor> {
        let minf = f32::NEG_INFINITY;
        let mask: Vec<_> = (0..tgt)
            .flat_map(|i| {
                (0..(tgt + offset)).map(move |j| if j <= i + offset { 0.0 } else { minf })
            })
            .collect();
        Tensor::from_slice(&mask, (1, 1, tgt, tgt + offset), &self.device)
    }

    /// Runs the text stack for `input` (`(batch, seq)` token ids) at absolute
    /// position `offset`. Each batch row is assumed to be a single, unpadded
    /// sequence: no attention padding mask is applied (the reference's
    /// `apply_mask_to_padding_states` is a no-op for unpadded input), so batched
    /// inference is only correct for equal-length, unpadded sequences.
    pub fn forward(&mut self, input: &Tensor, offset: usize) -> Result<Tensor> {
        let (_, l) = input.dims2()?;
        let mut h = self.embed_tokens.forward(input)?.to_dtype(DTYPE)?;
        if let Some(tf) = &self.embed_inv_tf {
            h = tf.inverse(&h)?;
        }
        let mask = if l == 1 {
            None
        } else {
            Some(self.causal_mask(l, offset)?)
        };
        for layer in &mut self.layers {
            h = layer.forward(&h, mask.as_ref(), offset)?;
        }
        let h = self.norm.forward(&h)?;
        let last_hidden = h.narrow(1, l - 1, 1)?;
        let last_hidden = hadamard::apply(&self.head_tf, &last_hidden)?;
        self.lm_head.forward(&last_hidden)?.squeeze(1)
    }

    pub fn clear_state(&mut self) {
        for layer in &mut self.layers {
            layer.clear_state();
        }
    }
}

/// Load the SwiGLU MLP projections for a layer from HuggingFace tensor names.
fn load_mlp(
    src: &mut SafeTensorsSource<'_>,
    prefix: &str,
    cfg: &Qwen35TextConfig,
    hadamard: &mut Option<HadamardLoader>,
) -> Result<MlpWeights> {
    let m = |n: &str| format!("{prefix}.mlp.{n}");
    let in_tf = shared_in_transform(
        hadamard,
        src,
        &[m("gate_proj"), m("up_proj")],
        cfg.hidden_size,
    )?;
    let down_tf = shared_in_transform(hadamard, src, &[m("down_proj")], cfg.intermediate_size)?;
    let gate_proj = QMatMul::from_weights(Arc::new(src.load_q2mlx(&m("gate_proj"))?))?;
    let up_proj = QMatMul::from_weights(Arc::new(src.load_q2mlx(&m("up_proj"))?))?;
    let down_proj = QMatMul::from_weights(Arc::new(src.load_q2mlx(&m("down_proj"))?))?;
    Ok(MlpWeights::from_qmatmuls(gate_proj, up_proj, down_proj).with_hadamard(in_tf, down_tf))
}

#[cfg(test)]
mod tests {
    use super::*;
    use safetensors::tensor::TensorView;
    use safetensors::{Dtype, SafeTensors};

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

    fn f32_bytes(values: &[f32]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    #[test]
    fn projection_dense_forward_matches_matmul() {
        let device = Device::Cpu;
        // w: [out=2, in=3], rows [1,2,3] and [4,5,6].
        let w = Tensor::from_vec(vec![1f32, 2., 3., 4., 5., 6.], (2, 3), &device).unwrap();
        let w_t = w.t().unwrap().contiguous().unwrap();
        let proj = Projection::Dense(w_t);
        let x = Tensor::from_vec(vec![1f32, 0., 0.], (1, 3), &device).unwrap();
        let y = proj.forward(&x).unwrap().to_vec2::<f32>().unwrap();
        assert_eq!(y, vec![vec![1.0, 4.0]]);
    }

    fn minimal_hadamard_manifest_json() -> &'static str {
        r#"{
            "prism.hadamard.version": 1,
            "prism.hadamard.block_size": 8,
            "prism.hadamard.transform": "normalized-sylvester-walsh-hadamard",
            "prism.hadamard.axis": "input-last-dimension",
            "prism.hadamard.sign_mode": "explicit",
            "prism.hadamard.weight_names": ["a.weight"],
            "prism.hadamard.inverse_weight_names": [],
            "prism.hadamard.sign_widths": [8],
            "prism.hadamard.sign_values": [1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0]
        }"#
    }

    #[test]
    fn hadamard_loader_finish_errors_on_unclaimed_name() {
        let device = Device::Cpu;
        let manifest = HadamardManifest::from_json(minimal_hadamard_manifest_json()).unwrap();
        let loader = HadamardLoader::new(manifest, &device).unwrap();
        // "a.weight" is listed in weight_names but transform_for was never called.
        assert!(loader.finish().is_err());
    }

    #[test]
    fn hadamard_loader_transform_for_errors_on_signs_mismatch() {
        let device = Device::Cpu;
        let manifest = HadamardManifest::from_json(minimal_hadamard_manifest_json()).unwrap();
        let mut loader = HadamardLoader::new(manifest, &device).unwrap();

        // "a.signs" disagrees with the manifest's sign vector for width 8
        // (all +1.0 instead of alternating +-1.0).
        let buf = build_safetensors(&[("a.signs", Dtype::F32, vec![8], f32_bytes(&[1.0; 8]))]);
        let st = SafeTensors::deserialize(&buf).unwrap();
        let src = SafeTensorsSource::new(st, device);

        assert!(loader.transform_for("a", 8, &src).is_err());
    }

    #[test]
    fn hadamard_loader_transform_for_succeeds_when_signs_match() {
        let device = Device::Cpu;
        let manifest = HadamardManifest::from_json(minimal_hadamard_manifest_json()).unwrap();
        let mut loader = HadamardLoader::new(manifest, &device).unwrap();

        let buf = build_safetensors(&[(
            "a.signs",
            Dtype::F32,
            vec![8],
            f32_bytes(&[1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0]),
        )]);
        let st = SafeTensors::deserialize(&buf).unwrap();
        let src = SafeTensorsSource::new(st, device);

        let tf = loader.transform_for("a", 8, &src).unwrap();
        assert!(tf.is_some());
        loader.finish().unwrap();
    }

    #[test]
    fn shared_in_transform_errors_when_group_is_partially_rotated() {
        let device = Device::Cpu;
        // Only "a.weight" is in weight_names; "b.weight" is not, so a group
        // sharing one input where only "a" is claimed must bail rather than
        // silently rotating the input for "b" too (or not rotating for "a").
        let manifest = HadamardManifest::from_json(minimal_hadamard_manifest_json()).unwrap();
        let mut hadamard = Some(HadamardLoader::new(manifest, &device).unwrap());

        let buf = build_safetensors(&[(
            "a.signs",
            Dtype::F32,
            vec![8],
            f32_bytes(&[1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0]),
        )]);
        let st = SafeTensors::deserialize(&buf).unwrap();
        let src = SafeTensorsSource::new(st, device);

        let bases = vec!["a".to_owned(), "b".to_owned()];
        let result = shared_in_transform(&mut hadamard, &src, &bases, 8);
        assert!(result.is_err());
    }

    #[test]
    fn shared_in_transform_ok_when_group_fully_unrotated() {
        let device = Device::Cpu;
        // Neither "x" nor "y" is in weight_names: the whole group is
        // legitimately un-rotated (e.g. a plain qwen3_5 checkpoint with no
        // hadamard manifest at all, or a group the manifest doesn't cover).
        let manifest = HadamardManifest::from_json(minimal_hadamard_manifest_json()).unwrap();
        let mut hadamard = Some(HadamardLoader::new(manifest, &device).unwrap());

        let buf = build_safetensors(&[("unused", Dtype::F32, vec![1], f32_bytes(&[0.0]))]);
        let st = SafeTensors::deserialize(&buf).unwrap();
        let src = SafeTensorsSource::new(st, device);

        let bases = vec!["x".to_owned(), "y".to_owned()];
        let result = shared_in_transform(&mut hadamard, &src, &bases, 8).unwrap();
        assert!(result.is_none());
    }
}
