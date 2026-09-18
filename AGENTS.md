# AGENTS.md

## What This Repo Contains

This workspace is a native Rust inference port for Bonsai models built on top of Candle. The candle-core and candle-metal-kernels crates are vendored and patched; candle-nn and candle-transformers come from crates.io with candle-core overridden via `[patch.crates-io]`.

Current intended path:

- load either a Bonsai GGUF file (`Bonsai-1.7B.gguf`, `Bonsai-8B.gguf`) or a
  HuggingFace/MLX directory (e.g. `prism-ml/Ternary-Bonsai-8B-mlx-2bit`,
  `prism-ml/Ternary-Bonsai-27B-mlx-2bit`)
- run inference on `CPU` or `Metal` for both formats
- use Candle-native GGUF loading / safetensors loading, tokenizer loading,
  sampling, and streaming output

Important constraints:

- this is inference-only
- two quantization formats are supported:
  - GGUF dtype `Q1_0_g128` — binary {−d, +d}, 1 bit/weight, block 128 (CPU + Metal)
  - runtime-only dtype `Q2MLX` — MLX 2-bit affine `w = scale·q + bias`
    (q ∈ {0,1,2,3}), group 128, loaded from MLX safetensors (CPU + Metal)
  - the Hadamard rotation used by `prism_hadamard_qwen35` (`Ternary-Bonsai-2-27B`)
    is **not** a third `GgmlDType` — it is an activation-side transform
    (`crates/bonsai-candle/src/hadamard.rs`) applied before/after the `Q2MLX`
    matmul; the packed weight format underneath is unchanged Q2MLX
- architectures: Qwen3 with YaRN rope scaling (GGUF + `Ternary-Bonsai-8B`), and
  the Qwen3.5 hybrid (`qwen3_5`, `Ternary-Bonsai-27B`; `prism_hadamard_qwen35`,
  `Ternary-Bonsai-2-27B`) — text-only, dispatched by `config.json` `model_type`

## Workspace Layout

- `Cargo.toml`
  - root workspace manifest
  - pins all workspace dependencies
  - points `candle` and `candle-metal-kernels` to `vendor/candle/...`
  - pulls `candle-nn` and `candle-transformers` from crates.io `0.10.2`
  - `[patch.crates-io]` redirects `candle-core` and `candle-metal-kernels` to vendor so upstream crates use the patched versions
- `crates/bonsai-candle`
  - library crate
  - owns model loading, prompt rendering, token streaming, sampling, repeat control, and generation stats
  - contains the Qwen3/Bonsai model implementation with YaRN rope scaling (`src/qwen3.rs`)
  - contains the Qwen3.5 hybrid implementation (`src/qwen3_5.rs`)
  - contains the Hadamard-rotation activation transform for `prism_hadamard_qwen35` (`src/hadamard.rs`)
  - contains `ConcatKvCache` (`src/kv_cache.rs`) and `LogitsProcessor`/`Sampling` (`src/generation.rs`)
- `crates/bonsai-cli`
  - CLI wrapper over `bonsai-candle`
  - accepts either `--prompt` or `--messages-file`
  - prints streamed generation plus final stats
- `vendor/candle`
  - contains only `candle-core` and `candle-metal-kernels`
  - do not assume upstream Candle has these patches

## Core Rust Components

### `bonsai-candle`

Main responsibilities:

- choose device via `DevicePreference`
- load GGUF and tokenizer from the model file
- use `crate::qwen3::ModelWeights` (local copy with YaRN support)
- render chat prompts from GGUF `tokenizer.chat_template`
- stream generated text to a writer
- collect `prompt_tokens`, `generated_tokens`, `prompt_tps`, `generation_tps`, and `peak_memory`
- reduce loops with:
  - `repeat_penalty`
  - `repeat_last_n`
  - `no_repeat_ngram_size`

Local modules (not from candle-transformers):

- `src/qwen3.rs` — quantized Qwen3 model with YaRN rope scaling; adapted from candle-transformers with extended context support
- `src/qwen3_5.rs` — Qwen3.5 hybrid (`qwen3_5` / `prism_hadamard_qwen35`) text stack: Gated DeltaNet linear-attention + gated full-attention, loaded from MLX `Q2MLX` safetensors
- `src/hadamard.rs` — blockwise normalized Sylvester Walsh–Hadamard transform (`prism_hadamard_qwen35` activation rotation) and its `hadamard.json` manifest parser/loader
- `src/kv_cache.rs` — `ConcatKvCache` using `Tensor::cat` for Metal/CUDA-optimized KV cache
- `src/generation.rs` — `LogitsProcessor` and `Sampling` enum (ArgMax, All, TopK, TopP, TopKThenTopP)

Important public types:

- `LoadOptions`
- `GenerateOptions`
- `GenerationStats`
- `ChatMessage`
- `MessageRole`
- `TemplateErrorMode`
- `BonsaiModel`

Current default generation behavior:

- `temperature = 0.5`
- `top_p = 0.85`
- `top_k = 20`
- `repeat_penalty = 1.1`
- `repeat_last_n = 256`
- `no_repeat_ngram_size = 6`
- `seed = rand::random()` (new random seed each run)

### `bonsai-cli`

Main responsibilities:

- parse CLI args
- build `GenerateOptions`
- choose `CPU` or `Metal`
- run prompt-based or message-based generation
- print final stats

Useful flags:

- `--model <path>` — either `.gguf` file or directory containing MLX/HF
  weights (`config.json`, `model.safetensors`, `tokenizer.json`,
  `chat_template.jinja`); autodetected
- `--device auto|cpu|metal`
- `--prompt "..."` or `--messages-file messages.json`
- `--raw-prompt`
- `--template-mode strict|warn-fallback`
- `--seed <u64>` (optional; random if omitted)
- `--repeat-penalty`
- `--repeat-last-n`
- `--no-repeat-ngram-size`

## Vendored Candle Patches

Only `candle-core` and `candle-metal-kernels` are vendored. `candle-nn` and `candle-transformers` come from crates.io. The `[patch.crates-io]` section in the root `Cargo.toml` ensures all crates (including upstream candle-nn/candle-transformers) resolve to the patched candle-core.

### Quantized GGUF support

Patched files:

- `vendor/candle/candle-core/src/quantized/mod.rs`
- `vendor/candle/candle-core/src/quantized/ggml_file.rs`
- `vendor/candle/candle-core/src/quantized/k_quants.rs`
- `vendor/candle/candle-core/src/quantized/metal.rs`
- `vendor/candle/candle-metal-kernels/src/metal_src/quantized.metal`
- `vendor/candle/candle-metal-kernels/src/kernels/quantized.rs`

What these patches do:

- add GGUF dtype `Q1_0_g128`
- map GGUF dtype id `41`
- load and store `BlockQ1_0_g128`
- support CPU quantized matmul for this format
- support Metal quantized kernels for this format
- add runtime-only dtype `Q2MLX` with `BlockQ2MLX { scale: f16, bias: f16,
  qs: [u8; 32] }` (36 bytes / 128 weights = 2.25 bpw). CPU: `to_float`,
  `vec_dot` via Q8_0 activations, `matmul_q2_mlx` helper. Metal:
  `dequantize_q2_mlx` + `kernel_mul_mv_q2_mlx_f32` + template instantiation
  `kernel_mul_mm_q2_mlx_f32`. Constructed from MLX safetensors via
  `QStorage::from_data(..., GgmlDType::Q2MLX)`; has no GGUF id.

Important rule:

- do not replace these with upstream Candle without verifying that upstream fully supports `Q1_0_g128`
- `candle-nn` and `candle-transformers` can be upgraded on crates.io independently as long as their API is compatible with the types used in `bonsai-candle/src/qwen3.rs`

### Qwen3 / Bonsai model

The quantized Qwen3 model lives in `crates/bonsai-candle/src/qwen3.rs`. It is not taken from candle-transformers at runtime — it is a local copy extended with YaRN rope scaling.

What to preserve:

- Bonsai GGUF loads through the quantized Qwen3 path in `crates/bonsai-candle/src/qwen3.rs`
- YaRN rope scaling reads `qwen3.rope.scaling.type`, `.factor`, `.original_context_length`, `.yarn_beta_fast`, `.yarn_beta_slow`, `.yarn_log_multiplier` from GGUF metadata
- `ConcatKvCache` in `src/kv_cache.rs` uses concatenation (not slice_set) — keep this for Metal performance

### Qwen3.5 hybrid model (`qwen3_5`)

`crates/bonsai-candle/src/qwen3_5.rs` implements text-only inference for the
`qwen3_5` architecture (`Ternary-Bonsai-27B`). `lib.rs` peeks `config.json`
`model_type` and dispatches to `Qwen35Weights` vs the existing `ModelWeights`
through the internal `Model` enum. Tensor names are `language_model.model.*` /
`language_model.lm_head` (the vision tower and `mtp.*` are skipped).

Structure (64 layers, one full-attention layer every `full_attention_interval = 4`):

- **Gated DeltaNet** (linear layers): `in_proj_qkv/z/a/b` (Q2MLX) + a plain-F16
  depthwise causal `conv1d` (kernel 4, applied as an explicit 4-tap MAC), silu,
  L2-normed q/k with `repeat_interleave` k-heads to the v-head count, `beta =
  sigmoid`, `g = -exp(A_log)·softplus(a + dt_bias)`, an explicit per-token
  delta-rule recurrence, then `RMSNormGated` and `out_proj`. Per-layer state
  (`conv_state`, `recurrent_state`) is F32 and carried across prefill/decode.
- **Gated full-attention** (full layers): `q_proj` packs `[query | gate]`
  interleaved per head, partial RoPE (rotates the first `partial_rotary_factor ×
  head_dim` dims), GQA, and a `sigmoid(gate)` output gate. Reuses `ConcatKvCache`.

Non-obvious details verified against the `transformers` `models/qwen3_5` reference:

- The MLX checkpoint **folds the `+1`** of `Qwen3_5RMSNorm`'s `(1 + weight)` into the
  stored weights (they are centered at ~1), so `Qwen35RmsNorm` applies plain
  `weight`. Do not add `1` again — that double-scales every norm and produces
  garbage output.
- The output gate is `sigmoid(gate)` even though `output_gate_type = "swish"`; the
  reference model does not read that config field.
- The DeltaNet query is scaled by `1/sqrt(key_head_dim)` after L2-norm; the key is not.
- The whole activation path runs in F32 — the vendored Metal quantized-matmul kernels
  require F32 input and always emit F32 (`quantized/metal.rs`), so there is no F16 path
  for `Q2MLX` projections. The one dense tensor is the embedding table, which is kept in
  F16 (`dequantize_f16`, ~2.5 GB instead of ~5 GB); its rows are cast to F32 at use.

The per-token recurrence is sequential, so 27B throughput is well below the
plain-attention models; chunked prefill is a future optimization.

### Hadamard-rotated variant (`prism_hadamard_qwen35`, `Ternary-Bonsai-2-27B`)

Same text topology as `qwen3_5` (`base_model_type = "qwen3_5"` in `config.json`,
byte-identical `text_config` besides `eos_token_id`/`mtp_num_hidden_layers`), but
every packed linear/embedding weight is quantized in a basis rotated by a
blockwise (block 1024) normalized Sylvester Walsh–Hadamard transform. `lib.rs`
dispatches both `"qwen3_5"` and `"prism_hadamard_qwen35"` `model_type` values to
`Qwen35Weights`; the manifest named by `config.json`'s `hadamard_config`
(`hadamard.json`) is parsed by `HadamardManifest::from_json` and required when
`model_type == "prism_hadamard_qwen35"`.

- **Transform**: `HadamardTransform::forward` = `(x * signs) @ (H_1024 / sqrt(1024))`;
  `::inverse` = `(x @ (H_1024 / sqrt(1024))) * signs` — signs-then-rotate forward,
  rotate-then-signs inverse. Getting this order backwards produces wrong (but
  still tensor-shaped) output, not a crash.
- **Sign vectors** are keyed by input width, not by tensor: 5120 (hidden state),
  6144 (mixer output: `o_proj`/`out_proj` input), 17408 (FFN-down input,
  `intermediate_size`). `HadamardLoader` cross-checks each tensor's `.signs`
  safetensors entry against the manifest's vector for that width rather than
  trusting either source alone.
- **Shared rotated inputs**: the transform is computed once per consumer group
  and reused — q/k/v share `ln1(x)`'s rotation, `in_proj_qkv`/`in_proj_z` share
  the DeltaNet input's rotation, `gate_proj`/`up_proj` share the MLP input's
  rotation — via `shared_in_transform` in `qwen3_5.rs`, not recomputed per
  projection.
- **`in_proj_a`/`in_proj_b` are NOT Hadamard-rotated.** They are also no longer
  Q2MLX in this checkpoint — they ship as plain dense F32 `[n_v_heads, hidden_size]`
  weights, detected by the *absence* of a `.scales` tensor (`Projection::load` in
  `qwen3_5.rs`), not by model type, so an old `qwen3_5` checkpoint (where they are
  still Q2MLX) keeps working unchanged. Both variants consume the **un-rotated**
  layer input — only `in_proj_qkv`/`in_proj_z` see the rotated input.
- **Small tensors are F32, not F16**: `input_layernorm`, `post_attention_layernorm`,
  `linear_attn.norm`, `q_norm`/`k_norm`, `A_log`, `dt_bias`, `conv1d.weight`,
  `model.norm` — the RMSNorm `+1` fold still applies (values still centered at
  ~1, not ~0). `SafeTensorsSource::plain_tensor` reads the real safetensors dtype
  instead of assuming F16 (the old `plain_f16` would silently misread or fail on
  these).
- **`eos_token_id` trap**: `text_config.eos_token_id = 248044` is `<|endoftext|>`
  (== bos); the real turn-end token `<|im_end|>` (248046) is only in
  `generation_config.json`. `lib.rs::read_generation_config_eos` takes priority
  over `config.json`'s value for both `qwen3_5` and `prism_hadamard_qwen35`
  checkpoints (harmless where they already agree).
- **Load-time completeness check**: `HadamardLoader::finish` errors if any
  `hadamard.json` `weight_names`/`inverse_weight_names` entry was never claimed
  by a loaded tensor — catches a future checkpoint variant that rotates a
  different set of projections instead of silently under-rotating.
- Vision tower tensors are present in the checkpoint but ignored, same as plain
  `qwen3_5`; there are no MTP tensors (`mtp_num_hidden_layers = 0`).

### Chat template compatibility

Library file:

- `crates/bonsai-candle/src/lib.rs`

What to preserve:

- prompt rendering uses `minijinja`
- Python-like string methods are provided through `minijinja-contrib` with `pycompat`
- `TemplateErrorMode::Strict` must fail loudly
- `TemplateErrorMode::WarnFallback` may fall back to the Bonsai/Qwen prompt format and surface the warning in stats

## Invariants To Keep

- Rust inference must continue to work without external llama.cpp or MLX calls
- `vendor/candle` contains only `candle-core` and `candle-metal-kernels`; `candle-nn` and `candle-transformers` come from crates.io
- `[patch.crates-io]` must always redirect `candle-core` and `candle-metal-kernels` to vendor
- model logic (qwen3, kv_cache, generation) lives in `bonsai-candle`, not in candle-transformers
- `bonsai-candle` remains the only place that owns generation logic
- CLI should stay a thin wrapper around the library
- streaming output must flush during generation, not only at the end
- anti-repeat logic must include both rolling repeat penalty and no-repeat n-gram blocking
- message-based generation must support `system`, `user`, and `assistant`
- raw prompt mode only supports a single user message

## Common Commands

### Build

```bash
cargo check -p bonsai-cli
```

### Tests

```bash
cargo test -p bonsai-candle -p bonsai-cli
```

### Run on CPU

```bash
cargo run --release -p bonsai-cli -- \
  --device cpu \
  --model /path/to/Bonsai-1.7B.gguf \
  --prompt "What is the capital of France?"
```

### Run on Metal

```bash
cargo run --release -p bonsai-cli -- \
  --device metal \
  --model /path/to/Bonsai-1.7B.gguf \
  --prompt "What is the capital of France?"
```

### Run with an MLX/HF directory (CPU or Metal)

```bash
huggingface-cli download prism-ml/Ternary-Bonsai-8B-mlx-2bit \
  --local-dir /tmp/ternary-bonsai-8b
cargo run --release -p bonsai-cli -- \
  --device metal \
  --model /tmp/ternary-bonsai-8b \
  --prompt "What is the capital of France?"
```

### Run with chat history

```bash
cargo run --release -p bonsai-cli -- \
  --device cpu \
  --model /path/to/Bonsai-1.7B.gguf \
  --messages-file messages.json
```

Minimal `messages.json` shape:

```json
[
  { "role": "system", "content": "You are concise." },
  { "role": "user", "content": "Say hello." }
]
```

## When Editing This Project

- prefer changing `crates/bonsai-candle` first and only then updating the CLI
- if behavior changes, add or update tests in the touched crate
- if you touch quantized loading or matmul, verify both CPU and Metal code paths
- if you touch `src/qwen3.rs`, remember it is not the candle-transformers version — it is a local copy with YaRN
- if you touch `src/qwen3_5.rs`, cross-check the recurrence/conv/gate math against the `transformers` `models/qwen3_5` reference, and remember the MLX RMSNorm weights already include the `+1` (do not re-add it)
- if you touch `src/hadamard.rs` or the `prism_hadamard_qwen35` wiring in `src/qwen3_5.rs`, verify against `runtime/runtime.py`'s `fwht()` in the checkpoint's bundled runtime (forward = signs then rotate; inverse = rotate then signs), and confirm `in_proj_a`/`in_proj_b` still receive the un-rotated input
- if you touch prompt rendering, verify both:
  - `--template-mode strict`
  - `--template-mode warn-fallback`
- if you touch streaming, verify that text appears before process exit
- if you touch repeat control, verify long generations do not collapse into repeated paragraphs

## Things That Are Easy To Break

- GGUF dtype parsing for `Q1_0_g128`
- tensor/block layout assumptions in quantized matmul
- Metal kernel name mapping between Rust and `.metal` code
- YaRN rope scaling metadata parsing in `src/qwen3.rs`
- `qwen3_5` RMSNorm weight handling (the MLX `+1` fold), the `sigmoid` output gate, the DeltaNet query `1/sqrt(head_dim)` scale, and the conv1d tap order
- `prism_hadamard_qwen35` transform order: signs-then-Hadamard forward, Hadamard-then-signs inverse — swapping these silently corrupts output instead of erroring
- `prism_hadamard_qwen35` `in_proj_a`/`in_proj_b` must see the un-rotated layer input, not `ln1(x)`'s rotated form used by `in_proj_qkv`/`in_proj_z`
- prompt rendering when the GGUF chat template uses Python-like string methods
- token streaming if buffering changes suppress intermediate flushes
- stats if timing starts after the first generated token
- `[patch.crates-io]` removal — would silently switch to upstream candle-core without Q1_0_g128

## Recommended Review Checklist

Before closing a Rust change, check:

- `cargo fmt`
- `cargo test -p bonsai-candle -p bonsai-cli`
- `cargo run --release -p bonsai-cli -- --device cpu --model ... --prompt "..."`
- if on macOS, also run the same command with `--device metal`

If the change touches `vendor/candle`, verify that the local workspace still builds and that no dependency silently switched back to upstream Candle (check `[patch.crates-io]` is intact).

If the change touches `candle-nn` or `candle-transformers` versions, verify that the types used in `src/qwen3.rs` (`QMatMul`, `RmsNorm`, `repeat_kv`) are still available and compatible.
