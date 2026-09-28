# bonsai-1bit-rs

Native Rust inference for [Bonsai](https://prismml.com) models — ultra-low-bit LLMs from Prism ML built on the Qwen3 and Qwen3.5 architectures, running on CPU and Apple Silicon Metal.

## Models

| Model | Format | Size | Notes |
|---|---|---|---|
| Bonsai-1.7B | GGUF `Q1_0_g128` | ~240 MB | 1-bit binary {−d, +d}, block 128 |
| Bonsai-8B | GGUF `Q1_0_g128` | ~1 GB | 1-bit binary {−d, +d}, block 128 |
| Ternary-Bonsai-8B | MLX safetensors `Q2MLX` | 2.15 GiB | Ternary 1.58-bit, 75.5 avg benchmark |
| Ternary-Bonsai-27B | MLX safetensors `Q2MLX` | 8.49 GiB | Qwen3.5 hybrid (Gated DeltaNet + gated full attention), text-only |
| Ternary-Bonsai-2-27B | MLX safetensors `Q2MLX` + Hadamard | 8.6 GiB | `prism_hadamard_qwen35`: same Qwen3.5 hybrid topology, but Q2MLX weights are quantized in a Hadamard-rotated basis, text-only |

Both quantization formats are custom extensions — `Q1_0_g128` and `Q2MLX` are not part of upstream Candle and are applied via vendored patches to `candle-core` and `candle-metal-kernels`.

## Requirements

- Rust 2021 edition
- macOS with Apple Silicon for Metal acceleration (CPU works on any platform)
- For GGUF models: `Bonsai-1.7B.gguf` or `Bonsai-8B.gguf`
- For MLX models: a directory with `config.json`, `model.safetensors`, `tokenizer.json`, `chat_template.jinja`
  - `prism_hadamard_qwen35` checkpoints (Ternary-Bonsai-2-27B) additionally need `hadamard.json`
    (named by `config.json`'s `hadamard_config`); `generation_config.json` is read if present
    and takes priority for `eos_token_id`

## Build

```sh
cargo build --release -p bonsai-cli
```

Metal support is compiled in automatically on macOS (Apple Silicon); select it at
runtime with `--device metal`. On other platforms the build is CPU-only.

## Usage

```sh
# Single prompt (GGUF, auto device)
bonsai-cli --model Bonsai-8B.gguf --prompt "Explain the Rust borrow checker."

# Single prompt (MLX directory, Metal)
bonsai-cli --model Ternary-Bonsai-8B-mlx-2bit --device metal --prompt "Hello!"

# Qwen3.5 hybrid 27B (MLX directory, Metal, text-only)
bonsai-cli --model Ternary-Bonsai-27B-mlx-2bit --device metal --prompt "Hello!"

# Qwen3.5 hybrid 27B v2, Hadamard-rotated Q2MLX (MLX directory, Metal, text-only)
bonsai-cli --model Ternary-Bonsai-2-27B-mlx-2bit --device metal --prompt "Hello!"

# Multi-turn conversation from a JSON file
bonsai-cli --model Bonsai-8B.gguf --messages-file messages.json

# Raw prompt (no chat template applied)
bonsai-cli --model Bonsai-8B.gguf --raw-prompt --prompt "<|im_start|>user\nHi<|im_end|>"
```

### messages.json format

```json
[
  { "role": "system", "content": "You are a helpful assistant." },
  { "role": "user",   "content": "What is 2 + 2?" }
]
```

### All flags

| Flag | Default | Description |
|---|---|---|
| `--model <path>` | required | `.gguf` file or MLX/HF weight directory |
| `--prompt <text>` | — | Single prompt (mutually exclusive with `--messages-file`) |
| `--messages-file <path>` | — | JSON file of `[{role, content}]` messages |
| `--device auto\|cpu\|metal` | `auto` | Inference device |
| `--max-new-tokens <n>` | `2048` | Maximum tokens to generate |
| `--temperature <f>` | `0.5` | Sampling temperature |
| `--top-p <f>` | `0.85` | Top-p nucleus sampling |
| `--top-k <n>` | `20` | Top-k sampling |
| `--repeat-penalty <f>` | `1.1` | Repetition penalty |
| `--repeat-last-n <n>` | `256` | Window for repetition penalty |
| `--no-repeat-ngram-size <n>` | `6` | Block repeated n-grams |
| `--seed <u64>` | random | RNG seed for reproducibility |
| `--raw-prompt` | off | Skip chat template, pass prompt verbatim |
| `--template-mode strict\|warn-fallback` | `warn-fallback` | Chat template error handling |

Generated tokens are streamed to stdout; stats are written to stderr on completion:

```
prompt_tokens: 42 (1234.56 tok/s)
generated_tokens: 128 (87.43 tok/s)
peak_memory: 1.03 GiB
```

## Architecture

```
bonsai-1bit-rs/
├── crates/
│   ├── bonsai-candle/   # model library (loading, generation, sampling)
│   │   └── src/
│   │       ├── lib.rs         # public API: BonsaiModel, GenerateOptions, model dispatch
│   │       ├── qwen3.rs       # quantized Qwen3 with YaRN rope scaling
│   │       ├── qwen3_5.rs     # Qwen3.5 hybrid (Gated DeltaNet + gated attention), text-only
│   │       ├── hadamard.rs    # blockwise Hadamard rotation (prism_hadamard_qwen35)
│   │       ├── kv_cache.rs    # ConcatKvCache (Tensor::cat based)
│   │       └── generation.rs  # LogitsProcessor, Sampling enum
│   └── bonsai-cli/      # CLI wrapper
└── vendor/candle/       # patched candle-core + candle-metal-kernels
```

### Key types (`bonsai-candle`)

- `BonsaiModel` — loads a model and drives generation
- `LoadOptions` — `device: DevicePreference`
- `GenerateOptions` — all sampling and generation parameters
- `GenerationStats` — `prompt_tokens`, `generated_tokens`, `prompt_tps`, `generation_tps`, `peak_memory_bytes`
- `ChatMessage` / `MessageRole` — structured chat input

### Vendored patches

`candle-core` and `candle-metal-kernels` are vendored under `vendor/candle/` with two custom quantization formats:

- **`Q1_0_g128`** — binary weights {−d, +d}, 1 bit/weight, block size 128. GGUF dtype id `41`. CPU matmul + Metal kernels.
- **`Q2MLX`** — MLX 2-bit affine `w = scale·q + bias` (q ∈ {0,1,2,3}), group 128. Loaded from MLX safetensors. CPU + Metal kernels. No GGUF id (runtime-only).

Do not replace the vendored crates with upstream Candle without verifying full support for these formats.

### Qwen3.5 hybrid (`Ternary-Bonsai-27B`)

`Ternary-Bonsai-27B` uses the `qwen3_5` architecture (`crates/bonsai-candle/src/qwen3_5.rs`):
64 decoder layers interleaving 48 **Gated DeltaNet** linear-attention layers with
16 **gated full-attention** layers (one full layer every `full_attention_interval = 4`).
Only the language model is loaded — the vision tower, MTP head and DSpark drafter are
ignored, so inference is text-only. The DeltaNet recurrence and its depthwise causal
convolution run as an explicit per-token scan (shared by prefill and decode), which is
correct but sequential, so throughput is lower than the plain-attention Qwen3 models;
chunked prefill is a possible future optimization.

### Hadamard-rotated Q2MLX (`Ternary-Bonsai-2-27B`)

`Ternary-Bonsai-2-27B` (`model_type = prism_hadamard_qwen35`) is the same Qwen3.5 hybrid
topology as `Ternary-Bonsai-27B`, but every packed linear/embedding weight is quantized
in a *rotated* basis: activations are passed through a blockwise (block 1024) normalized
Sylvester Walsh–Hadamard transform — sign flip then Hadamard rotation for linear inputs,
Hadamard rotation then sign flip for the embedding output — before/after they meet the
Q2MLX matmul (`crates/bonsai-candle/src/hadamard.rs`). The rotation is a fused sign flip +
O(n log n) butterfly FWHT (a Candle custom op: rayon-parallel CPU path, and a dedicated
`hadamard_block_fwht_f32` Metal kernel in the vendored `candle-metal-kernels`), applied once per
shared consumer group per layer (hidden state, mixer output, FFN-down input) rather than
once per projection. The dense `in_proj_a`/`in_proj_b` DeltaNet gates are not rotated and
are loaded as plain F32 weights instead of Q2MLX. `hadamard.json` is validated at load
time and every manifest-listed weight name must be claimed by a loaded tensor, so a
checkpoint variant that silently changes which projections are rotated fails to load
instead of producing wrong output.

## License

`candle-core` and `candle-metal-kernels` are MIT OR Apache-2.0 (HuggingFace).
Bonsai model weights are Apache-2.0 (Prism ML).
Code in `crates/` is MIT OR Apache-2.0.
