mod generation;
mod kv_cache;
mod qwen3;
mod qwen3_5;

use crate::generation::{LogitsProcessor, Sampling};
use crate::qwen3::{ModelWeights, Qwen3Config, SafeTensorsSource};
use crate::qwen3_5::{Qwen35Config, Qwen35Weights};
use anyhow::{anyhow, bail, Context, Result};
use candle::quantized::gguf_file;
use candle::quantized::tokenizer::TokenizerFromGguf;
use candle::Tensor;
use memmap2::Mmap;
use minijinja::{context, Environment};
use minijinja_contrib::pycompat;
use safetensors::SafeTensors;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::time::{Duration, Instant};
use tokenizers::Tokenizer;

/// Minimal probe to read `model_type` from a HuggingFace `config.json` before
/// committing to a full architecture-specific config parse.
#[derive(Debug, Deserialize)]
struct ModelTypeProbe {
    #[serde(default)]
    model_type: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DevicePreference {
    Auto,
    Cpu,
    Metal,
}

#[derive(Debug, Clone, Copy)]
pub struct LoadOptions {
    pub device: DevicePreference,
}

impl Default for LoadOptions {
    fn default() -> Self {
        Self {
            device: DevicePreference::Auto,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TemplateErrorMode {
    Strict,
    WarnFallback,
}

#[derive(Debug, Clone)]
pub struct GenerateOptions {
    pub max_new_tokens: usize,
    pub temperature: f64,
    pub top_p: f64,
    pub top_k: Option<usize>,
    pub repeat_penalty: f32,
    pub repeat_last_n: usize,
    pub no_repeat_ngram_size: usize,
    pub seed: u64,
    pub use_chat_template: bool,
    pub template_error_mode: TemplateErrorMode,
}

impl Default for GenerateOptions {
    fn default() -> Self {
        Self {
            max_new_tokens: 256,
            temperature: 0.5,
            top_p: 0.85,
            top_k: Some(20),
            repeat_penalty: 1.1,
            repeat_last_n: 256,
            no_repeat_ngram_size: 6,
            seed: rand::random(),
            use_chat_template: true,
            template_error_mode: TemplateErrorMode::Strict,
        }
    }
}

#[derive(Debug, Clone)]
pub struct GenerationStats {
    pub prompt_tokens: usize,
    pub generated_tokens: usize,
    pub prompt_tps: f64,
    pub generation_tps: f64,
    pub peak_memory_bytes: Option<u64>,
    pub template_warning: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MessageRole {
    System,
    User,
    Assistant,
}

impl MessageRole {
    fn as_str(self) -> &'static str {
        match self {
            Self::System => "system",
            Self::User => "user",
            Self::Assistant => "assistant",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: MessageRole,
    pub content: String,
}

impl ChatMessage {
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: MessageRole::System,
            content: content.into(),
        }
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: MessageRole::User,
            content: content.into(),
        }
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: MessageRole::Assistant,
            content: content.into(),
        }
    }
}

/// Loaded model weights for one of the supported architectures.
enum Model {
    Qwen3(ModelWeights),
    Qwen35(Qwen35Weights),
}

impl Model {
    fn forward(&mut self, input: &Tensor, offset: usize) -> candle::Result<Tensor> {
        match self {
            Model::Qwen3(m) => m.forward(input, offset),
            Model::Qwen35(m) => m.forward(input, offset),
        }
    }

    fn clear_kv_cache(&mut self) {
        match self {
            Model::Qwen3(m) => m.clear_kv_cache(),
            Model::Qwen35(m) => m.clear_state(),
        }
    }
}

pub struct BonsaiModel {
    model: Model,
    tokenizer: Tokenizer,
    prompt_formatter: PromptFormatter,
    device: candle::Device,
    eos_token_id: Option<u32>,
    max_seq_len: usize,
}

impl BonsaiModel {
    /// Loads a Bonsai model from either a `.gguf` file or a HuggingFace/MLX
    /// directory (containing `config.json`, `model.safetensors`, `tokenizer.json`,
    /// and a `chat_template.jinja`).
    pub fn load(path: impl AsRef<Path>, opts: LoadOptions) -> Result<Self> {
        let path = path.as_ref();
        if path.is_dir() {
            Self::load_from_hf_dir(path, opts)
        } else {
            Self::load_from_gguf(path, opts)
        }
    }

    pub fn load_from_hf_dir(dir: &Path, opts: LoadOptions) -> Result<Self> {
        let device = resolve_device(opts.device)?;
        let cfg_path = dir.join("config.json");
        let cfg_str = std::fs::read_to_string(&cfg_path)
            .with_context(|| format!("failed to read {}", cfg_path.display()))?;
        let probe: ModelTypeProbe = serde_json::from_str(&cfg_str)
            .with_context(|| format!("failed to parse {}", cfg_path.display()))?;

        let tokenizer_path = dir.join("tokenizer.json");
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow!("failed to load {}: {e}", tokenizer_path.display()))?;

        let weights_path = dir.join("model.safetensors");
        let weights_file = std::fs::File::open(&weights_path)
            .with_context(|| format!("failed to open {}", weights_path.display()))?;
        let mmap = unsafe {
            Mmap::map(&weights_file)
                .with_context(|| format!("failed to mmap {}", weights_path.display()))?
        };
        let st = SafeTensors::deserialize(&mmap)
            .map_err(|e| anyhow!("failed to parse safetensors: {e}"))?;
        let mut src = SafeTensorsSource::new(st, device.clone());

        let (model, config_eos, max_seq_len) = if probe.model_type.as_deref() == Some("qwen3_5") {
            let cfg: Qwen35Config = serde_json::from_str(&cfg_str)
                .with_context(|| format!("failed to parse {}", cfg_path.display()))?;
            let model = Qwen35Weights::from_safetensors(&mut src, &cfg, &device)
                .context("failed to load Bonsai qwen3_5 weights from MLX safetensors")?;
            let max_seq_len = cfg.max_position_embeddings();
            (Model::Qwen35(model), cfg.eos_token_id(), max_seq_len)
        } else {
            let cfg: Qwen3Config = serde_json::from_str(&cfg_str)
                .with_context(|| format!("failed to parse {}", cfg_path.display()))?;
            let model = ModelWeights::from_safetensors(&mut src, &cfg, &device)
                .context("failed to load Bonsai weights from MLX safetensors")?;
            (
                Model::Qwen3(model),
                cfg.eos_token_id,
                cfg.max_position_embeddings,
            )
        };

        let eos_token_id =
            config_eos.or_else(|| tokenizer.get_vocab(true).get("<|im_end|>").copied());
        let eos_token_str = eos_token_id
            .and_then(|id| tokenizer.id_to_token(id))
            .unwrap_or_default();

        let template_path = dir.join("chat_template.jinja");
        let template_str = std::fs::read_to_string(&template_path).ok();
        let prompt_formatter =
            PromptFormatter::from_template(template_str, String::new(), eos_token_str);

        Ok(Self {
            model,
            tokenizer,
            prompt_formatter,
            device,
            eos_token_id,
            max_seq_len,
        })
    }

    pub fn load_from_gguf(path: impl AsRef<Path>, opts: LoadOptions) -> Result<Self> {
        let path = path.as_ref();
        let device = resolve_device(opts.device)?;
        let mut file = std::fs::File::open(path)
            .with_context(|| format!("failed to open model file {}", path.display()))?;
        let content = gguf_file::Content::read(&mut file)
            .with_context(|| format!("failed to read {}", path.display()))?;
        let tokenizer =
            Tokenizer::from_gguf(&content).context("failed to build tokenizer from GGUF")?;
        let prompt_formatter = PromptFormatter::from_gguf(&content);
        let eos_token_id = metadata_u32(&content, "tokenizer.ggml.eos_token_id")
            .or_else(|| tokenizer.get_vocab(true).get("<|im_end|>").copied());
        let max_seq_len = metadata_u32(&content, "qwen3.context_length")
            .map(|v| v as usize)
            .unwrap_or(usize::MAX);
        let model = ModelWeights::from_gguf(content, &mut file, &device)
            .context("failed to load Bonsai weights")?;
        Ok(Self {
            model: Model::Qwen3(model),
            tokenizer,
            prompt_formatter,
            device,
            eos_token_id,
            max_seq_len,
        })
    }

    pub fn generate_stream<W: Write>(
        &mut self,
        prompt: &str,
        opts: &GenerateOptions,
        sink: &mut W,
    ) -> Result<GenerationStats> {
        if opts.use_chat_template {
            let messages = [ChatMessage::user(prompt)];
            self.generate_messages_stream(&messages, opts, sink)
        } else {
            self.generate_prompt_stream(prompt.to_owned(), opts, sink, None)
        }
    }

    pub fn generate_messages_stream<W: Write>(
        &mut self,
        messages: &[ChatMessage],
        opts: &GenerateOptions,
        sink: &mut W,
    ) -> Result<GenerationStats> {
        if !opts.use_chat_template {
            if let [message] = messages {
                if message.role == MessageRole::User {
                    return self.generate_prompt_stream(message.content.clone(), opts, sink, None);
                }
            }
            bail!("raw prompt mode only supports a single user message")
        }

        let rendered = self
            .prompt_formatter
            .apply_messages(messages, opts.template_error_mode)?;
        self.generate_prompt_stream(rendered.prompt_text, opts, sink, rendered.warning)
    }

    fn generate_prompt_stream<W: Write>(
        &mut self,
        prompt_text: String,
        opts: &GenerateOptions,
        sink: &mut W,
        template_warning: Option<String>,
    ) -> Result<GenerationStats> {
        self.model.clear_kv_cache();

        let encoding = self
            .tokenizer
            .encode(prompt_text, true)
            .map_err(|err| anyhow!("failed to tokenize prompt: {err}"))?;
        let prompt_tokens = encoding.get_ids().to_vec();
        if prompt_tokens.is_empty() {
            bail!("prompt encoded to zero tokens")
        }
        if prompt_tokens.len() >= self.max_seq_len {
            bail!(
                "prompt is {} tokens, exceeds model context length of {}",
                prompt_tokens.len(),
                self.max_seq_len
            );
        }
        let max_new_tokens = opts
            .max_new_tokens
            .min(self.max_seq_len - prompt_tokens.len());

        let mut token_stream = TokenOutputStream::new(self.tokenizer.clone());
        let sampling = sampling_from_options(opts);
        let mut logits_processor = LogitsProcessor::from_sampling(opts.seed, sampling);
        let mut repeat_context = RollingRepeatContext::new(&prompt_tokens, opts.repeat_last_n);
        let mut generated_sequence = Vec::with_capacity(max_new_tokens);
        let mut writer = StreamWriter::new(sink);

        let prompt_start = Instant::now();
        let input = Tensor::new(prompt_tokens.as_slice(), &self.device)?.unsqueeze(0)?;
        let logits = self.model.forward(&input, 0)?.squeeze(0)?;
        let prompt_dt = prompt_start.elapsed();

        if max_new_tokens == 0 {
            writer.finish()?;
            return Ok(GenerationStats {
                prompt_tokens: prompt_tokens.len(),
                generated_tokens: 0,
                prompt_tps: tokens_per_second(prompt_tokens.len(), prompt_dt),
                generation_tps: 0.0,
                peak_memory_bytes: peak_memory_bytes(),
                template_warning,
            });
        }

        let generation_start = Instant::now();
        let logits = apply_generation_constraints(
            logits,
            opts,
            repeat_context.tokens(),
            &generated_sequence,
        )?;
        let mut next_token = logits_processor.sample(&logits)?;
        let mut generated_tokens = 1;
        generated_sequence.push(next_token);
        repeat_context.push(next_token);

        if let Some(text) = token_stream.next_token(next_token)? {
            writer.push(&text)?;
        }

        for step in 1..max_new_tokens {
            if Some(next_token) == self.eos_token_id {
                break;
            }

            let input = Tensor::new(&[next_token], &self.device)?.unsqueeze(0)?;
            let logits = self
                .model
                .forward(&input, prompt_tokens.len() + step - 1)?
                .squeeze(0)?;
            let logits = apply_generation_constraints(
                logits,
                opts,
                repeat_context.tokens(),
                &generated_sequence,
            )?;
            next_token = logits_processor.sample(&logits)?;
            generated_tokens += 1;
            generated_sequence.push(next_token);
            repeat_context.push(next_token);

            if let Some(text) = token_stream.next_token(next_token)? {
                writer.push(&text)?;
            }
        }

        if let Some(rest) = token_stream.decode_rest()? {
            writer.push(&rest)?;
        }
        writer.finish()?;

        let generation_dt = generation_start.elapsed();
        Ok(GenerationStats {
            prompt_tokens: prompt_tokens.len(),
            generated_tokens,
            prompt_tps: tokens_per_second(prompt_tokens.len(), prompt_dt),
            generation_tps: tokens_per_second(generated_tokens, generation_dt),
            peak_memory_bytes: peak_memory_bytes(),
            template_warning,
        })
    }
}

fn resolve_device(preference: DevicePreference) -> Result<candle::Device> {
    match preference {
        DevicePreference::Cpu => Ok(candle::Device::Cpu),
        DevicePreference::Auto => {
            if candle::utils::metal_is_available() {
                return candle::Device::new_metal(0)
                    .context("Metal is available but device initialization failed");
            }
            Ok(candle::Device::Cpu)
        }
        DevicePreference::Metal => {
            if !candle::utils::metal_is_available() {
                bail!("Metal backend is not available in this build or on this machine")
            }
            candle::Device::new_metal(0).context("failed to initialize Metal device")
        }
    }
}

fn sampling_from_options(opts: &GenerateOptions) -> Sampling {
    if opts.temperature <= 0.0 {
        return Sampling::ArgMax;
    }
    match (opts.top_k, opts.top_p) {
        (Some(k), p) if p > 0.0 && p < 1.0 => Sampling::TopKThenTopP {
            k,
            p,
            temperature: opts.temperature,
        },
        (Some(k), _) => Sampling::TopK {
            k,
            temperature: opts.temperature,
        },
        (None, p) if p > 0.0 && p < 1.0 => Sampling::TopP {
            p,
            temperature: opts.temperature,
        },
        (None, _) => Sampling::All {
            temperature: opts.temperature,
        },
    }
}

fn apply_generation_constraints(
    logits: Tensor,
    opts: &GenerateOptions,
    repeat_context_tokens: &[u32],
    generated_tokens: &[u32],
) -> Result<Tensor> {
    let banned_tokens =
        banned_tokens_for_no_repeat_ngram(generated_tokens, opts.no_repeat_ngram_size);
    if (opts.repeat_penalty == 1.0 || repeat_context_tokens.is_empty()) && banned_tokens.is_empty()
    {
        return Ok(logits);
    }

    let device = logits.device().clone();
    let mut adjusted_logits = logits.to_dtype(candle::DType::F32)?.to_vec1::<f32>()?;

    if opts.repeat_penalty != 1.0 && !repeat_context_tokens.is_empty() {
        let mut already_seen = std::collections::HashSet::new();
        for token_id in repeat_context_tokens {
            if !already_seen.insert(*token_id) {
                continue;
            }
            if let Some(logit) = adjusted_logits.get_mut(*token_id as usize) {
                if *logit >= 0.0 {
                    *logit /= opts.repeat_penalty;
                } else {
                    *logit *= opts.repeat_penalty;
                }
            }
        }
    }

    for token_id in banned_tokens {
        if let Some(logit) = adjusted_logits.get_mut(token_id as usize) {
            *logit = f32::NEG_INFINITY;
        }
    }

    let logits_len = adjusted_logits.len();
    Ok(Tensor::from_vec(adjusted_logits, logits_len, &device)?)
}

fn banned_tokens_for_no_repeat_ngram(tokens: &[u32], ngram_size: usize) -> Vec<u32> {
    if ngram_size < 2 || tokens.len() + 1 < ngram_size {
        return Vec::new();
    }

    let prefix_len = ngram_size - 1;
    let prefix = &tokens[tokens.len() - prefix_len..];
    let mut banned = std::collections::BTreeSet::new();
    for window in tokens.windows(ngram_size) {
        if window[..prefix_len] == *prefix {
            banned.insert(window[prefix_len]);
        }
    }
    banned.into_iter().collect()
}

fn tokens_per_second(tokens: usize, dt: std::time::Duration) -> f64 {
    if tokens == 0 {
        0.0
    } else {
        tokens as f64 / dt.as_secs_f64().max(1e-9)
    }
}

fn peak_memory_bytes() -> Option<u64> {
    #[cfg(unix)]
    {
        let mut usage = std::mem::MaybeUninit::<libc::rusage>::uninit();
        let status = unsafe { libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) };
        if status != 0 {
            return None;
        }
        let usage = unsafe { usage.assume_init() };
        #[cfg(target_os = "macos")]
        {
            u64::try_from(usage.ru_maxrss).ok()
        }
        #[cfg(not(target_os = "macos"))]
        {
            u64::try_from(usage.ru_maxrss).ok()?.checked_mul(1024)
        }
    }
    #[cfg(not(unix))]
    {
        None
    }
}

fn metadata_u32(ct: &gguf_file::Content, key: &str) -> Option<u32> {
    ct.metadata.get(key).and_then(|v| v.to_u32().ok())
}

fn metadata_string(ct: &gguf_file::Content, key: &str) -> Option<String> {
    ct.metadata
        .get(key)
        .and_then(|v| v.to_string().ok())
        .cloned()
}

fn token_by_id(ct: &gguf_file::Content, key: &str) -> Option<String> {
    let token_id = metadata_u32(ct, key)? as usize;
    let tokens = ct.metadata.get("tokenizer.ggml.tokens")?.to_vec().ok()?;
    tokens.get(token_id)?.to_string().ok().cloned()
}

struct PromptFormatter {
    template_state: TemplateState,
    bos_token: String,
    eos_token: String,
}

impl PromptFormatter {
    fn from_gguf(ct: &gguf_file::Content) -> Self {
        let bos_token = token_by_id(ct, "tokenizer.ggml.bos_token_id").unwrap_or_default();
        let eos_token = token_by_id(ct, "tokenizer.ggml.eos_token_id").unwrap_or_default();
        let template_state = match metadata_string(ct, "tokenizer.chat_template") {
            Some(template) => match ChatTemplate::new(template) {
                Ok(template) => TemplateState::Ready(template),
                Err(err) => TemplateState::Broken(err.to_string()),
            },
            None => TemplateState::Missing,
        };
        Self {
            template_state,
            bos_token,
            eos_token,
        }
    }

    fn from_template(template: Option<String>, bos_token: String, eos_token: String) -> Self {
        let template_state = match template {
            Some(template) => match ChatTemplate::new(template) {
                Ok(template) => TemplateState::Ready(template),
                Err(err) => TemplateState::Broken(err.to_string()),
            },
            None => TemplateState::Missing,
        };
        Self {
            template_state,
            bos_token,
            eos_token,
        }
    }

    fn apply_messages(
        &self,
        messages: &[ChatMessage],
        mode: TemplateErrorMode,
    ) -> Result<RenderedPrompt> {
        let template_messages = messages
            .iter()
            .map(TemplateMessage::from_chat_message)
            .collect::<Vec<_>>();

        match &self.template_state {
            TemplateState::Ready(template) => match template.render_generation(
                &template_messages,
                &self.bos_token,
                &self.eos_token,
            ) {
                Ok(prompt_text) => Ok(RenderedPrompt {
                    prompt_text,
                    warning: None,
                }),
                Err(err) => self.handle_fallback(messages, mode, format!("{err:#}")),
            },
            TemplateState::Missing => self.handle_fallback(
                messages,
                mode,
                "GGUF tokenizer.chat_template metadata is missing".to_owned(),
            ),
            TemplateState::Broken(reason) => self.handle_fallback(messages, mode, reason.clone()),
        }
    }

    fn handle_fallback(
        &self,
        messages: &[ChatMessage],
        mode: TemplateErrorMode,
        reason: String,
    ) -> Result<RenderedPrompt> {
        match mode {
            TemplateErrorMode::Strict => Err(anyhow!(reason)),
            TemplateErrorMode::WarnFallback => Ok(RenderedPrompt {
                prompt_text: fallback_prompt(messages),
                warning: Some(format!("{reason}; using Bonsai/Qwen fallback chat prompt")),
            }),
        }
    }
}

enum TemplateState {
    Missing,
    Broken(String),
    Ready(ChatTemplate),
}

#[derive(Debug)]
struct RenderedPrompt {
    prompt_text: String,
    warning: Option<String>,
}

fn fallback_prompt(messages: &[ChatMessage]) -> String {
    let mut prompt = String::new();
    for message in messages {
        prompt.push_str("<|im_start|>");
        prompt.push_str(message.role.as_str());
        prompt.push('\n');
        prompt.push_str(&message.content);
        prompt.push_str("<|im_end|>\n");
    }
    prompt.push_str("<|im_start|>assistant\n<think>\n\n</think>\n\n");
    prompt
}

struct ChatTemplate {
    env: Environment<'static>,
}

impl ChatTemplate {
    fn new(template: String) -> Result<Self> {
        let mut env = Environment::new();
        env.add_function(
            "raise_exception",
            |msg: String| -> Result<String, minijinja::Error> {
                Err(minijinja::Error::new(
                    minijinja::ErrorKind::InvalidOperation,
                    msg,
                ))
            },
        );
        env.set_unknown_method_callback(pycompat::unknown_method_callback);
        env.add_template_owned("chat".to_owned(), template)
            .map_err(|err| anyhow!("failed to parse chat template: {err}"))?;
        Ok(Self { env })
    }

    fn render_generation(
        &self,
        messages: &[TemplateMessage<'_>],
        bos_token: &str,
        eos_token: &str,
    ) -> Result<String> {
        let template = self
            .env
            .get_template("chat")
            .map_err(|err| anyhow!("failed to load chat template: {err}"))?;
        template
            .render(context! {
                messages => messages,
                add_generation_prompt => true,
                continue_final_message => false,
                enable_thinking => false,
                bos_token => bos_token,
                eos_token => eos_token,
                tools => Vec::<serde_json::Value>::new(),
            })
            .map_err(|err| anyhow!("failed to render chat template: {err}"))
    }
}

#[derive(Debug, Clone, Serialize)]
struct TemplateMessage<'a> {
    role: &'a str,
    content: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_content: Option<&'a str>,
}

impl<'a> TemplateMessage<'a> {
    fn from_chat_message(message: &'a ChatMessage) -> Self {
        Self {
            role: message.role.as_str(),
            content: &message.content,
            reasoning_content: None,
        }
    }
}

struct RollingRepeatContext {
    tokens: VecDeque<u32>,
    limit: usize,
}

impl RollingRepeatContext {
    fn new(prompt_tokens: &[u32], limit: usize) -> Self {
        let tokens = if limit == 0 {
            VecDeque::new()
        } else {
            let start = prompt_tokens.len().saturating_sub(limit);
            VecDeque::from(prompt_tokens[start..].to_vec())
        };
        Self { tokens, limit }
    }

    fn tokens(&mut self) -> &[u32] {
        self.tokens.make_contiguous()
    }

    fn push(&mut self, token: u32) {
        if self.limit == 0 {
            return;
        }
        if self.tokens.len() == self.limit {
            self.tokens.pop_front();
        }
        self.tokens.push_back(token);
    }
}

struct StreamWriter<W: Write> {
    sink: BufWriter<W>,
    pending: String,
    last_flush: Instant,
    emitted_anything: bool,
}

impl<W: Write> StreamWriter<W> {
    fn new(sink: W) -> Self {
        Self {
            sink: BufWriter::new(sink),
            pending: String::new(),
            last_flush: Instant::now(),
            emitted_anything: false,
        }
    }

    fn push(&mut self, text: &str) -> Result<()> {
        if text.is_empty() {
            return Ok(());
        }
        self.pending.push_str(text);
        if !self.emitted_anything
            || self.pending.len() >= 64
            || text.contains('\n')
            || self.last_flush.elapsed() >= Duration::from_millis(50)
        {
            self.flush_pending()?;
        }
        Ok(())
    }

    fn finish(&mut self) -> Result<()> {
        self.flush_pending()?;
        self.sink.flush()?;
        Ok(())
    }

    fn flush_pending(&mut self) -> Result<()> {
        if self.pending.is_empty() {
            return Ok(());
        }
        self.sink.write_all(self.pending.as_bytes())?;
        self.pending.clear();
        self.sink.flush()?;
        self.last_flush = Instant::now();
        self.emitted_anything = true;
        Ok(())
    }
}

struct TokenOutputStream {
    tokenizer: Tokenizer,
    tokens: Vec<u32>,
    emitted_tokens: usize,
}

impl TokenOutputStream {
    fn new(tokenizer: Tokenizer) -> Self {
        Self {
            tokenizer,
            tokens: Vec::new(),
            emitted_tokens: 0,
        }
    }

    fn decode(&self, tokens: &[u32]) -> Result<String> {
        self.tokenizer
            .decode(tokens, true)
            .map_err(|err| anyhow!("failed to decode tokens: {err}"))
    }

    fn next_token(&mut self, token: u32) -> Result<Option<String>> {
        self.tokens.push(token);
        let text = self.decode(&self.tokens[self.emitted_tokens..])?;
        if text.is_empty() {
            return Ok(None);
        }
        if text.chars().last().is_some_and(char::is_alphanumeric) {
            self.emitted_tokens = self.tokens.len();
            Ok(Some(text))
        } else {
            Ok(None)
        }
    }

    fn decode_rest(&self) -> Result<Option<String>> {
        if self.emitted_tokens >= self.tokens.len() {
            return Ok(None);
        }
        let text = self.decode(&self.tokens[self.emitted_tokens..])?;
        if text.is_empty() {
            Ok(None)
        } else {
            Ok(Some(text))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;
    use std::rc::Rc;

    #[test]
    fn repeat_context_includes_prompt_tail_and_generated_tokens() {
        let mut ctx = RollingRepeatContext::new(&[1, 2, 3, 4], 3);
        assert_eq!(ctx.tokens(), &[2, 3, 4]);
        ctx.push(5);
        assert_eq!(ctx.tokens(), &[3, 4, 5]);
        ctx.push(6);
        assert_eq!(ctx.tokens(), &[4, 5, 6]);
    }

    #[test]
    fn fallback_prompt_formats_multi_turn_messages() {
        let prompt = fallback_prompt(&[
            ChatMessage::system("sys"),
            ChatMessage::user("hello"),
            ChatMessage::assistant("hi"),
        ]);
        assert_eq!(
            prompt,
            "<|im_start|>system\nsys<|im_end|>\n<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\nhi<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        );
    }

    #[test]
    fn strict_template_mode_returns_error() {
        let formatter = PromptFormatter {
            template_state: TemplateState::Broken("render failed".to_owned()),
            bos_token: String::new(),
            eos_token: String::new(),
        };
        let err = formatter
            .apply_messages(&[ChatMessage::user("hello")], TemplateErrorMode::Strict)
            .unwrap_err();
        assert!(err.to_string().contains("render failed"));
    }

    #[test]
    fn warn_template_mode_returns_warning_and_fallback() {
        let formatter = PromptFormatter {
            template_state: TemplateState::Broken("render failed".to_owned()),
            bos_token: String::new(),
            eos_token: String::new(),
        };
        let rendered = formatter
            .apply_messages(
                &[ChatMessage::user("hello")],
                TemplateErrorMode::WarnFallback,
            )
            .unwrap();
        assert_eq!(
            rendered.prompt_text,
            "<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        );
        assert_eq!(
            rendered.warning.as_deref(),
            Some("render failed; using Bonsai/Qwen fallback chat prompt")
        );
    }

    #[test]
    fn chat_template_supports_python_style_string_methods() {
        let template = ChatTemplate::new(
            "{{ value.startswith('foo') }}|{{ value.endswith('bar') }}|{{ value.split('|')[-1].lstrip('\\n').rstrip('\\n').strip('\\n') }}".to_owned(),
        )
        .unwrap();
        let template = template.env.get_template("chat").unwrap();
        let rendered = template
            .render(context! {
                value => "foo|\nbar",
            })
            .unwrap();
        assert_eq!(rendered, "true|true|bar");
    }

    #[test]
    fn no_repeat_ngram_blocks_next_token_for_seen_ngram() {
        let banned = banned_tokens_for_no_repeat_ngram(&[10, 11, 12, 20, 11, 12], 3);
        assert_eq!(banned, vec![20]);
    }

    #[test]
    fn no_repeat_ngram_ignores_short_history() {
        let banned = banned_tokens_for_no_repeat_ngram(&[10, 11], 4);
        assert!(banned.is_empty());
    }

    #[derive(Clone, Default)]
    struct RecordingSink {
        state: Rc<RefCell<RecordingSinkState>>,
    }

    #[derive(Default)]
    struct RecordingSinkState {
        bytes: Vec<u8>,
        flushes: usize,
    }

    impl RecordingSink {
        fn rendered(&self) -> String {
            String::from_utf8(self.state.borrow().bytes.clone()).unwrap()
        }

        fn flushes(&self) -> usize {
            self.state.borrow().flushes
        }
    }

    impl Write for RecordingSink {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            self.state.borrow_mut().bytes.extend_from_slice(buf);
            Ok(buf.len())
        }

        fn flush(&mut self) -> std::io::Result<()> {
            self.state.borrow_mut().flushes += 1;
            Ok(())
        }
    }

    #[test]
    fn stream_writer_flushes_first_chunk_immediately() {
        let sink = RecordingSink::default();
        let inspect = sink.clone();
        let mut writer = StreamWriter::new(sink);
        writer.push("hello").unwrap();
        assert_eq!(inspect.rendered(), "hello");
        assert!(inspect.flushes() >= 1);
    }

    #[test]
    fn stream_writer_flushes_pending_text_on_finish() {
        let sink = RecordingSink::default();
        let inspect = sink.clone();
        let mut writer = StreamWriter::new(sink);
        writer.push("hello").unwrap();
        writer.push(" world").unwrap();
        writer.finish().unwrap();
        assert_eq!(inspect.rendered(), "hello world");
    }
}
