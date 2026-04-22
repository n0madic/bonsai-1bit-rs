use anyhow::{bail, Context, Result};
use bonsai_candle::{
    BonsaiModel, ChatMessage, DevicePreference, GenerateOptions, LoadOptions, TemplateErrorMode,
};
use clap::{Parser, ValueEnum};
use std::fs;
use std::io::{stderr, stdout, Write};
use std::path::PathBuf;

#[derive(Debug, Clone, Copy, ValueEnum)]
enum DeviceArg {
    Auto,
    Cpu,
    Metal,
}

impl From<DeviceArg> for DevicePreference {
    fn from(value: DeviceArg) -> Self {
        match value {
            DeviceArg::Auto => DevicePreference::Auto,
            DeviceArg::Cpu => DevicePreference::Cpu,
            DeviceArg::Metal => DevicePreference::Metal,
        }
    }
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum TemplateModeArg {
    Strict,
    WarnFallback,
}

impl From<TemplateModeArg> for TemplateErrorMode {
    fn from(value: TemplateModeArg) -> Self {
        match value {
            TemplateModeArg::Strict => TemplateErrorMode::Strict,
            TemplateModeArg::WarnFallback => TemplateErrorMode::WarnFallback,
        }
    }
}

#[derive(Debug, Parser)]
#[command(author, version, about)]
struct Args {
    /// Path to a `.gguf` file (Bonsai GGUF) or a directory containing
    /// HuggingFace/MLX weights (`config.json`, `model.safetensors`,
    /// `tokenizer.json`, `chat_template.jinja`).
    #[arg(long)]
    model: PathBuf,

    #[arg(long)]
    prompt: Option<String>,

    #[arg(long)]
    messages_file: Option<PathBuf>,

    #[arg(long, value_enum, default_value_t = DeviceArg::Auto)]
    device: DeviceArg,

    #[arg(long, default_value_t = 2048)]
    max_new_tokens: usize,

    #[arg(long, default_value_t = 0.5)]
    temperature: f64,

    #[arg(long, default_value_t = 0.85)]
    top_p: f64,

    #[arg(long, default_value_t = 20)]
    top_k: usize,

    #[arg(long, default_value_t = 1.1)]
    repeat_penalty: f32,

    #[arg(long, default_value_t = 256)]
    repeat_last_n: usize,

    #[arg(long, default_value_t = 6)]
    no_repeat_ngram_size: usize,

    #[arg(long)]
    seed: Option<u64>,

    #[arg(long)]
    raw_prompt: bool,

    #[arg(long, value_enum, default_value_t = TemplateModeArg::WarnFallback)]
    template_mode: TemplateModeArg,
}

fn main() -> Result<()> {
    let args = Args::parse();
    validate_args(&args)?;
    let mut model = BonsaiModel::load(
        &args.model,
        LoadOptions {
            device: args.device.into(),
        },
    )?;

    let mut stdout = stdout();
    let base_opts = GenerateOptions {
        max_new_tokens: args.max_new_tokens,
        temperature: args.temperature,
        top_p: args.top_p,
        top_k: Some(args.top_k),
        repeat_penalty: args.repeat_penalty,
        repeat_last_n: args.repeat_last_n,
        no_repeat_ngram_size: args.no_repeat_ngram_size,
        seed: args.seed.unwrap_or_else(rand::random),
        use_chat_template: true,
        template_error_mode: args.template_mode.into(),
    };

    let stats = if let Some(prompt) = &args.prompt {
        let opts = GenerateOptions {
            use_chat_template: !args.raw_prompt,
            ..base_opts.clone()
        };
        model.generate_stream(prompt, &opts, &mut stdout)?
    } else {
        let messages = load_messages(
            args.messages_file
                .as_ref()
                .expect("validated messages_file presence"),
        )?;
        model.generate_messages_stream(&messages, &base_opts, &mut stdout)?
    };

    writeln!(stdout)?;
    stdout.flush()?;

    let mut stderr = stderr();
    writeln!(stderr)?;
    writeln!(
        stderr,
        "prompt_tokens: {} ({:.2} tok/s)",
        stats.prompt_tokens, stats.prompt_tps
    )?;
    writeln!(
        stderr,
        "generated_tokens: {} ({:.2} tok/s)",
        stats.generated_tokens, stats.generation_tps
    )?;
    if let Some(peak_memory_bytes) = stats.peak_memory_bytes {
        writeln!(stderr, "peak_memory: {}", format_bytes(peak_memory_bytes))?;
    }
    if let Some(warning) = &stats.template_warning {
        writeln!(stderr, "template_warning: {warning}")?;
    }
    stderr.flush()?;
    Ok(())
}

fn validate_args(args: &Args) -> Result<()> {
    match (&args.prompt, &args.messages_file) {
        (Some(_), Some(_)) => bail!("use either --prompt or --messages-file, not both"),
        (None, None) => bail!("one of --prompt or --messages-file is required"),
        _ => {}
    }
    if args.raw_prompt && args.prompt.is_none() {
        bail!("--raw-prompt is only supported together with --prompt");
    }
    Ok(())
}

fn load_messages(path: &PathBuf) -> Result<Vec<ChatMessage>> {
    let data = fs::read_to_string(path)
        .with_context(|| format!("failed to read messages file {}", path.display()))?;
    serde_json::from_str(&data)
        .with_context(|| format!("failed to parse messages JSON from {}", path.display()))
}

fn format_bytes(bytes: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KiB", "MiB", "GiB", "TiB"];
    let mut value = bytes as f64;
    let mut unit = 0;
    while value >= 1024.0 && unit < UNITS.len() - 1 {
        value /= 1024.0;
        unit += 1;
    }
    if unit == 0 {
        format!("{bytes} {}", UNITS[unit])
    } else {
        format!("{value:.2} {}", UNITS[unit])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validate_args_rejects_conflicting_inputs() {
        let args = Args {
            model: PathBuf::from("model.gguf"),
            prompt: Some("hello".to_owned()),
            messages_file: Some(PathBuf::from("messages.json")),
            device: DeviceArg::Auto,
            max_new_tokens: 16,
            temperature: 0.5,
            top_p: 0.85,
            top_k: 20,
            repeat_penalty: 1.1,
            repeat_last_n: 256,
            no_repeat_ngram_size: 6,
            seed: Some(1),
            raw_prompt: false,
            template_mode: TemplateModeArg::WarnFallback,
        };
        assert!(validate_args(&args).is_err());
    }

    #[test]
    fn format_bytes_scales_units_correctly() {
        assert_eq!(format_bytes(0), "0 B");
        assert_eq!(format_bytes(1023), "1023 B");
        assert_eq!(format_bytes(1024), "1.00 KiB");
        assert_eq!(format_bytes(1024 * 1024), "1.00 MiB");
        assert_eq!(format_bytes(1024 * 1024 * 1024), "1.00 GiB");
        assert_eq!(format_bytes(1536 * 1024), "1.50 MiB");
    }

    #[test]
    fn validate_args_rejects_raw_prompt_without_prompt() {
        let args = Args {
            model: PathBuf::from("model.gguf"),
            prompt: None,
            messages_file: Some(PathBuf::from("messages.json")),
            device: DeviceArg::Auto,
            max_new_tokens: 16,
            temperature: 0.5,
            top_p: 0.85,
            top_k: 20,
            repeat_penalty: 1.1,
            repeat_last_n: 256,
            no_repeat_ngram_size: 6,
            seed: Some(1),
            raw_prompt: true,
            template_mode: TemplateModeArg::WarnFallback,
        };
        assert!(validate_args(&args).is_err());
    }
}
