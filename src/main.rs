mod audio;
mod asr;
mod log;
mod pipeline;
mod protocol;
mod translator;
mod vad;

use std::io::{BufRead, BufReader};
use std::thread;

use anyhow::Result;
use clap::Parser;

#[derive(Parser)]
#[command(name = "live-translator")]
struct Cli {
    #[arg(long, default_value = "models")]
    model_dir: String,

    /// Whisper 模型路径 (tiny ~145MB, base ~277MB, small ~923MB)
    #[arg(long, default_value = "models/whisper-small")]
    asr_model: String,

    /// 翻译模型：本地目录路径
    #[arg(long, default_value = "models/opus-mt-en-zh")]
    translate_model: String,

    /// 音频设备名（为空则用默认麦克风，指定 monitor 设备则抓系统声音）
    #[arg(long, default_value = "")]
    audio_device: String,

    /// 列出所有音频设备后退出
    #[arg(long)]
    list_devices: bool,

    /// 启动后自动开始监听，无需 stdin 命令
    #[arg(long)]
    auto_start: bool,

    #[arg(long, default_value = "en")]
    auto_src: String,

    #[arg(long, default_value = "zh")]
    auto_tgt: String,

    /// VAD 能量阈值（0.0-1.0，默认 0.008）
    #[arg(long, default_value = "0.005")]
    vad_threshold: f32,

    /// 测试模式：加载 WAV 文件直接跑 ASR 识别，不启动音频抓取
    #[arg(long)]
    test_wav: Option<String>,

    /// 纯净模式：仅输出 "识别文本 翻译文本"，无调试日志和 JSON
    #[arg(long)]
    quiet: bool,
}

impl Cli {
    fn auto_start_cmd(&self) -> Option<protocol::Command> {
        if self.auto_start {
            Some(protocol::Command::Start {
                src: self.auto_src.clone(),
                tgt: self.auto_tgt.clone(),
            })
        } else {
            None
        }
    }
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    crate::log::set_quiet(cli.quiet);

    if cli.quiet {
        use std::os::unix::io::AsRawFd;
        let devnull = std::fs::File::open("/dev/null")
            .expect("open /dev/null");
        unsafe { libc::dup2(devnull.as_raw_fd(), libc::STDERR_FILENO); }
    }

    if cli.list_devices {
        return audio::list_devices();
    }

    if let Some(wav_path) = &cli.test_wav {
        return test_asr_wav(wav_path, &cli.asr_model, &cli.auto_src);
    }

    let (control_tx, control_rx) = crossbeam_channel::unbounded::<protocol::Command>();
    let (event_tx, event_rx) = crossbeam_channel::unbounded::<protocol::Event>();

    // Spawn stdin reader thread
    let control_tx_clone = control_tx.clone();
    let event_tx_stdin = event_tx.clone();
    thread::spawn(move || {
        let stdin = std::io::stdin();
        for line in BufReader::new(stdin).lines() {
            match line {
                Ok(line) if line.trim().is_empty() => continue,
                Ok(line) => match serde_json::from_str::<protocol::Command>(&line) {
                    Ok(cmd) => {
                        if control_tx_clone.send(cmd).is_err() {
                            break;
                        }
                    }
                    Err(e) => {
                        let _ = event_tx_stdin.send(protocol::Event::Error {
                            message: format!("Invalid command: {e}"),
                        });
                    }
                },
                Err(_) => break,
            }
        }
    });

    // Spawn event writer thread
    let quiet = cli.quiet;
    thread::spawn(move || {
        for event in event_rx {
            if quiet {
                if let protocol::Event::Final { text, translated, .. } = &event {
                    println!("{}\t{}", text, translated);
                }
            } else {
                // Human-readable to stderr (interactive terminal use)
                match &event {
                    protocol::Event::Status { state } => {
                        eprintln!("  [{}]", state);
                    }
                    protocol::Event::Partial { text, .. } => {
                        eprintln!("  ... {}", text);
                    }
                    protocol::Event::Final { text, translated, .. } => {
                        eprintln!("  ASR: {}", text);
                        eprintln!("  NMT: {}", translated);
                    }
                    protocol::Event::Error { message } => {
                        eprintln!("  ERR: {}", message);
                    }
                }
                // Machine-readable JSON to stdout (for QML integration)
                if let Ok(json) = serde_json::to_string(&event) {
                    println!("{json}");
                }
            }
        }
    });

    // Auto-start if requested
    if let Some(cmd) = cli.auto_start_cmd() {
        control_tx.send(cmd).ok();
    }

    // Run pipeline
    pipeline::run(cli, control_rx, event_tx)?;

    Ok(())
}


fn test_asr_wav(wav_path: &str, asr_model: &str, lang: &str) -> anyhow::Result<()> {
    use crate::asr::AsrRecognizer;

    log!("=== ASR WAV test: {wav_path} ===");
    let data = std::fs::read(wav_path)?;
    if data.len() < 44 || &data[0..4] != b"RIFF" {
        anyhow::bail!("not a valid WAV file");
    }
    let bits = u16::from_le_bytes([data[34], data[35]]);
    let channels = u16::from_le_bytes([data[22], data[23]]);
    let rate = u32::from_le_bytes([data[24], data[25], data[26], data[27]]);
    log!("WAV: {rate}Hz {channels}ch {bits}bit");

    let pcm = &data[44..];
    let samples: Vec<f32> = pcm
        .chunks_exact(2)
        .map(|c| i16::from_le_bytes([c[0], c[1]]) as f32 / 32768.0)
        .collect();
    let rms = (samples.iter().map(|s| s * s).sum::<f32>() / samples.len() as f32).sqrt();
    log!("audio: {} samples, {:.2}s, RMS={:.6}", samples.len(), samples.len() as f32 / 16000.0, rms);

    log!("loading ASR model...");
    let mut asr = AsrRecognizer::new(asr_model, lang)?;
    log!("running recognize...");
    match asr.recognize(&samples) {
        Ok(text) if text.is_empty() => log!("ASR result: EMPTY"),
        Ok(text) => println!("{text}"),
        Err(e) => log!("ASR error: {e}"),
    }
    Ok(())
}
