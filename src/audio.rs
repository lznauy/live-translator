use crate::log;
use std::io::Read;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use anyhow::{Context, Result};
use crossbeam_channel::Sender;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

const TARGET_RATE: u32 = 16000;
const TARGET_RMS: f32 = 0.1;

pub fn normalize_chunk(samples: &mut [f32]) {
    let sum_sq: f32 = samples.iter().map(|s| s * s).sum();
    let rms = (sum_sq / samples.len() as f32).sqrt();
    if rms > 0.0001 {
        let gain = (TARGET_RMS / rms).min(3.0);
        for s in samples.iter_mut() {
            *s = (*s * gain).clamp(-0.95, 0.95);
        }
    }
}

pub fn list_devices() -> Result<()> {
    let host = cpal::default_host();

    log!("(=== 输入设备 (Input) ===");
    if let Ok(devices) = host.input_devices() {
        for (i, d) in devices.enumerate() {
            let name = d.name().unwrap_or_default();
            let ch = d.default_input_config().map(|c| c.channels()).unwrap_or(0);
            let rate = d.default_input_config().map(|c| c.sample_rate().0).unwrap_or(0);
            log!("(  [{i}] {name}");
            log!("(       {ch}ch {rate}Hz");
        }
    }

    log!("(\n=== 输出设备 (Output) — 可用 *.monitor 源抓系统声音 ===");
    if let Ok(devices) = host.output_devices() {
        for (i, d) in devices.enumerate() {
            let name = d.name().unwrap_or_default();
            let ch = d.default_output_config().map(|c| c.channels()).unwrap_or(0);
            let rate = d.default_output_config().map(|c| c.sample_rate().0).unwrap_or(0);
            log!("(  [{i}] {name}");
            log!("(       {ch}ch {rate}Hz  → monitor: {name}.monitor");
        }
    }

    log!("(\n=== PulseAudio/PipeWire 所有输入源 (pactl) ===");
    if let Ok(output) = std::process::Command::new("pactl")
        .args(["list", "sources", "short"])
        .output()
    {
        if output.status.success() {
            let text = String::from_utf8_lossy(&output.stdout);
            for line in text.lines().filter(|l| !l.trim().is_empty()) {
                log!("(  {line}");
            }
        } else {
            log!("(  (pactl failed — 可能需要安装 pipewire-pulse)");
        }
    } else {
        log!("(  (pactl 不可用 — 试试 pw-cli list-objects)");
    }

    log!("(\n抓取 Chrome 浏览器声音:");
    log!("(  1. 创建虚拟 sink:  pactl load-module module-null-sink sink_name=chrome_sink");
    log!("(  2. 用 pavucontrol 把 Chrome 音频输出路由到 chrome_sink");
    log!("(  3. 启动:  live-translator --audio-device chrome_sink.monitor --auto-start");
    log!("(  或直接用系统默认 monitor:");
    log!("(     live-translator --audio-device \"monitor\" --auto-start");

    Ok(())
}

fn find_device(host: &cpal::Host, name: &str) -> Option<cpal::Device> {
    let name_lower = name.to_lowercase();

    // 1) Search input devices by substring match
    if let Ok(devices) = host.input_devices() {
        for d in devices {
            if let Ok(dname) = d.name() {
                if dname.to_lowercase().contains(&name_lower) {
                    return Some(d);
                }
            }
        }
    }

    // 2) Search output devices: match their real name OR their virtual {name}.monitor name.
    //    This handles queries like "monitor" or "analog.monitor" on PipeWire/ALSA where
    //    monitor sources may not be enumerated as standalone input devices.
    if let Ok(outputs) = host.output_devices() {
        for d in outputs {
            if let Ok(dname) = d.name() {
                let monitor_name = format!("{dname}.monitor");
                let monitor_lower = monitor_name.to_lowercase();

                // Match against output device name OR its derived monitor name
                if dname.to_lowercase().contains(&name_lower)
                    || monitor_lower.contains(&name_lower)
                {
                    // Try to find the monitor as an enumerated input device
                    if let Ok(inputs) = host.input_devices() {
                        for inp in inputs {
                            if inp.name().map(|n| n == monitor_name).unwrap_or(false) {
                                return Some(inp);
                            }
                        }
                    }
                    // Monitor not enumerated – fall back to the first matching input
                    // device whose name contains the monitor name (handles PipeWire
                    // where monitor ports use a slightly different naming scheme).
                    if let Ok(inputs) = host.input_devices() {
                        for inp in inputs {
                            if let Ok(iname) = inp.name() {
                                if iname.to_lowercase().contains(&monitor_lower) {
                                    return Some(inp);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    None
}


pub fn capture(tx: Sender<Vec<f32>>, device_name: &str, running: Arc<AtomicBool>) -> Result<()> {
    let host = cpal::default_host();
    let device = if device_name.is_empty() {
        host.default_input_device()
    } else {
        find_device(&host, device_name)
    };

    match device {
        Some(dev) => capture_via_cpal(&dev, tx, running),
        None => {
            log!("[audio] cpal 未找到设备 '{}'，尝试 pw-cat/parec...", device_name);
            capture_via_subprocess(device_name, tx, running)
        }
    }
}

fn capture_via_cpal(device: &cpal::Device, tx: Sender<Vec<f32>>, running: Arc<AtomicBool>) -> Result<()> {
    log!("[audio] using device: {}", device.name().unwrap_or_default());

    let mut supported = device.supported_input_configs()
        .context("Failed to query configs")?;
    let conf = supported.next().context("No supported config")?.with_max_sample_rate();
    let channels = conf.channels();
    let sample_rate = conf.sample_rate().0;
    let config: cpal::StreamConfig = conf.into();
    log!("[audio] {}ch {}Hz → mono {}Hz", channels, sample_rate, TARGET_RATE);

    let stream = device.build_input_stream(
        &config,
        move |data: &[f32], _: &cpal::InputCallbackInfo| {
            let mono: Vec<f32> = if channels > 1 {
                data.chunks(channels as usize).map(|c| c.iter().sum::<f32>() / channels as f32).collect()
            } else {
                data.to_vec()
            };
            let chunk = if sample_rate != TARGET_RATE {
                let ratio = TARGET_RATE as f64 / sample_rate as f64;
                let out_len = (mono.len() as f64 * ratio) as usize;
                let mut out = Vec::with_capacity(out_len);
                for i in 0..out_len {
                    let src = (i as f64 / ratio) as usize;
                    let frac = (i as f64 / ratio) - src as f64;
                    let a = mono[src.min(mono.len() - 1)];
                    let b = mono[(src + 1).min(mono.len() - 1)];
                    out.push((a as f64 * (1.0 - frac) + b as f64 * frac) as f32);
                }
                out
            } else {
                mono
            };
            let _ = tx.send(chunk);
        },
        |err| log!("(Audio error: {err}"),
        None,
    )?;
    stream.play()?;

    while running.load(Ordering::Relaxed) {
        std::thread::park_timeout(Duration::from_millis(100));
    }
    log!("[audio] stopped");
    Ok(())
}

/// Fallback capture via `pw-cat` (PipeWire native) or `parec` (PulseAudio compat).
/// Used when cpal cannot enumerate monitor sources.
///
/// Two modes for PipeWire:
///   "monitor"  → `-P '{stream.capture.sink=true}'` (default sink monitor)
///   "*.monitor" → `-P '{stream.capture.sink=true,node.target=*}'` (specific sink)
///   other      → `--target=<name>` then fallback to monitor mode
fn capture_via_subprocess(device_name: &str, tx: Sender<Vec<f32>>, running: Arc<AtomicBool>) -> Result<()> {
    let name_lower = device_name.to_lowercase();

    // Determine the capture strategy
    let is_monitor = name_lower == "monitor" || name_lower.ends_with(".monitor");

    if is_monitor {
        // Build PipeWire stream properties for monitor capture
        let props = if name_lower == "monitor" {
            // Default system audio: capture from the default sink's monitor
            log!("[audio] capturing from default sink monitor via PipeWire");
            "{stream.capture.sink=true}".to_string()
        } else {
            // Specific sink monitor: strip ".monitor" suffix
            let target = &device_name[..device_name.len() - ".monitor".len()];
            log!("[audio] capturing from monitor of PipeWire sink \"{target}\"");
            format!("{{stream.capture.sink=true,node.target={target}}}")
        };

        log!("[audio] trying pw-cat -P '{props}'...");
        let result = std::process::Command::new("pw-cat")
            .arg("--record")
            .arg("-P")
            .arg(&props)
            .arg("--rate=16000")
            .arg("--channels=1")
            .arg("--format=s16")
            .arg("--raw")  // ensure raw PCM, not WAV container
            .arg("-")
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .map_err(|e| anyhow::anyhow!(e))
            .and_then(|child| read_pcm_stdout(child, tx.clone(), &running));

        if result.is_ok() {
            return result;
        }

        // If monitor capture failed (e.g., pw-cat too old for -P flag),
        // fall through to try parec
        log!("[audio] PipeWire monitor capture failed, trying parec...");
    } else {
        // Non-monitor device: try --target first
        log!("[audio] trying pw-cat --target={device_name}...");
        let result = std::process::Command::new("pw-cat")
            .arg("--record")
            .arg("--target")
            .arg(device_name)
            .arg("--rate=16000")
            .arg("--channels=1")
            .arg("--format=s16")
            .arg("-")
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .map_err(|e| anyhow::anyhow!(e))
            .and_then(|child| read_pcm_stdout(child, tx.clone(), &running));

        if result.is_ok() {
            return result;
        }
    }

    // Final fallback: try parec (requires pipewire-pulse)
    log!("[audio] trying parec --device={device_name}...");
    std::process::Command::new("parec")
        .arg("--device")
        .arg(device_name)
        .arg("--format=s16le")
        .arg("--rate=16000")
        .arg("--channels=1")
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .spawn()
        .map_err(|e| anyhow::anyhow!(e))
        .and_then(|child| read_pcm_stdout(child, tx, &running))
        .context("pw-cat 和 parec 均未安装或启动失败")
}

/// Read raw s16le PCM from subprocess stdout and feed into channel.
fn read_pcm_stdout(
    mut child: std::process::Child,
    tx: Sender<Vec<f32>>,
    running: &Arc<AtomicBool>,
) -> Result<()> {
    let stdout = child.stdout.take().unwrap();
    // Spawn a thread to read subprocess stderr for diagnostics
    if let Some(stderr) = child.stderr.take() {
        std::thread::spawn(move || {
            use std::io::BufRead;
            let reader = std::io::BufReader::new(stderr);
            for line in reader.lines().map_while(Result::ok) {
                if !line.trim().is_empty() {
                    log!("[subprocess stderr] {}", line);
                }
            }
        });
    }
    let mut reader = std::io::BufReader::new(stdout);
    let mut buf = [0u8; 1024]; // 512 samples × 2 bytes (s16le)
    let mut chunk_count: u64 = 0;

    while running.load(Ordering::Relaxed) {
        reader.read_exact(&mut buf)?;
        let samples: Vec<f32> = buf
            .chunks_exact(2)
            .map(|c| i16::from_le_bytes([c[0], c[1]]) as f32 / 32768.0)
            .collect();
        // Debug: log chunk RMS every ~50 chunks (~1.6s)
        chunk_count += 1;
        if chunk_count % 50 == 1 {
            let rms = (samples.iter().map(|s| s * s).sum::<f32>() / samples.len() as f32).sqrt();
            log!("[audio] chunk #{chunk_count}, RMS={rms:.6}");
        }
        if tx.send(samples).is_err() {
            break;
        }
    }

    let _ = child.kill();
    let _ = child.wait();
    log!("[audio] subprocess stopped");
    Ok(())
}

pub fn window_size() -> usize { 512 }
pub fn sample_rate() -> i32 { TARGET_RATE as i32 }
