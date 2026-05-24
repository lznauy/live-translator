// Minimal candle-Whisper test: WAV → mel → encode → greedy decode.
// Strips away all pipeline/VAD/normalization to isolate the core ASR path.

use std::collections::HashMap;
use std::path::PathBuf;

use anyhow::Result;
use candle_core::{Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::whisper::{self as m, audio, model::Whisper, Config};

fn load_vocab(path: &std::path::Path) -> Result<(HashMap<String, u32>, HashMap<u32, String>)> {
    let raw = std::fs::read_to_string(path)?;
    let token_to_id: HashMap<String, u32> =
        serde_json::from_str(&raw).map_err(|e| anyhow::anyhow!("parse vocab: {e}"))?;
    let id_to_token: HashMap<u32, String> =
        token_to_id.iter().map(|(k, v)| (*v, k.clone())).collect();
    Ok((token_to_id, id_to_token))
}

fn load_mel_filters(base: &std::path::Path) -> Result<Vec<f32>> {
    let pp_path = base.join("preprocessor_config.json");
    if let Ok(raw) = std::fs::read_to_string(&pp_path) {
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(&raw) {
            if let Some(arr) = v["mel_filters"].as_array() {
                let filters: Vec<f32> = arr
                    .iter()
                    .flat_map(|row| {
                        row.as_array()
                            .map(|r| {
                                r.iter()
                                    .map(|x| x.as_f64().unwrap_or(0.0) as f32)
                                    .collect::<Vec<_>>()
                            })
                            .unwrap_or_default()
                    })
                    .collect();
                if !filters.is_empty() {
                    eprintln!(
                        "[test] loaded mel filters {}x{}",
                        arr.len(),
                        filters.len() / arr.len()
                    );
                    return Ok(filters);
                }
            }
        }
    }
    anyhow::bail!("no mel filters in preprocessor_config.json");
}

/// Max safe samples: ceil(samples/HOP_LENGTH) padded to 3000 mel frames,
/// after conv2 stride-2 → 1500 ≤ max_source_positions.
const MAX_SAFE_SAMPLES: usize = 240_000;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: {} <wav_path> <model_dir>", args[0]);
        std::process::exit(1);
    }
    let wav_path = &args[1];
    let model_dir = &args[2];

    // 1. Read WAV
    let data = std::fs::read(wav_path)?;
    if data.len() < 44 || &data[0..4] != b"RIFF" {
        anyhow::bail!("not a valid WAV file");
    }
    let channels = u16::from_le_bytes([data[22], data[23]]);
    let _rate = u32::from_le_bytes([data[24], data[25], data[26], data[27]]);
    let _bits = u16::from_le_bytes([data[34], data[35]]);
    eprintln!("WAV: {}ch {}Hz {}bit", channels, _rate, _bits);

    let pcm = &data[44..];
    let samples: Vec<f32> = pcm
        .chunks_exact(2)
        .map(|c| i16::from_le_bytes([c[0], c[1]]) as f32 / 32768.0)
        .collect();
    let rms = (samples.iter().map(|s| s * s).sum::<f32>() / samples.len() as f32).sqrt();
    eprintln!(
        "audio: {} samples, {:.2}s, RMS={:.6}",
        samples.len(),
        samples.len() as f32 / 16000.0,
        rms
    );

    // 2. Load model
    let device = Device::Cpu;
    let base = std::fs::canonicalize(model_dir).unwrap_or_else(|_| PathBuf::from(model_dir));
    let base = if base.is_absolute() {
        base
    } else {
        std::env::current_dir()?.join(&base)
    };

    let config: Config = {
        let raw = std::fs::read_to_string(base.join("config.json"))?;
        serde_json::from_str(&raw)?
    };
    eprintln!("[test] config: d_model={}", config.d_model);

    let (token_to_id, id_to_token) = load_vocab(&base.join("vocab.json"))?;
    eprintln!("[test] vocab: {} tokens", token_to_id.len());

    let model_path = base.join("model.safetensors");
    eprintln!("[test] loading model...");
    let vb =
        unsafe { VarBuilder::from_mmaped_safetensors(&[&model_path], m::DTYPE, &device)? };
    let mut model = Whisper::load(&vb, config.clone())?;
    eprintln!("[test] whisper ready");

    let mel_filters = load_mel_filters(&base)?;

    // 3. Truncate if needed
    let samples = if samples.len() > MAX_SAFE_SAMPLES {
        eprintln!("[test] truncating {}s → 15s", samples.len() / 16_000);
        &samples[samples.len() - MAX_SAFE_SAMPLES..]
    } else {
        &samples
    };

    // 4. Mel spectrogram (with gain normalization, matching asr.rs)
    let mut normalized = samples.to_vec();
    {
        let sum_sq: f32 = normalized.iter().map(|s| s * s).sum();
        let rms = (sum_sq / normalized.len() as f32).sqrt();
        if rms > 0.0001 {
            let gain = (0.1 / rms).min(3.0);
            for s in normalized.iter_mut() {
                *s = (*s * gain).clamp(-0.95, 0.95);
            }
        }
    }
    let mel = audio::pcm_to_mel(&config, &normalized, &mel_filters);
    let mel_len = mel.len() / config.num_mel_bins;
    eprintln!("[test] mel frames: {}", mel_len);
    let mel = Tensor::from_vec(mel, (1, config.num_mel_bins, mel_len), &device)?;

    // 5. Encode
    let audio_features = model.encoder.forward(&mel, true)?;
    let enc_frames = audio_features.dims3()?.1;
    eprintln!("[test] encoder output: {} frames", enc_frames);

    // 6. Decode
    let sot = *token_to_id.get(m::SOT_TOKEN).unwrap_or(&50257) as i64;
    let lang_token = token_to_id
        .get("<|en|>")
        .copied()
        .unwrap_or(50259) as i64;
    let transcribe = *token_to_id.get(m::TRANSCRIBE_TOKEN).unwrap_or(&50359) as i64;
    let no_ts = *token_to_id.get(m::NO_TIMESTAMPS_TOKEN).unwrap_or(&50363) as i64;
    let eot = *token_to_id.get(m::EOT_TOKEN).unwrap_or(&50257) as i64;

    model.reset_kv_cache();
    let mut tokens = vec![sot, lang_token, transcribe, no_ts];
    for step in 0..100 {
        let prev = *tokens.last().unwrap();
        if prev == eot {
            eprintln!("[test] EOT at step {step}");
            break;
        }
        let flush = step == 0;
        let tokens_u32: Vec<u32> = tokens.iter().map(|&t| t as u32).collect();
        let decoder_input =
            Tensor::new(tokens_u32.as_slice(), &device)?.unsqueeze(0)?;
        let logits = model.decoder.forward(&decoder_input, &audio_features, flush)?;
        let logits = model.decoder.final_linear(&logits)?;
        let logits_2d = logits.squeeze(0)?;
        let last_idx = logits_2d.dim(0)?.saturating_sub(1);
        let last_logits = logits_2d.i(last_idx)?;
        let selected = last_logits.argmax(0)?.to_scalar::<u32>()? as i64;

        // Show top-3 at step 0
        if step == 0 {
            let logp = candle_nn::ops::log_softmax(&last_logits, 0)?;
            let logp_vec = logp.to_vec1()?;
            let mut indexed: Vec<(usize, f32)> =
                logp_vec.iter().enumerate().map(|(i, &v)| (i, v)).collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            eprint!("[test] step0 top3: ");
            for (idx, val) in indexed.iter().take(3) {
                let tok = id_to_token
                    .get(&(*idx as u32))
                    .map(|s| s.replace('Ġ', "␣"))
                    .unwrap_or_else(|| "?".to_string());
                eprint!("{idx}:'{tok}'={val:.3}  ");
            }
            eprintln!();
        }

        tokens.push(selected);
    }

    // 7. Detokenize
    let text: String = tokens
        .iter()
        .filter(|&&t| t >= 0 && (t as u32) < 50257)
        .filter_map(|&t| id_to_token.get(&(t as u32)))
        .map(|s| s.replace('Ġ', " ").replace('Ċ', "\n").replace('ĉ', "\t"))
        .collect::<Vec<_>>()
        .join("");
    eprintln!("[test] text tokens: {}", tokens.len() - 4);
    eprintln!("[test] result: \"{}\"", text.trim());
    Ok(())
}
