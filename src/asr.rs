use crate::log;
use std::collections::HashMap;

use anyhow::Result;
use candle_core::{Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::whisper::{self as m, audio, model::Whisper, Config};

pub struct AsrRecognizer {
    model: Whisper,
    token_to_id: HashMap<String, u32>,
    id_to_token: HashMap<u32, String>,
    config: Config,
    device: Device,
    lang: String,
    mel_filters: Vec<f32>,
    max_decode_steps: usize,
}

impl AsrRecognizer {
    pub fn new(model_id: &str, lang: &str) -> Result<Self> {
        let device = Device::Cpu;
        let abs = std::fs::canonicalize(model_id)
            .unwrap_or_else(|_| std::path::PathBuf::from(model_id));
        let base = if abs.is_absolute() { abs } else {
            std::env::current_dir()?.join(&abs)
        };
        log!("[asr] base={}", base.display());

        let config_path = base.join("config.json");
        let model_path = base.join("model.safetensors");
        let vocab_path = base.join("vocab.json");

        let config_json = std::fs::read_to_string(&config_path)
            .map_err(|e| anyhow::anyhow!("read config: {e}"))?;
        let config: Config = serde_json::from_str(&config_json)
            .map_err(|e| anyhow::anyhow!("parse config: {e}"))?;
        log!("[asr] config: d_model={}", config.d_model);

        let (token_to_id, id_to_token) = Self::load_vocab(&vocab_path)?;
        log!("[asr] vocab: {} tokens", token_to_id.len());

        log!("[asr] loading model (this may take a while)...");
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[&model_path], m::DTYPE, &device)? };
        let model = Whisper::load(&vb, config.clone())?;
        log!("[asr] whisper ready: d_model={}, layers={}", config.d_model, config.encoder_layers);

        let mel_filters = Self::load_mel_filters(&base, &config)?;
        Ok(Self {
            model, token_to_id, id_to_token, config, device,
            lang: lang.to_string(),
            mel_filters,
            max_decode_steps: 100,
        })
    }

    pub fn set_language(&mut self, lang: &str) {
        self.lang = lang.to_string();
    }

    fn load_vocab(path: &std::path::Path) -> Result<(HashMap<String, u32>, HashMap<u32, String>)> {
        let raw = std::fs::read_to_string(path)
            .map_err(|e| anyhow::anyhow!("read vocab: {e}"))?;
        let token_to_id: HashMap<String, u32> = serde_json::from_str(&raw)
            .map_err(|e| anyhow::anyhow!("parse vocab: {e}"))?;
        let id_to_token: HashMap<u32, String> = token_to_id.iter()
            .map(|(k, v)| (*v, k.clone()))
            .collect();
        Ok((token_to_id, id_to_token))
    }

    /// Maximum safe audio length for Whisper tiny.
    /// `log_mel_spectrogram_` pads mel frames by 1500, and the encoder conv2
    /// (stride=2) halves the temporal dimension.  The encoderʼs
    /// `max_source_positions` is 1500, so we need:
    ///   (ceil(samples / HOP_LENGTH) + 1500) / 2  ≤ 1500
    ///   → samples  ≤ 1500 * HOP_LENGTH = 240_000  (15 s)
    const MAX_SAFE_SAMPLES: usize = 240_000;

    pub fn recognize(&mut self, samples: &[f32]) -> Result<String> {
        self.recognize_with_confidence(samples).map(|(text, _)| text)
    }

    pub fn recognize_with_confidence(&mut self, samples: &[f32]) -> Result<(String, f32)> {
        if samples.len() < 4800 { return Ok((String::new(), 0.0)); }
        let samples = if samples.len() > Self::MAX_SAFE_SAMPLES {
            log!(
                "[asr] truncating {}s → 15s to fit max_source_positions",
                samples.len() / 16_000
            );
            &samples[samples.len() - Self::MAX_SAFE_SAMPLES..]
        } else {
            samples
        };
        // Apply gain normalization at the segment level so VAD sees raw dynamics
        let mut normalized = samples.to_vec();
        crate::audio::normalize_chunk(&mut normalized);
        // After normalization, check if the audio is effectively silent
        let sum_sq: f32 = normalized.iter().map(|s| s * s).sum();
        let rms = (sum_sq / normalized.len() as f32).sqrt();
        if rms < 0.001 {
            return Ok((String::new(), 0.0));
        }
        let filters = &self.mel_filters;
        let mel = audio::pcm_to_mel(&self.config, &normalized, filters);
        let mel_len = mel.len() / self.config.num_mel_bins;
        let mel = Tensor::from_vec(mel, (1, self.config.num_mel_bins, mel_len), &self.device)?;
        let audio_features = self.model.encoder.forward(&mel, true)?;
        let sot = *self.token_to_id.get(m::SOT_TOKEN).unwrap_or(&50257) as i64;
        let lang_token = self.token_to_id
            .get(&format!("<|{}|>", match self.lang.as_str() {
                "zh" => "zh", "ja" => "ja", "en" => "en", "ko" => "ko", _ => "en",
            })).copied().unwrap_or(50259) as i64;
        let transcribe = *self.token_to_id.get(m::TRANSCRIBE_TOKEN).unwrap_or(&50359) as i64;
        let no_ts = *self.token_to_id.get(m::NO_TIMESTAMPS_TOKEN).unwrap_or(&50363) as i64;
        let eot = *self.token_to_id.get(m::EOT_TOKEN).unwrap_or(&50257) as i64;
        let nospeech_id = *self.token_to_id.get("<|nospeech|>").unwrap_or(&50362) as i64;
        let nocaptions_id = *self.token_to_id.get("<|nocaptions|>").unwrap_or(&50361) as i64;
        self.model.reset_kv_cache();
        let mut tokens = vec![sot, lang_token, transcribe, no_ts];
        let mut log_probs: Vec<f32> = Vec::new();
        for step in 0..self.max_decode_steps {
            let prev = *tokens.last().unwrap();
            if prev == eot { break; }
            if step % 10 == 0 {
                log!("[asr] decoding step {step}/{}...", self.max_decode_steps);
            }
            // Pass full token sequence so decoder self-attention has full context.
            // Cross-attention KV cache is flushed only on the first step.
            let flush = step == 0;
            let tokens_u32: Vec<u32> = tokens.iter().map(|&t| t as u32).collect();
            let decoder_input = Tensor::new(tokens_u32.as_slice(), &self.device)?.unsqueeze(0)?;
            let logits = self.model.decoder.forward(&decoder_input, &audio_features, flush)?;
            let logits = self.model.decoder.final_linear(&logits)?;
            // Logits shape: (1, seq_len, vocab_size) — take the last position
            let logits_2d = logits.squeeze(0)?;
            let last_idx = logits_2d.dim(0)?.saturating_sub(1);
            let last_logits = logits_2d.i(last_idx)?;
            let selected = last_logits.argmax(0)?.to_scalar::<u32>()? as i64;
            let logp = candle_nn::ops::log_softmax(&last_logits, 0)?;
            let lp = logp.get(selected as usize).map(|t| t.to_scalar::<f32>().unwrap_or(0.0)).unwrap_or(0.0);
            log_probs.push(lp);
            tokens.push(selected);
        }
        // Check for no_speech tokens
        let has_nospeech = tokens.iter().any(|&t| t == nospeech_id || t == nocaptions_id);
        if has_nospeech {
            return Ok((String::new(), 0.0));
        }
        // Debug: show token IDs and mel/encoder shape
        let text_tokens: Vec<&i64> = tokens
            .iter()
            .filter(|&&t| t >= 0 && (t as u32) < 50257)
            .collect();
        let all_ids: Vec<i64> = tokens.to_vec();
        let avg_logprob = if log_probs.is_empty() { 0.0 }
            else { log_probs.iter().sum::<f32>() / log_probs.len() as f32 };
        log!(
            "[asr] mel_frames={}, enc_frames={}, tokens={} (text={}), avg_logprob={:.3}, ids={:?}",
            mel_len,
            audio_features.dims3().map(|(_, s, _)| s).unwrap_or(0),
            tokens.len(),
            text_tokens.len(),
            avg_logprob,
            &all_ids[all_ids.len().saturating_sub(10)..],
        );

        let text: String = tokens.iter()
            .filter(|&&t| t >= 0 && (t as u32) < 50257)
            .filter_map(|&t| self.id_to_token.get(&(t as u32)))
            .map(|s| s.replace('Ġ', " ").replace('Ċ', "\n").replace('ĉ', "\t"))
            .collect::<Vec<_>>().join("");
        let text = Self::clean_text(text.trim());
        Ok((text, avg_logprob))
    }

    /// Strip Whisper formatting artifacts: parenthetical notes (laughs), (Music), etc.
    fn clean_text(text: &str) -> String {
        let mut out = String::with_capacity(text.len());
        let mut depth: usize = 0;
        for ch in text.chars() {
            match ch {
                '(' => depth += 1,
                ')' if depth > 0 => depth -= 1,
                _ if depth == 0 => out.push(ch),
                _ => {}
            }
        }
        // If we have an unmatched '(' or ')' at the end, don't include trailing garbage
        out.trim().to_string()
    }

    /// Load mel filters from the model's preprocessor_config.json.
    /// Falls back to computing them if the file is missing or has no mel_filters.
    fn load_mel_filters(base: &std::path::Path, config: &Config) -> Result<Vec<f32>> {
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
                        log!("[asr] loaded mel filters from preprocessor_config.json ({} x {})",
                            arr.len(), filters.len() / arr.len());
                        return Ok(filters);
                    }
                }
            }
        }
        log!("[asr] preprocessor_config.json not found, computing mel filters");
        Ok(Self::compute_mel_filters(config))
    }

    /// Compute mel filterbank (fallback when preprocessor_config.json is unavailable).
    /// Uses n_mels+2 equally-spaced mel points matching librosa.filters.mel.
    fn compute_mel_filters(config: &Config) -> Vec<f32> {
        let n_mels = config.num_mel_bins;
        let n_fft = m::N_FFT;
        let sample_rate = m::SAMPLE_RATE as f32;
        let n_freqs = n_fft / 2 + 1;

        let f_min = 0.0f32;
        let f_max = sample_rate / 2.0;
        let mel_min = 2595.0 * (1.0 + f_min / 700.0).log10();
        let mel_max = 2595.0 * (1.0 + f_max / 700.0).log10();

        let mel_points: Vec<f32> = (0..n_mels + 2)
            .map(|i| {
                let m = mel_min + (mel_max - mel_min) * i as f32 / (n_mels + 1) as f32;
                700.0 * (10.0f32.powf(m / 2595.0) - 1.0)
            })
            .collect();

        let mut filters = vec![0.0f32; n_mels * n_freqs];
        for j in 0..n_freqs {
            let freq = j as f32 * sample_rate / n_fft as f32;
            for i in 0..n_mels {
                let lower = mel_points[i];
                let center = mel_points[i + 1];
                let upper = mel_points[i + 2];
                if freq > lower && freq < center {
                    filters[i * n_freqs + j] = (freq - lower) / (center - lower);
                } else if freq >= center && freq < upper {
                    filters[i * n_freqs + j] = (upper - freq) / (upper - center);
                }
            }
        }

        filters
    }
}
