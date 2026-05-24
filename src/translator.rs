use crate::log;
use std::collections::HashMap;
use std::path::Path;

use anyhow::Result;
use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::marian::{Config, MTModel};

pub struct TranslatorEngine {
    model: MTModel,
    token_to_id: HashMap<String, u32>,
    id_to_token: HashMap<u32, String>,
    device: Device,
    decoder_start_token_id: u32,
    eos_token_id: u32,
    max_token_len: usize,
}

impl TranslatorEngine {
    pub fn new(model_id: &str) -> Result<Self> {
        let base = Path::new(model_id);
        if !base.is_dir() {
            return Err(anyhow::anyhow!("Model dir not found: {model_id}"));
        }

        let config_json = std::fs::read_to_string(base.join("config.json"))?;
        let mut config_value: serde_json::Value = serde_json::from_str(&config_json)
            .map_err(|e| anyhow::anyhow!("parse config.json: {e}"))?;
        if !config_value.as_object().is_some_and(|o| o.contains_key("share_encoder_decoder_embeddings")) {
            if let Some(obj) = config_value.as_object_mut() {
                obj.insert("share_encoder_decoder_embeddings".into(), serde_json::Value::Bool(true));
            }
        }
        let config: Config = serde_json::from_value(config_value)
            .map_err(|e| anyhow::anyhow!("config: {e}"))?;

        let (token_to_id, id_to_token) = Self::load_vocab(base)?;
        let max_token_len = token_to_id.keys().map(|k| k.chars().count()).max().unwrap_or(16);

        let device = Device::Cpu;
        log!("[nmt] loading Marian model...");
        // SAFETY: Memory-mapping a safetensors file is safe because the file
        // is not modified during inference and the mmap region is read-only.
        // The lifetime of the mmap is tied to VarBuilder, which owns the mapping.
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[base.join("model.safetensors")], candle_core::DType::F32, &device)?
        };
        let model = MTModel::new(&config, vb)?;
        log!("[nmt] Marian loaded: d_model={}, enc={}, dec={}",
            config.d_model, config.encoder_layers, config.decoder_layers);

        Ok(Self {
            model, token_to_id, id_to_token, device,
            decoder_start_token_id: config.decoder_start_token_id,
            eos_token_id: config.eos_token_id,
            max_token_len,
        })
    }

    fn load_vocab(base: &Path) -> Result<(HashMap<String, u32>, HashMap<u32, String>)> {
        let raw = std::fs::read_to_string(base.join("vocab.json"))
            .map_err(|e| anyhow::anyhow!("read vocab.json: {e}"))?;
        let token_to_id: HashMap<String, u32> = serde_json::from_str(&raw)
            .map_err(|e| anyhow::anyhow!("parse vocab.json: {e}"))?;
        let id_to_token: HashMap<u32, String> = token_to_id.iter()
            .map(|(k, v)| (*v, k.clone()))
            .collect();
        Ok((token_to_id, id_to_token))
    }

    fn tokenize(&self, text: &str) -> Vec<u32> {
        // Only lowercase ASCII text (Marian models are trained on lowercased
        // English). Non-ASCII source text (e.g. CJK) is left as-is.
        let text = text.trim();
        let text = if text.is_ascii() {
            text.to_lowercase()
        } else {
            text.to_string()
        };
        let mut input = String::with_capacity(text.len() + 10);
        // SentencePiece preprocessing: prepend ▁, replace spaces with ▁
        input.push('\u{2581}');
        for ch in text.chars() {
            if ch == ' ' {
                input.push('\u{2581}');
            } else {
                input.push(ch);
            }
        }

        let chars: Vec<char> = input.chars().collect();
        let mut ids = Vec::with_capacity(chars.len() / 2 + 1);
        let mut pos = 0;
        while pos < chars.len() {
            let remaining = chars.len() - pos;
            let max_len = remaining.min(self.max_token_len);
            let mut matched = false;
            for len in (1..=max_len).rev() {
                let piece: String = chars[pos..pos + len].iter().collect();
                if let Some(&id) = self.token_to_id.get(&piece) {
                    ids.push(id);
                    pos += len;
                    matched = true;
                    break;
                }
            }
            if !matched {
                pos += 1;
            }
        }
        ids.push(self.eos_token_id);
        ids
    }

    pub fn translate(&mut self, text: &str) -> Result<String> {
        if text.trim().is_empty() {
            return Ok(String::new());
        }

        let input_ids = self.tokenize(text);

        let input = Tensor::new(&input_ids[..], &self.device)?.unsqueeze(0)?;
        let encoder_output = self.model.encoder().forward(&input, 0)?;

        let mut output_ids = vec![self.decoder_start_token_id];
        for _ in 0..256 {
            let prev = *output_ids.last().unwrap();
            let decoder_input = Tensor::new(&[prev], &self.device)?.unsqueeze(0)?;
            let logits = self.model.decode(&decoder_input, &encoder_output, output_ids.len() - 1)?;
            let next = logits.squeeze(0)?.squeeze(0)?.argmax(0)?.to_scalar::<u32>()?;
            if next == self.eos_token_id { break; }
            output_ids.push(next);
        }

        self.model.reset_kv_cache();

        let text: String = output_ids[1..].iter()
            .filter_map(|id| self.id_to_token.get(id))
            .map(|s| s.as_str())
            .collect::<Vec<_>>()
            .join("")
            .replace('\u{2581}', " ")
            .trim()
            .to_string();
        Ok(text)
    }
}
