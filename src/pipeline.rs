use crate::log;
use std::io::Write;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::thread;
use std::time::{Duration, Instant};

use anyhow::Result;
use crossbeam_channel::{Receiver, Sender};

use crate::audio;
use crate::asr::AsrRecognizer;
use crate::protocol::{Command, Event};
use crate::translator::TranslatorEngine;
use crate::vad::Vad;

/// Maximum speech buffer size: 15 seconds at 16kHz mono.
/// Must not exceed what `log_mel_spectrogram_` can handle after padding —
/// the Whisper‑tiny encoder caps at max_source_positions = 1500.
const MAX_SPEECH_BUFFER: usize = 16_000 * 15;

/// Number of segments to save as WAV files for debugging.
const DEBUG_SAVE_SEGMENTS: u32 = 2;

enum State {
    Idle,
    Listening,
    Speaking,
}

struct TranslationJob {
    text: String,
    confidence: f32,
    src_lang: String,
    generation: u64,
}

struct TranslationResult {
    text: String,
    confidence: f32,
    translated: String,
    src_lang: String,
    generation: u64,
}

pub fn run(
    cli: crate::Cli,
    control: Receiver<Command>,
    event_tx: Sender<Event>,
) -> Result<()> {
    log!("[pipeline] loading VAD (threshold={})...", cli.vad_threshold);
    let mut vad = Vad::new(cli.vad_threshold)?;
    log!("[pipeline] loading ASR...");
    let mut asr = AsrRecognizer::new(&cli.asr_model, "auto")?;
    log!("[pipeline] loading NMT...");
    let translate_model = cli.translate_model.clone();
    let mut translator = TranslatorEngine::new(&translate_model)?;
    log!("[pipeline] all models loaded");

    let (audio_tx, audio_rx) = crossbeam_channel::bounded::<Vec<f32>>(128);

    // Translation channels
    let (translate_tx, translate_rx) = crossbeam_channel::unbounded::<TranslationJob>();
    let (result_tx, result_rx) = crossbeam_channel::unbounded::<TranslationResult>();

    // Clone receiver so the main loop can drain stale jobs on language switch
    let translate_rx_drain = translate_rx.clone();

    // Spawn translator thread (supports model reload on language switch)
    let (translator_reload_tx, translator_reload_rx) = crossbeam_channel::unbounded::<String>();
    let translator_thread = thread::spawn(move || {
        loop {
            crossbeam_channel::select! {
                recv(translate_rx) -> job => {
                    match job {
                        Ok(job) => {
                            let t0 = std::time::Instant::now();
                            let translated = match translator.translate(&job.text) {
                                Ok(t) => t,
                                Err(e) => {
                                    log!("[translator] error translating '{}': {e}", &job.text);
                                    String::new()
                                }
                            };
                            let nmt_ms = t0.elapsed().as_millis();
                            log!("[perf] NMT: {nmt_ms}ms, src=\"{}\"", &job.text);
                            let _ = result_tx.send(TranslationResult {
                                text: job.text,
                                confidence: job.confidence,
                                translated,
                                src_lang: job.src_lang,
                                generation: job.generation,
                            });
                        }
                        Err(_) => break,
                    }
                }
                recv(translator_reload_rx) -> model_path => {
                    if let Ok(path) = model_path {
                        match TranslatorEngine::new(&path) {
                            Ok(new_translator) => {
                                translator = new_translator;
                                log!("[pipeline] translator reloaded: {}", path);
                            }
                            Err(e) => log!("[pipeline] failed to reload translator: {e}"),
                        }
                    }
                }
            }
        }
    });

    // Audio capture thread
    let running = Arc::new(AtomicBool::new(true));
    let running_clone = running.clone();
    let device_name = cli.audio_device.clone();
    let event_tx_audio = event_tx.clone();
    log!("[pipeline] audio device: '{}'", if device_name.is_empty() { "default" } else { &device_name });
    let audio_thread = thread::spawn(move || {
        if let Err(e) = audio::capture(audio_tx, &device_name, running_clone) {
            log!("[audio] capture failed: {e}");
            let _ = event_tx_audio.send(Event::Error {
                message: format!("Audio capture failed: {e}"),
            });
        }
    });

    let mut state = State::Idle;
    let mut src_lang = String::from("auto");
    let mut translation_generation: u64 = 0;
    let debug_segment_count = AtomicU32::new(0);

    let mut speech_buffer: Vec<f32> = Vec::new();
    let mut last_partial = Instant::now();
    let partial_fast = Duration::from_millis(500);
    let partial_slow = Duration::from_millis(1000);
    let partial_threshold = (audio::sample_rate() * 2) as usize;

    event_tx.send(Event::Status { state: "idle".into() }).ok();

    loop {
        crossbeam_channel::select! {
            recv(control) -> msg => {
                match msg {
                    Ok(Command::Start { src, tgt: _tgt }) => {
                        translation_generation = translation_generation.wrapping_add(1);
                        if src != src_lang {
                            let first_start = src_lang == "auto";
                            asr.set_language(&src);
                            if !first_start {
                                // Drain stale translation jobs before reloading model
                                while translate_rx_drain.try_recv().is_ok() {}
                                let _ = translator_reload_tx.send(cli.translate_model.clone());
                            }
                        }
                        src_lang = src;
                        vad.reset();
                        // Discard stale audio that piled up during blocking reload
                        let stale: usize = audio_rx.try_iter().count();
                        if stale > 0 {
                            log!("[pipeline] discarded {stale} stale audio chunks (reload backlog)");
                        }
                        speech_buffer.clear();
                        state = State::Listening;
                        event_tx.send(Event::Status { state: "listening".into() }).ok();
                    }
                    Ok(Command::Stop) => {
                        state = State::Idle;
                        vad.reset();
                        speech_buffer.clear();
                        while let Ok(result) = result_rx.try_recv() {
                            if result.generation == translation_generation {
                                event_tx.send(Event::Final {
                                    text: result.text,
                                    translated: result.translated,
                                    lang: result.src_lang,
                                    confidence: Some(result.confidence),
                                }).ok();
                            }
                        }
                        event_tx.send(Event::Status { state: "idle".into() }).ok();
                    }
                    Ok(Command::Quit) => {
                        event_tx.send(Event::Status { state: "exiting".into() }).ok();
                        break;
                    }
                    Err(_) => break,
                }
            }
            recv(audio_rx) -> msg => {
                match msg {
                    Ok(chunk) => {
                        if matches!(state, State::Idle) { continue; }

                        vad.accept_waveform(&chunk);

                        let was_speaking = matches!(state, State::Speaking);

                        if vad.is_speech() {
                            if !was_speaking {
                                log!("[vad] speech start (buffer={} samples)", speech_buffer.len());
                            }
                            state = State::Speaking;
                            // Drain backlog and feed each chunk through VAD too,
                            // otherwise VAD misses audio and segments are misaligned.
                            audio_rx.try_iter().for_each(|extra| {
                                vad.accept_waveform(&extra);
                                speech_buffer.extend(extra);
                            });
                            speech_buffer.extend_from_slice(&chunk);

                            // Prevent unbounded growth when ASR falls behind
                            if speech_buffer.len() > MAX_SPEECH_BUFFER {
                                let excess = speech_buffer.len() - MAX_SPEECH_BUFFER;
                                speech_buffer.drain(..excess);
                            }

                            let interval = if speech_buffer.len() >= partial_threshold {
                                partial_slow
                            } else {
                                partial_fast
                            };
                            if last_partial.elapsed() >= interval
                                && speech_buffer.len() > audio::window_size() * 4
                            {
                                let partial_window = 16_000 * 5; // last 5 seconds for partial
                                let partial_input = if speech_buffer.len() > partial_window {
                                    &speech_buffer[speech_buffer.len() - partial_window..]
                                } else {
                                    &speech_buffer
                                };
                                let t0 = std::time::Instant::now();
                                let result = asr.recognize_with_confidence(partial_input);
                                let asr_ms = t0.elapsed().as_millis();
                                match result {
                                    Ok((text, conf)) if !text.trim().is_empty() && conf > -3.0 => {
                                        log!("[perf] partial ASR: {asr_ms}ms, text=\"{text}\"");
                                        event_tx.send(Event::Partial {
                                            text,
                                            lang: src_lang.clone(),
                                            confidence: Some(conf),
                                        }).ok();
                                    }
                                    Ok(_) => {
                                        log!(
                                            "[vad] partial ASR empty ({asr_ms}ms, buffer={:.1}s)",
                                            speech_buffer.len() as f32 / audio::sample_rate() as f32
                                        );
                                    }
                                    Err(e) => {
                                        log!("[pipeline] ASR partial error: {e}");
                                    }
                                }
                                last_partial = Instant::now();
                            }
                        } else if was_speaking {
                            vad.flush();
                        }

                        while vad.has_segment() {
                            let segment = vad.take_segment().unwrap();
                            let segment_samples = segment.samples();
                            let seg_len = segment_samples.len();
                            log!(
                                "[vad] segment: {:.1}s, ASR running...",
                                seg_len as f32 / audio::sample_rate() as f32
                            );

                            // Save first N segments as WAV for debugging
                            let n = debug_segment_count.fetch_add(1, Ordering::Relaxed);
                            if n < DEBUG_SAVE_SEGMENTS {
                                let path = format!("/tmp/asr_debug_segment_{n}.wav");
                                if let Err(e) = save_wav(&path, segment_samples) {
                                    log!("[debug] failed to save {path}: {e}");
                                } else {
                                    log!("[debug] saved {path} ({:.1}s)", seg_len as f32 / audio::sample_rate() as f32);
                                }
                            }

                            let t0 = std::time::Instant::now();
                            let result = asr.recognize_with_confidence(segment_samples);
                            let asr_ms = t0.elapsed().as_millis();
                            match result {
                                Ok((text, confidence)) if !text.trim().is_empty() => {
                                    log!("[vad] segment ASR ({asr_ms}ms): \"{}\" (conf={:.3})", text, confidence);
                                    translate_tx.send(TranslationJob {
                                        text: text.clone(),
                                        confidence,
                                        src_lang: src_lang.clone(),
                                        generation: translation_generation,
                                    }).ok();
                                }
                                Ok(_) => {
                                    log!("[vad] segment ASR empty ({:.1}s)", seg_len as f32 / audio::sample_rate() as f32);
                                }
                                Err(e) => {
                                    log!("[pipeline] ASR segment error: {e}");
                                }
                            }

                            if speech_buffer.len() > seg_len {
                                speech_buffer.drain(..seg_len);
                            } else {
                                speech_buffer.clear();
                            }
                        }

                        if matches!(state, State::Speaking) && !vad.is_speech() && !vad.has_segment() {
                            state = State::Listening;
                            let keep = (audio::sample_rate() as f32 * 0.3) as usize;
                            if speech_buffer.len() > keep {
                                speech_buffer.drain(..speech_buffer.len() - keep);
                            }
                            event_tx.send(Event::Status { state: "listening".into() }).ok();
                        }
                    }
                    Err(_) => {
                        // Channel closed: audio capture failed or stopped
                        log!("[pipeline] audio disconnected, stopping");
                        break;
                    }
                }
            }
            recv(result_rx) -> msg => {
                if let Ok(result) = msg {
                    if result.generation == translation_generation {
                        event_tx.send(Event::Final {
                            text: result.text,
                            translated: result.translated,
                            lang: result.src_lang,
                            confidence: Some(result.confidence),
                        }).ok();
                    }
                }
            }
        }
    }

    // Graceful shutdown
    running.store(false, Ordering::Relaxed);
    audio_thread.thread().unpark();
    let _ = audio_thread.join();

    drop(translate_tx);
    drop(translator_reload_tx);
    translator_thread.join().ok();

    Ok(())
}

/// Write raw 16kHz mono f32 samples as a WAV file for debugging.
fn save_wav(path: &str, samples: &[f32]) -> std::io::Result<()> {
    let mut f = std::fs::File::create(path)?;
    let data_len = (samples.len() * 2) as u32;
    // WAV header
    f.write_all(b"RIFF")?;
    f.write_all(&(36u32 + data_len).to_le_bytes())?;
    f.write_all(b"WAVE")?;
    f.write_all(b"fmt ")?;
    f.write_all(&16u32.to_le_bytes())?; // chunk size
    f.write_all(&1u16.to_le_bytes())?;  // PCM
    f.write_all(&1u16.to_le_bytes())?;  // mono
    f.write_all(&16_000u32.to_le_bytes())?; // sample rate
    f.write_all(&32_000u32.to_le_bytes())?; // byte rate
    f.write_all(&2u16.to_le_bytes())?;  // block align
    f.write_all(&16u16.to_le_bytes())?; // bits per sample
    f.write_all(b"data")?;
    f.write_all(&data_len.to_le_bytes())?;
    // PCM samples
    for &s in samples {
        let v = (s.clamp(-1.0, 1.0) * 32767.0) as i16;
        f.write_all(&v.to_le_bytes())?;
    }
    Ok(())
}
