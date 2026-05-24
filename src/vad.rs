use anyhow::Result;

const SAMPLE_RATE: usize = 16000;

pub struct Vad {
    threshold: f32,
    speech_active: bool,
    speech_buffer: Vec<f32>,
    silence_frames: usize,
    min_speech_samples: usize,  // 0.15s of speech before triggering
    max_silence_frames: usize,  // 0.5s of silence before ending segment
    max_speech_samples: usize,  // 10s of continuous speech before forced cut
    pending_segments: Vec<Vec<f32>>,
}

impl Vad {
    pub fn new(threshold: f32) -> Result<Self> {
        Ok(Self {
            threshold,
            speech_active: false,
            speech_buffer: Vec::new(),
            silence_frames: 0,
            min_speech_samples: (SAMPLE_RATE as f32 * 0.15) as usize,
            max_silence_frames: ((SAMPLE_RATE as f32 * 1.0) / 512.0) as usize,
            max_speech_samples: SAMPLE_RATE * 10, // 10 seconds
            pending_segments: Vec::new(),
        })
    }

    pub fn accept_waveform(&mut self, samples: &[f32]) {
        let sum_sq: f32 = samples.iter().map(|s| s * s).sum();
        let rms = (sum_sq / samples.len() as f32).sqrt();
        let is_voice = rms > self.threshold;

        if is_voice {
            self.silence_frames = 0;
            if !self.speech_active {
                self.speech_active = true;
                self.speech_buffer.clear();
            }
            self.speech_buffer.extend_from_slice(samples);

            // Force a segment cut when speech continues too long without a pause.
            // This is essential for continuous speech (e.g. video lectures) where
            // silence-based segmentation never triggers.
            if self.speech_buffer.len() >= self.max_speech_samples {
                self.pending_segments.push(self.speech_buffer.clone());
                self.speech_buffer.clear();
                // speech_active stays true — speech is still ongoing
            }
        } else if self.speech_active {
            self.silence_frames += 1;
            self.speech_buffer.extend_from_slice(samples);

            if self.silence_frames >= self.max_silence_frames {
                if self.speech_buffer.len() >= self.min_speech_samples {
                    let trim_len = self.max_silence_frames * 512;
                    let seg_len = self.speech_buffer.len().saturating_sub(trim_len);
                    if seg_len >= self.min_speech_samples {
                        self.pending_segments.push(self.speech_buffer[..seg_len].to_vec());
                    }
                }
                self.speech_active = false;
                self.speech_buffer.clear();
                self.silence_frames = 0;
            }
        }
    }

    pub fn is_speech(&self) -> bool {
        self.speech_active
    }

    pub fn has_segment(&self) -> bool {
        !self.pending_segments.is_empty()
    }

    pub fn take_segment(&mut self) -> Option<SpeechSegment> {
        if self.pending_segments.is_empty() {
            None
        } else {
            Some(SpeechSegment { samples: self.pending_segments.remove(0) })
        }
    }

    pub fn flush(&mut self) {
        if self.speech_active && self.speech_buffer.len() >= self.min_speech_samples {
            self.pending_segments.push(self.speech_buffer.clone());
        }
        self.speech_active = false;
        self.speech_buffer.clear();
        self.silence_frames = 0;
    }

    pub fn reset(&mut self) {
        self.speech_active = false;
        self.speech_buffer.clear();
        self.silence_frames = 0;
        self.pending_segments.clear();
    }
}

pub struct SpeechSegment {
    samples: Vec<f32>,
}

impl SpeechSegment {
    pub fn samples(&self) -> &[f32] {
        &self.samples
    }
}
