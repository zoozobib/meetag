use anyhow::{Context, Result};
use std::path::PathBuf;
use tauri::Manager;
use webrtc_vad::{Vad as WebRtcVadImp, VadMode};

/// Unified interface for VAD backends
pub trait VadEngine: Send + Sync {
    /// Process a frame of audio.
    /// Returns true if voice is detected, false otherwise.
    fn is_voice_segment(&mut self, audio_frame: &[i16]) -> Result<bool>;

    /// Reset internal state if applicable
    fn reset(&mut self);
}

// ─── WebRTC VAD ───────────────────────────────────────────────────────

/// Wrapper for WebRTC VAD (traditional signal-processing)
pub struct WebRtcVadWrapper {
    inner: WebRtcVadImp,
}

// SAFETY: WebRtcVadImp wraps a raw pointer to Fvad.
// Since we own it and only access it via mutable reference (exclusive access),
// it is safe to send between threads.
unsafe impl Send for WebRtcVadWrapper {}
unsafe impl Sync for WebRtcVadWrapper {}

impl WebRtcVadWrapper {
    pub fn new() -> Self {
        Self {
            inner: WebRtcVadImp::new_with_rate_and_mode(
                webrtc_vad::SampleRate::Rate16kHz,
                VadMode::VeryAggressive,
            ),
        }
    }
}

impl VadEngine for WebRtcVadWrapper {
    fn is_voice_segment(&mut self, audio_frame: &[i16]) -> Result<bool> {
        match self.inner.is_voice_segment(audio_frame) {
            Ok(is_voice) => Ok(is_voice),
            Err(_) => Ok(false),
        }
    }

    fn reset(&mut self) {
        self.inner = WebRtcVadImp::new_with_rate_and_mode(
            webrtc_vad::SampleRate::Rate16kHz,
            VadMode::VeryAggressive,
        );
    }
}

// ─── Sherpa-ONNX Silero VAD ───────────────────────────────────────────

/// Wrapper for Silero VAD via sherpa-onnx (neural network, high accuracy).
/// Uses the same ONNX Runtime as diarization — zero version conflict.
pub struct SherpaVadWrapper {
    vad: sherpa_onnx::VoiceActivityDetector,
}

unsafe impl Send for SherpaVadWrapper {}
unsafe impl Sync for SherpaVadWrapper {}

impl SherpaVadWrapper {
    pub fn new(model_path: PathBuf, threshold: f32) -> Result<Self> {
        let config = sherpa_onnx::VadModelConfig {
            silero_vad: sherpa_onnx::SileroVadModelConfig {
                model: Some(
                    model_path
                        .to_str()
                        .context("Invalid model path")?
                        .to_string(),
                ),
                threshold,
                min_silence_duration: 0.25, // 250ms silence to end speech
                min_speech_duration: 0.1,   // 100ms minimum speech
                window_size: 512,           // Silero default window
                max_speech_duration: 30.0,  // 30s max segment
            },
            sample_rate: 16000,
            num_threads: 1,
            provider: Some("cpu".to_string()),
            debug: false,
            ..Default::default()
        };

        let vad = sherpa_onnx::VoiceActivityDetector::create(&config, 30.0)
            .context("Failed to create Sherpa Silero VAD")?;

        println!("✅ [VAD] Sherpa Silero VAD initialized (threshold={:.2})", threshold);
        Ok(Self { vad })
    }
}

impl VadEngine for SherpaVadWrapper {
    fn is_voice_segment(&mut self, audio_frame: &[i16]) -> Result<bool> {
        // Convert i16 → f32 (sherpa-onnx VAD expects f32 samples)
        let f32_samples: Vec<f32> = audio_frame
            .iter()
            .map(|&s| s as f32 / 32768.0)
            .collect();

        self.vad.accept_waveform(&f32_samples);
        Ok(self.vad.detected())
    }

    fn reset(&mut self) {
        self.vad.reset();
    }
}

// ─── Factory ──────────────────────────────────────────────────────────

pub fn create_vad(
    app: &tauri::AppHandle,
    backend: crate::settings::VadBackend,
    threshold: f32,
) -> Result<Box<dyn VadEngine>> {
    match backend {
        crate::settings::VadBackend::WebRtc => Ok(Box::new(WebRtcVadWrapper::new())),
        crate::settings::VadBackend::Silero => {
            // Use sherpa-onnx built-in Silero VAD (shares ORT with diarization)
            let model_path = app
                .path()
                .resolve(
                    "resources/silero_vad.onnx",
                    tauri::path::BaseDirectory::Resource,
                )
                .map_err(|e| anyhow::anyhow!("Failed to resolve VAD model path: {}", e))?;

            if !model_path.exists() {
                return Err(anyhow::anyhow!(
                    "Silero VAD model not found at: {}",
                    model_path.display()
                ));
            }

            Ok(Box::new(SherpaVadWrapper::new(model_path, threshold)?))
        }
    }
}
