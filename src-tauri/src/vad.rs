use anyhow::{Context, Result};
use std::path::PathBuf;
use tauri::Manager;
use ten_vad_rs::TenVad;
use webrtc_vad::{Vad as WebRtcVadImp, VadMode};

/// Unified interface for VAD backends
pub trait VadEngine: Send + Sync {
    /// Process a frame of audio.
    /// Returns true if voice is detected, false otherwise.
    fn is_voice_segment(&mut self, audio_frame: &[i16]) -> Result<bool>;

    /// Reset internal state if applicable
    fn reset(&mut self);
}

/// Wrapper for WebRTC VAD
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
        // WebRTC VAD is strict about frame sizes (10/20/30ms).
        match self.inner.is_voice_segment(audio_frame) {
            Ok(is_voice) => Ok(is_voice),
            Err(_) => {
                // If frame size is wrong, return false to avoid crash
                Ok(false)
            }
        }
    }

    fn reset(&mut self) {
        self.inner = WebRtcVadImp::new_with_rate_and_mode(
            webrtc_vad::SampleRate::Rate16kHz,
            VadMode::VeryAggressive,
        );
    }
}

/// Wrapper for Ten (Silero) VAD
pub struct TenVadWrapper {
    inner: TenVad,
    threshold: f32,
}

// TenVad is likely Send, but let's ensure it.
// If ten-vad-rs doesn't impl Send, we might need a wrapper or Mutex.
// Assuming it does (standard Rust struct).

impl TenVadWrapper {
    pub fn new(model_path: PathBuf, threshold: f32) -> Result<Self> {
        let vad = TenVad::new(model_path.to_str().context("Invalid path")?, 16000)
            .map_err(|e| anyhow::anyhow!("Failed to init TenVad: {:?}", e))?;

        Ok(Self {
            inner: vad,
            threshold,
        })
    }
}

impl VadEngine for TenVadWrapper {
    fn is_voice_segment(&mut self, audio_frame: &[i16]) -> Result<bool> {
        match self.inner.process_frame(audio_frame) {
            Ok(probability) => Ok(probability >= self.threshold),
            Err(e) => {
                eprintln!("TenVad error: {:?}", e);
                Ok(false)
            }
        }
    }

    fn reset(&mut self) {
        self.inner.reset();
    }
}

pub fn create_vad(
    app: &tauri::AppHandle,
    backend: crate::settings::VadBackend,
    threshold: f32,
) -> Result<Box<dyn VadEngine>> {
    match backend {
        crate::settings::VadBackend::WebRtc => Ok(Box::new(WebRtcVadWrapper::new())),
        crate::settings::VadBackend::Silero => {
            // Locate model file using Tauri path resolver
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

            Ok(Box::new(TenVadWrapper::new(model_path, threshold)?))
        }
    }
}
