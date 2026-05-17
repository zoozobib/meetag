use anyhow::Result;
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

pub fn create_vad(
    _app: &tauri::AppHandle,
    backend: crate::settings::VadBackend,
    _threshold: f32,
) -> Result<Box<dyn VadEngine>> {
    match backend {
        crate::settings::VadBackend::WebRtc => Ok(Box::new(WebRtcVadWrapper::new())),
        crate::settings::VadBackend::Silero => {
            // Silero VAD (ten-vad-rs) was removed due to ort crate version conflict
            // with sherpa-onnx. Fall back to WebRTC VAD.
            println!("⚠️ [VAD] Silero VAD unavailable (ort conflict), falling back to WebRTC VAD");
            Ok(Box::new(WebRtcVadWrapper::new()))
        }
    }
}
