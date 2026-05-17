//! Voice Activity Detection — thin wrappers around sherpa-onnx Silero VAD.
//!
//! The primary VAD consumer (asr.rs) uses sherpa_onnx::VoiceActivityDetector
//! directly for streaming segment detection. This module exists only for
//! the legacy settings plumbing and any non-streaming use cases.

pub use sherpa_onnx::{VadModelConfig, SileroVadModelConfig, VoiceActivityDetector};
