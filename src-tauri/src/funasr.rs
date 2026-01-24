//! FunASR (SenseVoice) integration module.
//!
//! This module provides a thread-safe singleton (`SenseVoiceManager`) for managing
//! the SenseVoice model via sherpa-rs.

use anyhow::{Context, Result};
use once_cell::sync::OnceCell;
use sherpa_rs::sense_voice::{SenseVoiceConfig, SenseVoiceRecognizer};
use std::path::Path;
use std::sync::Mutex;
use std::time::Instant;

/// Global singleton for SenseVoiceManager
static SENSE_VOICE_MANAGER: OnceCell<SenseVoiceManager> = OnceCell::new();

/// Thread-safe wrapper around SenseVoiceRecognizer
pub struct SenseVoiceManager {
    recognizer: Mutex<SenseVoiceRecognizer>,
}

/// Result of a transcription operation
#[derive(Debug, Clone)]
pub struct FunAsrTranscriptionResult {
    pub text: String,
    pub inference_time_ms: u64,
}

impl SenseVoiceManager {
    /// Initialize the global SenseVoiceManager.
    pub fn init(base_resource_dir: &Path) -> Result<()> {
        println!("🚀 [FUNASR] Initializing SenseVoiceManager...");
        let start = Instant::now();

        // Construct paths relative to resource dir
        // Expected structure:
        // resources/
        //   sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/
        //     model.onnx
        //     tokens.txt
        let model_dir =
            base_resource_dir.join("sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17");
        let model_path = model_dir.join("model.int8.onnx");
        let tokens_path = model_dir.join("tokens.txt");

        if !model_path.exists() || !tokens_path.exists() {
            return Err(anyhow::anyhow!(
                "SenseVoice model files not found at: {}",
                model_dir.display()
            ));
        }

        println!("📂 [FUNASR] Loading model from: {}", model_dir.display());

        let config = SenseVoiceConfig {
            model: model_path.to_string_lossy().to_string(),
            tokens: tokens_path.to_string_lossy().to_string(),
            language: "".to_string(), // Auto-detect
            use_itn: true,
            provider: None, // CPU
            num_threads: Some(4),
            debug: false,
        };

        let recognizer = SenseVoiceRecognizer::new(config)
            .map_err(|e| anyhow::anyhow!("Failed to create SenseVoice recognizer: {}", e))?;

        let load_time = start.elapsed();
        println!(
            "✅ [FUNASR] Model loaded successfully in {:.2}s",
            load_time.as_secs_f32()
        );

        // Store in global singleton
        SENSE_VOICE_MANAGER
            .set(SenseVoiceManager {
                recognizer: Mutex::new(recognizer),
            })
            .map_err(|_| anyhow::anyhow!("SenseVoiceManager already initialized"))?;

        Ok(())
    }

    /// Get the global SenseVoiceManager instance.
    /// Panics if not initialized.
    pub fn get() -> &'static SenseVoiceManager {
        SENSE_VOICE_MANAGER
            .get()
            .expect("SenseVoiceManager not initialized. Call SenseVoiceManager::init() first.")
    }

    /// Check if initialized
    pub fn is_initialized() -> bool {
        SENSE_VOICE_MANAGER.get().is_some()
    }

    /// Transcribe audio samples.
    ///
    /// # Arguments
    /// * `samples` - PCM16 audio samples at 16kHz
    ///
    /// # Returns
    /// * `FunAsrTranscriptionResult`
    pub fn transcribe(&self, samples: &[i16]) -> Result<FunAsrTranscriptionResult> {
        let start = Instant::now();
        let duration_s = samples.len() as f32 / 16000.0;

        // println!(
        //     "🚀 [FUNASR] Starting transcription: {:.2}s of audio",
        //     duration_s
        // );

        // Convert i16 samples to f32 (normalized)
        let samples_f32: Vec<f32> = samples.iter().map(|&s| s as f32 / 32768.0).collect();

        // Acquire lock and run inference
        let mut recognizer = self
            .recognizer
            .lock()
            .map_err(|e| anyhow::anyhow!("Lock poisoned: {}", e))?;

        let result = recognizer.transcribe(16000, &samples_f32);

        let inference_time = start.elapsed();
        let inference_time_ms = inference_time.as_millis() as u64;

        if !result.text.trim().is_empty() {
            println!(
                "✅ [FUNASR] Transcription complete: {:.2}s inference ({:.1}x realtime) -> \"{}\"",
                inference_time.as_secs_f32(),
                duration_s / inference_time.as_secs_f32(),
                result.text.trim()
            );
        }

        Ok(FunAsrTranscriptionResult {
            text: result.text.trim().to_string(),
            inference_time_ms,
        })
    }
}
