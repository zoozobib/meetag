//! FunASR (SenseVoice) integration module.
//!
//! This module provides a thread-safe singleton (`SenseVoiceManager`) for managing
//! the SenseVoice model via sherpa-onnx (migrated from sherpa-rs).

use anyhow::{Context, Result};
use once_cell::sync::OnceCell;
use sherpa_onnx::{OfflineRecognizer, OfflineRecognizerConfig, OfflineSenseVoiceModelConfig};
use std::path::Path;
use std::sync::Mutex;
use std::time::Instant;

/// Global singleton for SenseVoiceManager
static SENSE_VOICE_MANAGER: OnceCell<SenseVoiceManager> = OnceCell::new();

/// Thread-safe wrapper around OfflineRecognizer (SenseVoice)
pub struct SenseVoiceManager {
    recognizer: Mutex<OfflineRecognizer>,
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

        let mut config = OfflineRecognizerConfig::default();
        config.model_config.sense_voice = OfflineSenseVoiceModelConfig {
            model: Some(model_path.to_string_lossy().to_string()),
            language: Some("zh".to_string()),
            use_itn: true,
        };
        config.model_config.tokens = Some(tokens_path.to_string_lossy().to_string());
        config.model_config.num_threads = 4;
        config.model_config.debug = false;
        config.model_config.provider = Some("cpu".to_string());

        let recognizer = OfflineRecognizer::create(&config)
            .ok_or_else(|| anyhow::anyhow!("Failed to create SenseVoice recognizer. Check model paths."))?;

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

        // Convert i16 samples to f32 (normalized)
        let samples_f32: Vec<f32> = samples.iter().map(|&s| s as f32 / 32768.0).collect();

        // Acquire lock and run inference
        let recognizer = self
            .recognizer
            .lock()
            .map_err(|e| anyhow::anyhow!("Lock poisoned: {}", e))?;

        let stream = recognizer.create_stream();
        stream.accept_waveform(16000, &samples_f32);
        recognizer.decode(&stream);

        let result = stream.get_result()
            .ok_or_else(|| anyhow::anyhow!("SenseVoice returned no result"))?;

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
