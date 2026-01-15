//! Whisper-rs integration module for speech-to-text transcription.
//!
//! This module provides a thread-safe singleton (`WhisperManager`) for managing
//! the Whisper model and performing transcription with Metal GPU acceleration.

use anyhow::{Context, Result};
use once_cell::sync::OnceCell;
use std::path::Path;
use std::sync::Mutex;
use std::time::Instant;
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

/// Global singleton for WhisperContext
static WHISPER_MANAGER: OnceCell<WhisperManager> = OnceCell::new();

/// Thread-safe wrapper around WhisperContext
pub struct WhisperManager {
    ctx: Mutex<WhisperContext>,
}

/// Result of a transcription operation
#[derive(Debug, Clone)]
pub struct TranscriptionResult {
    pub text: String,
    pub segments: Vec<TranscriptionSegment>,
    pub inference_time_ms: u64,
}

/// Individual segment with timing and confidence
#[derive(Debug, Clone)]
pub struct TranscriptionSegment {
    pub text: String,
    pub start_ms: i64,
    pub end_ms: i64,
    pub avg_logprob: f32,
}

impl WhisperManager {
    /// Initialize the global WhisperManager with the given model path.
    /// This should be called once at application startup.
    pub fn init(model_path: &Path) -> Result<()> {
        println!("🎙️ [WHISPER] Initializing WhisperManager...");
        let start = Instant::now();

        // Log model path
        println!("📂 [WHISPER] Loading model from: {}", model_path.display());

        // Create context parameters with GPU acceleration
        let params = WhisperContextParameters::default();

        // Load model
        let ctx = WhisperContext::new_with_params(
            model_path.to_str().context("Invalid model path")?,
            params,
        )
        .map_err(|e| anyhow::anyhow!("Failed to load Whisper model: {}", e))?;

        let load_time = start.elapsed();
        println!(
            "✅ [WHISPER] Model loaded successfully in {:.2}s",
            load_time.as_secs_f32()
        );

        // Store in global singleton
        WHISPER_MANAGER
            .set(WhisperManager {
                ctx: Mutex::new(ctx),
            })
            .map_err(|_| anyhow::anyhow!("WhisperManager already initialized"))?;

        Ok(())
    }

    /// Get the global WhisperManager instance.
    /// Panics if not initialized.
    pub fn get() -> &'static WhisperManager {
        WHISPER_MANAGER
            .get()
            .expect("WhisperManager not initialized. Call WhisperManager::init() first.")
    }

    /// Check if the WhisperManager has been initialized.
    pub fn is_initialized() -> bool {
        WHISPER_MANAGER.get().is_some()
    }

    /// Transcribe audio samples.
    ///
    /// # Arguments
    /// * `samples` - PCM16 audio samples at 16kHz
    /// * `language` - Language code (e.g., "zh", "en", "auto")
    /// * `initial_prompt` - Optional prompt to guide transcription
    ///
    /// # Returns
    /// * `TranscriptionResult` with text and segments
    pub fn transcribe(
        &self,
        samples: &[i16],
        language: &str,
        initial_prompt: Option<&str>,
    ) -> Result<TranscriptionResult> {
        let start = Instant::now();
        let duration_s = samples.len() as f32 / 16000.0;

        println!(
            "🎙️ [WHISPER] Starting transcription: {:.2}s of audio, lang={}",
            duration_s, language
        );

        // Convert i16 samples to f32 (whisper-rs expects normalized f32)
        let samples_f32: Vec<f32> = samples.iter().map(|&s| s as f32 / 32768.0).collect();

        // Configure parameters
        let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });

        // Language settings
        if language != "auto" {
            params.set_language(Some(language));
        }
        params.set_translate(false);

        // Set prompt if provided
        if let Some(prompt) = initial_prompt {
            params.set_initial_prompt(prompt);
        }

        // Performance settings
        params.set_n_threads(4);
        params.set_print_progress(false);
        params.set_print_realtime(false);
        params.set_print_timestamps(false);
        params.set_print_special(false);

        // Acquire lock and run inference
        let ctx = self
            .ctx
            .lock()
            .map_err(|e| anyhow::anyhow!("Lock poisoned: {}", e))?;

        // Create a new state for this transcription
        let mut state = ctx
            .create_state()
            .map_err(|e| anyhow::anyhow!("Failed to create state: {}", e))?;

        // Run inference
        state
            .full(params, &samples_f32)
            .map_err(|e| anyhow::anyhow!("Whisper inference failed: {}", e))?;

        // Extract results
        let num_segments = state
            .full_n_segments()
            .map_err(|e| anyhow::anyhow!("Failed to get segment count: {}", e))?;

        let mut segments = Vec::with_capacity(num_segments as usize);
        let mut full_text = String::new();

        for i in 0..num_segments {
            let text = state
                .full_get_segment_text(i)
                .map_err(|e| anyhow::anyhow!("Failed to get segment text: {}", e))?;

            let start_t = state
                .full_get_segment_t0(i)
                .map_err(|e| anyhow::anyhow!("Failed to get segment start: {}", e))?;
            let end_t = state
                .full_get_segment_t1(i)
                .map_err(|e| anyhow::anyhow!("Failed to get segment end: {}", e))?;

            // Note: whisper-rs doesn't expose avg_logprob directly through public API
            // We set a default value; confidence filtering can be done at higher level if needed
            let avg_logprob = -0.5; // Default moderate confidence

            full_text.push_str(&text);
            segments.push(TranscriptionSegment {
                text,
                start_ms: (start_t * 10) as i64, // whisper timestamps are in 10ms units
                end_ms: (end_t * 10) as i64,
                avg_logprob,
            });
        }

        let inference_time = start.elapsed();
        let inference_time_ms = inference_time.as_millis() as u64;

        println!(
            "✅ [WHISPER] Transcription complete: {} segments, {:.2}s inference ({:.1}x realtime)",
            segments.len(),
            inference_time.as_secs_f32(),
            duration_s / inference_time.as_secs_f32()
        );

        if !full_text.trim().is_empty() {
            println!("📝 [WHISPER] Result: \"{}\"", full_text.trim());
        }

        Ok(TranscriptionResult {
            text: full_text.trim().to_string(),
            segments,
            inference_time_ms,
        })
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_silence_transcription() {
        // Test with silence - should return empty
        // This is a placeholder; real tests would need model initialization
    }
}
