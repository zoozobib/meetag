//! Audio processing module for noise reduction and loudness normalization.
//!
//! This module provides professional-grade audio processing to improve ASR accuracy:
//! - **RNNoise**: Neural network-based noise suppression
//! - **EBU R128**: Broadcast-standard loudness normalization

use ebur128::{EbuR128, Mode};
use nnnoiseless::DenoiseState;
use std::sync::Mutex;

/// Audio processing pipeline with noise reduction and loudness normalization.
pub struct AudioProcessor {
    /// RNNoise denoiser state (48kHz, mono)
    denoiser: Mutex<Box<DenoiseState<'static>>>,
    /// EBU R128 loudness meter
    loudness_meter: Mutex<EbuR128>,
    /// Sample rate for processing
    sample_rate: u32,
}

/// Result of noise reduction processing
#[derive(Debug)]
pub struct DenoiseResult {
    pub samples: Vec<f32>,
    pub rms_before: f32,
    pub rms_after: f32,
    pub noise_reduction_db: f32,
}

/// Result of loudness normalization
#[derive(Debug)]
pub struct LoudnessResult {
    pub samples: Vec<i16>,
    pub loudness_lufs: f64,
    pub gain_applied_db: f64,
}

impl AudioProcessor {
    /// Create a new audio processor.
    ///
    /// # Arguments
    /// * `sample_rate` - Sample rate for EBU R128 meter (typically 16000 for ASR)
    pub fn new(sample_rate: u32) -> Self {
        println!(
            "🔧 [AUDIO_PROCESSOR] Initializing with sample_rate={}Hz",
            sample_rate
        );

        // Create RNNoise denoiser (always operates at 48kHz internally)
        let denoiser = DenoiseState::new();
        println!("✅ [AUDIO_PROCESSOR] RNNoise denoiser initialized");

        // Create EBU R128 loudness meter
        let loudness_meter = EbuR128::new(1, sample_rate, Mode::I | Mode::LRA | Mode::SAMPLE_PEAK)
            .expect("Failed to create EBU R128 meter");
        println!("✅ [AUDIO_PROCESSOR] EBU R128 meter initialized (target: -23 LUFS)");

        Self {
            denoiser: Mutex::new(denoiser),
            loudness_meter: Mutex::new(loudness_meter),
            sample_rate,
        }
    }

    /// Apply RNNoise neural network noise reduction to audio samples.
    ///
    /// Note: RNNoise expects 48kHz audio. This function handles resampling internally
    /// for 16kHz input, but for simplicity we assume input is already normalized f32.
    ///
    /// # Arguments
    /// * `samples_f32` - Normalized f32 samples in range [-1.0, 1.0]
    ///
    /// # Returns
    /// * `DenoiseResult` with processed samples and metrics
    pub fn denoise(&self, samples_f32: &[f32]) -> DenoiseResult {
        let rms_before = Self::calculate_rms(samples_f32);
        println!(
            "🔇 [AUDIO_PROCESSOR] Denoise START: {} samples, RMS={:.4}",
            samples_f32.len(),
            rms_before
        );

        // RNNoise processes in frames of 480 samples (10ms at 48kHz)
        // For 16kHz input, we need to process in chunks of 160 samples (10ms at 16kHz)
        // and the denoiser will handle the internal 3x upsampling

        let frame_size = DenoiseState::FRAME_SIZE; // 480 samples for 48kHz
        let mut denoiser = self.denoiser.lock().unwrap();

        // For 16kHz audio, we work in 160-sample chunks and let RNNoise handle it
        // Actually nnnoiseless expects 480 samples at 48kHz, so we need to resample
        // For simplicity, we'll process what we have and pad if needed

        let mut output = Vec::with_capacity(samples_f32.len());
        let mut input_buffer = Vec::with_capacity(frame_size);
        let mut output_buffer = vec![0.0f32; frame_size];

        // Simple 3x upsampling for 16kHz -> 48kHz (linear interpolation)
        let upsampled: Vec<f32> = if self.sample_rate == 16000 {
            samples_f32
                .windows(2)
                .flat_map(|w| {
                    let a = w[0];
                    let b = w[1];
                    [a, a + (b - a) / 3.0, a + 2.0 * (b - a) / 3.0]
                })
                .chain(std::iter::once(*samples_f32.last().unwrap_or(&0.0)))
                .collect()
        } else {
            samples_f32.to_vec()
        };

        // Process in 480-sample frames
        for chunk in upsampled.chunks(frame_size) {
            input_buffer.clear();
            input_buffer.extend_from_slice(chunk);

            // Pad with zeros if needed
            while input_buffer.len() < frame_size {
                input_buffer.push(0.0);
            }

            // Apply RNNoise
            denoiser.process_frame(&mut output_buffer, &input_buffer);
            output.extend_from_slice(&output_buffer[..chunk.len()]);
        }

        // Downsample back to 16kHz if we upsampled
        let final_output: Vec<f32> = if self.sample_rate == 16000 {
            output.iter().step_by(3).copied().collect()
        } else {
            output
        };

        // Trim or pad to match original length
        let mut result: Vec<f32> = final_output.into_iter().take(samples_f32.len()).collect();
        while result.len() < samples_f32.len() {
            result.push(0.0);
        }

        let rms_after = Self::calculate_rms(&result);
        let noise_reduction_db = if rms_before > 0.0 && rms_after > 0.0 {
            20.0 * (rms_before / rms_after).log10()
        } else {
            0.0
        };

        println!(
            "✅ [AUDIO_PROCESSOR] Denoise COMPLETE: RMS {:.4} -> {:.4} (reduction: {:.1} dB)",
            rms_before, rms_after, noise_reduction_db
        );

        DenoiseResult {
            samples: result,
            rms_before,
            rms_after,
            noise_reduction_db,
        }
    }

    /// Measure and normalize audio loudness to EBU R128 standard (-23 LUFS).
    ///
    /// # Arguments
    /// * `samples_i16` - PCM16 audio samples
    ///
    /// # Returns
    /// * `LoudnessResult` with normalized samples and metrics
    pub fn normalize_loudness(&self, samples_i16: &[i16]) -> LoudnessResult {
        const TARGET_LUFS: f64 = -23.0; // EBU R128 broadcast standard

        // Convert i16 to f32 for measurement
        let samples_f32: Vec<f32> = samples_i16.iter().map(|&s| s as f32 / 32768.0).collect();

        // Measure current loudness
        let mut meter = self.loudness_meter.lock().unwrap();

        // Reset meter for new measurement
        meter.reset();

        // Add samples to meter
        if let Err(e) = meter.add_frames_f32(&samples_f32) {
            eprintln!("⚠️ [AUDIO_PROCESSOR] Failed to measure loudness: {:?}", e);
            return LoudnessResult {
                samples: samples_i16.to_vec(),
                loudness_lufs: -70.0,
                gain_applied_db: 0.0,
            };
        }

        // Get integrated loudness
        let loudness_lufs = meter.loudness_global().unwrap_or(-70.0);
        println!(
            "📊 [AUDIO_PROCESSOR] Loudness measurement: {:.1} LUFS (target: {:.1} LUFS)",
            loudness_lufs, TARGET_LUFS
        );

        // Calculate required gain
        let gain_db = TARGET_LUFS - loudness_lufs;

        // Limit gain to prevent clipping and excessive amplification
        let gain_db_clamped = gain_db.clamp(-12.0, 24.0);

        if (gain_db - gain_db_clamped).abs() > 0.1 {
            println!(
                "⚠️ [AUDIO_PROCESSOR] Gain clamped: {:.1} dB -> {:.1} dB",
                gain_db, gain_db_clamped
            );
        }

        // Apply gain
        let gain_linear = 10.0_f64.powf(gain_db_clamped / 20.0);
        let normalized: Vec<i16> = samples_i16
            .iter()
            .map(|&s| {
                let amplified = (s as f64 * gain_linear).round();
                amplified.clamp(-32768.0, 32767.0) as i16
            })
            .collect();

        println!(
            "✅ [AUDIO_PROCESSOR] Loudness normalized: {:.1} LUFS -> ~{:.1} LUFS (gain: {:.1} dB)",
            loudness_lufs,
            loudness_lufs + gain_db_clamped,
            gain_db_clamped
        );

        LoudnessResult {
            samples: normalized,
            loudness_lufs,
            gain_applied_db: gain_db_clamped,
        }
    }

    /// Calculate RMS (Root Mean Square) of audio samples.
    fn calculate_rms(samples: &[f32]) -> f32 {
        if samples.is_empty() {
            return 0.0;
        }
        let sum_sq: f32 = samples.iter().map(|&s| s * s).sum();
        (sum_sq / samples.len() as f32).sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rms_calculation() {
        let samples = vec![0.5, -0.5, 0.5, -0.5];
        let rms = AudioProcessor::calculate_rms(&samples);
        assert!((rms - 0.5).abs() < 0.001);
    }
}
