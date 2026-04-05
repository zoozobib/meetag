//! Speaker Diarization module.
//!
//! This module provides functionality to extract speaker embeddings using an ONNX model
//! and group them into different speaker identities using clustering.

use anyhow::Result;
use ndarray::{Array2, Axis};
use once_cell::sync::OnceCell;
use ort::session::Session;
use ort::value::Value;
use rustfft::{num_complex::Complex, FftPlanner};
use std::path::Path;
use std::sync::Mutex;

/// Global singleton for SpeakerExtractor
static SPEAKER_EXTRACTOR: OnceCell<SpeakerExtractor> = OnceCell::new();

pub struct SpeakerExtractor {
    session: Mutex<Session>,
    input_name: String,
}

impl SpeakerExtractor {
    /// Initialize the global SpeakerExtractor.
    pub fn init(model_path: &Path) -> Result<()> {
        println!("🚀 [DIARIZATION] Initializing SpeakerExtractor...");

        let session = Session::builder()?
            .commit_from_file(model_path)
            .map_err(|e| anyhow::anyhow!("Failed to load speaker model: {}", e))?;

        // Dynamically get the input node name and shape from the model
        let input_info = session
            .inputs()
            .first()
            .ok_or_else(|| anyhow::anyhow!("Model has no input nodes"))?;
        let input_name = input_info.name().to_string();

        println!("🔍 [DIARIZATION] Detected input node name: {}", input_name);

        SPEAKER_EXTRACTOR
            .set(SpeakerExtractor {
                session: Mutex::new(session),
                input_name,
            })
            .map_err(|_| anyhow::anyhow!("SpeakerExtractor already initialized"))?;

        println!(
            "✅ [DIARIZATION] Speaker model loaded successfully from: {}",
            model_path.display()
        );
        Ok(())
    }

    /// Get the global SpeakerExtractor instance.
    pub fn get() -> &'static SpeakerExtractor {
        SPEAKER_EXTRACTOR
            .get()
            .expect("SpeakerExtractor not initialized. Call SpeakerExtractor::init() first.")
    }

    /// Extract speaker embedding from PCM16 audio samples.
    /// Samples must be 16kHz, mono.
    pub fn extract_embedding(&self, samples: &[i16]) -> Result<Vec<f32>> {
        // 1. MINIMUM LENGTH CHECK
        // 16kHz * 0.2s = 3200 samples.
        if samples.len() < 3200 {
            return Err(anyhow::anyhow!(
                "Audio segment too short for embedding extraction ({} samples < 3200)",
                samples.len()
            ));
        }

        // 2. PRE-PROCESSING: Convert raw audio to Mel-spectrogram [1, time, 80]
        let samples_f32: Vec<f32> = samples.iter().map(|&s| s as f32 / 32768.0).collect();
        let mel_spec = compute_mel_spectrogram(&samples_f32, 16000, 80)?;

        let time_steps = mel_spec.len_of(Axis(0));
        let input_shape = [1, time_steps, 80];

        let input_value =
            Value::from_array((input_shape, mel_spec.into_raw_vec())).map_err(|e| {
                eprintln!("❌ [DIARIZATION] Failed to create input tensor: {}", e);
                e
            })?;

        let mut session = self
            .session
            .lock()
            .map_err(|_| anyhow::anyhow!("Lock poisoned"))?;

        // 3. INFERENCE
        let inputs = vec![(&self.input_name, input_value)];
        let outputs = session.run(inputs).map_err(|e| {
            eprintln!(
                "❌ [DIARIZATION] Session run failed for model {}: {}",
                self.input_name, e
            );
            e
        })?;

        // In ort 2.0, SessionOutputs doesn't have a direct index accessor.
        // We use the output name. Most CAM++ models use "output" or a specific name.
        let output_value = outputs
            .get("output")
            .or_else(|| outputs.get("embedding"))
            .ok_or_else(|| {
                eprintln!("❌ [DIARIZATION] Model produced no outputs or output name mismatch");
                anyhow::anyhow!("Model produced no outputs")
            })?;

        // 4. POST-PROCESSING
        let (_, embedding_slice) = output_value.try_extract_tensor::<f32>().map_err(|e| {
            eprintln!(
                "❌ [DIARIZATION] Failed to extract tensor from output: {}",
                e
            );
            e
        })?;

        let embedding: Vec<f32> = embedding_slice.to_vec();
        if embedding.is_empty() {
            return Err(anyhow::anyhow!("Extracted embedding is empty"));
        }

        let norm = embedding.iter().map(|&x| x * x).sum::<f32>().sqrt();
        if norm > 1e-6 {
            Ok(embedding.into_iter().map(|x| x / norm).collect())
        } else {
            eprintln!("⚠️ [DIARIZATION] Embedding norm too low: {}", norm);
            Ok(embedding)
        }
    }
}

fn compute_mel_spectrogram(
    samples: &[f32],
    sample_rate: usize,
    n_mels: usize,
) -> Result<Array2<f32>> {
    let fft_size = 512;
    let hop_size = 160; // 10ms
    let win_len = 512;

    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(fft_size);

    let num_frames = (samples.len() - win_len) / hop_size + 1;
    let mut mel_spec = Array2::<f32>::zeros((num_frames, n_mels));

    // Precompute Mel filterbank
    let mel_filters = create_mel_filterbank(sample_rate, fft_size, n_mels);

    for (i, frame_start) in (0..num_frames).map(|i| i * hop_size).enumerate() {
        let mut window = vec![0.0f32; fft_size];
        for j in 0..win_len {
            // Hann window
            let multiplier = 0.5
                * (1.0 - (2.0 * std::f32::consts::PI * j as f32 / (win_len as f32 - 1.0)).cos());
            window[j] = samples[frame_start + j] * multiplier;
        }

        let mut complex_window: Vec<Complex<f32>> =
            window.iter().map(|&x| Complex { re: x, im: 0.0 }).collect();

        fft.process(&mut complex_window);

        // Power spectrum
        let power_spec: Vec<f32> = complex_window
            .iter()
            .take(fft_size / 2 + 1)
            .map(|c| (c.re * c.re + c.im * c.im) / (fft_size as f32))
            .collect();

        // Apply Mel filters
        for (m, filter) in mel_filters.iter().enumerate() {
            let mut mel_val = 0.0;
            for (k, &weight) in filter.iter().enumerate() {
                if k < power_spec.len() {
                    mel_val += weight * power_spec[k];
                }
            }
            mel_spec[[i, m]] = mel_val.max(1e-10).ln(); // Log-mel
        }
    }

    Ok(mel_spec)
}

fn create_mel_filterbank(sample_rate: usize, fft_size: usize, n_mels: usize) -> Vec<Vec<f32>> {
    let min_mel = 0.0;
    let max_mel = 2595.0 * (sample_rate as f32 / 2.0).log10(); // Simplified mel scale

    let mel_points = (0..=n_mels)
        .map(|i| min_mel + i as f32 * (max_mel - min_mel) / n_mels as f32)
        .collect::<Vec<_>>();

    let hz_points: Vec<f32> = mel_points
        .iter()
        .map(|&m| 700.0 * (10.0f32.powf(m / 2595.0) - 1.0))
        .collect();

    let bin_size = sample_rate as f32 / (2.0 * fft_size as f32);
    let bins = hz_points
        .iter()
        .map(|&hz| ((hz / bin_size).round() as usize).min(fft_size / 2))
        .collect::<Vec<_>>();

    let mut filters = vec![vec![0.0f32; fft_size / 2 + 1]; n_mels];
    for m in 0..n_mels {
        let left = bins[m];
        let center = bins[m + 1];
        let right = if m < n_mels - 1 {
            bins[m + 2]
        } else {
            bins[m + 1] + 1
        };

        for k in left..center {
            filters[m][k] = (k as f32 - left as f32) / (center as f32 - left as f32);
        }
        for k in center..right {
            filters[m][k] = (right as f32 - k as f32) / (right as f32 - center as f32);
        }
    }
    filters
}

/// Simple Diarization Engine to handle clustering of embeddings in real-time.
pub struct DiarizationEngine {
    centroids: Mutex<Vec<Vec<f32>>>,
}

impl DiarizationEngine {
    pub fn new() -> Self {
        Self {
            centroids: Mutex::new(Vec::new()),
        }
    }

    pub fn label_embedding(&self, embedding: &[f32], threshold: f32) -> usize {
        let mut centroids = self.centroids.lock().unwrap();
        let mut best_speaker = None;
        let mut max_sim = -1.0f32;

        for (idx, centroid) in centroids.iter().enumerate() {
            let sim = cosine_similarity(embedding, centroid);
            if sim > max_sim {
                max_sim = sim;
                best_speaker = Some(idx);
            }
        }

        if let Some(speaker_id) = best_speaker {
            if max_sim > threshold {
                let learning_rate = 0.1f32;
                let centroid = &mut centroids[speaker_id];
                for i in 0..centroid.len() {
                    centroid[i] =
                        centroid[i] * (1.0 - learning_rate) + embedding[i] * learning_rate;
                }
                return speaker_id;
            }
        }

        centroids.push(embedding.to_vec());
        centroids.len() - 1
    }

    pub fn clear(&self) {
        self.centroids.lock().unwrap().clear();
    }
}

static DIARIZATION_ENGINE: OnceCell<DiarizationEngine> = OnceCell::new();

impl DiarizationEngine {
    pub fn get() -> &'static DiarizationEngine {
        DIARIZATION_ENGINE.get_or_init(|| DiarizationEngine::new())
    }
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
    let norm_a = a.iter().map(|&x| x * x).sum::<f32>().sqrt();
    let norm_b = b.iter().map(|&x| x * x).sum::<f32>().sqrt();
    if norm_a > 1e-6 && norm_b > 1e-6 {
        dot / (norm_a * norm_b)
    } else {
        0.0
    }
}
