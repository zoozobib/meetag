//! Speaker Diarization module.
//!
//! This module provides functionality to extract speaker embeddings using an ONNX model
//! and group them into different speaker identities using clustering.

use anyhow::Result;
use once_cell::sync::OnceCell;
use ort::session::Session;
use ort::value::Value;
use std::path::Path;
use std::sync::Mutex;

/// Global singleton for SpeakerExtractor
static SPEAKER_EXTRACTOR: OnceCell<SpeakerExtractor> = OnceCell::new();

pub struct SpeakerExtractor {
    session: Mutex<Session>,
}

impl SpeakerExtractor {
    /// Initialize the global SpeakerExtractor.
    pub fn init(model_path: &Path) -> Result<()> {
        println!("🚀 [DIARIZATION] Initializing SpeakerExtractor...");

        // In ort 2.0, commit_from_file is often the method to load and finalize the session
        let session = Session::builder()?
            .commit_from_file(model_path)
            .map_err(|e| anyhow::anyhow!("Failed to load speaker model: {}", e))?;

        SPEAKER_EXTRACTOR
            .set(SpeakerExtractor {
                session: Mutex::new(session),
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
        // Convert i16 to f32 and normalize
        let samples_f32: Vec<f32> = samples.iter().map(|&s| s as f32 / 32768.0).collect();

        // CAM++ expects input shape [1, samples_len]
        // In ort 2.0, Value::from_array takes a tuple (shape, data)
        let input_shape = [1, samples_f32.len()];
        let input_value = Value::from_array((input_shape, samples_f32))?;

        let mut session = self
            .session
            .lock()
            .map_err(|_| anyhow::anyhow!("Lock poisoned"))?;

        // In ort 2.0, we pass a Vec of (name, value) tuples for session inputs
        // Most CAM++ models use "input" as the input node name.
        let inputs = vec![("input", input_value)];
        let outputs = session.run(inputs)?;

        // try_extract_tensor returns a tuple (&Shape, &[T])
        let (_, embedding_slice) = outputs[0].try_extract_tensor::<f32>()?;

        // Convert the slice to a Vec
        let embedding = embedding_slice.to_vec();

        // Normalize embedding (L2 norm) for cosine similarity
        let norm = embedding.iter().map(|&x| x * x).sum::<f32>().sqrt();
        if norm > 1e-6 {
            Ok(embedding.into_iter().map(|x| x / norm).collect())
        } else {
            Ok(embedding)
        }
    }
}

/// Simple Diarization Engine to handle clustering of embeddings in real-time.
pub struct DiarizationEngine {
    /// Stores the average embedding (centroid) for each discovered speaker.
    centroids: Mutex<Vec<Vec<f32>>>,
}

impl DiarizationEngine {
    pub fn new() -> Self {
        Self {
            centroids: Mutex::new(Vec::new()),
        }
    }

    /// Assign a label to a new embedding in real-time.
    /// Returns the speaker ID (0, 1, 2...).
    pub fn label_embedding(&self, embedding: &[f32], threshold: f32) -> usize {
        let mut centroids = self.centroids.lock().unwrap();

        let mut best_speaker = None;
        let mut max_sim = -1.0f32;

        // Compare new embedding with all existing speaker centroids
        for (idx, centroid) in centroids.iter().enumerate() {
            let sim = cosine_similarity(embedding, centroid);
            if sim > max_sim {
                max_sim = sim;
                best_speaker = Some(idx);
            }
        }

        if let Some(speaker_id) = best_speaker {
            if max_sim > threshold {
                // Update the centroid using a simple moving average to adapt to speaking style
                let learning_rate = 0.1f32;
                let centroid = &mut centroids[speaker_id];
                for i in 0..centroid.len() {
                    centroid[i] =
                        centroid[i] * (1.0 - learning_rate) + embedding[i] * learning_rate;
                }
                return speaker_id;
            }
        }

        // No match found or similarity too low -> create a new speaker
        centroids.push(embedding.to_vec());
        centroids.len() - 1
    }

    pub fn clear(&self) {
        self.centroids.lock().unwrap().clear();
    }
}

/// Global singleton for DiarizationEngine
static DIARIZATION_ENGINE: OnceCell<DiarizationEngine> = OnceCell::new();

impl DiarizationEngine {
    /// Get the global DiarizationEngine instance.
    pub fn get() -> &'static DiarizationEngine {
        DIARIZATION_ENGINE.get_or_init(|| DiarizationEngine::new())
    }
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
    dot
}
