//! Speaker Diarization v4.0 — Bayesian evidence accumulation.
//!
//! Key design:
//!   - Replaces rule-based mismatch counter with continuous evidence accumulator
//!   - Density → confidence weighting (no hard gate, low density = low weight)
//!   - Evidence decays over time (old mismatches fade naturally)
//!   - A match reduces evidence proportionally, NOT resets it to zero
//!   - Anti-proliferation guards:
//!     1. Minimum embedding density (10%) — below this, embedding is garbage
//!     2. Conservative reinforcement (only from score ≥ 0.60)
//!     3. Max speaker cap (8)
//!     4. Minimum speech for registration (0.3s)

use anyhow::Result;
use once_cell::sync::OnceCell;
use sherpa_rs::embedding_manager::EmbeddingManager;
use sherpa_rs::speaker_id::{EmbeddingExtractor, ExtractorConfig};
use std::sync::Mutex;

static DIARIZATION_PIPELINE: OnceCell<DiarizationPipeline> = OnceCell::new();

/// Score at or above which we consider it a match.
const MATCH_THRESHOLD: f32 = 0.45;
/// Score at or above which we also strengthen the speaker profile.
const REINFORCE_THRESHOLD: f32 = 0.60;
/// Minimum density to extract embedding at all (below this, embedding is garbage).
const MIN_EMBEDDING_DENSITY: f32 = 0.10;
/// Minimum speech seconds for registering a new speaker.
const MIN_REGISTER_SPEECH_SECS: f32 = 0.3;
/// Maximum number of speakers we'll track.
const MAX_SPEAKERS: usize = 8;
/// Evidence threshold for creating a new speaker.
const EVIDENCE_THRESHOLD: f32 = 1.5;
/// Time decay factor applied to evidence each segment.
const EVIDENCE_DECAY: f32 = 0.9;

/// Cosine similarity threshold for "Me" voice verification.
/// Below this, mic audio is considered echo leakage from speaker.
const ME_VERIFY_THRESHOLD: f32 = 0.40;
/// Maximum reinforcement updates for Me anchor embedding.
const ME_ANCHOR_MAX_REINFORCEMENTS: usize = 5;
/// Minimum speech seconds needed to register/verify Me anchor.
const ME_ANCHOR_MIN_SPEECH_SECS: f32 = 1.0;

/// Compute cosine similarity between two embedding vectors.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}

pub struct DiarizationPipeline {
    extractor: Mutex<EmbeddingExtractor>,
    manager: Mutex<EmbeddingManager>,
    speaker_count: Mutex<usize>,
    last_speaker: Mutex<String>,
    /// Bayesian evidence accumulator for new speaker detection.
    /// Positive evidence = likely a new speaker. Decays over time.
    /// When this exceeds EVIDENCE_THRESHOLD, a new speaker is created.
    pending_evidence: Mutex<f32>,
    /// Stored embedding from the best unmatched segment (candidate for new speaker).
    pending_embedding: Mutex<Option<Vec<f32>>>,
    /// Stored embedding anchor for the local user's voice ("Me").
    me_anchor: Mutex<Option<Vec<f32>>>,
    /// Number of embeddings averaged into the Me anchor.
    me_anchor_count: Mutex<usize>,
}

impl DiarizationPipeline {
    pub fn init(embedding_model: &std::path::Path, _threshold: f32) -> Result<()> {
        println!("🚀 [DIARIZATION] Initializing v4.0 (Bayesian)...");
        println!("📂 [DIARIZATION] Model: {}", embedding_model.display());
        println!(
            "📂 [DIARIZATION] match≥{:.2}, reinforce≥{:.2}, min_density≥{:.0}%, evidence_threshold={:.1}, decay={:.1}",
            MATCH_THRESHOLD, REINFORCE_THRESHOLD, MIN_EMBEDDING_DENSITY * 100.0, EVIDENCE_THRESHOLD, EVIDENCE_DECAY
        );

        let config = ExtractorConfig {
            model: embedding_model.to_string_lossy().to_string(),
            provider: None,
            num_threads: None,
            debug: false,
        };

        let extractor = EmbeddingExtractor::new(config)
            .map_err(|e| anyhow::anyhow!("Failed to create EmbeddingExtractor: {}", e))?;

        let dim = extractor.embedding_size as i32;
        println!("📐 [DIARIZATION] Embedding dim: {}", dim);

        DIARIZATION_PIPELINE
            .set(DiarizationPipeline {
                extractor: Mutex::new(extractor),
                manager: Mutex::new(EmbeddingManager::new(dim)),
                speaker_count: Mutex::new(0),
                last_speaker: Mutex::new("Speaker 1".to_string()),
                pending_evidence: Mutex::new(0.0),
                pending_embedding: Mutex::new(None),
                me_anchor: Mutex::new(None),
                me_anchor_count: Mutex::new(0),
            })
            .map_err(|_| anyhow::anyhow!("DiarizationPipeline already initialized"))?;

        println!("✅ [DIARIZATION] Ready");
        Ok(())
    }

    pub fn get() -> &'static DiarizationPipeline {
        DIARIZATION_PIPELINE
            .get()
            .expect("DiarizationPipeline not initialized")
    }

    pub fn try_get() -> Option<&'static DiarizationPipeline> {
        DIARIZATION_PIPELINE.get()
    }

    pub fn identify_speaker_i16(&self, samples: &[i16], speech_density: f32) -> Result<String> {
        // ── Guard: extremely low density → embedding is garbage ──────────
        if speech_density < MIN_EMBEDDING_DENSITY {
            let last = self.get_last();
            println!(
                "⏩ [DIARIZATION] Density too low ({:.0}%), reusing: {}",
                speech_density * 100.0, last
            );
            return Ok(last);
        }

        // ── Confidence from density (replaces hard gate) ────────────────
        // density=50%+ → confidence=1.0, density=10% → confidence=0.2
        let confidence = (speech_density / 0.5).clamp(0.1, 1.0);

        let total_secs = samples.len() as f32 / 16000.0;
        let speech_secs = total_secs * speech_density;

        // ── Extract embedding ───────────────────────────────────────────
        let samples_f32: Vec<f32> = samples.iter().map(|&s| s as f32 / 32768.0).collect();
        let mut embedding = {
            let mut ext = self.extractor.lock()
                .map_err(|_| anyhow::anyhow!("Extractor lock poisoned"))?;
            ext.compute_speaker_embedding(samples_f32, 16000)
                .map_err(|e| anyhow::anyhow!("Embedding failed: {}", e))?
        };

        // ── Search for match ────────────────────────────────────────────
        let mut mgr = self.manager.lock()
            .map_err(|_| anyhow::anyhow!("Manager lock poisoned"))?;
        let mut cnt = self.speaker_count.lock()
            .map_err(|_| anyhow::anyhow!("Count lock poisoned"))?;

        let n_speakers = *cnt;
        let best = mgr.get_best_matches(&embedding, MATCH_THRESHOLD, 3);

        // Get the raw best score (including below-threshold) for evidence calculation
        let best_raw_score: f32;
        if !best.is_empty() {
            let strs: Vec<String> = best.iter()
                .map(|m| format!("{}={:.3}", m.name, m.score))
                .collect();
            println!("📊 [DIARIZATION] Matches(≥{:.2}): [{}] conf={:.2}", MATCH_THRESHOLD, strs.join(", "), confidence);
            best_raw_score = best[0].score;
        } else if n_speakers > 0 {
            let all = mgr.get_best_matches(&embedding, 0.0, 3);
            if !all.is_empty() {
                let strs: Vec<String> = all.iter()
                    .map(|m| format!("{}={:.3}", m.name, m.score))
                    .collect();
                println!("📊 [DIARIZATION] No match (all below {:.2}): [{}] conf={:.2}", MATCH_THRESHOLD, strs.join(", "), confidence);
                best_raw_score = all[0].score;
            } else {
                best_raw_score = 0.0;
            }
        } else {
            best_raw_score = 0.0;
        }

        // ── MATCH FOUND ─────────────────────────────────────────────────
        if !best.is_empty() {
            let name = best[0].name.clone();
            let score = best[0].score;

            // Reinforce only from confident matches
            if score >= REINFORCE_THRESHOLD {
                let _ = mgr.add(name.clone(), &mut embedding);
            }

            // Decay evidence proportionally (NOT reset to zero)
            // A strong match (0.70) decays more than a weak match (0.46)
            let decay_amount = (score - MATCH_THRESHOLD) * confidence * 0.5;
            let evidence = self.decay_evidence(decay_amount);

            self.set_last(&name);
            println!("🔊 [DIARIZATION] ✓ {} (score={:.3}, evidence={:.2})", name, score, evidence);
            return Ok(name);
        }

        // ── NO MATCH — accumulate evidence for new speaker ──────────────

        // Guard: max speakers
        if n_speakers >= MAX_SPEAKERS {
            let last = self.get_last();
            println!("🛑 [DIARIZATION] Max speakers ({}), reusing: {}", MAX_SPEAKERS, last);
            return Ok(last);
        }

        // Accumulate evidence: lower score = stronger evidence for new speaker
        // score=0.09 → delta=0.91*conf, score=0.40 → delta=0.60*conf
        let delta = (1.0 - best_raw_score) * confidence;
        let evidence = self.accumulate_evidence(delta);

        // Store the best embedding we've seen (prefer longer speech segments)
        if speech_secs >= MIN_REGISTER_SPEECH_SECS {
            self.store_pending_embedding(embedding.clone());
        }

        // ── Decision: create new speaker? ───────────────────────────────

        // First speaker: always create immediately (no evidence needed)
        if n_speakers == 0 {
            if speech_secs < MIN_REGISTER_SPEECH_SECS {
                let last = self.get_last();
                println!(
                    "⏩ [DIARIZATION] First speaker: speech too short ({:.1}s), reusing: {}",
                    speech_secs, last
                );
                return Ok(last);
            }
            // Fall through to CREATE below
        } else {
            // Subsequent speakers: need sufficient evidence
            if evidence < EVIDENCE_THRESHOLD {
                let last = self.get_last();
                println!(
                    "📈 [DIARIZATION] Evidence {:.2}/{:.1} (+{:.2}), reusing: {}",
                    evidence, EVIDENCE_THRESHOLD, delta, last
                );
                return Ok(last);
            }

            // Evidence threshold met — but need a usable embedding to register
            if speech_secs < MIN_REGISTER_SPEECH_SECS && !self.has_pending_embedding() {
                let last = self.get_last();
                println!(
                    "📈 [DIARIZATION] Evidence {:.2} ✓ but no usable embedding (speech={:.1}s), reusing: {}",
                    evidence, speech_secs, last
                );
                return Ok(last);
            }
        }

        // ── CREATE NEW SPEAKER ──────────────────────────────────────────
        let register_embedding = self.take_pending_embedding().unwrap_or(embedding);
        *cnt += 1;
        let new_name = format!("Speaker {}", *cnt);

        let mut reg_emb = register_embedding;
        mgr.add(new_name.clone(), &mut reg_emb)
            .map_err(|e| anyhow::anyhow!("Failed to register: {}", e))?;

        self.reset_evidence();
        self.set_last(&new_name);

        if n_speakers == 0 {
            println!(
                "🆕 [DIARIZATION] First: {} (speech={:.1}s, density={:.0}%)",
                new_name, speech_secs, speech_density * 100.0
            );
        } else {
            println!(
                "🆕 [DIARIZATION] Confirmed: {} (evidence={:.2}, speech={:.1}s, density={:.0}%)",
                new_name, evidence, speech_secs, speech_density * 100.0
            );
        }

        Ok(new_name)
    }

    fn get_last(&self) -> String {
        self.last_speaker.lock().map(|s| s.clone()).unwrap_or_else(|_| "Speaker 1".to_string())
    }

    fn set_last(&self, name: &str) {
        if let Ok(mut s) = self.last_speaker.lock() {
            *s = name.to_string();
        }
    }

    /// Accumulate positive evidence for a new speaker.
    /// Applies time decay first, then adds new evidence.
    fn accumulate_evidence(&self, delta: f32) -> f32 {
        if let Ok(mut e) = self.pending_evidence.lock() {
            *e = (*e * EVIDENCE_DECAY + delta).max(0.0);
            *e
        } else {
            0.0
        }
    }

    /// Decay evidence when a match is found.
    /// A match provides counter-evidence, reducing the new-speaker signal.
    fn decay_evidence(&self, amount: f32) -> f32 {
        if let Ok(mut e) = self.pending_evidence.lock() {
            *e = (*e * EVIDENCE_DECAY - amount).max(0.0);
            *e
        } else {
            0.0
        }
    }

    /// Reset evidence to zero (after creating a new speaker).
    fn reset_evidence(&self) {
        if let Ok(mut e) = self.pending_evidence.lock() {
            *e = 0.0;
        }
        if let Ok(mut e) = self.pending_embedding.lock() {
            *e = None;
        }
    }

    fn store_pending_embedding(&self, emb: Vec<f32>) {
        if let Ok(mut e) = self.pending_embedding.lock() {
            *e = Some(emb);
        }
    }

    fn take_pending_embedding(&self) -> Option<Vec<f32>> {
        if let Ok(mut e) = self.pending_embedding.lock() {
            e.take()
        } else {
            None
        }
    }

    fn has_pending_embedding(&self) -> bool {
        self.pending_embedding.lock().map(|e| e.is_some()).unwrap_or(false)
    }

    /// Check if audio matches any registered system speaker (read-only, no side effects).
    /// Used for echo detection: if mic audio matches a system speaker, it's echo leakage.
    /// Returns Ok(Some((name, score))) if matches, Ok(None) if no match or no speakers.
    pub fn matches_any_speaker(&self, samples: &[i16]) -> anyhow::Result<Option<(String, f32)>> {
        let cnt = self.speaker_count.lock()
            .map_err(|_| anyhow::anyhow!("Count lock poisoned"))?;
        if *cnt == 0 {
            return Ok(None); // No speakers registered yet
        }
        drop(cnt);

        let samples_f32: Vec<f32> = samples.iter().map(|&s| s as f32 / 32768.0).collect();
        let embedding = {
            let mut ext = self.extractor.lock()
                .map_err(|_| anyhow::anyhow!("Extractor lock poisoned"))?;
            ext.compute_speaker_embedding(samples_f32, 16000)
                .map_err(|e| anyhow::anyhow!("Embedding failed: {}", e))?
        };

        let mut mgr = self.manager.lock()
            .map_err(|_| anyhow::anyhow!("Manager lock poisoned"))?;
        let best = mgr.get_best_matches(&embedding, MATCH_THRESHOLD, 1);
        if !best.is_empty() {
            Ok(Some((best[0].name.clone(), best[0].score)))
        } else {
            Ok(None)
        }
    }

    /// Check if a Me anchor has been registered.
    pub fn has_me_anchor(&self) -> bool {
        self.me_anchor.lock().map(|a| a.is_some()).unwrap_or(false)
    }

    /// Register or reinforce the "Me" voice anchor from mic channel audio.
    pub fn register_me_anchor(&self, samples: &[i16], speech_density: f32) -> anyhow::Result<()> {
        let total_secs = samples.len() as f32 / 16000.0;
        let speech_secs = total_secs * speech_density;
        // Use moderate density threshold for anchor registration.
        // Real mic segments typically have 10-48% density (lots of buffered silence),
        // so 50% was too strict and prevented anchor from ever being registered.
        const ME_ANCHOR_MIN_DENSITY: f32 = 0.30;
        if speech_secs < ME_ANCHOR_MIN_SPEECH_SECS || speech_density < ME_ANCHOR_MIN_DENSITY {
            return Ok(()); // Not enough speech to establish anchor
        }

        // Extract embedding
        let samples_f32: Vec<f32> = samples.iter().map(|&s| s as f32 / 32768.0).collect();
        let embedding = {
            let mut ext = self.extractor.lock()
                .map_err(|_| anyhow::anyhow!("Extractor lock poisoned"))?;
            ext.compute_speaker_embedding(samples_f32, 16000)
                .map_err(|e| anyhow::anyhow!("Embedding failed: {}", e))?
        };

        let mut anchor = self.me_anchor.lock()
            .map_err(|_| anyhow::anyhow!("Me anchor lock poisoned"))?;
        let mut count = self.me_anchor_count.lock()
            .map_err(|_| anyhow::anyhow!("Me anchor count lock poisoned"))?;

        if let Some(ref mut existing) = *anchor {
            // Reinforce: running average
            if *count < ME_ANCHOR_MAX_REINFORCEMENTS {
                let n = *count as f32;
                for (i, val) in existing.iter_mut().enumerate() {
                    if i < embedding.len() {
                        *val = (*val * n + embedding[i]) / (n + 1.0);
                    }
                }
                *count += 1;
                println!("🔊 [DIARIZATION] Me anchor reinforced ({}/{})", *count, ME_ANCHOR_MAX_REINFORCEMENTS);
            }
        } else {
            // First registration
            *anchor = Some(embedding);
            *count = 1;
            println!(
                "🆕 [DIARIZATION] Me anchor registered (speech={:.1}s, density={:.0}%)",
                speech_secs, speech_density * 100.0
            );
        }

        Ok(())
    }

    /// Verify if audio from mic channel sounds like the registered "Me" anchor.
    /// Returns: Ok(Some(true)) = Me, Ok(Some(false)) = not Me (echo), Ok(None) = can't determine.
    pub fn verify_is_me(&self, samples: &[i16], speech_density: f32) -> anyhow::Result<Option<bool>> {
        let anchor = self.me_anchor.lock()
            .map_err(|_| anyhow::anyhow!("Me anchor lock poisoned"))?;
        let anchor_ref = match anchor.as_ref() {
            Some(a) => a,
            None => return Ok(None), // No anchor yet
        };

        // Very low guard: we want to verify nearly all mic segments.
        // The similarity gap between real voice (~0.77) and echo (~0.05-0.08)
        // is so large that even low-quality embeddings can distinguish them.
        let total_secs = samples.len() as f32 / 16000.0;
        let speech_secs = total_secs * speech_density;
        if speech_secs < 0.3 || speech_density < 0.05 {
            return Ok(None);
        }

        // Extract embedding
        let samples_f32: Vec<f32> = samples.iter().map(|&s| s as f32 / 32768.0).collect();
        let embedding = {
            let mut ext = self.extractor.lock()
                .map_err(|_| anyhow::anyhow!("Extractor lock poisoned"))?;
            ext.compute_speaker_embedding(samples_f32, 16000)
                .map_err(|e| anyhow::anyhow!("Embedding failed: {}", e))?
        };

        let similarity = cosine_similarity(anchor_ref, &embedding);
        println!(
            "🎤 [DIARIZATION] Me verification: similarity={:.3} (threshold={:.2})",
            similarity, ME_VERIFY_THRESHOLD
        );

        Ok(Some(similarity >= ME_VERIFY_THRESHOLD))
    }

    pub fn clear(&self) {
        if let (Ok(mut m), Ok(mut c), Ok(e), Ok(mut l), Ok(mut p)) = (
            self.manager.lock(),
            self.speaker_count.lock(),
            self.extractor.lock(),
            self.last_speaker.lock(),
            self.pending_evidence.lock(),
        ) {
            let dim = e.embedding_size as i32;
            *m = EmbeddingManager::new(dim);
            *c = 0;
            *l = "Speaker 1".to_string();
            *p = 0.0;
            println!("🧹 [DIARIZATION] Reset (dim={})", dim);
        }
        if let Ok(mut e) = self.pending_embedding.lock() {
            *e = None;
        }
        // Reset Me anchor
        if let Ok(mut a) = self.me_anchor.lock() {
            *a = None;
        }
        if let Ok(mut c) = self.me_anchor_count.lock() {
            *c = 0;
        }
    }
}
