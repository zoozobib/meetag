//! Speaker Diarization v3.3 — Fast speaker switching with strong-mismatch bypass.
//!
//! Key design:
//!   - Single threshold (0.45): match or no-match, no gray zone
//!   - Anti-proliferation via GUARDS, not via over-broad score ranges:
//!     1. Density gating (skip < 25% speech density)
//!     2. Confirmation (require 2 cumulative unmatched segments — counter, not consecutive)
//!        BUT: if best score < 0.20 (strong mismatch), skip confirmation entirely
//!     3. Conservative reinforcement (only from score ≥ 0.60)
//!     4. Max speaker cap (8)
//!     5. Minimum speech duration for new speaker (0.5s)

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
/// Skip diarization when speech density is below this.
const MIN_SPEECH_DENSITY: f32 = 0.25;
/// Minimum seconds of actual speech content before registering a new speaker.
const MIN_NEW_SPEAKER_SPEECH_SECS: f32 = 0.5;
/// Maximum number of speakers we'll track.
const MAX_SPEAKERS: usize = 8;
/// When the best speaker score is below this, it's a strong mismatch.
/// Skip pending confirmation and create a new speaker immediately.
const STRONG_MISMATCH_THRESHOLD: f32 = 0.20;

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
    /// Cumulative count of unmatched segments (not necessarily consecutive).
    /// New speakers require 2+ mismatches before creation.
    /// A successful match resets this to 0.
    pending_mismatch_count: Mutex<u8>,
    /// Stored embedding from the last unmatched segment (candidate for new speaker).
    pending_embedding: Mutex<Option<Vec<f32>>>,
    /// Stored embedding anchor for the local user's voice ("Me").
    me_anchor: Mutex<Option<Vec<f32>>>,
    /// Number of embeddings averaged into the Me anchor.
    me_anchor_count: Mutex<usize>,
}

impl DiarizationPipeline {
    pub fn init(embedding_model: &std::path::Path, _threshold: f32) -> Result<()> {
        println!("🚀 [DIARIZATION] Initializing v3.3...");
        println!("📂 [DIARIZATION] Model: {}", embedding_model.display());
        println!(
            "📂 [DIARIZATION] match≥{:.2}, reinforce≥{:.2}, density≥{:.0}%, max_speakers={}",
            MATCH_THRESHOLD, REINFORCE_THRESHOLD, MIN_SPEECH_DENSITY * 100.0, MAX_SPEAKERS
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
                pending_mismatch_count: Mutex::new(0),
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
        // ── Guard: low density → unreliable, skip ───────────────────────
        if speech_density < MIN_SPEECH_DENSITY {
            let last = self.get_last();
            println!(
                "⏩ [DIARIZATION] Low density ({:.0}%), reusing: {}",
                speech_density * 100.0, last
            );
            return Ok(last);
        }

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

        // Debug: always log scores
        // Also retrieve ALL scores (including below-threshold) for strong-mismatch detection
        let best_raw_score: f32;
        if !best.is_empty() {
            let strs: Vec<String> = best.iter()
                .map(|m| format!("{}={:.3}", m.name, m.score))
                .collect();
            println!("📊 [DIARIZATION] Matches(≥{:.2}): [{}]", MATCH_THRESHOLD, strs.join(", "));
            best_raw_score = best[0].score;
        } else if n_speakers > 0 {
            // Also show what the best score WAS (below threshold) for debugging
            let all = mgr.get_best_matches(&embedding, 0.0, 3);
            if !all.is_empty() {
                let strs: Vec<String> = all.iter()
                    .map(|m| format!("{}={:.3}", m.name, m.score))
                    .collect();
                println!("📊 [DIARIZATION] No match (all below {:.2}): [{}]", MATCH_THRESHOLD, strs.join(", "));
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

            // A match resets the mismatch counter
            self.reset_mismatch_counter();
            self.set_last(&name);
            println!("🔊 [DIARIZATION] ✓ {} (score={:.3})", name, score);
            return Ok(name);
        }

        // ── NO MATCH — new speaker candidate ────────────────────────────

        // Guard: max speakers
        if n_speakers >= MAX_SPEAKERS {
            let last = self.get_last();
            println!("🛑 [DIARIZATION] Max speakers ({}), reusing: {}", MAX_SPEAKERS, last);
            return Ok(last);
        }

        // Detect strong mismatch: best score is far below threshold.
        // e.g., score=0.047 means this is clearly a completely different person.
        let is_strong_mismatch = n_speakers > 0 && best_raw_score < STRONG_MISMATCH_THRESHOLD;

        // Increment mismatch counter for ANY unmatched segment (even short ones).
        // This ensures short segments during speaker transitions contribute
        // to new speaker detection instead of resetting the process.
        let mismatches = self.increment_mismatch_counter();

        // Store the best embedding we've seen (prefer longer speech segments)
        if speech_secs >= MIN_NEW_SPEAKER_SPEECH_SECS {
            self.store_pending_embedding(embedding.clone());
        }

        // Guard: minimum speech content — short segments count toward
        // mismatch but can't register a new speaker by themselves
        if speech_secs < MIN_NEW_SPEAKER_SPEECH_SECS {
            let last = self.get_last();
            println!(
                "⏩ [DIARIZATION] Speech too short ({:.1}s < {:.1}s), mismatch {}/2{}, reusing: {}",
                speech_secs, MIN_NEW_SPEAKER_SPEECH_SECS, mismatches,
                if is_strong_mismatch { " [STRONG]" } else { "" }, last
            );
            return Ok(last);
        }

        // Guard: confirmation (skip for the very first speaker)
        // EXCEPTION: strong mismatch (best score < 0.20) skips confirmation entirely.
        // When the score is 0.047 it's obviously a completely different person —
        // waiting for a second mismatch wastes 3-5 seconds.
        if n_speakers > 0 && mismatches < 2 && !is_strong_mismatch {
            let last = self.get_last();
            println!(
                "⏳ [DIARIZATION] Pending new speaker (mismatch {}/2), reusing: {}",
                mismatches, last
            );
            return Ok(last);
        }

        // ── CREATE NEW SPEAKER ──────────────────────────────────────────
        // Use stored pending embedding if available (may be from a longer segment)
        let register_embedding = self.take_pending_embedding().unwrap_or(embedding);
        *cnt += 1;
        let new_name = format!("Speaker {}", *cnt);

        let mut reg_emb = register_embedding;
        mgr.add(new_name.clone(), &mut reg_emb)
            .map_err(|e| anyhow::anyhow!("Failed to register: {}", e))?;

        self.reset_mismatch_counter();
        self.set_last(&new_name);

        if n_speakers == 0 {
            println!(
                "🆕 [DIARIZATION] First: {} (speech={:.1}s, density={:.0}%)",
                new_name, speech_secs, speech_density * 100.0
            );
        } else {
            println!(
                "🆕 [DIARIZATION] Confirmed: {} (speech={:.1}s, density={:.0}%)",
                new_name, speech_secs, speech_density * 100.0
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

    fn increment_mismatch_counter(&self) -> u8 {
        if let Ok(mut c) = self.pending_mismatch_count.lock() {
            *c = c.saturating_add(1);
            *c
        } else {
            1
        }
    }

    fn reset_mismatch_counter(&self) {
        if let Ok(mut c) = self.pending_mismatch_count.lock() {
            *c = 0;
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
            self.pending_mismatch_count.lock(),
        ) {
            let dim = e.embedding_size as i32;
            *m = EmbeddingManager::new(dim);
            *c = 0;
            *l = "Speaker 1".to_string();
            *p = 0;
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
