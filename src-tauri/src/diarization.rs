//! Speaker Diarization v3.1 — Balanced streaming speaker identification.
//!
//! Key design:
//!   - Single threshold (0.45): match or no-match, no gray zone
//!   - Anti-proliferation via GUARDS, not via over-broad score ranges:
//!     1. Density gating (skip < 25% speech density)
//!     2. Confirmation (require 2 consecutive unmatched segments)
//!     3. Conservative reinforcement (only from score ≥ 0.60)
//!     4. Max speaker cap (8)
//!     5. Minimum speech duration for new speaker (1.0s)

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
const MIN_NEW_SPEAKER_SPEECH_SECS: f32 = 1.0;
/// Maximum number of speakers we'll track.
const MAX_SPEAKERS: usize = 8;

pub struct DiarizationPipeline {
    extractor: Mutex<EmbeddingExtractor>,
    manager: Mutex<EmbeddingManager>,
    speaker_count: Mutex<usize>,
    last_speaker: Mutex<String>,
    /// True when the previous segment was unmatched.
    /// New speakers require 2 consecutive unmatched segments.
    pending_new: Mutex<bool>,
}

impl DiarizationPipeline {
    pub fn init(embedding_model: &std::path::Path, _threshold: f32) -> Result<()> {
        println!("🚀 [DIARIZATION] Initializing v3.1...");
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
                pending_new: Mutex::new(false),
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
        if !best.is_empty() {
            let strs: Vec<String> = best.iter()
                .map(|m| format!("{}={:.3}", m.name, m.score))
                .collect();
            println!("📊 [DIARIZATION] Matches(≥{:.2}): [{}]", MATCH_THRESHOLD, strs.join(", "));
        } else if n_speakers > 0 {
            // Also show what the best score WAS (below threshold) for debugging
            let all = mgr.get_best_matches(&embedding, 0.0, 3);
            if !all.is_empty() {
                let strs: Vec<String> = all.iter()
                    .map(|m| format!("{}={:.3}", m.name, m.score))
                    .collect();
                println!("📊 [DIARIZATION] No match (all below {:.2}): [{}]", MATCH_THRESHOLD, strs.join(", "));
            }
        }

        // ── MATCH FOUND ─────────────────────────────────────────────────
        if !best.is_empty() {
            let name = best[0].name.clone();
            let score = best[0].score;

            // Reinforce only from confident matches
            if score >= REINFORCE_THRESHOLD {
                let _ = mgr.add(name.clone(), &mut embedding);
            }

            self.set_pending(false);
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

        // Guard: minimum speech content
        if speech_secs < MIN_NEW_SPEAKER_SPEECH_SECS {
            let last = self.get_last();
            println!(
                "⏩ [DIARIZATION] Speech too short ({:.1}s < {:.1}s), reusing: {}",
                speech_secs, MIN_NEW_SPEAKER_SPEECH_SECS, last
            );
            return Ok(last);
        }

        // Guard: confirmation (skip for the very first speaker)
        if n_speakers > 0 && !self.is_pending() {
            self.set_pending(true);
            let last = self.get_last();
            println!(
                "⏳ [DIARIZATION] Pending new speaker (need 1 more mismatch), reusing: {}",
                last
            );
            return Ok(last);
        }

        // ── CREATE NEW SPEAKER ──────────────────────────────────────────
        *cnt += 1;
        let new_name = format!("Speaker {}", *cnt);

        mgr.add(new_name.clone(), &mut embedding)
            .map_err(|e| anyhow::anyhow!("Failed to register: {}", e))?;

        self.set_pending(false);
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

    fn is_pending(&self) -> bool {
        self.pending_new.lock().map(|p| *p).unwrap_or(false)
    }

    fn set_pending(&self, val: bool) {
        if let Ok(mut p) = self.pending_new.lock() {
            *p = val;
        }
    }

    pub fn clear(&self) {
        if let (Ok(mut m), Ok(mut c), Ok(e), Ok(mut l), Ok(mut p)) = (
            self.manager.lock(),
            self.speaker_count.lock(),
            self.extractor.lock(),
            self.last_speaker.lock(),
            self.pending_new.lock(),
        ) {
            let dim = e.embedding_size as i32;
            *m = EmbeddingManager::new(dim);
            *c = 0;
            *l = "Speaker 1".to_string();
            *p = false;
            println!("🧹 [DIARIZATION] Reset (dim={})", dim);
        }
    }
}
