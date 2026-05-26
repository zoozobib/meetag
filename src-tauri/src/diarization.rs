//! Speaker Diarization v10.0 — Lightweight offline enhancement.
//!
//! Architecture:
//!   - During recording: real-time ASR writes transcript.jsonl (the primary product).
//!   - After recording: lightweight enhancement (no re-ASR):
//!     1. system.wav → OfflineSpeakerDiarization (speaker labels)
//!     2. Text-level LCS echo cleanup (using real-time system entries)
//!     3. Sliding window deduplication
//!   - On next startup: catch up any sessions that weren't enhanced before app quit.

use anyhow::Result;
use once_cell::sync::OnceCell;
use sherpa_onnx::{
    FastClusteringConfig, OfflineSpeakerDiarization, OfflineSpeakerDiarizationConfig,
    OfflineSpeakerDiarizationSegment, OfflineSpeakerSegmentationModelConfig,
    OfflineSpeakerSegmentationPyannoteModelConfig, SpeakerEmbeddingExtractorConfig,
};
use std::path::Path;

static DIARIZER: OnceCell<OfflineSpeakerDiarization> = OnceCell::new();

// ══════════════════════════════════════════════════════════════════════════

/// Initialize the offline diarizer with Pyannote segmentation + embedding models.
/// Call once at startup.
pub fn init(segmentation_model: &Path, embedding_model: &Path) -> Result<()> {
    println!("🚀 [DIARIZATION] Initializing v9.0 (Dual-Channel Post-Processing)...");
    println!(
        "📂 [DIARIZATION] Segmentation model: {}",
        segmentation_model.display()
    );
    println!(
        "📂 [DIARIZATION] Embedding model: {}",
        embedding_model.display()
    );

    if !segmentation_model.exists() {
        return Err(anyhow::anyhow!(
            "Segmentation model not found: {}",
            segmentation_model.display()
        ));
    }
    if !embedding_model.exists() {
        return Err(anyhow::anyhow!(
            "Embedding model not found: {}",
            embedding_model.display()
        ));
    }

    // Dynamic thread count based on CPU cores (capped 2-6)
    let num_threads = std::thread::available_parallelism()
        .map(|n| (n.get() as i32).min(6).max(2))
        .unwrap_or(2);
    println!("🧵 [DIARIZATION] Using {} threads (dynamic)", num_threads);

    let config = OfflineSpeakerDiarizationConfig {
        segmentation: OfflineSpeakerSegmentationModelConfig {
            pyannote: OfflineSpeakerSegmentationPyannoteModelConfig {
                model: Some(segmentation_model.to_string_lossy().into_owned()),
            },
            num_threads,
            debug: false,
            provider: Some("cpu".to_string()),
        },
        embedding: SpeakerEmbeddingExtractorConfig {
            model: Some(embedding_model.to_string_lossy().into_owned()),
            num_threads,
            debug: false,
            provider: Some("cpu".to_string()),
        },
        clustering: FastClusteringConfig {
            num_clusters: -1, // auto-detect number of speakers
            threshold: 0.65, // 0.65 is the sweet spot for distinguishing multiple speakers (0.80 was too aggressive, merging 3 people into 1-2; 0.5 was too low)
        },
        min_duration_on: 0.3,  // minimum speech duration (seconds)
        min_duration_off: 0.5, // minimum silence duration (seconds)
    };

    let diarizer = OfflineSpeakerDiarization::create(&config)
        .ok_or_else(|| anyhow::anyhow!("Failed to create OfflineSpeakerDiarization"))?;

    let sample_rate = diarizer.sample_rate();
    println!(
        "✅ [DIARIZATION] Ready (v8.0 Offline, sample_rate={})",
        sample_rate
    );

    DIARIZER
        .set(diarizer)
        .map_err(|_| anyhow::anyhow!("Diarizer already initialized"))?;

    Ok(())
}

/// Check if diarizer is initialized.
pub fn is_initialized() -> bool {
    DIARIZER.get().is_some()
}

// ── Post-recording diarization ─────────────────────────────────────────

/// Run offline speaker diarization on a complete WAV file.
/// Returns sorted segments with (start_time, end_time, speaker_id).
pub fn process_wav(wav_path: &Path) -> Result<Vec<DiarizationSegment>> {
    let diarizer = DIARIZER
        .get()
        .ok_or_else(|| anyhow::anyhow!("Diarizer not initialized"))?;

    println!(
        "🔄 [DIARIZATION] Processing complete recording: {}",
        wav_path.display()
    );

    // Read WAV file
    let (samples, file_sample_rate) = read_wav_f32(wav_path)?;
    let expected_sr = diarizer.sample_rate();

    println!(
        "📊 [DIARIZATION] Audio: {:.1}s, {} samples, {}Hz (expected {}Hz)",
        samples.len() as f32 / file_sample_rate as f32,
        samples.len(),
        file_sample_rate,
        expected_sr
    );

    // Resample if needed
    let samples = if file_sample_rate != expected_sr as u32 {
        println!(
            "🔄 [DIARIZATION] Resampling {}Hz → {}Hz",
            file_sample_rate, expected_sr
        );
        resample(&samples, file_sample_rate, expected_sr as u32)
    } else {
        samples
    };

    // Run diarization on the complete audio
    let start = std::time::Instant::now();
    let result = diarizer
        .process(&samples)
        .ok_or_else(|| anyhow::anyhow!("Diarization returned no result"))?;
    let elapsed = start.elapsed();

    let num_speakers = result.num_speakers();
    let segments_raw: Vec<OfflineSpeakerDiarizationSegment> = result.sort_by_start_time();

    println!(
        "✅ [DIARIZATION] Done in {:.1}s: {} speakers, {} segments",
        elapsed.as_secs_f32(),
        num_speakers,
        segments_raw.len()
    );

    // ── Post-processing: merge noise speakers & renumber ──────────────
    // Step 1: Calculate total speaking time per raw speaker ID
    let mut speaker_durations: std::collections::HashMap<i32, f32> =
        std::collections::HashMap::new();
    for s in &segments_raw {
        *speaker_durations.entry(s.speaker).or_insert(0.0) += s.end - s.start;
    }
    let total_duration: f32 = speaker_durations.values().sum();

    // Step 2: Identify noise speakers (< 3% of total speaking time)
    let mut noise_speakers: std::collections::HashSet<i32> = std::collections::HashSet::new();
    for (&spk, &dur) in &speaker_durations {
        let pct = dur / total_duration * 100.0;
        if pct < 3.0 {
            println!(
                "🧹 [DIARIZATION] Merging Speaker {} ({:.1}s, {:.1}% — noise)",
                spk + 1,
                dur,
                pct
            );
            noise_speakers.insert(spk);
        }
    }

    // Step 3: Build segments, replacing noise speakers with nearest valid neighbor
    let mut segments: Vec<DiarizationSegment> = Vec::new();
    for s in &segments_raw {
        let speaker_id = if noise_speakers.contains(&s.speaker) {
            // Find the nearest non-noise segment by time proximity
            let mid = (s.start + s.end) / 2.0;
            segments_raw
                .iter()
                .filter(|other| !noise_speakers.contains(&other.speaker))
                .min_by(|a, b| {
                    let dist_a = (mid - (a.start + a.end) / 2.0).abs();
                    let dist_b = (mid - (b.start + b.end) / 2.0).abs();
                    dist_a.partial_cmp(&dist_b).unwrap()
                })
                .map(|nearest| nearest.speaker)
                .unwrap_or(s.speaker)
        } else {
            s.speaker
        };
        segments.push(DiarizationSegment {
            start: s.start,
            end: s.end,
            speaker: format!("raw_{}", speaker_id), // temporary label
        });
    }

    // Step 4: Renumber speakers sequentially (raw_5, raw_0 → Speaker 1, Speaker 2)
    let mut seen_ids: Vec<i32> = Vec::new();
    for s in &segments_raw {
        if !noise_speakers.contains(&s.speaker) && !seen_ids.contains(&s.speaker) {
            seen_ids.push(s.speaker);
        }
    }
    let id_map: std::collections::HashMap<String, String> = seen_ids
        .iter()
        .enumerate()
        .map(|(i, &raw_id)| (format!("raw_{}", raw_id), format!("Speaker {}", i + 1)))
        .collect();

    for seg in &mut segments {
        if let Some(label) = id_map.get(&seg.speaker) {
            seg.speaker = label.clone();
        }
    }

    // Step 5: Resolve overlapping segments
    // Sort by start time, then trim overlaps (earlier segment's end clipped to next start)
    segments.sort_by(|a, b| a.start.partial_cmp(&b.start).unwrap());
    for i in 0..segments.len().saturating_sub(1) {
        if segments[i].end > segments[i + 1].start {
            segments[i].end = segments[i + 1].start;
        }
    }
    // Remove segments that became zero or negative duration after trimming
    segments.retain(|s| s.end - s.start > 0.1);

    // Step 6: Merge adjacent same-speaker segments with gap < 1.5s
    // This consolidates fragmented turns like "因为" + "那部片" + "大红之后" into one segment
    let mut merged: Vec<DiarizationSegment> = Vec::new();
    for seg in segments {
        if let Some(last) = merged.last_mut() {
            if last.speaker == seg.speaker && (seg.start - last.end) < 1.5 {
                // Same speaker, small gap → extend
                last.end = seg.end;
                continue;
            }
        }
        merged.push(seg);
    }
    let segments = merged;

    // Step 7: Merge fragmented speakers (non-overlapping merge)
    // If two speakers NEVER overlap temporally, they're likely the same person
    // whose voice was split by the diarization model due to short segments.
    // Merge the smaller speaker into the larger one, iteratively.
    let mut segments = segments;
    loop {
        // Collect unique speakers and their total durations
        let mut speaker_dur: std::collections::HashMap<String, f32> =
            std::collections::HashMap::new();
        for s in &segments {
            *speaker_dur.entry(s.speaker.clone()).or_insert(0.0) += s.end - s.start;
        }

        if speaker_dur.len() <= 2 {
            break; // Already 2 or fewer speakers, done
        }

        // Sort speakers by duration (ascending) — try to merge smallest first
        let mut speakers_sorted: Vec<(String, f32)> = speaker_dur.into_iter().collect();
        speakers_sorted.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        let mut merged_any = false;

        // Try to merge the smallest speaker with another non-overlapping speaker
        'outer: for i in 0..speakers_sorted.len() {
            let (ref small_spk, _) = speakers_sorted[i];

            // Get all time intervals for small speaker
            let small_intervals: Vec<(f32, f32)> = segments
                .iter()
                .filter(|s| &s.speaker == small_spk)
                .map(|s| (s.start, s.end))
                .collect();

            // Check against each other speaker (prefer merging into the largest)
            for j in (0..speakers_sorted.len()).rev() {
                if i == j {
                    continue;
                }
                let (ref candidate_spk, _) = speakers_sorted[j];

                let candidate_intervals: Vec<(f32, f32)> = segments
                    .iter()
                    .filter(|s| &s.speaker == candidate_spk)
                    .map(|s| (s.start, s.end))
                    .collect();

                // Check if they ever overlap
                let overlaps = small_intervals.iter().any(|&(s1, e1)| {
                    candidate_intervals.iter().any(|&(s2, e2)| {
                        s1 < e2 && s2 < e1 // standard interval overlap test
                    })
                });

                if !overlaps {
                    // No overlap → merge small into candidate
                    println!(
                        "🔗 [DIARIZATION] Merging {} into {} (non-overlapping, likely same speaker)",
                        small_spk, candidate_spk
                    );
                    let target = candidate_spk.clone();
                    let source = small_spk.clone();
                    for seg in &mut segments {
                        if seg.speaker == source {
                            seg.speaker = target.clone();
                        }
                    }
                    merged_any = true;
                    break 'outer;
                }
            }
        }

        if !merged_any {
            break; // No more merges possible
        }

        // After merging, re-merge adjacent same-speaker segments
        segments.sort_by(|a, b| a.start.partial_cmp(&b.start).unwrap());
        let mut re_merged: Vec<DiarizationSegment> = Vec::new();
        for seg in segments {
            if let Some(last) = re_merged.last_mut() {
                if last.speaker == seg.speaker && (seg.start - last.end) < 1.5 {
                    last.end = seg.end;
                    continue;
                }
            }
            re_merged.push(seg);
        }
        segments = re_merged;
    }

    // Step 8: Final renumber (Speaker 1, Speaker 2, ...)
    // After merging, speaker labels may have gaps — renumber by first appearance
    let mut seen_labels: Vec<String> = Vec::new();
    for s in &segments {
        if !seen_labels.contains(&s.speaker) {
            seen_labels.push(s.speaker.clone());
        }
    }
    if seen_labels.len() >= 2 {
        // Only renumber if needed
        let label_map: std::collections::HashMap<String, String> = seen_labels
            .iter()
            .enumerate()
            .map(|(i, old)| (old.clone(), format!("Speaker {}", i + 1)))
            .collect();
        for seg in &mut segments {
            if let Some(new_label) = label_map.get(&seg.speaker) {
                seg.speaker = new_label.clone();
            }
        }
    }

    // Print summary
    let final_speakers: std::collections::HashSet<&str> =
        segments.iter().map(|s| s.speaker.as_str()).collect();
    println!(
        "✅ [DIARIZATION] Final: {} speakers, {} segments (merged {} noise speakers)",
        final_speakers.len(),
        segments.len(),
        noise_speakers.len()
    );
    for seg in &segments {
        println!("  📌 {:.1}s - {:.1}s: {}", seg.start, seg.end, seg.speaker);
    }

    Ok(segments)
}

// ── Data types ─────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct DiarizationSegment {
    pub start: f32,
    pub end: f32,
    pub speaker: String,
}

/// Read a WAV file into f32 samples (mono).
fn read_wav_f32(path: &Path) -> Result<(Vec<f32>, u32)> {
    let mut reader = hound::WavReader::open(path)?;
    let spec = reader.spec();
    let sample_rate = spec.sample_rate;

    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Int => {
            let max_val = (1 << (spec.bits_per_sample - 1)) as f32;
            reader
                .samples::<i32>()
                .filter_map(|s| s.ok())
                .map(|s| s as f32 / max_val)
                .collect()
        }
        hound::SampleFormat::Float => reader.samples::<f32>().filter_map(|s| s.ok()).collect(),
    };

    // If stereo, take only left channel (or average)
    let mono = if spec.channels > 1 {
        samples
            .chunks(spec.channels as usize)
            .map(|chunk| chunk.iter().sum::<f32>() / chunk.len() as f32)
            .collect()
    } else {
        samples
    };

    Ok((mono, sample_rate))
}

/// Simple linear resampling.
fn resample(samples: &[f32], from_rate: u32, to_rate: u32) -> Vec<f32> {
    if from_rate == to_rate {
        return samples.to_vec();
    }
    let ratio = from_rate as f64 / to_rate as f64;
    let new_len = (samples.len() as f64 / ratio) as usize;
    let mut output = Vec::with_capacity(new_len);
    for i in 0..new_len {
        let src_idx = i as f64 * ratio;
        let idx = src_idx as usize;
        let frac = src_idx - idx as f64;
        let sample = if idx + 1 < samples.len() {
            samples[idx] * (1.0 - frac as f32) + samples[idx + 1] * frac as f32
        } else if idx < samples.len() {
            samples[idx]
        } else {
            0.0
        };
        output.push(sample);
    }
    output
}

// ══════════════════════════════════════════════════════════════════════════
// Lightweight Offline Enhancement (v10.0)
//
// Architecture:
//   - Real-time transcript is the primary product (no re-ASR)
//   - Offline processing only:
//     1. Adds speaker labels via diarization
//     2. Removes echo entries via text-level LCS (using real-time system text)
//     3. Deduplicates sliding window overlaps
//   - Total time: < 30 seconds (vs 30-60 minutes with old re-ASR approach)
// ══════════════════════════════════════════════════════════════════════════

/// Lightweight offline enhancement — no ASR, uses real-time transcript data.
///
/// Steps:
///   1. Load existing transcript.jsonl (real-time entries)
///   2. Run diarization on system.wav → speaker segments
///   3. Map speaker labels to system entries (timestamp overlap)
///   4. Echo cleanup: LCS compare user entries against real-time system text
///   5. Deduplicate sliding window overlaps
///   6. Write enhanced transcript + optionally emit to frontend
pub fn enhance_transcript(
    system_wav: &Path,
    transcript_path: &Path,
    app: &tauri::AppHandle,
    emit_to_frontend: bool,
) -> Result<usize> {
    use std::io::{BufRead, Write};
    use tauri::Emitter;

    println!("═══ ENHANCE v10.0: Lightweight offline enhancement ═══");

    // ═══ Step 1: Load existing transcript.jsonl ═══
    let all_entries: Vec<serde_json::Value> = if transcript_path.exists() {
        std::fs::File::open(transcript_path)
            .ok()
            .map(|f| {
                std::io::BufReader::new(f)
                    .lines()
                    .filter_map(|line| line.ok())
                    .filter_map(|line| serde_json::from_str::<serde_json::Value>(&line).ok())
                    .collect()
            })
            .unwrap_or_default()
    } else {
        Vec::new()
    };

    // ═══ Guard: skip sessions already processed by old pipeline ═══
    // Old pipeline wrote "Speaker 1", "Speaker 2" etc. If these exist,
    // the transcript is already enhanced — don't re-process or we'll lose labels.
    let has_old_labels = all_entries.iter().any(|e| {
        let s = e.get("speaker").and_then(|v| v.as_str()).unwrap_or("");
        s.starts_with("Speaker ")
    });
    if has_old_labels {
        println!("📋 [ENHANCE] Session already has speaker labels from old pipeline, skipping");
        if let Some(session_dir) = transcript_path.parent() {
            let _ = std::fs::File::create(session_dir.join(".enhanced"));
        }
        return Ok(all_entries.len());
    }

    // Separate into system and user entries
    let mut system_entries: Vec<serde_json::Value> = Vec::new();
    let mut user_entries: Vec<serde_json::Value> = Vec::new();
    for entry in &all_entries {
        let speaker = entry.get("speaker").and_then(|s| s.as_str()).unwrap_or("");
        if speaker == "system" {
            system_entries.push(entry.clone());
        } else {
            user_entries.push(entry.clone());
        }
    }
    println!(
        "📋 [ENHANCE] Loaded {} entries: {} system, {} user",
        all_entries.len(),
        system_entries.len(),
        user_entries.len()
    );

    // ═══ Step 2: Run diarization on system.wav ═══
    let diar_segments = if system_wav.exists() {
        match process_wav(system_wav) {
            Ok(segs) => {
                println!("✅ [ENHANCE] Diarization: {} speaker segments", segs.len());
                segs
            }
            Err(e) => {
                eprintln!(
                    "⚠️ [ENHANCE] Diarization failed: {}, skipping speaker labels",
                    e
                );
                Vec::new()
            }
        }
    } else {
        println!("⚠️ [ENHANCE] No system.wav, skipping diarization");
        Vec::new()
    };

    // ═══ Step 3: Map speaker labels to system entries ═══
    if !diar_segments.is_empty() {
        for entry in &mut system_entries {
            let entry_start = entry.get("start").and_then(|v| v.as_f64()).unwrap_or(0.0) as f32;
            let entry_end = entry.get("end").and_then(|v| v.as_f64()).unwrap_or(0.0) as f32;
            let entry_mid = (entry_start + entry_end) / 2.0;

            // Find the diarization segment that contains the midpoint of this entry
            let best_speaker = diar_segments
                .iter()
                .filter(|seg| entry_mid >= seg.start && entry_mid <= seg.end)
                .max_by(|a, b| {
                    // If multiple segments contain the midpoint, pick the one with most overlap
                    let overlap_a = a.end.min(entry_end) - a.start.max(entry_start);
                    let overlap_b = b.end.min(entry_end) - b.start.max(entry_start);
                    overlap_a.partial_cmp(&overlap_b).unwrap()
                })
                .map(|seg| seg.speaker.as_str());

            if let Some(speaker) = best_speaker {
                entry.as_object_mut().unwrap().insert(
                    "speaker".to_string(),
                    serde_json::Value::String(speaker.to_string()),
                );
            }
        }
        println!(
            "✅ [ENHANCE] Speaker labels mapped to {} system entries",
            system_entries.len()
        );
    }

    // ═══ Step 4: Echo cleanup using text-level LCS ═══
    let pre_echo_count = user_entries.len();
    let user_entries = filter_echo_by_text(user_entries, &system_entries);
    println!(
        "🔇 [ENHANCE] Echo cleanup: {} → {} user entries ({} removed)",
        pre_echo_count,
        user_entries.len(),
        pre_echo_count - user_entries.len()
    );

    // ═══ Step 5: Deduplicate sliding window overlaps ═══
    let pre_dedup_sys = system_entries.len();
    let pre_dedup_usr = user_entries.len();
    let system_entries = dedup_overlapping(system_entries);
    let user_entries = dedup_overlapping(user_entries);
    println!(
        "🔗 [ENHANCE] Dedup: system {} → {}, user {} → {}",
        pre_dedup_sys,
        system_entries.len(),
        pre_dedup_usr,
        user_entries.len()
    );

    // ═══ Step 6: Merge, sort, write, emit ═══
    let mut final_entries: Vec<serde_json::Value> = Vec::new();
    final_entries.extend(system_entries);
    final_entries.extend(user_entries);

    // Sort by start time
    final_entries.sort_by(|a, b| {
        let a_start = a.get("start").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let b_start = b.get("start").and_then(|v| v.as_f64()).unwrap_or(0.0);
        a_start
            .partial_cmp(&b_start)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Write to temp file first, then atomic rename to protect against crash/quit.
    // If interrupted, the original real-time transcript.jsonl remains intact.
    let tmp_path = transcript_path.with_extension("jsonl.tmp");
    let mut file = std::fs::File::create(&tmp_path)?;
    let mut user_count = 0;
    let mut system_count = 0;
    for entry in &final_entries {
        let speaker = entry.get("speaker").and_then(|s| s.as_str()).unwrap_or("");
        if speaker == "user" {
            user_count += 1;
        } else {
            system_count += 1;
        }
        if let Ok(line) = serde_json::to_string(entry) {
            writeln!(file, "{}", line)?;
        }
    }
    file.flush()?;
    drop(file); // Close before rename
    std::fs::rename(&tmp_path, transcript_path)?;

    // Emit to frontend only during active recording session (not during startup catchup)
    if emit_to_frontend {
        for entry in &final_entries {
            let text = entry.get("text").and_then(|t| t.as_str()).unwrap_or("");
            let speaker = entry
                .get("speaker")
                .and_then(|s| s.as_str())
                .unwrap_or("system");
            let payload = serde_json::json!({
                "text": text,
                "source": speaker,
                "is_final": true
            });
            let _ = app.emit("asr_final", &payload);
        }
    }

    // Write .enhanced marker so we don't re-process on next startup
    if let Some(session_dir) = transcript_path.parent() {
        let marker = session_dir.join(".enhanced");
        let _ = std::fs::File::create(&marker);
    }

    println!(
        "✅ [ENHANCE v10.0] Complete: {} user + {} system = {} total",
        user_count,
        system_count,
        final_entries.len()
    );

    Ok(final_entries.len())
}

/// Filter user entries whose text is mostly echo of system audio.
/// Uses LCS (longest common substring) comparison against real-time system entries
/// in the same time range.
fn filter_echo_by_text(
    user_entries: Vec<serde_json::Value>,
    system_entries: &[serde_json::Value],
) -> Vec<serde_json::Value> {
    user_entries
        .into_iter()
        .filter(|user_entry| {
            let user_text = user_entry
                .get("text")
                .and_then(|t| t.as_str())
                .unwrap_or("");
            let user_start = user_entry
                .get("start")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0) as f32;
            let user_end = user_entry
                .get("end")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0) as f32;

            // Extract meaningful characters (Chinese + alphanumeric)
            let user_chars: Vec<char> = user_text
                .chars()
                .filter(|c| {
                    ('\u{4e00}'..='\u{9fff}').contains(c)
                        || ('\u{3400}'..='\u{4dbf}').contains(c)
                        || c.is_ascii_alphanumeric()
                })
                .collect();

            if user_chars.is_empty() || user_chars.len() < 3 {
                return true; // Too short to judge, keep
            }

            // Collect system text from temporally overlapping entries
            let mut combined_system_text = String::new();
            for sys_entry in system_entries {
                let sys_start = sys_entry
                    .get("start")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.0) as f32;
                let sys_end = sys_entry.get("end").and_then(|v| v.as_f64()).unwrap_or(0.0) as f32;

                let overlap_start = user_start.max(sys_start);
                let overlap_end = user_end.min(sys_end);
                if overlap_end > overlap_start {
                    if let Some(text) = sys_entry.get("text").and_then(|t| t.as_str()) {
                        combined_system_text.push_str(text);
                    }
                }
            }

            if combined_system_text.is_empty() {
                return true; // No overlapping system entries → not echo
            }

            let system_chars: Vec<char> = combined_system_text
                .chars()
                .filter(|c| {
                    ('\u{4e00}'..='\u{9fff}').contains(c)
                        || ('\u{3400}'..='\u{4dbf}').contains(c)
                        || c.is_ascii_alphanumeric()
                })
                .collect();

            let lcs_len = longest_common_subsequence(&user_chars, &system_chars);
            let echo_ratio = lcs_len as f32 / user_chars.len() as f32;

            if echo_ratio > 0.5 {
                let display: String = user_text.chars().take(30).collect();
                println!(
                    "  🔇 ECHO [{:.1}s-{:.1}s]: \"{}\" (LCS={}/{}, ratio={:.0}%)",
                    user_start,
                    user_end,
                    display,
                    lcs_len,
                    user_chars.len(),
                    echo_ratio * 100.0
                );
                false // Remove
            } else {
                true // Keep
            }
        })
        .collect()
}

/// Compute the length of the longest common subsequence between two char slices.
fn longest_common_subsequence(a: &[char], b: &[char]) -> usize {
    if a.is_empty() || b.is_empty() {
        return 0;
    }
    let m = a.len();
    let n = b.len();
    let mut prev = vec![0usize; n + 1];
    let mut curr = vec![0usize; n + 1];

    for i in 1..=m {
        for j in 1..=n {
            if a[i - 1] == b[j - 1] {
                curr[j] = prev[j - 1] + 1;
            } else {
                curr[j] = prev[j].max(curr[j - 1]);
            }
        }
        std::mem::swap(&mut prev, &mut curr);
    }

    prev[n]
}

/// Deduplicate overlapping entries from the same channel.
/// When consecutive entries overlap >50% in time, keep the later one (more complete context).
fn dedup_overlapping(mut entries: Vec<serde_json::Value>) -> Vec<serde_json::Value> {
    if entries.len() < 2 {
        return entries;
    }

    // Sort by start time
    entries.sort_by(|a, b| {
        let a_start = a.get("start").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let b_start = b.get("start").and_then(|v| v.as_f64()).unwrap_or(0.0);
        a_start
            .partial_cmp(&b_start)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut result: Vec<serde_json::Value> = Vec::with_capacity(entries.len());

    for entry in entries {
        let entry_start = entry.get("start").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let entry_end = entry.get("end").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let entry_dur = entry_end - entry_start;

        if let Some(last) = result.last() {
            let last_start = last.get("start").and_then(|v| v.as_f64()).unwrap_or(0.0);
            let last_end = last.get("end").and_then(|v| v.as_f64()).unwrap_or(0.0);

            let overlap_start = entry_start.max(last_start);
            let overlap_end = entry_end.min(last_end);
            let overlap = (overlap_end - overlap_start).max(0.0);

            let shorter_dur = entry_dur.min(last_end - last_start);
            let overlap_ratio = if shorter_dur > 0.0 {
                overlap / shorter_dur
            } else {
                0.0
            };

            if overlap_ratio > 0.5 {
                // Significant overlap: keep the later entry (more complete context)
                // but extend its start to cover the earlier entry's time range
                let mut merged = entry;
                if let Some(obj) = merged.as_object_mut() {
                    obj.insert("start".to_string(), serde_json::json!(last_start));
                }
                *result.last_mut().unwrap() = merged;
                continue;
            }
        }
        result.push(entry);
    }

    result
}

// ══════════════════════════════════════════════════════════════════════════
// Startup Enhancement — process sessions that weren't enhanced before app quit
// ══════════════════════════════════════════════════════════════════════════

/// Scan sessions directory for unenhanced sessions and process them.
/// Called once at startup, after diarization models are loaded.
/// Runs in the calling thread (caller should spawn a background thread).
/// Does NOT emit to frontend (startup catchup is silent).
pub fn enhance_pending_sessions(app: &tauri::AppHandle) {
    use tauri::Manager;

    let sessions_dir = match app.path().app_data_dir() {
        Ok(base) => base.join("sessions"),
        Err(e) => {
            eprintln!("⚠️ [ENHANCE] Cannot resolve app data dir: {}", e);
            return;
        }
    };

    if !sessions_dir.exists() {
        return;
    }

    // Collect session dirs, sort newest first
    let mut session_dirs: Vec<std::path::PathBuf> = match std::fs::read_dir(&sessions_dir) {
        Ok(entries) => entries
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.is_dir())
            .collect(),
        Err(_) => return,
    };
    session_dirs.sort_by(|a, b| b.cmp(a)); // newest first

    // Only process the most recent N unenhanced sessions.
    // Older sessions get a marker so we don't scan them every startup.
    let max_to_process = 3;
    let mut enhanced_count = 0;
    let mut skipped_old = 0;

    for session_dir in &session_dirs {
        let marker = session_dir.join(".enhanced");
        if marker.exists() {
            continue; // Already enhanced
        }

        let transcript = session_dir.join("transcript.jsonl");

        if !transcript.exists() {
            // No transcript — mark and skip
            let _ = std::fs::File::create(&marker);
            continue;
        }

        if enhanced_count >= max_to_process {
            // Older sessions: mark as skipped to avoid re-scanning
            let _ = std::fs::File::create(&marker);
            skipped_old += 1;
            continue;
        }

        let system_wav = session_dir.join("system.wav");
        let session_name = session_dir
            .file_name()
            .unwrap_or_default()
            .to_string_lossy();
        println!("🔄 [STARTUP] Enhancing pending session: {}", session_name);

        match enhance_transcript(&system_wav, &transcript, app, false) {
            Ok(n) => {
                println!(
                    "✅ [STARTUP] Session {} enhanced: {} entries",
                    session_name, n
                );
                enhanced_count += 1;
            }
            Err(e) => {
                eprintln!("⚠️ [STARTUP] Failed to enhance {}: {}", session_name, e);
                // Don't write marker — will retry next startup
            }
        }
    }

    if skipped_old > 0 {
        println!("⏭️ [STARTUP] Skipped {} older session(s)", skipped_old);
    }

    if enhanced_count > 0 {
        println!(
            "✅ [STARTUP] Enhanced {} pending session(s)",
            enhanced_count
        );
    }
}
