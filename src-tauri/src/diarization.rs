//! Speaker Diarization v8.0 — Offline post-processing via sherpa-onnx OfflineSpeakerDiarization.
//!
//! Architecture:
//!   - During recording: no speaker identification at all. Just transcribe with timestamps.
//!   - After recording: run OfflineSpeakerDiarization on the complete mix.wav.
//!     Pyannote segmentation + 3D-Speaker embeddings + clustering on the FULL recording
//!     gives far better accuracy than per-segment streaming matching.
//!   - Re-label transcript.jsonl entries by aligning timestamps with diarization segments.

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
    println!("🚀 [DIARIZATION] Initializing v8.0 (Offline Post-Processing)...");
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

    let config = OfflineSpeakerDiarizationConfig {
        segmentation: OfflineSpeakerSegmentationModelConfig {
            pyannote: OfflineSpeakerSegmentationPyannoteModelConfig {
                model: Some(segmentation_model.to_string_lossy().into_owned()),
            },
            num_threads: 2,
            debug: false,
            provider: Some("cpu".to_string()),
        },
        embedding: SpeakerEmbeddingExtractorConfig {
            model: Some(embedding_model.to_string_lossy().into_owned()),
            num_threads: 2,
            debug: false,
            provider: Some("cpu".to_string()),
        },
        clustering: FastClusteringConfig {
            num_clusters: -1, // auto-detect number of speakers
            threshold: 0.80,  // higher = fewer speakers (0.5 was too aggressive, splitting 2 people into dozens)
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
    let mut speaker_durations: std::collections::HashMap<i32, f32> = std::collections::HashMap::new();
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
                spk + 1, dur, pct
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
        let mut speaker_dur: std::collections::HashMap<String, f32> = std::collections::HashMap::new();
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
            let small_intervals: Vec<(f32, f32)> = segments.iter()
                .filter(|s| &s.speaker == small_spk)
                .map(|s| (s.start, s.end))
                .collect();

            // Check against each other speaker (prefer merging into the largest)
            for j in (0..speakers_sorted.len()).rev() {
                if i == j { continue; }
                let (ref candidate_spk, _) = speakers_sorted[j];

                let candidate_intervals: Vec<(f32, f32)> = segments.iter()
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
    let final_speakers: std::collections::HashSet<&str> = segments.iter().map(|s| s.speaker.as_str()).collect();
    println!(
        "✅ [DIARIZATION] Final: {} speakers, {} segments (merged {} noise speakers)",
        final_speakers.len(), segments.len(), noise_speakers.len()
    );
    for seg in &segments {
        println!(
            "  📌 {:.1}s - {:.1}s: {}",
            seg.start, seg.end, seg.speaker
        );
    }

    Ok(segments)
}

/// Transcribe each diarization segment independently using ASR.
///
/// This is the correct architecture: instead of relabeling real-time transcript entries
/// (which mix multiple speakers), we extract audio for each diarization segment and
/// run ASR on it. Each resulting text entry is guaranteed to belong to one speaker.
///
/// Returns the number of segments successfully transcribed.
pub fn transcribe_segments(
    wav_path: &Path,
    segments: &[DiarizationSegment],
    transcript_path: &Path,
    app: &tauri::AppHandle,
) -> Result<usize> {
    use std::io::Write;
    use tauri::Emitter;

    if segments.is_empty() {
        println!("⚠️ [DIARIZATION] No segments to transcribe");
        return Ok(0);
    }

    // Read the full audio (reuse the same WAV we already loaded for diarization)
    let (all_samples, file_sample_rate) = read_wav_f32(wav_path)?;

    // Resample to 16kHz if needed (ASR expects 16kHz)
    let (samples_16k, sr) = if file_sample_rate != 16000 {
        (resample(&all_samples, file_sample_rate, 16000), 16000u32)
    } else {
        (all_samples, file_sample_rate)
    };

    let total_samples = samples_16k.len();
    println!(
        "🔄 [DIARIZATION] Transcribing {} segments from {:.1}s audio",
        segments.len(),
        total_samples as f32 / sr as f32
    );

    // Determine which ASR backend to use
    let backend = crate::settings::SETTINGS.read().unwrap().asr.backend;
    let mut transcribed = 0;

    // Read existing mic entries BEFORE overwriting transcript file
    let realtime_mic_entries: Vec<serde_json::Value> = if transcript_path.exists() {
        use std::io::BufRead;
        std::fs::File::open(transcript_path)
            .ok()
            .map(|f| {
                std::io::BufReader::new(f)
                    .lines()
                    .filter_map(|line| line.ok())
                    .filter_map(|line| serde_json::from_str::<serde_json::Value>(&line).ok())
                    .filter(|e| e.get("speaker").and_then(|s| s.as_str()) == Some("user"))
                    .filter(|e| e.get("rms").is_some()) // only real-time entries have RMS
                    .collect()
            })
            .unwrap_or_default()
    } else {
        Vec::new()
    };
    println!(
        "📋 [DIARIZATION] Found {} mic entries for 'me' merge",
        realtime_mic_entries.len()
    );

    // Overwrite the transcript file with diarization-based results
    let mut file = std::fs::File::create(transcript_path)?;

    for (i, seg) in segments.iter().enumerate() {
        // Extract audio slice for this segment
        let start_sample = (seg.start * sr as f32) as usize;
        let end_sample = ((seg.end * sr as f32) as usize).min(total_samples);

        if start_sample >= end_sample || start_sample >= total_samples {
            continue;
        }

        let segment_audio = &samples_16k[start_sample..end_sample];
        let duration_s = segment_audio.len() as f32 / sr as f32;

        // Skip very short segments (< 0.3s — likely noise)
        if duration_s < 0.3 {
            continue;
        }

        // Run ASR on this segment's audio
        let text = match backend {
            crate::settings::AsrBackend::FunAsr => {
                let funasr = crate::funasr::SenseVoiceManager::get();
                let samples_i16: Vec<i16> = segment_audio
                    .iter()
                    .map(|&s| (s * 32767.0).clamp(-32768.0, 32767.0) as i16)
                    .collect();
                match funasr.transcribe(&samples_i16) {
                    Ok(result) => result.text,
                    Err(e) => {
                        eprintln!("  ⚠️ Segment {}: ASR error: {}", i + 1, e);
                        continue;
                    }
                }
            }
            crate::settings::AsrBackend::Whisper => {
                let whisper = crate::whisper::WhisperManager::get();
                match whisper.transcribe_f32(segment_audio, "zh", None) {
                    Ok(result) => result.text,
                    Err(e) => {
                        eprintln!("  ⚠️ Segment {}: ASR error: {}", i + 1, e);
                        continue;
                    }
                }
            }
        };

        let text = text.trim().to_string();

        // Skip empty or hallucinated text
        if text.is_empty() || crate::text_filter::is_hallucination(&text) {
            continue;
        }

        // Skip non-Chinese output
        let chinese_chars = text.chars().filter(|c| ('\u{4e00}'..='\u{9fff}').contains(c) || ('\u{3400}'..='\u{4dbf}').contains(c)).count();
        let total_chars = text.chars().filter(|c| !c.is_ascii_punctuation() && !c.is_whitespace()).count();
        if total_chars > 0 && (chinese_chars as f32 / total_chars as f32) < 0.5 {
            println!("  🗑️ Skipping non-Chinese segment: {:?}", text);
            continue;
        }

        // Safe UTF-8 truncation for log display
        let display_text: String = text.chars().take(30).collect();
        println!(
            "  📝 {:.1}s-{:.1}s [{}]: {}",
            seg.start, seg.end, seg.speaker, display_text
        );

        // Write to transcript.jsonl
        let entry = serde_json::json!({
            "start": seg.start,
            "end": seg.end,
            "speaker": seg.speaker,
            "text": text,
            "backend": match backend {
                crate::settings::AsrBackend::FunAsr => "funasr",
                crate::settings::AsrBackend::Whisper => "whisper",
            }
        });
        if let Ok(line) = serde_json::to_string(&entry) {
            writeln!(file, "{}", line)?;
        }

        // Emit to frontend for display
        let payload = serde_json::json!({
            "text": text,
            "source": seg.speaker,
        });
        let _ = app.emit("asr_final", payload.to_string());

        transcribed += 1;
    }

    // ── Add user ("me") entries from real-time mic transcript ────────────
    // Read the pre-diarization transcript.jsonl (saved before overwrite) for mic entries.
    // Only include entries with RMS > 0.06 (genuine user speech, not echo).
    let mut user_entries = 0;
    for entry in &realtime_mic_entries {
        let rms = entry.get("rms").and_then(|v| v.as_f64()).unwrap_or(0.0);
        if rms <= 0.06 {
            continue; // echo — skip
        }
        if let Some(text) = entry.get("text").and_then(|t| t.as_str()) {
            if text.is_empty() {
                continue;
            }
            let ts_ms = entry.get("timestamp").and_then(|v| v.as_u64()).unwrap_or(0);
            let backend_name = entry.get("backend").and_then(|b| b.as_str()).unwrap_or("unknown");

            let me_entry = serde_json::json!({
                "start": 0.0, // real-time entries don't have WAV-relative timestamps
                "end": 0.0,
                "speaker": "me",
                "text": text,
                "timestamp": ts_ms,
                "backend": backend_name,
            });
            if let Ok(line) = serde_json::to_string(&me_entry) {
                writeln!(file, "{}", line)?;
            }

            // Emit to frontend
            let payload = serde_json::json!({ "text": text, "source": "me" });
            let _ = app.emit("asr_final", payload.to_string());

            println!("  👤 [me] (RMS={:.4}): {}", rms, text);
            user_entries += 1;
        }
    }

    println!(
        "✅ [DIARIZATION] Transcribed {}/{} system segments + {} user entries",
        transcribed, segments.len(), user_entries
    );

    Ok(transcribed + user_entries)
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
        hound::SampleFormat::Float => reader
            .samples::<f32>()
            .filter_map(|s| s.ok())
            .collect(),
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
