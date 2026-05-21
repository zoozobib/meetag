//! Speaker Diarization v9.0 — Offline post-processing via sherpa-onnx OfflineSpeakerDiarization.
//!
//! Architecture:
//!   - During recording: no speaker identification at all. Just transcribe with timestamps.
//!   - After recording: run dual-channel processing:
//!     1. system.wav → OfflineSpeakerDiarization (Pyannote + embedding + clustering)
//!     2. mic.wav → Silero VAD (lightweight) + echo filtering + ASR
//!     3. Merge both by timestamp into transcript.jsonl
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

    // Collect all entries (both new system segments and existing mic entries)
    let mut final_entries: Vec<serde_json::Value> = realtime_mic_entries;

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

        // Emit to frontend for display
        let payload = serde_json::json!({
            "text": text,
            "source": seg.speaker,
            "is_final": true
        });
        let _ = app.emit("asr_final", &payload);

        final_entries.push(entry);
        transcribed += 1;
    }

    // Sort all entries chronologically by start time
    final_entries.sort_by(|a, b| {
        let a_start = a.get("start").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let b_start = b.get("start").and_then(|v| v.as_f64()).unwrap_or(0.0);
        a_start.partial_cmp(&b_start).unwrap_or(std::cmp::Ordering::Equal)
    });

    // Write the perfectly sorted timeline to transcript.jsonl
    let mut user_entries = 0;
    for entry in final_entries {
        if entry.get("speaker").and_then(|s| s.as_str()) == Some("user") {
            user_entries += 1;
        }
        if let Ok(line) = serde_json::to_string(&entry) {
            writeln!(file, "{}", line)?;
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

// ══════════════════════════════════════════════════════════════════════════
// Dual-Channel Post-Processing (v9.1 — Gap-based approach)
// ══════════════════════════════════════════════════════════════════════════

/// Run Silero VAD on a complete audio buffer (16kHz mono).
/// Returns a list of (start_sec, end_sec) speech segments.
fn run_vad_offline(
    samples_16k: &[f32],
    app: &tauri::AppHandle,
) -> Result<Vec<(f32, f32)>> {
    use tauri::Manager;

    let model_path = app
        .path()
        .resource_dir()
        .map_err(|e| anyhow::anyhow!("Failed to get resource dir: {}", e))?
        .join("resources/silero_vad.onnx");

    let vad_config = sherpa_onnx::VadModelConfig {
        silero_vad: sherpa_onnx::SileroVadModelConfig {
            model: Some(model_path.to_string_lossy().to_string()),
            threshold: 0.5,
            min_silence_duration: 0.8,
            min_speech_duration: 0.3,
            window_size: 512,
            max_speech_duration: 30.0,
        },
        sample_rate: 16000,
        num_threads: 1,
        provider: Some("cpu".to_string()),
        debug: false,
        ..Default::default()
    };

    let vad = sherpa_onnx::VoiceActivityDetector::create(&vad_config, 60.0)
        .ok_or_else(|| anyhow::anyhow!("Failed to create Silero VAD for offline processing"))?;

    // Feed all audio through VAD in chunks of 512 samples (window_size)
    let window_size = 512;
    for chunk in samples_16k.chunks(window_size) {
        if chunk.len() == window_size {
            vad.accept_waveform(chunk);
        }
    }
    // Flush remaining speech
    vad.flush();

    // Collect all detected speech segments
    let mut segments = Vec::new();
    while !vad.is_empty() {
        if let Some(seg) = vad.front() {
            let start_sec = seg.start() as f32 / 16000.0;
            let duration_sec = seg.samples().len() as f32 / 16000.0;
            let end_sec = start_sec + duration_sec;
            segments.push((start_sec, end_sec));
        }
        vad.pop();
    }

    Ok(segments)
}

/// Main dual-channel post-processing function (v9.1 — Gap-based approach).
///
/// Strategy:
///   - Preserve real-time user entries (they used raw audio switch, already quality-filtered)
///   - Run diarization + ASR on system.wav (for remote speakers)
///   - Find silence gaps in system timeline (no remote speaker active)
///   - Only process mic.wav during these gaps (zero echo risk)
///   - De-duplicate + merge + write transcript.jsonl
pub fn process_dual_channel(
    system_wav: &Path,
    mic_wav: &Path,
    transcript_path: &Path,
    app: &tauri::AppHandle,
) -> Result<usize> {
    use std::io::Write;
    use tauri::Emitter;

    println!("🔄 [DUAL-CHANNEL v9.1] Starting gap-based post-processing...");
    println!("  system.wav: {}", system_wav.display());
    println!("  mic.wav: {}", mic_wav.display());

    let backend = crate::settings::SETTINGS.read().unwrap().asr.backend;

    // ═══════════════════════════════════════════════════════════════════════
    // PHASE 0: Read real-time user entries from existing transcript.jsonl
    // ═══════════════════════════════════════════════════════════════════════
    println!("═══ PHASE 0: Reading real-time user entries ═══");
    let realtime_user_entries: Vec<serde_json::Value> = if transcript_path.exists() {
        use std::io::BufRead;
        std::fs::File::open(transcript_path)
            .ok()
            .map(|f| {
                std::io::BufReader::new(f)
                    .lines()
                    .filter_map(|line| line.ok())
                    .filter_map(|line| serde_json::from_str::<serde_json::Value>(&line).ok())
                    .filter(|e| e.get("speaker").and_then(|s| s.as_str()) == Some("user"))
                    .filter(|e| e.get("rms").is_some()) // only real-time entries have RMS field
                    .collect()
            })
            .unwrap_or_default()
    } else {
        Vec::new()
    };
    println!("📋 [DUAL-CHANNEL] Found {} real-time user entries to preserve", realtime_user_entries.len());

    // ═══════════════════════════════════════════════════════════════════════
    // PHASE 1: Process system.wav through diarization pipeline
    // ═══════════════════════════════════════════════════════════════════════
    println!("═══ PHASE 1: System audio diarization ═══");
    let system_entries = process_system_channel(system_wav, backend, app)?;
    println!("✅ [DUAL-CHANNEL] System channel: {} entries", system_entries.len());

    // ═══════════════════════════════════════════════════════════════════════
    // PHASE 2: Find silence gaps in system timeline
    // ═══════════════════════════════════════════════════════════════════════
    println!("═══ PHASE 2: Computing silence gaps ═══");
    let gaps = find_system_gaps(&system_entries, mic_wav)?;
    println!("📊 [DUAL-CHANNEL] Found {} silence gaps (system inactive)", gaps.len());
    for (i, &(gs, ge)) in gaps.iter().enumerate() {
        println!("  Gap {}: {:.1}s - {:.1}s ({:.1}s)", i + 1, gs, ge, ge - gs);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // PHASE 3: Process mic.wav ONLY during silence gaps
    // ═══════════════════════════════════════════════════════════════════════
    println!("═══ PHASE 3: Gap-based mic processing ═══");
    let gap_entries = process_mic_gaps(mic_wav, &gaps, &realtime_user_entries, backend, app)?;
    println!("✅ [DUAL-CHANNEL] Gap processing: {} new user entries", gap_entries.len());

    // ═══════════════════════════════════════════════════════════════════════
    // PHASE 4: Merge all + write
    // ═══════════════════════════════════════════════════════════════════════
    println!("═══ PHASE 4: Merging and writing ═══");
    let mut final_entries: Vec<serde_json::Value> = Vec::new();
    final_entries.extend(realtime_user_entries);
    final_entries.extend(system_entries);
    final_entries.extend(gap_entries);

    // Sort by start time
    final_entries.sort_by(|a, b| {
        let a_start = a.get("start").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let b_start = b.get("start").and_then(|v| v.as_f64()).unwrap_or(0.0);
        a_start.partial_cmp(&b_start).unwrap_or(std::cmp::Ordering::Equal)
    });

    // Write to transcript.jsonl
    let mut file = std::fs::File::create(transcript_path)?;
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

    // Emit all entries to frontend for display
    for entry in &final_entries {
        let text = entry.get("text").and_then(|t| t.as_str()).unwrap_or("");
        let speaker = entry.get("speaker").and_then(|s| s.as_str()).unwrap_or("system");
        let payload = serde_json::json!({
            "text": text,
            "source": speaker,
            "is_final": true
        });
        let _ = app.emit("asr_final", &payload);
    }

    println!(
        "✅ [DUAL-CHANNEL v9.1] Complete: {} user (realtime+gap) + {} system = {} total",
        user_count, system_count, final_entries.len()
    );

    Ok(final_entries.len())
}

/// Find silence gaps in the system timeline.
/// Returns a list of (start_sec, end_sec) intervals where NO system speaker is active.
fn find_system_gaps(
    system_entries: &[serde_json::Value],
    mic_wav: &Path,
) -> Result<Vec<(f32, f32)>> {
    // Get total audio duration from mic.wav
    let (mic_samples, mic_sr) = read_wav_f32(mic_wav)?;
    let total_duration = mic_samples.len() as f32 / mic_sr as f32;

    if system_entries.is_empty() {
        // No system speech at all → entire recording is a gap
        return Ok(vec![(0.0, total_duration)]);
    }

    // Extract system active intervals
    let mut active_intervals: Vec<(f32, f32)> = system_entries
        .iter()
        .filter_map(|e| {
            let start = e.get("start").and_then(|v| v.as_f64())? as f32;
            let end = e.get("end").and_then(|v| v.as_f64())? as f32;
            Some((start, end))
        })
        .collect();

    // Sort by start time
    active_intervals.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    // Merge overlapping intervals
    let mut merged: Vec<(f32, f32)> = Vec::new();
    for (start, end) in active_intervals {
        if let Some(last) = merged.last_mut() {
            if start <= last.1 + 0.5 {
                // Allow 0.5s tolerance for near-continuous speech
                last.1 = last.1.max(end);
                continue;
            }
        }
        merged.push((start, end));
    }

    // Compute gaps (intervals between merged active intervals)
    let mut gaps: Vec<(f32, f32)> = Vec::new();

    // Gap before first active interval
    if let Some(&(first_start, _)) = merged.first() {
        if first_start > 0.5 {
            gaps.push((0.0, first_start));
        }
    }

    // Gaps between active intervals
    for window in merged.windows(2) {
        let gap_start = window[0].1;
        let gap_end = window[1].0;
        if gap_end - gap_start > 0.5 {
            gaps.push((gap_start, gap_end));
        }
    }

    // Gap after last active interval
    if let Some(&(_, last_end)) = merged.last() {
        if total_duration - last_end > 0.5 {
            gaps.push((last_end, total_duration));
        }
    }

    Ok(gaps)
}

/// Process mic.wav ONLY during silence gaps.
/// Since system is silent during these gaps, there's zero echo — mic audio is pure user speech.
fn process_mic_gaps(
    mic_wav: &Path,
    gaps: &[(f32, f32)],
    realtime_entries: &[serde_json::Value],
    backend: crate::settings::AsrBackend,
    app: &tauri::AppHandle,
) -> Result<Vec<serde_json::Value>> {
    let mut entries = Vec::new();

    if gaps.is_empty() {
        println!("  ℹ️ No silence gaps found — system spoke continuously");
        return Ok(entries);
    }

    // Read and resample mic audio
    let (mic_raw, mic_sr) = read_wav_f32(mic_wav)?;
    let mic_16k = if mic_sr != 16000 {
        resample(&mic_raw, mic_sr, 16000)
    } else {
        mic_raw
    };
    let mic_total = mic_16k.len();

    println!("  🎤 Mic audio: {:.1}s, processing {} gaps", mic_total as f32 / 16000.0, gaps.len());

    for (gap_idx, &(gap_start, gap_end)) in gaps.iter().enumerate() {
        let start_sample = (gap_start * 16000.0) as usize;
        let end_sample = ((gap_end * 16000.0) as usize).min(mic_total);

        if start_sample >= end_sample || start_sample >= mic_total {
            continue;
        }

        let gap_audio = &mic_16k[start_sample..end_sample];
        let gap_duration = gap_audio.len() as f32 / 16000.0;

        if gap_duration < 0.5 {
            continue;
        }

        // Run VAD on this gap's audio
        let vad_segments = run_vad_offline(gap_audio, app)?;

        for (seg_idx, &(vad_start, vad_end)) in vad_segments.iter().enumerate() {
            // Convert VAD-relative times to absolute times
            let abs_start = gap_start + vad_start;
            let abs_end = gap_start + vad_end;
            let seg_duration = abs_end - abs_start;

            if seg_duration < 0.3 {
                continue;
            }

            // De-duplicate: check if a real-time entry already covers this time
            let already_covered = realtime_entries.iter().any(|e| {
                let rt_start = e.get("start").and_then(|v| v.as_f64()).unwrap_or(0.0) as f32;
                let rt_end = e.get("end").and_then(|v| v.as_f64()).unwrap_or(0.0) as f32;
                // Overlap check
                let overlap_start = abs_start.max(rt_start);
                let overlap_end = abs_end.min(rt_end);
                let overlap = (overlap_end - overlap_start).max(0.0);
                overlap / seg_duration > 0.5
            });

            if already_covered {
                println!("  ⏭️ Gap {} seg {}: already covered by real-time entry", gap_idx + 1, seg_idx + 1);
                continue;
            }

            // Extract audio for this VAD segment (relative to gap audio)
            let seg_start_sample = (vad_start * 16000.0) as usize;
            let seg_end_sample = ((vad_end * 16000.0) as usize).min(gap_audio.len());
            if seg_start_sample >= seg_end_sample {
                continue;
            }
            let seg_audio = &gap_audio[seg_start_sample..seg_end_sample];

            // Run ASR
            let text = match run_asr_on_segment(seg_audio, backend, seg_idx) {
                Ok(t) => t.trim().to_string(),
                Err(_) => continue,
            };

            if text.is_empty() || crate::text_filter::is_hallucination(&text) {
                continue;
            }

            // Skip non-Chinese output
            let chinese_chars = text.chars().filter(|c| ('\u{4e00}'..='\u{9fff}').contains(c) || ('\u{3400}'..='\u{4dbf}').contains(c)).count();
            let total_chars = text.chars().filter(|c| !c.is_ascii_punctuation() && !c.is_whitespace()).count();
            if total_chars > 0 && (chinese_chars as f32 / total_chars as f32) < 0.5 {
                continue;
            }

            let display_text: String = text.chars().take(30).collect();
            println!("  📝 GAP {:.1}s-{:.1}s [user]: {}", abs_start, abs_end, display_text);

            entries.push(serde_json::json!({
                "start": abs_start,
                "end": abs_end,
                "speaker": "user",
                "text": text,
                "source": "gap_supplement",
                "backend": match backend {
                    crate::settings::AsrBackend::FunAsr => "funasr",
                    crate::settings::AsrBackend::Whisper => "whisper",
                }
            }));
        }
    }

    println!("  📊 Gap processing: {} entries found in {} gaps", entries.len(), gaps.len());
    Ok(entries)
}

/// Process system.wav: diarization + per-segment ASR.
fn process_system_channel(
    system_wav: &Path,
    backend: crate::settings::AsrBackend,
    _app: &tauri::AppHandle,
) -> Result<Vec<serde_json::Value>> {
    let mut entries = Vec::new();

    // Run diarization (existing pipeline)
    let segments = process_wav(system_wav)?;

    // Read and resample audio for ASR
    let (all_samples, file_sample_rate) = read_wav_f32(system_wav)?;
    let samples_16k = if file_sample_rate != 16000 {
        resample(&all_samples, file_sample_rate, 16000)
    } else {
        all_samples
    };
    let total_samples = samples_16k.len();

    for (i, seg) in segments.iter().enumerate() {
        let start_sample = (seg.start * 16000.0) as usize;
        let end_sample = ((seg.end * 16000.0) as usize).min(total_samples);

        if start_sample >= end_sample || start_sample >= total_samples {
            continue;
        }

        let segment_audio = &samples_16k[start_sample..end_sample];
        let duration_s = segment_audio.len() as f32 / 16000.0;
        if duration_s < 0.3 {
            continue;
        }

        // Run ASR
        let text = run_asr_on_segment(segment_audio, backend, i)?;
        let text = text.trim().to_string();

        if text.is_empty() || crate::text_filter::is_hallucination(&text) {
            continue;
        }

        // Skip non-Chinese output
        let chinese_chars = text.chars().filter(|c| ('\u{4e00}'..='\u{9fff}').contains(c) || ('\u{3400}'..='\u{4dbf}').contains(c)).count();
        let total_chars = text.chars().filter(|c| !c.is_ascii_punctuation() && !c.is_whitespace()).count();
        if total_chars > 0 && (chinese_chars as f32 / total_chars as f32) < 0.5 {
            println!("  🗑️ Skipping non-Chinese system segment: {:?}", text);
            continue;
        }

        let display_text: String = text.chars().take(30).collect();
        println!("  📝 SYS {:.1}s-{:.1}s [{}]: {}", seg.start, seg.end, seg.speaker, display_text);

        entries.push(serde_json::json!({
            "start": seg.start,
            "end": seg.end,
            "speaker": seg.speaker,
            "text": text,
            "backend": match backend {
                crate::settings::AsrBackend::FunAsr => "funasr",
                crate::settings::AsrBackend::Whisper => "whisper",
            }
        }));
    }

    Ok(entries)
}

/// Run ASR on a single audio segment (16kHz mono f32).
fn run_asr_on_segment(
    segment_audio: &[f32],
    backend: crate::settings::AsrBackend,
    seg_index: usize,
) -> Result<String> {
    match backend {
        crate::settings::AsrBackend::FunAsr => {
            let funasr = crate::funasr::SenseVoiceManager::get();
            let samples_i16: Vec<i16> = segment_audio
                .iter()
                .map(|&s| (s * 32767.0).clamp(-32768.0, 32767.0) as i16)
                .collect();
            match funasr.transcribe(&samples_i16) {
                Ok(result) => Ok(result.text),
                Err(e) => {
                    eprintln!("  ⚠️ Segment {}: ASR error: {}", seg_index + 1, e);
                    Err(anyhow::anyhow!("ASR error: {}", e))
                }
            }
        }
        crate::settings::AsrBackend::Whisper => {
            let whisper = crate::whisper::WhisperManager::get();
            match whisper.transcribe_f32(segment_audio, "zh", None) {
                Ok(result) => Ok(result.text),
                Err(e) => {
                    eprintln!("  ⚠️ Segment {}: ASR error: {}", seg_index + 1, e);
                    Err(anyhow::anyhow!("ASR error: {}", e))
                }
            }
        }
    }
}
