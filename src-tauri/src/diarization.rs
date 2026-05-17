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
            threshold: 0.5,   // default clustering threshold
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

    let segments: Vec<DiarizationSegment> = segments_raw
        .into_iter()
        .map(|s| {
            let label = format!("Speaker {}", s.speaker + 1);
            DiarizationSegment {
                start: s.start,
                end: s.end,
                speaker: label,
            }
        })
        .collect();

    // Print summary
    for seg in &segments {
        println!(
            "  📌 {:.1}s - {:.1}s: {}",
            seg.start, seg.end, seg.speaker
        );
    }

    Ok(segments)
}

/// Re-label a transcript.jsonl file using diarization results.
/// Each line in the JSONL has a "timestamp" (millis since epoch) and "speaker" field.
/// We replace "speaker" with the diarization-derived label.
pub fn relabel_transcript(
    transcript_path: &Path,
    segments: &[DiarizationSegment],
    recording_start_ms: u128,
) -> Result<usize> {
    use std::io::{BufRead, Write};

    if segments.is_empty() {
        println!("⚠️ [DIARIZATION] No segments to relabel with");
        return Ok(0);
    }

    let file = std::fs::File::open(transcript_path)?;
    let reader = std::io::BufReader::new(file);
    let mut updated_lines: Vec<String> = Vec::new();
    let mut relabeled_count = 0;

    for line in reader.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }

        match serde_json::from_str::<serde_json::Value>(&line) {
            Ok(mut entry) => {
                if let Some(ts_ms) = entry.get("timestamp").and_then(|v| v.as_u64()) {
                    // Convert absolute timestamp to relative seconds from recording start
                    let relative_secs =
                        (ts_ms as f64 - recording_start_ms as f64) / 1000.0;

                    // Find which diarization segment this timestamp falls into
                    if let Some(speaker) = find_speaker_at(segments, relative_secs as f32) {
                        entry["speaker"] = serde_json::Value::String(speaker.clone());
                        relabeled_count += 1;
                    }
                }
                updated_lines.push(serde_json::to_string(&entry)?);
            }
            Err(_) => {
                updated_lines.push(line);
            }
        }
    }

    // Write back
    let mut file = std::fs::File::create(transcript_path)?;
    for line in &updated_lines {
        writeln!(file, "{}", line)?;
    }

    println!(
        "✅ [DIARIZATION] Relabeled {}/{} transcript entries",
        relabeled_count,
        updated_lines.len()
    );

    Ok(relabeled_count)
}

// ── Data types ─────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct DiarizationSegment {
    pub start: f32,
    pub end: f32,
    pub speaker: String,
}

// ── Helpers ─────────────────────────────────────────────────────────────

/// Find which speaker is talking at a given time (seconds from recording start).
fn find_speaker_at(segments: &[DiarizationSegment], time_secs: f32) -> Option<&String> {
    // Find the segment that contains this timestamp
    for seg in segments {
        if time_secs >= seg.start && time_secs <= seg.end {
            return Some(&seg.speaker);
        }
    }

    // If no exact match, find the closest segment
    let mut best: Option<(f32, &String)> = None;
    for seg in segments {
        let mid = (seg.start + seg.end) / 2.0;
        let dist = (time_secs - mid).abs();
        if best.is_none() || dist < best.unwrap().0 {
            best = Some((dist, &seg.speaker));
        }
    }

    // Only use closest if within 5 seconds
    best.filter(|(dist, _)| *dist < 5.0).map(|(_, speaker)| speaker)
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
