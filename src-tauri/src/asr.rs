use anyhow::Result;
use once_cell::sync::OnceCell;
use tauri::{Emitter, Manager};

use crate::text_filter;

/// Global singleton for AudioProcessor (lazily initialized)
static AUDIO_PROCESSOR: OnceCell<crate::audio_processor::AudioProcessor> = OnceCell::new();

/// Realtime ASR worker — powered by Sherpa-ONNX Silero VAD.
///
/// Architecture:
/// - System channel ("system"): results are emitted to frontend immediately for real-time display.
/// - Mic channel ("user"): results are written to transcript.jsonl ONLY.
///   No frontend emission — echo is handled at the capture layer (raw signal routing).
///   The offline diarization produces the final speaker-attributed result.
pub fn realtime_inference_worker(
    app: tauri::AppHandle,
    stop: std::sync::Arc<std::sync::atomic::AtomicBool>,
    mut rx: std::sync::mpsc::Receiver<i16>,
    source: String,
    transcript_writer: std::sync::Arc<std::sync::Mutex<std::fs::File>>,
) -> Result<(), String> {
    use std::sync::atomic::Ordering;
    use std::time::Duration;

    // ── Initialize Sherpa VAD (streaming mode) ──────────────────────────
    let model_path = app
        .path()
        .resource_dir()
        .map_err(|e| format!("Failed to get resource dir: {}", e))?
        .join("resources/silero_vad.onnx");

    let vad_config = sherpa_onnx::VadModelConfig {
        silero_vad: sherpa_onnx::SileroVadModelConfig {
            model: Some(model_path.to_string_lossy().to_string()),
            threshold: 0.5,
            min_silence_duration: 0.5,
            min_speech_duration: 0.25,
            window_size: 512,
            max_speech_duration: 10.0,
        },
        sample_rate: 16000,
        num_threads: 1,
        provider: Some("cpu".to_string()),
        debug: false,
        ..Default::default()
    };

    let vad = sherpa_onnx::VoiceActivityDetector::create(&vad_config, 60.0)
        .ok_or_else(|| "Failed to create Sherpa VAD".to_string())?;

    println!("✅ [ASR/{}] Sherpa Silero VAD initialized (streaming mode)", source);

    // ── State ────────────────────────────────────────────────────────────
    let mut last_transcript = String::new();
    let mut sample_buf: Vec<f32> = Vec::with_capacity(16000);
    let mut segment_count: u64 = 0;
    let is_system = source == "system";

    // ── Main loop ────────────────────────────────────────────────────────
    while !stop.load(Ordering::Relaxed) {
        match rx.recv_timeout(Duration::from_millis(100)) {
            Ok(s) => {
                let f32_sample = s as f32 / 32768.0;
                sample_buf.push(f32_sample);

                if sample_buf.len() >= 512 {
                    vad.accept_waveform(&sample_buf);
                    sample_buf.clear();

                    while !vad.is_empty() {
                        if let Some(segment) = vad.front() {
                            let samples = segment.samples();
                            let duration_s = samples.len() as f32 / 16000.0;
                            segment_count += 1;

                            println!(
                                "🎤 [VAD/{}] Segment #{}: {:.2}s ({} samples)",
                                source, segment_count, duration_s, samples.len()
                            );

                            process_segment(
                                &app, samples, &source, &transcript_writer,
                                &mut last_transcript, is_system,
                            );
                        }
                        vad.pop();
                    }
                }
            }
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                if !sample_buf.is_empty() {
                    vad.accept_waveform(&sample_buf);
                    sample_buf.clear();
                }
            }
            Err(_) => break,
        }
    }

    // ── Final flush ──────────────────────────────────────────────────────
    if !sample_buf.is_empty() {
        vad.accept_waveform(&sample_buf);
    }
    vad.flush();

    while !vad.is_empty() {
        if let Some(segment) = vad.front() {
            let samples = segment.samples();
            segment_count += 1;
            println!(
                "🎤 [VAD/{}] Final segment #{}: {:.2}s",
                source, segment_count, samples.len() as f32 / 16000.0
            );
            process_segment(
                &app, samples, &source, &transcript_writer,
                &mut last_transcript, is_system,
            );
        }
        vad.pop();
    }

    println!("✅ [ASR/{}] Worker stopped. {} segments processed.", source, segment_count);
    Ok(())
}

/// Process a single VAD segment: denoise → ASR → emit (system only) + log.
fn process_segment(
    app: &tauri::AppHandle,
    samples_f32: &[f32],
    source: &str,
    transcript_writer: &std::sync::Arc<std::sync::Mutex<std::fs::File>>,
    last_transcript: &mut String,
    emit_to_frontend: bool,
) {
    // Add 300ms silence padding
    let mut padded = samples_f32.to_vec();
    padded.resize(padded.len() + 4800, 0.0);

    // Noise reduction
    let audio_processor =
        AUDIO_PROCESSOR.get_or_init(|| crate::audio_processor::AudioProcessor::new(16000));
    let result = audio_processor.denoise(&padded);
    let denoised = &result.samples;

    println!(
        "🔇 [ASR] Noise reduction: RMS {:.4} → {:.4} ({:.1} dB)",
        result.rms_before, result.rms_after, result.noise_reduction_db
    );

    let backend = crate::settings::SETTINGS.read().unwrap().asr.backend;

    let (text, backend_name, inference_time_ms) = match backend {
        crate::settings::AsrBackend::FunAsr => {
            if !crate::funasr::SenseVoiceManager::is_initialized() {
                eprintln!("❌ [ASR] FunASR not initialized!");
                return;
            }
            let funasr = crate::funasr::SenseVoiceManager::get();
            let i16_samples: Vec<i16> = denoised
                .iter()
                .map(|&s| (s * 32767.0).clamp(-32768.0, 32767.0) as i16)
                .collect();
            match funasr.transcribe(&i16_samples) {
                Ok(r) => (r.text.trim().to_string(), "funasr", r.inference_time_ms),
                Err(e) => { eprintln!("❌ [FUNASR] Error: {}", e); return; }
            }
        }
        crate::settings::AsrBackend::Whisper => {
            let whisper = crate::whisper::WhisperManager::get();
            let prompt = if last_transcript.is_empty() {
                "这是一段会议记录。".to_string()
            } else {
                last_transcript.chars().rev().take(100).collect::<Vec<_>>().into_iter().rev().collect()
            };
            match whisper.transcribe_f32(denoised, "zh", Some(&prompt)) {
                Ok(r) => {
                    if r.segments.iter().any(|s| s.avg_logprob < -1.0) { return; }
                    (r.text.trim().to_string(), "whisper", r.inference_time_ms)
                }
                Err(e) => { eprintln!("❌ [WHISPER] Error: {}", e); return; }
            }
        }
    };

    if text.is_empty() { return; }

    if crate::text_filter::is_hallucination(&text) {
        println!("🗑️ [{}] Discarding hallucination: {:?}", backend_name.to_uppercase(), text);
        return;
    }

    *last_transcript = text.clone();

    let segment_rms = result.rms_before;

    // Determine whether to emit to frontend:
    // - System channel: always emit (real-time display)
    // - Mic channel: emit ONLY if RMS > 0.06 (user actually speaking into mic)
    //   Echo through speakers has RMS ~0.04, direct speech has RMS ~0.08-0.12.
    //   This 2x gap makes the gate robust across hardware configurations.
    let should_emit = if emit_to_frontend {
        true // system channel
    } else {
        // mic channel: energy gate
        if segment_rms > 0.06 {
            println!("🎙️ [ASR/user] User speech detected (RMS={:.4}): {:?}", segment_rms, text);
            true
        } else {
            println!("🔇 [ASR/user] Echo discarded (RMS={:.4}): {:?}", segment_rms, text);
            false
        }
    };

    if should_emit {
        let emit_source = if emit_to_frontend { source } else { "me" };
        let payload = serde_json::json!({ "text": text, "source": emit_source });
        let _ = app.emit("asr_final", payload.to_string());
    }

    // Write to transcript.jsonl (always, with RMS for diarization filtering)
    let entry = serde_json::json!({
        "timestamp": std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis(),
        "speaker": source,
        "text": text,
        "rms": segment_rms,
        "inference_time_ms": inference_time_ms,
        "backend": backend_name
    });
    if let Ok(line) = serde_json::to_string(&entry) {
        use std::io::Write;
        if let Ok(mut w) = transcript_writer.lock() {
            let _ = writeln!(w, "{}", line);
        }
    }
}
