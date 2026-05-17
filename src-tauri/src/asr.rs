use anyhow::Result;
use once_cell::sync::OnceCell;
use tauri::{Emitter, Manager};

use crate::text_filter;

/// Global singleton for AudioProcessor (lazily initialized)
static AUDIO_PROCESSOR: OnceCell<crate::audio_processor::AudioProcessor> = OnceCell::new();

/// Realtime ASR worker — powered by Sherpa-ONNX Silero VAD for intelligent segmentation.
///
/// The VAD model handles all speech boundary detection internally:
///   - Speech/silence classification (neural network, not energy-based rules)
///   - Minimum speech/silence duration filtering
///   - Automatic segment boundary detection
///
/// No hand-written debounce, no adaptive thresholds, no RMS gates, no density filters.
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
        .resolve(
            "resources/silero_vad.onnx",
            tauri::path::BaseDirectory::Resource,
        )
        .map_err(|e| format!("Failed to resolve VAD model: {}", e))?;

    if !model_path.exists() {
        return Err(format!(
            "Silero VAD model not found: {}",
            model_path.display()
        ));
    }

    let vad_config = sherpa_onnx::VadModelConfig {
        silero_vad: sherpa_onnx::SileroVadModelConfig {
            model: Some(model_path.to_string_lossy().into_owned()),
            threshold: 0.5,              // speech probability threshold (model default)
            min_silence_duration: 0.5,   // 500ms silence = sentence boundary
            min_speech_duration: 0.25,   // ignore speech shorter than 250ms
            window_size: 512,            // Silero model window size
            max_speech_duration: 30.0,   // force-cut at 30s (safety limit)
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
    let mut current_speaker = source.clone();
    let mut last_transcript = String::new();
    let mut sample_buf: Vec<f32> = Vec::with_capacity(16000); // 1s accumulator
    let mut segment_count: u64 = 0;

    // ── Main loop ────────────────────────────────────────────────────────
    while !stop.load(Ordering::Relaxed) {
        match rx.recv_timeout(Duration::from_millis(100)) {
            Ok(s) => {
                // Convert i16 → f32 and feed to VAD
                let f32_sample = s as f32 / 32768.0;
                sample_buf.push(f32_sample);

                // Feed in chunks (VAD window_size = 512 samples = 32ms)
                if sample_buf.len() >= 512 {
                    vad.accept_waveform(&sample_buf);
                    sample_buf.clear();

                    // Check if VAD has produced any complete speech segments
                    while !vad.is_empty() {
                        if let Some(segment) = vad.front() {
                            let samples = segment.samples();
                            let duration_s = samples.len() as f32 / 16000.0;
                            segment_count += 1;

                            println!(
                                "🎤 [VAD/{}] Segment #{}: {:.2}s ({} samples)",
                                source, segment_count, duration_s, samples.len()
                            );

                            // Send to ASR (the segment is already clean speech)
                            send_audio_to_asr(
                                &app,
                                samples,
                                &source,
                                &transcript_writer,
                                &mut current_speaker,
                                &mut last_transcript,
                            );
                        }
                        vad.pop();
                    }
                }
            }
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                // Feed any remaining samples on timeout
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
            send_audio_to_asr(
                &app,
                samples,
                &source,
                &transcript_writer,
                &mut current_speaker,
                &mut last_transcript,
            );
        }
        vad.pop();
    }

    println!("✅ [ASR/{}] Worker stopped. {} segments processed.", source, segment_count);
    Ok(())
}

fn send_audio_to_asr(
    app: &tauri::AppHandle,
    samples_f32: &[f32],
    source: &str,
    transcript_writer: &std::sync::Arc<std::sync::Mutex<std::fs::File>>,
    current_speaker: &mut String,
    last_transcript: &mut String,
) {
    // Speaker label = audio source (post-recording diarization will relabel)
    let speaker_label = source.to_string();
    *current_speaker = speaker_label.clone();

    // Add 300ms silence padding to help ASR complete the last word
    let mut padded_f32 = samples_f32.to_vec();
    padded_f32.resize(padded_f32.len() + 4800, 0.0);

    // === NOISE REDUCTION ===
    let audio_processor =
        AUDIO_PROCESSOR.get_or_init(|| crate::audio_processor::AudioProcessor::new(16000));

    let denoise_result = audio_processor.denoise(&padded_f32);
    let denoised_f32: &[f32] = &denoise_result.samples;

    println!(
        "🔇 [ASR] Noise reduction: RMS {:.4} → {:.4} ({:.1} dB)",
        denoise_result.rms_before, denoise_result.rms_after, denoise_result.noise_reduction_db
    );

    // Check backend setting
    let backend = crate::settings::SETTINGS.read().unwrap().asr.backend;

    match backend {
        crate::settings::AsrBackend::FunAsr => {
            if !crate::funasr::SenseVoiceManager::is_initialized() {
                eprintln!("❌ [ASR] FunASR selected but not initialized!");
                return;
            }
            let funasr = crate::funasr::SenseVoiceManager::get();
            let denoised_i16: Vec<i16> = denoised_f32
                .iter()
                .map(|&s| (s * 32767.0).clamp(-32768.0, 32767.0) as i16)
                .collect();
            match funasr.transcribe(&denoised_i16) {
                Ok(result) => {
                    emit_and_log(
                        app,
                        result.text.trim(),
                        &speaker_label,
                        "funasr",
                        result.inference_time_ms,
                        transcript_writer,
                        last_transcript,
                    );
                }
                Err(e) => {
                    eprintln!("❌ [FUNASR] Transcription error: {}", e);
                }
            }
        }
        crate::settings::AsrBackend::Whisper => {
            let whisper = crate::whisper::WhisperManager::get();

            let initial_prompt = if last_transcript.is_empty() {
                "这是一段会议记录。".to_string()
            } else {
                last_transcript
                    .chars()
                    .rev()
                    .take(100)
                    .collect::<Vec<_>>()
                    .into_iter()
                    .rev()
                    .collect()
            };

            match whisper.transcribe_f32(denoised_f32, "zh", Some(&initial_prompt)) {
                Ok(result) => {
                    let is_reliable = result.segments.iter().all(|seg| {
                        if seg.avg_logprob < -1.0 {
                            eprintln!(
                                "⚠️ ASR Low Confidence: logprob={:.3} < -1.0, text={:?}",
                                seg.avg_logprob, seg.text
                            );
                            false
                        } else {
                            true
                        }
                    });

                    if is_reliable {
                        emit_and_log(
                            app,
                            result.text.trim(),
                            &speaker_label,
                            "whisper",
                            result.inference_time_ms,
                            transcript_writer,
                            last_transcript,
                        );
                    }
                }
                Err(e) => {
                    eprintln!("❌ [WHISPER] Transcription error: {}", e);
                }
            }
        }
    }
}

/// Common output handler for both ASR backends.
/// Filters hallucinations, emits to frontend, writes to transcript.jsonl.
fn emit_and_log(
    app: &tauri::AppHandle,
    text: &str,
    speaker_label: &str,
    backend: &str,
    inference_time_ms: u64,
    transcript_writer: &std::sync::Arc<std::sync::Mutex<std::fs::File>>,
    last_transcript: &mut String,
) {
    if text.is_empty() {
        return;
    }

    // Text Post-processing (Blacklist/Repetition)
    if crate::text_filter::is_hallucination(text) {
        println!(
            "🗑️ [{}] Discarding hallucination: {:?}",
            backend.to_uppercase(),
            text
        );
        return;
    }

    // Emit to frontend
    let payload = serde_json::json!({
        "text": text,
        "source": speaker_label,
    });
    let _ = app.emit("asr_final", payload.to_string());

    // Update context for next inference
    *last_transcript = text.to_string();

    // Append to transcript.jsonl
    let entry = serde_json::json!({
        "timestamp": std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis(),
        "speaker": speaker_label,
        "text": text,
        "inference_time_ms": inference_time_ms,
        "backend": backend
    });
    if let Ok(line) = serde_json::to_string(&entry) {
        use std::io::Write;
        if let Ok(mut w) = transcript_writer.lock() {
            let _ = writeln!(w, "{}", line);
        }
    }
}
