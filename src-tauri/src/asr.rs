use anyhow::Result;
use once_cell::sync::OnceCell;
use tauri::Emitter;

use crate::text_filter;

/// Global singleton for AudioProcessor (lazily initialized)
static AUDIO_PROCESSOR: OnceCell<crate::audio_processor::AudioProcessor> = OnceCell::new();

pub fn realtime_inference_worker(
    app: tauri::AppHandle,
    stop: std::sync::Arc<std::sync::atomic::AtomicBool>,
    mut rx: std::sync::mpsc::Receiver<i16>,
    source: String,
    transcript_writer: std::sync::Arc<std::sync::Mutex<std::fs::File>>,
) -> Result<(), String> {
    use std::sync::atomic::Ordering;
    use std::time::Duration;

    // VAD Initialization
    // We use the configured VAD backend (WebRTC or Silero/TEN)
    let settings = crate::settings::SETTINGS.read().unwrap().audio.clone();
    let vad_backend = settings.vad_backend;
    let vad_threshold = settings.vad_threshold;

    let mut vad: Box<dyn crate::vad::VadEngine> =
        match crate::vad::create_vad(&app, vad_backend, vad_threshold) {
            Ok(v) => {
                println!("✅ [ASR] Initialized VAD backend: {}", vad_backend);
                v
            }
            Err(e) => {
                eprintln!(
                    "❌ [ASR] Failed to init VAD backend {:?}: {}",
                    vad_backend, e
                );
                eprintln!("⚠️ [ASR] Fallback to WebRTC VAD");
                Box::new(crate::vad::WebRtcVadWrapper::new())
            }
        };

    // Frame size: WebRTC likes 10/20/30ms. TenVad is flexible but 20ms (320 samples) is safe for both.
    let vad_frame_size = 320; // 20ms @ 16kHz
    let mut vad_accum: Vec<i16> = Vec::with_capacity(vad_frame_size);

    // Debounce: require consecutive speech frames to trigger
    let min_speech_frames = 5; // ~100ms

    // Adaptive VAD Parameters
    // We want to be conservative at first (wait for clear end),
    // but become aggressive if the person keeps talking (to reduce latency).\
    let samples_stage_1 = 16000 * 4; // 0-4s: High quality mode
    let samples_stage_2 = 16000 * 10; // 4-10s: Normal mode
                                      // >10s: Low latency mode

    let frames_stage_1 = 16; // ~320ms (Wait for clear sentence finish) (1 frame = 20ms)
    let frames_stage_2 = 10; // ~200ms
    let frames_stage_3 = 8; // ~160ms

    let max_len_samples = 16000 * 15; // 15 seconds max (avoid cutting sentences)
    let max_silence_buffer_samples = 16000 * 5;

    // State
    let mut buf: Vec<i16> = Vec::with_capacity(max_len_samples);
    let mut is_speaking = false;
    let mut silence_frames = 0;
    let mut speech_run_count = 0; // for debounce
    let mut speech_frames_count = 0; // Total speech frames in current buffer
    let mut current_speaker = "Speaker 1".to_string();
    let mut last_transcript = String::new(); // Context for Whisper prompt
    // frame_accum removed, using vad_accum

    while !stop.load(Ordering::Relaxed) {
        match rx.recv_timeout(Duration::from_millis(100)) {
            Ok(s) => {
                buf.push(s);
                vad_accum.push(s);

                // Process VAD every 20ms
                if vad_accum.len() >= vad_frame_size {
                    // 0. RMS Gate (Simple Noise Gate)
                    let sq_sum: f64 = vad_accum.iter().map(|&x| (x as f64).powi(2)).sum();
                    let rms = (sq_sum / vad_accum.len() as f64).sqrt();

                    // DEBUG: Print RMS to fine-tune threshold
                    // if speech_run_count % 10 == 0 {
                    //     println!("📊 VAD Input RMS: {:.1}", rms);
                    // }

                    let mut is_voice = vad.is_voice_segment(&vad_accum).unwrap_or(false);

                    // Force silence if energy is too low (e.g. background hiss)
                    // Bumped to 1000.0: Stronger noise filter.
                    if rms < 1000.0 {
                        is_voice = false;
                    }

                    if speech_run_count % 50 == 0 && is_voice {
                        // Debug log occasionally to check RMS levels during speech
                        println!("🔉 VAD [{}]: Speech frame RMS={:.1}", source, rms);
                    }

                    vad_accum.clear();

                    if is_voice {
                        speech_run_count += 1;
                        speech_frames_count += 1;
                    } else {
                        speech_run_count = 0;
                    }

                    // Trigger "speaking" state only after stability
                    if speech_run_count >= min_speech_frames {
                        if !is_speaking {
                            is_speaking = true;
                            println!("🎤 VAD [{}]: Speech START", source);
                        }
                        if silence_frames > 0 {
                            println!(
                                "🔄 VAD [{}]: Silence RESET by speech run ({} frames)",
                                source, speech_run_count
                            );
                        }
                        silence_frames = 0;
                    } else {
                        // If we are already speaking, this counts as silence frame
                        if is_speaking {
                            silence_frames += 1;
                            // Debug log every 10 frames of silence (~200ms) to track why it's not cutting
                            if silence_frames % 5 == 0 {
                                println!(
                                    "... VAD [{}]: Silence frame {} (Threshold: ...)",
                                    source, silence_frames
                                );
                            }
                        }
                    }

                    // --- Dynamic Segmentation Logic (Adaptive Urgency) ---
                    // As the buffer grows, we tolerate shorter silences to "get the text out".
                    let len = buf.len();
                    let current_threshold = if len > samples_stage_2 {
                        frames_stage_3 // >10s: Urgent, cut on 160ms
                    } else if len > samples_stage_1 {
                        frames_stage_2 // 4-10s: Normal, cut on 200ms
                    } else {
                        frames_stage_1 // <4s: Conservative, wait 320ms
                    };

                    // 1. End of Sentence (Speaking -> Pause > limit)
                    if is_speaking {
                        if silence_frames % 5 == 0 {
                            println!(
                                "📊 VAD Status [{}]: len={} samples, silence={} frames, limit={}",
                                source,
                                buf.len(),
                                silence_frames,
                                current_threshold
                            );
                        }
                    }
                    if is_speaking && silence_frames >= current_threshold {
                        if buf.len() > 1000 {
                            // Check speech density
                            // One frame = 320 samples.
                            // total_frames = buf.len() / 320
                            // density = speech_frames_count / total_frames
                            let total_frames = buf.len() / 320;
                            let density = if total_frames > 0 {
                                speech_frames_count as f32 / total_frames as f32
                            } else {
                                0.0
                            };

                            // Filter out low density (e.g. < 15% speech) if buffer is long enough (>2s)
                            if buf.len() > 32000 && density < 0.15 {
                                println!(
                                     "⚠️ Low speech density [{}]: {:.1}% (len={}ms) - Sending anyway to preserve latency",
                                     source,
                                     density * 100.0,
                                     buf.len() / 16
                                 );
                                send_audio_to_asr(
                                    &app,
                                    &buf,
                                    &source,
                                    &transcript_writer,
                                    &mut current_speaker,
                                    density,
                                    &mut last_transcript,
                                );
                            } else {
                                println!("🚀 Sending audio to ASR [{}](Condition 1: Silence Cut), density={:.1}%", source, density * 100.0);
                                // 1000 samples ~ 60ms
                                send_audio_to_asr(
                                    &app,
                                    &buf,
                                    &source,
                                    &transcript_writer,
                                    &mut current_speaker,
                                    density,
                                    &mut last_transcript,
                                );
                            }
                        }
                        buf.clear();
                        is_speaking = false;
                        silence_frames = 0;
                        speech_run_count = 0;
                        speech_frames_count = 0;
                    }
                }

                // 2. Max Length (Force send at 5s)
                if buf.len() >= max_len_samples {
                    if is_speaking {
                        let total_frames_ml = buf.len() / 320;
                        let density_ml = if total_frames_ml > 0 {
                            speech_frames_count as f32 / total_frames_ml as f32
                        } else { 0.0 };
                        send_audio_to_asr(
                            &app,
                            &buf,
                            &source,
                            &transcript_writer,
                            &mut current_speaker,
                            density_ml,
                            &mut last_transcript,
                        );
                    }
                    buf.clear();
                    is_speaking = false;
                    silence_frames = 0;
                    speech_run_count = 0;
                    speech_frames_count = 0;
                }

                // 3. Garbage Collection (Smart Pre-roll)
                if !is_speaking && buf.len() >= max_silence_buffer_samples {
                    // Don't clear everything! Keep last 0.5s as "pre-roll" for next sentence
                    let keep_len = 8000; // 500ms
                    if buf.len() > keep_len {
                        let drain_end = buf.len() - keep_len;
                        buf.drain(0..drain_end);
                    }
                    silence_frames = 0;
                    speech_run_count = 0;
                    speech_frames_count = 0; // Reset for new segment
                }
            }
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {}
            Err(_) => break,
        }
    }

    // Final flush
    if !buf.is_empty() && is_speaking {
        let total_frames_fl = buf.len() / 320;
        let density_fl = if total_frames_fl > 0 {
            speech_frames_count as f32 / total_frames_fl as f32
        } else { 0.0 };
        send_audio_to_asr(
            &app,
            &buf,
            &source,
            &transcript_writer,
            &mut current_speaker,
            density_fl,
            &mut last_transcript,
        );
    }

    Ok(())
}

fn send_audio_to_asr(
    app: &tauri::AppHandle,
    samples: &[i16],
    source: &str,
    transcript_writer: &std::sync::Arc<std::sync::Mutex<std::fs::File>>,
    current_speaker: &mut String,
    speech_density: f32,
    last_transcript: &mut String,
) {
    // 1. SPEAKER DIARIZATION
    //
    // Strategy:
    // - "user" (mic) channel: Always label as "Me" — the mic only captures the local user.
    // - "system" channel: Use sherpa-onnx embedding + manager. The pipeline internally
    //   handles low-density filtering and minimum-speech-for-new-speaker checks.
    // - Short segments (<1s): Skip diarization, reuse last known speaker label.
    let speaker_label = if source == "user" {
        // Mic channel: verify voice matches "Me" anchor to detect echo leakage
        match crate::diarization::DiarizationPipeline::try_get() {
            Some(pipeline) => {
                if !pipeline.has_me_anchor() {
                    // === PRE-ANCHOR PHASE ===
                    // Me anchor not yet established. Must be careful:
                    // mic can pick up system audio echo which should NOT be labeled "Me".

                    // 1. Low density → almost certainly noise/echo, discard silently
                    if speech_density < 0.25 {
                        println!(
                            "🔇 [ASR] Pre-anchor: low density ({:.0}%), discarding mic segment",
                            speech_density * 100.0
                        );
                        return;
                    }

                    // 2. Cross-check against known system speakers (echo detection)
                    //    If mic audio matches Speaker 1/2/etc, it's echo leakage.
                    if samples.len() >= 8000 {
                        match pipeline.matches_any_speaker(samples) {
                            Ok(Some((speaker, score))) => {
                                println!(
                                    "🔇 [ASR] Pre-anchor: mic matches {} (score={:.3}), echo leakage, discarding",
                                    speaker, score
                                );
                                return;
                            }
                            Ok(None) => {} // No match → could be real user speech
                            Err(e) => eprintln!("⚠️ [DIARIZATION] matches_any_speaker error: {}", e),
                        }
                    }

                    // 3. Passed echo checks → try to register as Me anchor
                    if let Err(e) = pipeline.register_me_anchor(samples, speech_density) {
                        eprintln!("⚠️ [DIARIZATION] Failed to register Me anchor: {}", e);
                    }

                    if pipeline.has_me_anchor() {
                        // Registration succeeded → this is real user speech
                        let label = "Me".to_string();
                        *current_speaker = label.clone();
                        label
                    } else {
                        // Registration failed (density/duration too low), discard
                        println!("🔇 [ASR] Pre-anchor: Me registration failed, discarding mic segment");
                        return;
                    }
                } else {
                    // === POST-ANCHOR PHASE ===
                    // Verify this sounds like the registered Me anchor
                    match pipeline.verify_is_me(samples, speech_density) {
                        Ok(Some(true)) => {
                            // Confirmed as Me — reinforce anchor
                            let _ = pipeline.register_me_anchor(samples, speech_density);
                            let label = "Me".to_string();
                            *current_speaker = label.clone();
                            label
                        }
                        Ok(Some(false)) => {
                            // Doesn't sound like Me — echo leakage, discard
                            println!(
                                "🔇 [ASR] Mic audio doesn't match Me anchor (echo leakage?), discarding segment"
                            );
                            return;
                        }
                        Ok(None) => {
                            // Too short/low-quality to verify — discard to be safe
                            // (previously defaulted to "Me" which caused false positives)
                            println!(
                                "🔇 [ASR] Post-anchor: mic segment too short to verify, discarding"
                            );
                            return;
                        }
                        Err(e) => {
                            eprintln!("⚠️ [DIARIZATION] Me verify error: {}, discarding", e);
                            return;
                        }
                    }
                }
            }
            None => {
                // Diarization not initialized, fallback to Me
                let label = "Me".to_string();
                *current_speaker = label.clone();
                label
            }
        }
    } else if samples.len() < 8000 {
        // Too short for reliable diarization (< 0.5s at 16kHz)
        println!(
            "⏩ [DIARIZATION] Segment too short ({:.1}s), reusing: {}",
            samples.len() as f32 / 16000.0,
            current_speaker
        );
        current_speaker.clone()
    } else {
        // System channel: use sherpa-onnx diarization pipeline with speech density info
        match crate::diarization::DiarizationPipeline::get().identify_speaker_i16(samples, speech_density) {
            Ok(label) => {
                println!(
                    "🎯 [DIARIZATION] {} -> {}",
                    source, label
                );
                *current_speaker = label.clone();
                label
            }
            Err(e) => {
                eprintln!("⚠️ [DIARIZATION] Pipeline failed: {}, reusing: {}", e, current_speaker);
                current_speaker.clone()
            }
        }
    };

    // Add 300ms silence padding to end (Post-roll) to help ASR complete the last word
    // Convert to f32 upfront — all subsequent processing stays in f32
    let mut padded_f32: Vec<f32> = samples.iter().map(|&s| s as f32 / 32768.0).collect();
    padded_f32.resize(padded_f32.len() + 4800, 0.0); // 300ms silence padding

    // === NOISE REDUCTION ===
    // Apply RNNoise neural network denoising before transcription
    let audio_processor =
        AUDIO_PROCESSOR.get_or_init(|| crate::audio_processor::AudioProcessor::new(16000));

    let denoise_result = audio_processor.denoise(&padded_f32);

    // Keep denoised samples as f32 — no unnecessary f32→i16→f32 round-trips
    let denoised_f32: &[f32] = &denoise_result.samples;

    println!(
        "🔇 [ASR] Noise reduction applied: RMS {:.4} -> {:.4} ({:.1} dB reduction)",
        denoise_result.rms_before, denoise_result.rms_after, denoise_result.noise_reduction_db
    );

    // Check backend setting
    let backend = crate::settings::SETTINGS.read().unwrap().asr.backend;

    match backend {
        crate::settings::AsrBackend::FunAsr => {
            // Use FunASR (SenseVoice)
            if !crate::funasr::SenseVoiceManager::is_initialized() {
                eprintln!("❌ [ASR] FunASR selected but not initialized!");
                return;
            }
            let funasr = crate::funasr::SenseVoiceManager::get();
            // FunASR expects i16, convert from denoised f32
            let denoised_i16: Vec<i16> = denoised_f32
                .iter()
                .map(|&s| (s * 32767.0).clamp(-32768.0, 32767.0) as i16)
                .collect();
            match funasr.transcribe(&denoised_i16) {
                Ok(result) => {
                    let t = result.text.trim();
                    if !t.is_empty() {
                        // Text Post-processing (Blacklist/Repetition)
                        if crate::text_filter::is_hallucination(t) {
                            println!("🗑️ [FunASR] Discarding hallucination: {:?}", t);
                        } else {
                            let payload = serde_json::json!({
                                "text": t,
                                "source": speaker_label,
                            });
                            let _ = app.emit("asr_final", payload.to_string());

                            // Update context for next inference
                            *last_transcript = t.to_string();

                            // LOGGING: Append to transcript.jsonl
                            let entry = serde_json::json!({
                                "timestamp": std::time::SystemTime::now()
                                    .duration_since(std::time::UNIX_EPOCH)
                                    .unwrap()
                                    .as_millis(),
                                "speaker": speaker_label,
                                "text": t,
                                "inference_time_ms": result.inference_time_ms,
                                "backend": "funasr"
                            });
                            if let Ok(line) = serde_json::to_string(&entry) {
                                use std::io::Write;
                                if let Ok(mut w) = transcript_writer.lock() {
                                    let _ = writeln!(w, "{}", line);
                                }
                            }
                        }
                    }
                }
                Err(e) => {
                    eprintln!("❌ [FUNASR] Transcription error: {}", e);
                }
            }
        }
        crate::settings::AsrBackend::Whisper => {
            // Use whisper-rs with f32 path (no redundant i16 conversion)
            let whisper = crate::whisper::WhisperManager::get();

            // Build dynamic initial_prompt with previous context
            let initial_prompt = if last_transcript.is_empty() {
                "这是一段会议记录。".to_string()
            } else {
                // Use last ~100 characters as context for continuity
                let ctx: String = last_transcript.chars()
                    .rev().take(100).collect::<Vec<_>>()
                    .into_iter().rev().collect();
                ctx
            };

            // Use transcribe_f32 to avoid redundant f32→i16→f32 conversion
            match whisper.transcribe_f32(denoised_f32, "zh", Some(&initial_prompt)) {
                Ok(result) => {
                    // Check if any segment has low confidence (avg_logprob < -1.0)
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
                        let t = result.text.trim();
                        if !t.is_empty() {
                            // Text Post-processing (Blacklist/Repetition)
                            if crate::text_filter::is_hallucination(t) {
                                println!("🗑️ [Whisper] Discarding hallucination: {:?}", t);
                            } else {
                                let payload = serde_json::json!({
                                    "text": t,
                                    "source": speaker_label,
                                });
                                let _ = app.emit("asr_final", payload.to_string());

                                // Update context for next inference
                                *last_transcript = t.to_string();

                                // LOGGING: Append to transcript.jsonl
                                let entry = serde_json::json!({
                                    "timestamp": std::time::SystemTime::now()
                                        .duration_since(std::time::UNIX_EPOCH)
                                        .unwrap()
                                        .as_millis(),
                                    "speaker": speaker_label,
                                    "text": t,
                                    "inference_time_ms": result.inference_time_ms,
                                    "backend": "whisper"
                                });
                                if let Ok(line) = serde_json::to_string(&entry) {
                                    use std::io::Write;
                                    if let Ok(mut w) = transcript_writer.lock() {
                                        let _ = writeln!(w, "{}", line);
                                    }
                                }
                            }
                        }
                    }
                }
                Err(e) => {
                    eprintln!("❌ [WHISPER] Transcription error: {}", e);
                }
            }
        }
    }
}
