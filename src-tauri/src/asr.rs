use anyhow::Result;
use once_cell::sync::OnceCell;
use tauri::{Emitter, Manager};

use crate::text_filter;

pub type RecentSystemTexts =
    std::sync::Arc<std::sync::Mutex<(std::collections::VecDeque<(u128, String)>, String)>>;

pub fn new_system_text_buffer() -> RecentSystemTexts {
    std::sync::Arc::new(std::sync::Mutex::new((
        std::collections::VecDeque::new(),
        String::new(),
    )))
}

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
    recording_start_ms: u128,
    system_texts: RecentSystemTexts,
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
            min_silence_duration: 1.0,
            min_speech_duration: 0.25,
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
        .ok_or_else(|| "Failed to create Sherpa VAD".to_string())?;

    println!(
        "✅ [ASR/{}] Sherpa Silero VAD initialized (streaming mode)",
        source
    );

    // ── State ────────────────────────────────────────────────────────────
    let mut last_transcript = String::new();
    let mut last_committed_text = String::new();
    let mut sample_buf: Vec<f32> = Vec::with_capacity(16000);
    let mut interim_buf: Vec<f32> = Vec::with_capacity(16000 * 30);
    let mut segment_count: u64 = 0;
    let is_system = source == "system";
    let mut last_interim_samples = 0;
    let mut current_interim_interval = 1.5;

    // ── Main loop ────────────────────────────────────────────────────────
    while !stop.load(Ordering::Relaxed) {
        match rx.recv_timeout(Duration::from_millis(100)) {
            Ok(s) => {
                let f32_sample = s as f32 / 32768.0;
                sample_buf.push(f32_sample);
                interim_buf.push(f32_sample);

                // Drain all available samples from the channel to catch up!
                while let Ok(extra_s) = rx.try_recv() {
                    let f32_sample = extra_s as f32 / 32768.0;
                    sample_buf.push(f32_sample);
                    interim_buf.push(f32_sample);
                }

                // Sliding Window Bounds
                let max_window_samples = 8 * 16000;
                let overlap_samples = 3 * 16000;
                let is_window_full = interim_buf.len() >= max_window_samples;

                // Check interim conditions (simulated streaming)
                let elapsed_since_last =
                    interim_buf.len() as f32 / 16000.0 - last_interim_samples as f32 / 16000.0;
                
                if is_window_full || elapsed_since_last >= current_interim_interval {
                    let latest_chunk = &interim_buf[last_interim_samples..];
                    let mut sum_sq = 0.0;
                    if !latest_chunk.is_empty() {
                        for &x in latest_chunk {
                            sum_sq += x * x;
                        }
                    }
                    let rms = if latest_chunk.is_empty() { 0.0 } else { (sum_sq / latest_chunk.len() as f32).sqrt() };

                    if rms > 0.01 || is_window_full {
                        // Has energy or forced by sliding window
                        let result = process_segment(
                            &app,
                            &interim_buf,
                            &source,
                            &transcript_writer,
                            &mut last_transcript,
                            is_system,
                            is_window_full,
                            recording_start_ms,
                            &system_texts,
                            &mut last_committed_text,
                        );

                        match result {
                            Err(_) => {
                                // Hallucination triggered: clear everything
                                interim_buf.clear();
                                last_committed_text.clear();
                                last_interim_samples = 0;
                                current_interim_interval = 1.5;
                            }
                            Ok(was_finalized) => {
                                if was_finalized || is_window_full {
                                    if is_window_full {
                                        segment_count += 1;
                                        println!(
                                            "[WINDOW/{}] Slide #{}: {:.2}s -> keep {:.2}s overlap",
                                            source,
                                            segment_count,
                                            interim_buf.len() as f32 / 16000.0,
                                            overlap_samples as f32 / 16000.0
                                        );
                                        let keep = std::cmp::min(interim_buf.len(), overlap_samples);
                                        let tail = interim_buf[interim_buf.len() - keep..].to_vec();
                                        interim_buf = tail;
                                    } else {
                                        interim_buf.clear();
                                        last_committed_text.clear();
                                    }
                                    last_interim_samples = 0;
                                    current_interim_interval = 1.5;
                                } else {
                                    last_interim_samples = interim_buf.len();
                                    current_interim_interval += 0.5;
                                    if current_interim_interval > 3.0 {
                                        current_interim_interval = 3.0;
                                    }
                                }
                            }
                        }
                    }
                }

                if sample_buf.len() >= 512 {
                    vad.accept_waveform(&sample_buf);
                    sample_buf.clear();

                    let mut vad_triggered = false;
                    while !vad.is_empty() {
                        vad_triggered = true;
                        vad.pop();
                    }

                    if vad_triggered && !interim_buf.is_empty() {
                        segment_count += 1;
                        println!(
                            "🎤 [VAD/{}] Segment #{}: {:.2}s ({} samples) - Silence detected",
                            source,
                            segment_count,
                            interim_buf.len() as f32 / 16000.0,
                            interim_buf.len()
                        );

                        // Unified transcription: always use interim_buf for finalization to keep diff-stitching consistent
                        let _ = process_segment(
                            &app,
                            &interim_buf,
                            &source,
                            &transcript_writer,
                            &mut last_transcript,
                            is_system,
                            true,
                            recording_start_ms,
                            &system_texts,
                            &mut last_committed_text,
                        );

                        interim_buf.clear();
                        last_committed_text.clear();
                        last_interim_samples = 0;
                        current_interim_interval = 1.5;
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
    if !interim_buf.is_empty() {
        segment_count += 1;
        let _ = process_segment(
            &app,
            &interim_buf,
            &source,
            &transcript_writer,
            &mut last_transcript,
            is_system,
            true,
            recording_start_ms,
            &system_texts,
            &mut last_committed_text,
        );
    }

    println!(
        "✅ [ASR/{}] Worker stopped. {} segments processed.",
        source, segment_count
    );
    Ok(())
}

fn longest_common_substring(s1: &str, s2: &str) -> String {
    let c1: Vec<char> = s1.chars().collect();
    let c2: Vec<char> = s2.chars().collect();
    if c1.is_empty() || c2.is_empty() {
        return String::new();
    }
    let mut m = vec![vec![0; c2.len() + 1]; c1.len() + 1];
    let mut max_len = 0;
    let mut end_pos = 0;
    for i in 1..=c1.len() {
        for j in 1..=c2.len() {
            if c1[i - 1] == c2[j - 1] {
                m[i][j] = m[i - 1][j - 1] + 1;
                if m[i][j] > max_len {
                    max_len = m[i][j];
                    end_pos = i;
                }
            }
        }
    }
    c1[end_pos - max_len..end_pos].iter().collect()
}

fn overlapping_stitch(s1: &str, s2: &str) -> String {
    let c1: Vec<char> = s1.chars().collect();
    let c2: Vec<char> = s2.chars().collect();
    
    if c1.is_empty() || c2.is_empty() {
        return s2.to_string();
    }
    
    let look_len = 25; // Look at up to 25 chars
    let s1_tail_start = c1.len().saturating_sub(look_len);
    let s1_tail: String = c1[s1_tail_start..].iter().collect();
    
    let s2_head_end = std::cmp::min(c2.len(), look_len);
    let s2_head: String = c2[0..s2_head_end].iter().collect();
    
    let overlap = longest_common_substring(&s1_tail, &s2_head);
    
    // If overlap is significant (>= 3 chars), strip it and any preceding noise
    if overlap.chars().count() >= 3 {
        if let Some(idx) = s2.find(&overlap) {
            let remain = &s2[idx + overlap.len()..];
            return remain.trim_start().to_string();
        }
    }
    
    s2.to_string()
}

/// Process a single VAD segment: denoise → ASR → emit (system only) + log.
/// Returns true if the segment was finalized (either explicitly or via semantic chunking).
fn process_segment(
    app: &tauri::AppHandle,
    samples_f32: &[f32],
    source: &str,
    transcript_writer: &std::sync::Arc<std::sync::Mutex<std::fs::File>>,
    last_transcript: &mut String,
    is_system: bool,
    is_final: bool,
    recording_start_ms: u128,
    system_texts: &RecentSystemTexts,
    last_committed_text: &mut String,
) -> Result<bool, ()> {
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
                return Ok(false);
            }
            let funasr = crate::funasr::SenseVoiceManager::get();
            let i16_samples: Vec<i16> = denoised
                .iter()
                .map(|&s| (s * 32767.0).clamp(-32768.0, 32767.0) as i16)
                .collect();
            match funasr.transcribe(&i16_samples) {
                Ok(r) => (r.text.trim().to_string(), "funasr", r.inference_time_ms),
                Err(e) => {
                    eprintln!("❌ [FUNASR] Error: {}", e);
                    return Ok(false);
                }
            }
        }
        crate::settings::AsrBackend::Whisper => {
            let whisper = crate::whisper::WhisperManager::get();
            let prompt = if last_transcript.is_empty() {
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
            match whisper.transcribe_f32(denoised, "zh", Some(&prompt)) {
                Ok(r) => {
                    if r.segments.iter().any(|s| s.avg_logprob < -1.0) {
                        return Ok(false);
                    }
                    (r.text.trim().to_string(), "whisper", r.inference_time_ms)
                }
                Err(e) => {
                    eprintln!("❌ [WHISPER] Error: {}", e);
                    return Ok(false);
                }
            }
        }
    };

    if text.is_empty() {
        return Ok(false);
    }

    if crate::text_filter::is_hallucination(&text) {
        println!(
            "🗑️ [{}] Discarding hallucination: {:?}",
            backend_name.to_uppercase(),
            text
        );
        // Clear frontend interim bubble to avoid stuck text
        let emit_source = if is_system { source } else { "me" };
        let payload = serde_json::json!({
            "text": "",
            "source": emit_source,
            "is_final": false
        });
        let _ = app.emit("asr_final", payload.to_string());
        return Err(()); // Force clear interim_buf and VAD to break the poisoned buffer deadlock
    }

    *last_transcript = text.clone();

    let segment_rms = result.rms_before;

    // --- 文本级回声消除与状态记录 ---
    let mut final_text = text.trim().to_string();

    // 移除破损的、基于结尾标点的语义断句规则。现在完全依赖上层的强制滑动窗口或真正的 VAD 静音。
    let effective_final = is_final;

    // --- Diff-Stitching: only on final, not interim ---
    if is_final && !last_committed_text.is_empty() {
        final_text = overlapping_stitch(last_committed_text, &final_text);
    }

    if !is_system {
        // me channel: textual echo subtraction
        if let Ok(mut st) = system_texts.lock() {
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_millis();
            st.0.retain(|(ts, _)| now.saturating_sub(*ts) < 15000); // keep last 15s

            // Subtract from finalized texts
            for (_, sys_txt) in st.0.iter() {
                let overlap = longest_common_substring(&final_text, sys_txt);
                // If overlap is significant (e.g. >= 4 chars), subtract it
                if overlap.chars().count() >= 4 {
                    final_text = final_text.replace(&overlap, "").trim().to_string();
                }
            }

            // Subtract from current interim text
            let overlap = longest_common_substring(&final_text, &st.1);
            if overlap.chars().count() >= 4 {
                final_text = final_text.replace(&overlap, "").trim().to_string();
            }
        }

        // Basic RMS gate: If it's too quiet (<0.04), it's likely residual silence/hum, don't emit
        if segment_rms <= 0.04 || final_text.is_empty() {
            return Ok(false);
        }
    } else {
        // system channel: save to shared buffer for me-channel cancellation
        if let Ok(mut st) = system_texts.lock() {
            if effective_final {
                let now = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_millis();
                st.0.push_back((now, final_text.clone()));
                st.1.clear();
            } else {
                st.1 = final_text.clone();
            }
        }
    }

    // --- 前端发送 ---
    let emit_source = if is_system { source } else { "me" };
    let payload = serde_json::json!({
        "text": final_text,
        "source": emit_source,
        "is_final": effective_final
    });
    let _ = app.emit("asr_final", payload.to_string());

    if effective_final {
        // Store FULL original transcription (pre-stitch) so next stitch has correct comparison base
        *last_committed_text = text.trim().to_string();
    }

    // --- 写入日志 (ONLY FINAL) ---
    if effective_final {
        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis();
        let relative_end = (now_ms - recording_start_ms) as f32 / 1000.0;
        let duration_s = samples_f32.len() as f32 / 16000.0;
        let relative_start = (relative_end - duration_s).max(0.0);

        let entry = serde_json::json!({
            "start": relative_start,
            "end": relative_end,
            "speaker": source,
            "text": final_text,
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
    Ok(effective_final)
}
