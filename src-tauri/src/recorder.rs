//! Recording lifecycle — start/stop recording, thread management, manual transcript.
//!
//! Extracted from main.rs to keep the entry point lean.

use once_cell::sync::Lazy;
use std::sync::atomic::{AtomicBool, Ordering};
use std::{
    path::PathBuf,
    sync::{Arc, Mutex},
};

use crate::audio;
use crate::asr;
use crate::capture;
use crate::wav::WavWriter;
use tauri::Manager;

static RECORDER: Lazy<Mutex<Option<RecorderHandle>>> = Lazy::new(|| Mutex::new(None));

struct RecorderHandle {
    stop: Arc<AtomicBool>,
    rec_join: std::thread::JoinHandle<()>,
    mixer_join: std::thread::JoinHandle<()>,
    asr_user_join: std::thread::JoinHandle<()>,
    asr_system_join: std::thread::JoinHandle<()>,
    base_dir: PathBuf,
    mic_path: PathBuf,
    system_path: PathBuf,
    mix_path: PathBuf,
    mix_asr_path: PathBuf,
    transcript_writer: Arc<Mutex<std::fs::File>>,
    transcript_path: PathBuf,
    recording_start_ms: u128,
}

// Global PCM channels for real-time mixer
// The mixer thread reads from these.
// Writing to MIC_PCM_TX is done by capture::start_mic_stream (and main passing the sender) now.
// Writing to SYS_PCM_TX is done by main system loop.
static MIC_PCM_TX: Lazy<Mutex<Option<std::sync::mpsc::Sender<i16>>>> =
    Lazy::new(|| Mutex::new(None));
static SYS_PCM_TX: Lazy<Mutex<Option<std::sync::mpsc::Sender<i16>>>> =
    Lazy::new(|| Mutex::new(None));

// =====================
// Tauri command: record N seconds -> (system.wav, mic.wav)
// NOTE: sync function to avoid Send future issues
// =====================
#[tauri::command]
pub fn start_recording(app: tauri::AppHandle) -> Result<(String, String), String> {
    use tauri::Emitter;
    println!("\n========================================");
    println!("▶ [LIFECYCLE: start_recording] Called");
    println!("========================================");
    let _ = app.emit("tray-log", "▶ start_recording...");

    let mut guard = RECORDER.lock().unwrap();
    if guard.is_some() {
        println!("❌ [LIFECYCLE: start_recording] Already running, returning error");
        return Err("recording already running".into());
    }
    println!("✅ [LIFECYCLE: start_recording] No existing recorder, proceeding...");

    let res = (|| -> Result<(String, String), String> {
        // Output dir
        // Output dir: app_data/sessions/YYYY-MM-DD_HH-mm-ss
        let base_app_data: PathBuf = app.path().app_data_dir().map_err(|e| e.to_string())?;

        // Generate session timestamp
        let now = std::time::SystemTime::now();
        let dt: chrono::DateTime<chrono::Local> = now.into();
        let folder_name = dt.format("%Y-%m-%d_%H-%M-%S").to_string();
        let session_dir = base_app_data.join("sessions").join(folder_name);

        std::fs::create_dir_all(&session_dir).map_err(|e| e.to_string())?;

        // v8.0: No per-session speaker state to clear — offline diarization is stateless.

        let system_path = session_dir.join("system.wav");
        let mic_path = session_dir.join("mic.wav");
        let mic_asr_path = session_dir.join("mic_asr_16k_mono.wav");
        let mix_path = session_dir.join("mix.wav");
        let mix_asr_path = session_dir.join("mix_asr_16k_mono.wav");
        let transcript_path = session_dir.join("transcript.jsonl");

        // Create transcript writer
        let transcript_file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&transcript_path)
            .map_err(|e| e.to_string())?;
        let transcript_writer = Arc::new(Mutex::new(transcript_file));

        // Create writers
        let system_writer = Arc::new(Mutex::new(
            WavWriter::create(&system_path).map_err(|e| e.to_string())?,
        ));
        let mic_writer = Arc::new(Mutex::new(
            WavWriter::create(&mic_path).map_err(|e| e.to_string())?,
        ));
        let mic_asr_writer = Arc::new(Mutex::new(
            WavWriter::create(&mic_asr_path).map_err(|e| e.to_string())?,
        ));

        // Stop flag
        let stop = Arc::new(AtomicBool::new(false));
        // Realtime PCM channels
        let (mic_pcm_tx, mic_pcm_rx) = std::sync::mpsc::channel::<i16>();
        let (sys_pcm_tx, sys_pcm_rx) = std::sync::mpsc::channel::<i16>();
        let (mix_pcm_tx, mix_pcm_rx) = std::sync::mpsc::channel::<i16>();

        // Valid for ASR (Independent channels)
        let (asr_mic_tx, asr_mic_rx) = std::sync::mpsc::channel::<i16>();
        let (asr_sys_tx, asr_sys_rx) = std::sync::mpsc::channel::<i16>();

        *MIC_PCM_TX.lock().unwrap() = Some(mic_pcm_tx.clone());
        *SYS_PCM_TX.lock().unwrap() = Some(sys_pcm_tx.clone());

        // Mixer thread
        let stop_mix = stop.clone();
        let mixer_join = std::thread::spawn(move || {
            let mut s_last: i16 = 0;
            // Synchronize on Mic stream (16kHz clock)
            while let Ok(mic_sample) = mic_pcm_rx.recv() {
                if stop_mix.load(Ordering::Acquire) {
                    break;
                }
                if let Ok(v) = sys_pcm_rx.try_recv() {
                    s_last = v;
                } else {
                    // If system stream is slower/empty
                    if s_last != 0 {
                        s_last = 0;
                    }
                }

                // Saturation mix
                let sum = mic_sample as i32 + s_last as i32;
                let mixed = sum.clamp(i16::MIN as i32, i16::MAX as i32) as i16;
                let _ = mix_pcm_tx.send(mixed);
            }
        });

        // ASR worker thread 1: User (Mic)
        let asr_app_1 = app.clone();
        let stop_asr_1 = stop.clone();
        let tw_1 = transcript_writer.clone();
        let asr_user_join = std::thread::spawn(move || {
            let _ = asr::realtime_inference_worker(
                asr_app_1,
                stop_asr_1,
                asr_mic_rx,
                "user".to_string(),
                tw_1,
            );
        });

        // ASR worker thread 2: System (Speaker)
        let asr_app_2 = app.clone();
        let stop_asr_2 = stop.clone();
        let tw_2 = transcript_writer.clone();
        let asr_system_join = std::thread::spawn(move || {
            let _ = asr::realtime_inference_worker(
                asr_app_2,
                stop_asr_2,
                asr_sys_rx,
                "system".to_string(),
                tw_2,
            );
        });

        let stop2 = stop.clone();
        let mic_path_t = mic_path.clone();
        let system_path_t = system_path.clone();
        let mix_path_t = mix_path.clone();
        let mix_asr_path_t = mix_asr_path.clone();
        let transcript_path_t = transcript_path.clone();

        // Use mic_pcm_tx.clone() to pass to valid stream
        let mic_tx_for_capture = mic_pcm_tx.clone();

        // Shared Atomic Gate for AEC (Energy Interlock)
        let system_speaking = Arc::new(AtomicBool::new(false));

        // Record the start timestamp for post-processing alignment
        let recording_start_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis();

        // Spawn recording thread
        let sys_speaking_mic = system_speaking.clone();
        let rec_app = app.clone();
        println!("\n🔄 [LIFECYCLE: start_recording] Spawning main recording thread...");
        let rec_join = std::thread::spawn(move || {
            println!("🔄 [LIFECYCLE: rec_join thread ENTER] Recording thread started");
            // 1) MIC stream
            println!("🎤 [LIFECYCLE: rec_join] Creating mic stream...");
            let mic_stream = match capture::start_mic_stream(
                mic_writer.clone(),
                mic_asr_writer.clone(),
                mic_tx_for_capture,
                asr_mic_tx, // Send to User ASR
                sys_speaking_mic,
            ) {
                Ok(s) => {
                    println!("✅ [LIFECYCLE: rec_join] Mic stream created successfully");
                    s
                }
                Err(e) => {
                    eprintln!("❌ start_mic_stream failed: {e:?}");
                    return;
                }
            };
            if let Err(e) = cpal::traits::StreamTrait::play(&mic_stream) {
                eprintln!("❌ mic_stream.play failed: {e:?}");
                return;
            }
            println!("✅ [LIFECYCLE: rec_join] Mic stream playing");

            // 2) System stream (Hybrid: SCKit or CoreAudio)
            println!("🔊 [LIFECYCLE: rec_join] Creating system audio stream...");
            let mut system_stream: audio::capture::SystemAudioStream =
                match tauri::async_runtime::block_on(audio::capture::start_system_audio_capture()) {
                    Ok(s) => {
                        println!("✅ [LIFECYCLE: rec_join] System stream created successfully");
                        s
                    }
                    Err(e) => {
                        eprintln!("❌ SystemAudioCapture failed: {e:?}");
                        return;
                    }
                };
            // WAV writer will be initialized lazily in the audio loop with detected sample rate

            // 3) Tokio Runtime for System Capture
            let rt = match tokio::runtime::Builder::new_current_thread()
                .enable_time()
                .build()
            {
                Ok(rt) => rt,
                Err(e) => {
                    eprintln!("❌ tokio runtime build failed: {e:?}");
                    return;
                }
            };

            let system_writer2 = system_writer.clone();
            let sys_speaking_loop = system_speaking.clone();

            rt.block_on(async move {
                use futures_util::StreamExt;
                use tokio::time::{timeout, Duration, Instant};

                // Lazy initialization - will be set on first audio packet
                let mut wav_initialized = false;
                let mut buf: Vec<i16> = Vec::new();
                let mut last_flush = Instant::now();

                // Resampler state - ratio will be calculated on first packet
                let mut rs_phase: f32 = 0.0;
                let mut ratio: f32 = 3.0; // Default, will be updated
                let mut prev_sample: f32 = 0.0;
                let mut actual_sr: u32 = 48000; // Will be updated

                // AEC Gate State
                let mut rms_window_sum = 0.0;
                let mut rms_window_count = 0;
                let mut rms_window_size = 480; // Will be updated based on actual SR
                let gate_threshold = 0.05;
                let mut hang_timer = 0;
                let hang_duration = 5;

                // Eager initialization - ensure valid WAV header even if silent
                actual_sr = system_stream.sample_rate();
                
                println!("╔══════════════════════════════════════════════════════════════╗");
                println!("║           SYSTEM CAPTURE STARTED                             ║");
                println!("╠══════════════════════════════════════════════════════════════╣");
                println!("║ SAMPLE RATE INFO:                                            ║");
                println!("║   Detected from stream: {} Hz", actual_sr);
                
                println!("╠══════════════════════════════════════════════════════════════╣");
                println!("║ WAV WRITER CONFIG:                                           ║");
                println!("║   Output sample rate: {} Hz", actual_sr);
                println!("║   Channels: 1 (mono)                                         ║");
                println!("║   Format: PCM 16-bit                                         ║");
                
                // Initialize WAV writer with correct sample rate
                match system_writer2.lock().unwrap().init_pcm16(actual_sr, 1) {
                    Ok(_) => println!("║   Status: ✅ Initialized successfully                        ║"),
                    Err(e) => println!("║   Status: ❌ Failed: {:?}", e),
                }

                println!("╠══════════════════════════════════════════════════════════════╣");
                println!("║ RESAMPLING CONFIG (for ASR):                                 ║");
                ratio = actual_sr as f32 / 16_000.0;
                println!("║   Source: {} Hz -> Target: 16000 Hz", actual_sr);
                println!("║   Ratio: {:.4}", ratio);

                println!("╠══════════════════════════════════════════════════════════════╣");
                println!("║ AEC CONFIG:                                                  ║");
                rms_window_size = (actual_sr / 100) as usize;
                println!("║   RMS window: {} samples (~10ms)", rms_window_size);
                println!("║   Gate threshold: {}", gate_threshold);

                println!("╠══════════════════════════════════════════════════════════════╣");
                println!("║ BUFFER CONFIG:                                               ║");
                buf = Vec::with_capacity(actual_sr as usize);
                println!("║   Capacity: {} samples (~1 sec)", actual_sr);
                println!("╚══════════════════════════════════════════════════════════════╝");
                println!("");
                println!("⚠️ If WAV playback has wrong pitch:");
                println!("   - Higher pitch (cartoon): WAV SR > actual data SR");
                println!("   - Lower pitch (slow): WAV SR < actual data SR");
                println!("   Check SCK format diagnostic above for correct interpretation.");
                println!("");

                wav_initialized = true;

                while !stop2.load(Ordering::Acquire) {
                    match timeout(Duration::from_millis(200), system_stream.next()).await {
                        Ok(Some(s)) => {
                            let s: f32 = s;

                            // Lazy initialization removed (done above)

                            // DEBUG: Trace data arrival with stats
                            static MAIN_LOG_COUNTER: std::sync::atomic::AtomicUsize =
                                std::sync::atomic::AtomicUsize::new(0);
                            static SAMPLE_SUM: std::sync::atomic::AtomicU64 =
                                std::sync::atomic::AtomicU64::new(0);
                            static SAMPLE_SQ_SUM: std::sync::atomic::AtomicU64 =
                                std::sync::atomic::AtomicU64::new(0);
                            static SAMPLE_MIN: std::sync::atomic::AtomicI32 =
                                std::sync::atomic::AtomicI32::new(i32::MAX);
                            static SAMPLE_MAX: std::sync::atomic::AtomicI32 =
                                std::sync::atomic::AtomicI32::new(i32::MIN);
                                
                            let count = MAIN_LOG_COUNTER.fetch_add(1, Ordering::Relaxed);
                            
                            // Track sample statistics
                            let s_int = (s * 1000000.0) as i64;
                            SAMPLE_SUM.fetch_add(s_int.unsigned_abs(), Ordering::Relaxed);
                            SAMPLE_SQ_SUM.fetch_add((s * s * 1000000.0) as u64, Ordering::Relaxed);
                            SAMPLE_MIN.fetch_min((s * 1000000.0) as i32, Ordering::Relaxed);
                            SAMPLE_MAX.fetch_max((s * 1000000.0) as i32, Ordering::Relaxed);
                            
                            // Log every 48000 samples (~1 second at 48kHz)
                            if count > 0 && count % 48000 == 0 {
                                let n = count as f64;
                                let avg = SAMPLE_SUM.load(Ordering::Relaxed) as f64 / n / 1000000.0;
                                let rms_val = (SAMPLE_SQ_SUM.load(Ordering::Relaxed) as f64 / n / 1000000.0).sqrt();
                                let min_val = SAMPLE_MIN.load(Ordering::Relaxed) as f64 / 1000000.0;
                                let max_val = SAMPLE_MAX.load(Ordering::Relaxed) as f64 / 1000000.0;
                                println!("📊 [MAIN_LOOP] Stats after {} samples: avg_abs={:.6}, RMS={:.6}, range=[{:.6}, {:.6}]",
                                         count, avg, rms_val, min_val, max_val);
                            }

                            let v = (s.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
                            buf.push(v);

                            // --- AEC Logic: Update Energy State ---
                            rms_window_sum += s * s;
                            rms_window_count += 1;
                            if rms_window_count >= rms_window_size {
                                let rms = (rms_window_sum / rms_window_count as f32).sqrt();
                                if rms > gate_threshold {
                                    sys_speaking_loop.store(true, Ordering::Relaxed);
                                    hang_timer = hang_duration;
                                } else if hang_timer > 0 {
                                    hang_timer -= 1;
                                } else {
                                    sys_speaking_loop.store(false, Ordering::Relaxed);
                                }
                                rms_window_sum = 0.0;
                                rms_window_count = 0;
                            }

                            // --- Resample to 16kHz for ASR ---
                            rs_phase += 1.0 / ratio;
                            while rs_phase >= 1.0 {
                                let t = 1.0 - (rs_phase - 1.0);
                                let y = prev_sample + (s - prev_sample) * t;
                                let v_asr = (y.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;

                                if let Some(tx) = SYS_PCM_TX.lock().unwrap().as_ref() {
                                    let _ = tx.send(v_asr);
                                }
                                let _ = asr_sys_tx.send(v_asr);
                                rs_phase -= 1.0;
                            }
                            prev_sample = s;

                            // Flush WAV periodically
                            if buf.len() >= actual_sr as usize || last_flush.elapsed() >= Duration::from_secs(1)
                            {
                                crate::wav::flush_i16(system_writer2.clone(), &mut buf);
                                last_flush = Instant::now();
                            }
                        }
                        Ok(None) => break,
                        Err(_) => {
                            // timeout: check stop flag again
                        }
                    }
                }

                if !buf.is_empty() {
                    crate::wav::flush_i16(system_writer2.clone(), &mut buf);
                }

                // Give mic callback some time to flush tail
                tokio::time::sleep(Duration::from_millis(50)).await;
                println!("✅ System capture loop ended, flushing complete.");
                println!("🛑 [LIFECYCLE: rt.block_on async] About to exit async block, system_stream will be dropped...");
                // system_stream is moved into this async block and will be dropped when the block ends
            });
            println!("✅ System capture runtime finished.");
            println!("🛑 [LIFECYCLE: rt.block_on] Async block exited, system_stream should have been dropped");

            // 4) Clean up
            println!("\n🛑 [LIFECYCLE: rec_join] System capture loop ended, starting cleanup...");
            println!("🛑 Pausing mic stream...");
            if let Err(e) = cpal::traits::StreamTrait::pause(&mic_stream) {
                eprintln!("⚠️ Failed to pause mic stream: {:?}", e);
            }
            println!("🛑 Dropping mic stream...");
            drop(mic_stream);
            println!("✅ Mic stream dropped.");

            // Finalize WAVs
            println!("💾 [LIFECYCLE: rec_join] Finalizing WAV files...");
            let _ = system_writer.lock().unwrap().finalize();
            let _ = mic_writer.lock().unwrap().finalize();
            let _ = mic_asr_writer.lock().unwrap().finalize();
            println!("✅ [LIFECYCLE: rec_join] WAV files finalized");

            // NOTE: system_stream was already moved into rt.block_on(async move {...})
            // and was dropped when that block returned. The SystemAudioStream::drop
            // should have sent a signal to the tokio forwarding task.

            // 5) Mix + Offline Diarization (Background Thread)
            // We spawn a new thread so the join handle returns immediately,
            // allowing stop_recording to return to UI without waiting for mix.
            let mix_app = rec_app.clone();
            let transcript_path_bg = transcript_path_t;
            std::thread::spawn(move || {
                use tauri::Emitter;
                // Notify start of mixing (optional, mainly for debug logs)
                println!("⏳ Starting background mix...");

                if let Err(e) = crate::wav::mix_pcm16_wav(&mic_path_t, &system_path_t, &mix_path_t)
                {
                    eprintln!("❌ mix failed: {e:?}");
                    let _ = mix_app.emit("tray-log", format!("❌ Background mix failed: {}", e));
                } else {
                    println!("✅ mix.wav: {}", mix_path_t.display());

                    // Convert mix.wav -> mix_asr_16k_mono.wav
                    match crate::wav::convert_wav_to_16k_mono_pcm16(&mix_path_t, &mix_asr_path_t) {
                        Ok(()) => {
                            println!("✅ mix_asr_16k_mono.wav: {}", mix_asr_path_t.display());
                            let _ = mix_app.emit(
                                "tray-log",
                                format!(
                                    "✅ Mixing & Resampling complete: {}",
                                    mix_path_t.file_name().unwrap_or_default().to_string_lossy()
                                ),
                            );
                        }
                        Err(e) => {
                            eprintln!("❌ convert mix_asr failed: {e:?}");
                            let _ = mix_app
                                .emit("tray-log", format!("❌ Convert mix_asr failed: {}", e));
                        }
                    }

                    // ── v8.0: Post-recording offline diarization ──────────────
                    if crate::diarization::is_initialized() {
                        println!("🔄 [DIARIZATION] Starting offline post-processing...");
                        match crate::diarization::process_wav(&mix_path_t) {
                            Ok(segments) => {
                                println!("✅ [DIARIZATION] Got {} segments, relabeling transcript...", segments.len());
                                match crate::diarization::relabel_transcript(
                                    &transcript_path_bg,
                                    &segments,
                                    recording_start_ms,
                                ) {
                                    Ok(n) => {
                                        println!("✅ [DIARIZATION] Relabeled {} entries", n);
                                        let _ = mix_app.emit("tray-log", format!("✅ Speaker diarization complete: {} entries relabeled", n));
                                    }
                                    Err(e) => eprintln!("⚠️ [DIARIZATION] Relabel failed: {}", e),
                                }
                            }
                            Err(e) => eprintln!("⚠️ [DIARIZATION] Offline processing failed: {}", e),
                        }
                    }
                }
            });
            println!("🔄 [LIFECYCLE: rec_join thread EXIT] Recording thread ending");
        });


        *guard = Some(RecorderHandle {
            stop,
            rec_join,
            mixer_join,
            asr_user_join,
            asr_system_join,
            base_dir: session_dir.clone(),
            mic_path: mic_path.clone(),
            system_path: system_path.clone(),
            mix_path: mix_path.clone(),
            mix_asr_path: mix_asr_path.clone(),
            transcript_writer: transcript_writer.clone(),
            transcript_path: transcript_path.clone(),
            recording_start_ms,
        });

        Ok((
            system_path.display().to_string(),
            mic_path.display().to_string(),
        ))
    })();

    match &res {
        Ok((sys, mic)) => {
            let msg = format!("started: [\"{}\",\"{}\"]", sys, mic);
            let _ = app.emit("tray-log", &msg);
        }
        Err(e) => {
            let msg = format!("❌ start_recording failed: {}", e);
            let _ = app.emit("tray-log", &msg);
        }
    }
    res
}

#[tauri::command]
pub async fn stop_recording(app: tauri::AppHandle) -> Result<(String, String, String), String> {
    use tauri::Emitter;
    println!("\n========================================");
    println!("▶ [LIFECYCLE: stop_recording] Called");
    println!("========================================");
    let _ = app.emit("tray-log", "▶ stop_recording...");

    let res = (async || -> Result<(String, String, String), String> {
        let handle = {
            let mut guard = RECORDER.lock().unwrap();
            match guard.take() {
                Some(h) => {
                    println!("✅ [LIFECYCLE: stop_recording] Got recorder handle");
                    h
                }
                None => {
                    println!("❌ [LIFECYCLE: stop_recording] No recorder handle found!");
                    return Err("recording is not running".into());
                }
            }
        };

        println!("🛑 [LIFECYCLE: stop_recording] Setting stop flag to true...");
        handle.stop.store(true, Ordering::Release);
        println!("✅ [LIFECYCLE: stop_recording] Stop flag set");

        // Join in blocking thread
        let system_path = handle.system_path.clone();
        let mic_path = handle.mic_path.clone();
        let mix_path = handle.mix_path.clone();
        let transcript_path = handle.transcript_path.clone();
        let recording_start_ms = handle.recording_start_ms;

        println!("🔄 [LIFECYCLE: stop_recording] Spawning blocking task for thread joins...");
        tauri::async_runtime::spawn_blocking(move || {
            println!("🔄 [LIFECYCLE: spawn_blocking ENTER] Starting thread joins...");

            // 1. Join 主录音线程
            println!("⏳ [LIFECYCLE: spawn_blocking] Joining rec_join thread...");
            let rec_result = handle.rec_join.join();
            println!(
                "✅ [LIFECYCLE: spawn_blocking] rec_join thread joined: {:?}",
                rec_result.is_ok()
            );

            // 2. 清理全局 PCM 通道，关闭 channel 唤醒阻塞的 mixer 线程
            println!("🧹 [LIFECYCLE: spawn_blocking] Clearing global PCM channels...");
            *MIC_PCM_TX.lock().unwrap() = None;
            *SYS_PCM_TX.lock().unwrap() = None;
            println!("✅ [LIFECYCLE: spawn_blocking] PCM channels cleared");

            // 3. Join 所有辅助线程
            println!("⏳ [LIFECYCLE: spawn_blocking] Joining mixer thread...");
            let mixer_result = handle.mixer_join.join();
            println!(
                "✅ [LIFECYCLE: spawn_blocking] mixer_join thread joined: {:?}",
                mixer_result.is_ok()
            );

            println!("⏳ [LIFECYCLE: spawn_blocking] Joining asr_user thread...");
            let asr_user_result = handle.asr_user_join.join();
            println!(
                "✅ [LIFECYCLE: spawn_blocking] asr_user_join thread joined: {:?}",
                asr_user_result.is_ok()
            );

            println!("⏳ [LIFECYCLE: spawn_blocking] Joining asr_system thread...");
            let asr_sys_result = handle.asr_system_join.join();
            println!(
                "✅ [LIFECYCLE: spawn_blocking] asr_system_join thread joined: {:?}",
                asr_sys_result.is_ok()
            );

            println!("🔄 [LIFECYCLE: spawn_blocking EXIT] All threads joined!");

            (system_path, mic_path, mix_path)
        })
        .await
        .map_err(|e| e.to_string())
        .map(|(s, m, x)| {
            (
                s.display().to_string(),
                m.display().to_string(),
                x.display().to_string(),
            )
        })
    })()
    .await;

    match &res {
        Ok((sys, mic, mix)) => {
            let msg = format!("stopped: [\"{}\",\"{}\",\"{}\"]", sys, mic, mix);
            println!("✅ [LIFECYCLE: stop_recording] Success: {}", msg);
            let _ = app.emit("tray-log", &msg);
        }
        Err(e) => {
            let msg = format!("❌ stop_recording failed: {}", e);
            println!("❌ [LIFECYCLE: stop_recording] Failed: {}", e);
            let _ = app.emit("tray-log", &msg);
        }
    }

    println!("========================================");
    println!("▶ [LIFECYCLE: stop_recording] Returning");
    println!("========================================\n");
    res
}

#[derive(serde::Serialize)]
struct ManualTranscriptEntry {
    text: String,
    source: String,
    is_manual: bool,
}

#[tauri::command]
pub fn add_manual_transcript(
    app: tauri::AppHandle,
    text: String,
    session_id: Option<String>,
) -> Result<(), String> {
    use std::io::Write;
    use tauri::Manager;

    // Create entry JSON
    let entry = ManualTranscriptEntry {
        text: text.clone(),
        source: "user".to_string(),
        is_manual: true,
    };
    let json_line = serde_json::to_string(&entry).map_err(|e| e.to_string())? + "\n";

    let mut handled_active = false;

    // Check active recorder
    {
        let guard = RECORDER.lock().unwrap();
        if let Some(handle) = guard.as_ref() {
            let active_path = &handle.base_dir;
            let active_id = active_path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or_default();

            let target_id = session_id.as_deref().unwrap_or(active_id);

            if active_id == target_id {
                let mut writer = handle
                    .transcript_writer
                    .lock()
                    .map_err(|_| "Failed to lock transcript writer".to_string())?;
                writer
                    .write_all(json_line.as_bytes())
                    .map_err(|e| e.to_string())?;
                writer.flush().map_err(|e| e.to_string())?;
                handled_active = true;
            }
        }
    }

    if handled_active {
        use tauri::Emitter;
        let _ = app.emit("asr_final", &json_line);
        return Ok(());
    }

    if let Some(sid) = session_id {
        let base_app_data = app.path().app_data_dir().map_err(|e| e.to_string())?;
        let session_dir = base_app_data.join("sessions").join(&sid);
        if !session_dir.exists() {
            return Err("Session not found".into());
        }
        let transcript_path = session_dir.join("transcript.jsonl");

        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(transcript_path)
            .map_err(|e| e.to_string())?;

        file.write_all(json_line.as_bytes())
            .map_err(|e| e.to_string())?;
        return Ok(());
    }

    Err("No active recording and no session ID provided".into())
}

// App entry wrapper for last record base (helpers if needed, but unused in main flow)
static LAST_RECORD_BASE: Lazy<Mutex<Option<std::path::PathBuf>>> = Lazy::new(|| Mutex::new(None));
#[allow(dead_code)]
fn set_last_record_base(p: std::path::PathBuf) {
    *LAST_RECORD_BASE.lock().unwrap() = Some(p);
}
#[allow(dead_code)]
fn get_last_record_base() -> Option<std::path::PathBuf> {
    LAST_RECORD_BASE.lock().unwrap().clone()
}
