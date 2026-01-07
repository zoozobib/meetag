#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use anyhow::{Context, Result};
use once_cell::sync::Lazy;
use std::sync::atomic::{AtomicBool, Ordering};
use std::{
    path::PathBuf,
    sync::{Arc, Mutex},
};

mod asr;
mod audio;
mod capture;
mod history;
mod llm;
mod text_filter;
mod tray;
mod vad;
mod wav;

use crate::wav::WavWriter;
use tauri::{Listener, Manager};

static RECORDER: Lazy<Mutex<Option<RecorderHandle>>> = Lazy::new(|| Mutex::new(None));

struct RecorderHandle {
    stop: Arc<AtomicBool>,
    join: std::thread::JoinHandle<()>,
    base_dir: PathBuf,
    mic_path: PathBuf,
    system_path: PathBuf,
    mix_path: PathBuf,
    mix_asr_path: PathBuf,
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
fn start_recording(app: tauri::AppHandle) -> Result<(String, String), String> {
    let mut guard = RECORDER.lock().unwrap();
    if guard.is_some() {
        return Err("recording already running".into());
    }

    // Output dir
    // Output dir: app_data/sessions/YYYY-MM-DD_HH-mm-ss
    let base_app_data: PathBuf = app.path().app_data_dir().map_err(|e| e.to_string())?;

    // Generate session timestamp
    let now = std::time::SystemTime::now();
    let dt: chrono::DateTime<chrono::Local> = now.into();
    let folder_name = dt.format("%Y-%m-%d_%H-%M-%S").to_string();
    let session_dir = base_app_data.join("sessions").join(folder_name);

    std::fs::create_dir_all(&session_dir).map_err(|e| e.to_string())?;

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
    std::thread::spawn(move || {
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
    std::thread::spawn(move || {
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
    std::thread::spawn(move || {
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

    // Use mic_pcm_tx.clone() to pass to valid stream
    let mic_tx_for_capture = mic_pcm_tx.clone();

    // Shared Atomic Gate for AEC (Energy Interlock)
    let system_speaking = Arc::new(AtomicBool::new(false));

    // Spawn recording thread
    let sys_speaking_mic = system_speaking.clone();
    let rec_app = app.clone();
    let join = std::thread::spawn(move || {
        // 1) MIC stream
        let mic_stream = match capture::start_mic_stream(
            mic_writer.clone(),
            mic_asr_writer.clone(),
            mic_tx_for_capture,
            asr_mic_tx, // Send to User ASR
            sys_speaking_mic,
        ) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("❌ start_mic_stream failed: {e:?}");
                return;
            }
        };
        if let Err(e) = cpal::traits::StreamTrait::play(&mic_stream) {
            eprintln!("❌ mic_stream.play failed: {e:?}");
            return;
        }

        // 2) System stream (CoreAudio)
        let mut system_stream =
            match audio::capture::core_audio::CoreAudioCapture::new().and_then(|c| c.stream()) {
                Ok(s) => s,
                Err(e) => {
                    eprintln!("❌ CoreAudioCapture failed: {e:?}");
                    return;
                }
            };

        let sr = system_stream.sample_rate();
        if let Err(e) = system_writer.lock().unwrap().init_pcm16(sr, 1) {
            eprintln!("❌ init system wav failed: {e:?}");
            return;
        }

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

            let mut buf: Vec<i16> = Vec::with_capacity(48000);
            let mut last_flush = Instant::now();

            let mut rs_phase: f32 = 0.0;
            let ratio = sr as f32 / 16_000.0;
            let mut prev_sample: f32 = 0.0;

            // AEC Gate State
            let mut rms_window_sum = 0.0;
            let mut rms_window_count = 0;
            let rms_window_size = 480; // ~10ms at 48kHz (adjust based on SR)
            let gate_threshold = 0.05; // Adjust this sensitivity!
            let mut hang_timer = 0;
            let hang_duration = 5; // Hold gate for ~5 windows (50ms) after loud sound

            while !stop2.load(Ordering::Acquire) {
                // Don't block forever
                match timeout(Duration::from_millis(200), system_stream.next()).await {
                    Ok(Some(s)) => {
                        let s: f32 = s;
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
                        // --------------------------------------

                        // --- Resample to 16kHz for ASR Mixer ---
                        rs_phase += 1.0 / ratio;
                        while rs_phase >= 1.0 {
                            let t = 1.0 - (rs_phase - 1.0);
                            let y = prev_sample + (s - prev_sample) * t;
                            let v_asr = (y.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;

                            if let Some(tx) = SYS_PCM_TX.lock().unwrap().as_ref() {
                                let _ = tx.send(v_asr);
                            }
                            // Also send to System ASR
                            let _ = asr_sys_tx.send(v_asr);
                            rs_phase -= 1.0;
                        }
                        prev_sample = s;
                        // ---------------------------------------

                        if buf.len() >= 48000 || last_flush.elapsed() >= Duration::from_secs(1) {
                            crate::wav::flush_i16(system_writer2.clone(), &mut buf);
                            last_flush = Instant::now();
                        }
                    }
                    Ok(None) => break, // stream ended
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
        });
        println!("✅ System capture runtime finished.");

        // 4) Clean up
        println!("🛑 Pausing mic stream...");
        if let Err(e) = cpal::traits::StreamTrait::pause(&mic_stream) {
            eprintln!("⚠️ Failed to pause mic stream: {:?}", e);
        }
        println!("🛑 Dropping mic stream...");
        drop(mic_stream);
        println!("✅ Mic stream dropped.");

        // Finalize WAVs
        let _ = system_writer.lock().unwrap().finalize();
        let _ = mic_writer.lock().unwrap().finalize();
        let _ = mic_asr_writer.lock().unwrap().finalize();

        // 5) Mix (Background Thread)
        // We spawn a new thread so the join handle returns immediately,
        // allowing stop_recording to return to UI without waiting for mix.
        let mix_app = rec_app.clone();
        std::thread::spawn(move || {
            use tauri::Emitter;
            // Notify start of mixing (optional, mainly for debug logs)
            println!("⏳ Starting background mix...");

            if let Err(e) = crate::wav::mix_pcm16_wav(&mic_path_t, &system_path_t, &mix_path_t) {
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
                        let _ =
                            mix_app.emit("tray-log", format!("❌ Convert mix_asr failed: {}", e));
                    }
                }
            }
        });
    });

    *guard = Some(RecorderHandle {
        stop,
        join,
        base_dir: session_dir.clone(),
        mic_path: mic_path.clone(),
        system_path: system_path.clone(),
        mix_path: mix_path.clone(),
        mix_asr_path: mix_asr_path.clone(),
    });

    Ok((
        system_path.display().to_string(),
        mic_path.display().to_string(),
    ))
}

#[tauri::command]
async fn stop_recording() -> Result<(String, String, String), String> {
    let handle = {
        let mut guard = RECORDER.lock().unwrap();
        guard.take().ok_or("recording is not running")?
    };

    handle.stop.store(true, Ordering::Release);

    // Join in blocking thread
    let system_path = handle.system_path.clone();
    let mic_path = handle.mic_path.clone();
    let mix_path = handle.mix_path.clone();

    tauri::async_runtime::spawn_blocking(move || {
        let _ = handle.join.join();
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

// Global handle for the whisper sidecar process
static WHISPER_PROCESS: Lazy<Mutex<Option<tauri_plugin_shell::process::CommandChild>>> =
    Lazy::new(|| Mutex::new(None));

fn main() {
    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .on_window_event(|window, event| {
            if let tauri::WindowEvent::CloseRequested { api, .. } = event {
                // For now, adhere to the user's existing logic (hide on close),
                // but this contributes to the "zombie process" feeling if not careful.
                // However, the main fix is ensuring real exit kills the child.
                window.hide().unwrap();
                api.prevent_close();
            }
        })
        .setup(|app| {
            let handle = app.handle().clone();

            // Init Tray
            tray::create_tray(&handle)?;

            // Listen for Tray Events
            let h1 = handle.clone();
            handle.listen("tray-open-window", move |_| {
                use tauri::Manager;
                if let Some(window) = h1.get_webview_window("main") {
                    let _ = window.show();
                    let _ = window.set_focus();
                }
            });

            let h2 = handle.clone();
            handle.listen("tray-record-start", move |_| {
                use tauri::Emitter;
                let _ = h2.emit("tray-log", "▶ start_recording...");
                println!("▶ start_recording...");
                match start_recording(h2.clone()) {
                    Ok((sys, mic)) => {
                        let msg = format!("started: [\"{}\",\"{}\"]", sys, mic);
                        let _ = h2.emit("tray-log", &msg);
                        println!("✅ {}", msg);
                    }
                    Err(e) => {
                        let msg = format!("❌ start_recording failed: {}", e);
                        let _ = h2.emit("tray-log", &msg);
                        eprintln!("{}", msg);
                    }
                }
            });

            let h3 = handle.clone();
            handle.listen("tray-record-stop", move |_| {
                use tauri::Emitter;
                let _ = h3.emit("tray-log", "▶ stop_recording...");
                println!("▶ stop_recording...");
                let h_stop = h3.clone();
                tauri::async_runtime::spawn(async move {
                    match stop_recording().await {
                        Ok((sys, mic, mix)) => {
                            let msg = format!("stopped: [\"{}\",\"{}\",\"{}\"]", sys, mic, mix);
                            let _ = h_stop.emit("tray-log", &msg);
                            println!("{}", msg);
                        }
                        Err(e) => {
                            let msg = format!("❌ stop_recording failed: {}", e);
                            let _ = h_stop.emit("tray-log", &msg);
                            eprintln!("{}", msg);
                        }
                    }
                });
            });

            tauri::async_runtime::spawn(async move {
                use tauri::Manager;
                use tauri_plugin_shell::ShellExt;

                let resource_path = handle
                    .path()
                    .resolve(
                        "resources/ggml-small.bin",
                        tauri::path::BaseDirectory::Resource,
                    )
                    .unwrap();

                // Start whisper server sidecar
                let sidecar_command = handle.shell().sidecar("whisper-server").unwrap().args([
                    "-m",
                    resource_path.to_str().unwrap(),
                    "--port",
                    "8178",
                    "--host",
                    "127.0.0.1",
                ]);

                let (mut rx, child) = sidecar_command
                    .spawn()
                    .expect("Failed to spawn whisper sidecar");

                println!(
                    "🚀 Whisper sidecar spawned with PID: {:?} on port 8178",
                    child.pid()
                );

                // Store the child process handle globally
                *WHISPER_PROCESS.lock().unwrap() = Some(child);

                // Continuously read the sidecar's output to prevent pipe blocking
                use tauri_plugin_shell::process::CommandEvent;
                while let Some(event) = rx.recv().await {
                    match event {
                        CommandEvent::Stdout(line) => {
                            // Only print if needed, or just consume it
                            let log = String::from_utf8_lossy(&line);
                            println!("[Whisper] {}", log.trim());
                        }
                        CommandEvent::Stderr(line) => {
                            let log = String::from_utf8_lossy(&line);
                            eprintln!("[Whisper Err] {}", log.trim());
                        }
                        _ => {}
                    }
                }
                println!("⚠️ Whisper sidecar channel closed");
            });
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            start_recording,
            stop_recording,
            history::get_sessions,
            history::get_sessions,
            history::get_session_detail,
            llm::generate_summary,
            llm::save_summary,
            llm::get_summary
        ])
        .build(tauri::generate_context!())
        .expect("error while building tauri application")
        .run(|_app_handle, event| {
            if let tauri::RunEvent::Exit = event {
                // Cleanup sidecar on exit
                let mut guard = WHISPER_PROCESS.lock().unwrap();
                if let Some(child) = guard.take() {
                    println!("🛑 Killing whisper sidecar (PID: {:?})", child.pid());
                    let _ = child.kill();
                }
            }
        });
}
