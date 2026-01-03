#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use anyhow::{Context, Result};
use futures_util::StreamExt;
use once_cell::sync::Lazy;
use std::sync::atomic::{AtomicBool, Ordering};
use std::{
    fs::File,
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
};

mod asr;
mod audio;
mod capture;
mod vad;
mod wav;

use crate::wav::WavWriter;
use tauri::Manager;

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
    let base: PathBuf = app.path().app_data_dir().map_err(|e| e.to_string())?;
    std::fs::create_dir_all(&base).map_err(|e| e.to_string())?;

    let system_path = base.join("system.wav");
    let mic_path = base.join("mic.wav");
    let mic_asr_path = base.join("mic_asr_16k_mono.wav");
    let mix_path = base.join("mix.wav");
    let mix_asr_path = base.join("mix_asr_16k_mono.wav");

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

    // ASR worker thread (chunks -> /inference)
    let asr_app = app.clone();
    let stop_asr = stop.clone();
    std::thread::spawn(move || {
        let _ = asr::realtime_inference_worker(asr_app, stop_asr, mix_pcm_rx);
    });

    let stop2 = stop.clone();
    let mic_path_t = mic_path.clone();
    let system_path_t = system_path.clone();
    let mix_path_t = mix_path.clone();
    let mix_asr_path_t = mix_asr_path.clone();

    // Use mic_pcm_tx.clone() to pass to valid stream
    let mic_tx_for_capture = mic_pcm_tx.clone();

    // Spawn recording thread
    let join = std::thread::spawn(move || {
        // 1) MIC stream
        let mic_stream = match capture::start_mic_stream(
            mic_writer.clone(),
            mic_asr_writer.clone(),
            mic_tx_for_capture,
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

        rt.block_on(async move {
            use futures_util::StreamExt;
            use tokio::time::{timeout, Duration, Instant};

            let mut buf: Vec<i16> = Vec::with_capacity(48000);
            let mut last_flush = Instant::now();

            // System Resample State
            let mut rs_phase: f32 = 0.0;
            let ratio = sr as f32 / 16_000.0;
            let mut prev_sample: f32 = 0.0;

            while !stop2.load(Ordering::Acquire) {
                // Don't block forever
                match timeout(Duration::from_millis(200), system_stream.next()).await {
                    Ok(Some(s)) => {
                        let s: f32 = s;
                        let v = (s.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
                        buf.push(v);

                        // --- Resample to 16kHz for ASR Mixer ---
                        rs_phase += 1.0 / ratio;
                        while rs_phase >= 1.0 {
                            let t = 1.0 - (rs_phase - 1.0);
                            let y = prev_sample + (s - prev_sample) * t;
                            let v_asr = (y.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;

                            if let Some(tx) = SYS_PCM_TX.lock().unwrap().as_ref() {
                                let _ = tx.send(v_asr);
                            }
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
        });

        // 4) Clean up
        drop(mic_stream);

        // Finalize WAVs
        let _ = system_writer.lock().unwrap().finalize();
        let _ = mic_writer.lock().unwrap().finalize();
        let _ = mic_asr_writer.lock().unwrap().finalize();

        // 5) Mix
        if let Err(e) = crate::wav::mix_pcm16_wav(&mic_path_t, &system_path_t, &mix_path_t) {
            eprintln!("❌ mix failed: {e:?}");
        } else {
            println!("✅ mix.wav: {}", mix_path_t.display());

            // Convert mix.wav -> mix_asr_16k_mono.wav
            match crate::wav::convert_wav_to_16k_mono_pcm16(&mix_path_t, &mix_asr_path_t) {
                Ok(()) => println!("✅ mix_asr_16k_mono.wav: {}", mix_asr_path_t.display()),
                Err(e) => eprintln!("❌ convert mix_asr failed: {e:?}"),
            }
        }
    });

    *guard = Some(RecorderHandle {
        stop,
        join,
        base_dir: base.clone(),
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

fn main() {
    tauri::Builder::default()
        // NOTE: start_demo_recording removed or can be re-added if needed, but it was commented out in original file mostly?
        // User asked to clean up, so only keeping active commands.
        // Wait, start_demo_recording WAS there but commented out in the last view?
        // Let's checking View 256. Lines 1154-1261 are commented out?
        // Lines 1154 starts `// fn start_demo_recording`.
        // Yes, it was commented out. So I can remove it or keep it commented. Removing is cleaner.
        .invoke_handler(tauri::generate_handler![start_recording, stop_recording])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
