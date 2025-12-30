#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use anyhow::{Context, Result};
use futures_util::StreamExt;
use std::{
    fs::File,
    io::{Seek, SeekFrom, Write},
    path::PathBuf,
    sync::{Arc, Mutex},
    time::Duration as StdDuration,
};
use tauri::Manager;

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

mod audio;

// =====================
// Minimal WAV writer (PCM16)
// =====================
struct WavWriter {
    f: File,
    data_bytes: u32,
    initialized: bool,
}

impl WavWriter {
    fn create(path: &std::path::Path) -> Result<Self> {
        Ok(Self {
            f: File::create(path)?,
            data_bytes: 0,
            initialized: false,
        })
    }

    fn init_pcm16(&mut self, sample_rate: u32, channels: u16) -> Result<()> {
        if self.initialized {
            return Ok(());
        }
        let bits: u16 = 16;
        let byte_rate = sample_rate * channels as u32 * (bits as u32 / 8);
        let block_align = channels * (bits / 8);

        self.f.write_all(b"RIFF")?;
        self.f.write_all(&0u32.to_le_bytes())?; // placeholder
        self.f.write_all(b"WAVE")?;

        self.f.write_all(b"fmt ")?;
        self.f.write_all(&16u32.to_le_bytes())?; // PCM fmt chunk size
        self.f.write_all(&1u16.to_le_bytes())?; // PCM format
        self.f.write_all(&channels.to_le_bytes())?;
        self.f.write_all(&sample_rate.to_le_bytes())?;
        self.f.write_all(&byte_rate.to_le_bytes())?;
        self.f.write_all(&block_align.to_le_bytes())?;
        self.f.write_all(&bits.to_le_bytes())?;

        self.f.write_all(b"data")?;
        self.f.write_all(&0u32.to_le_bytes())?; // placeholder
        self.initialized = true;
        Ok(())
    }

    fn write_data(&mut self, bytes: &[u8]) {
        if !self.initialized {
            return;
        }
        if self.f.write_all(bytes).is_ok() {
            self.data_bytes = self.data_bytes.saturating_add(bytes.len() as u32);
        }
    }

    fn finalize(&mut self) -> Result<()> {
        if !self.initialized {
            return Ok(());
        }
        let riff_size = 36 + self.data_bytes;

        self.f.seek(SeekFrom::Start(4))?;
        self.f.write_all(&riff_size.to_le_bytes())?;

        self.f.seek(SeekFrom::Start(40))?;
        self.f.write_all(&self.data_bytes.to_le_bytes())?;

        self.f.flush()?;
        Ok(())
    }
}

fn flush_i16(writer: Arc<Mutex<WavWriter>>, buf: &mut Vec<i16>) {
    let mut bytes = Vec::with_capacity(buf.len() * 2);
    for &v in buf.iter() {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    writer.lock().unwrap().write_data(&bytes);
    buf.clear();
}

// =====================
// MIC capture (CPAL) -> mic.wav
// =====================
fn start_mic_stream(writer: Arc<Mutex<WavWriter>>) -> Result<cpal::Stream> {
    let host = cpal::default_host();
    let dev = host
        .default_input_device()
        .context("no default input device")?;

    let cfg = dev
        .default_input_config()
        .context("no default input config")?;

    let sample_rate = cfg.sample_rate().0;
    let channels = cfg.channels() as u16;

    writer.lock().unwrap().init_pcm16(sample_rate, channels)?;

    let stream_config: cpal::StreamConfig = cfg.clone().into();
    let err_fn = |err| eprintln!("mic stream error: {err}");

    match cfg.sample_format() {
        cpal::SampleFormat::F32 => {
            let w = writer.clone();
            let stream = dev.build_input_stream(
                &stream_config,
                move |data: &[f32], _| {
                    // 你要更响可以把 gain 调大，比如 4.0/6.0
                    let gain: f32 = 1.0;

                    let mut bytes = Vec::with_capacity(data.len() * 2);
                    for &x in data {
                        let v = ((x * gain).clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
                        bytes.extend_from_slice(&v.to_le_bytes());
                    }
                    w.lock().unwrap().write_data(&bytes);
                },
                err_fn,
                None,
            )?;
            Ok(stream)
        }
        other => anyhow::bail!("mic format {:?} not handled in demo", other),
    }
}

// =====================
// Tauri command: record N seconds -> (system.wav, mic.wav)
// NOTE: sync function to avoid Send future issues
// =====================
#[tauri::command]
fn start_demo_recording(app: tauri::AppHandle, seconds: u64) -> Result<(String, String), String> {
    let (tx, rx) = std::sync::mpsc::channel();

    std::thread::spawn(move || {
        let res: Result<(String, String), String> = (|| {
            let base: PathBuf = app.path().app_data_dir().map_err(|e| e.to_string())?;
            std::fs::create_dir_all(&base).map_err(|e| e.to_string())?;

            let system_path = base.join("system.wav");
            let mic_path = base.join("mic.wav");

            let system_writer = Arc::new(Mutex::new(
                WavWriter::create(&system_path).map_err(|e| e.to_string())?,
            ));
            let mic_writer = Arc::new(Mutex::new(
                WavWriter::create(&mic_path).map_err(|e| e.to_string())?,
            ));

            // 1) MIC stream (CPAL)
            let mic_stream = start_mic_stream(mic_writer.clone()).map_err(|e| e.to_string())?;
            mic_stream.play().map_err(|e| e.to_string())?;

            // 2) System audio stream (CoreAudio Tap)
            let mut system_stream = audio::capture::core_audio::CoreAudioCapture::new()
                .map_err(|e| format!("CoreAudioCapture::new failed: {e:?}"))?
                .stream()
                .map_err(|e| format!("CoreAudioCapture::stream failed: {e:?}"))?;

            let sr = system_stream.sample_rate();
            system_writer
                .lock()
                .unwrap()
                .init_pcm16(sr, 1)
                .map_err(|e| e.to_string())?;

            // 3) Run async loop on a local tokio runtime (current-thread)
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_time()
                .build()
                .map_err(|e| e.to_string())?;

            let system_writer2 = system_writer.clone();

            rt.block_on(async move {
                use tokio::time::{Duration, Instant};

                let deadline = Instant::now() + Duration::from_secs(seconds);
                let mut buf: Vec<i16> = Vec::with_capacity(48000);

                while Instant::now() < deadline {
                    if let Some(s) = system_stream.next().await {
                        // 强制类型，避免推断失败
                        let s: f32 = s;

                        let v = (s.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
                        buf.push(v);

                        if buf.len() >= 48000 {
                            flush_i16(system_writer2.clone(), &mut buf);
                        }
                    } else {
                        break;
                    }
                }

                if !buf.is_empty() {
                    flush_i16(system_writer2.clone(), &mut buf);
                }

                // 小睡一下让 CPAL 回调把最后一小段写进去（可选）
                tokio::time::sleep(Duration::from_millis(50)).await;
            });

            // 4) stop mic
            drop(mic_stream);

            // 5) finalize wav
            system_writer
                .lock()
                .unwrap()
                .finalize()
                .map_err(|e| e.to_string())?;
            mic_writer
                .lock()
                .unwrap()
                .finalize()
                .map_err(|e| e.to_string())?;

            Ok((
                system_path.display().to_string(),
                mic_path.display().to_string(),
            ))
        })();

        let _ = tx.send(res);
    });

    rx.recv().map_err(|e| e.to_string())?
}

// =====================
// App entry
// =====================
// fn main() {
//     tauri::Builder::default()
//         // 如果你想“0 UI 自动开始录音”，可以在 setup 里直接调用 start_demo_recording
//         .invoke_handler(tauri::generate_handler![start_demo_recording])
//         .run(tauri::generate_context!())
//         .expect("error while running tauri application");
// }

fn main() {
    tauri::Builder::default()
        .setup(|app| {
            // 一启动就开始录音：10 秒
            let handle = app.handle().clone();

            std::thread::spawn(move || {
                // 这里直接调用我们写的 command 函数即可
                match start_demo_recording(handle, 10) {
                    Ok((system_path, mic_path)) => {
                        println!("✅ system.wav: {}", system_path);
                        println!("✅ mic.wav: {}", mic_path);
                    }
                    Err(e) => {
                        eprintln!("❌ recording failed: {}", e);
                    }
                }
            });

            Ok(())
        })
        .invoke_handler(tauri::generate_handler![start_demo_recording])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
