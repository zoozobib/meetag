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
// Minimal WAV PCM16 Reader
// =====================
#[derive(Debug, Clone)]
struct Pcm16Wav {
    sample_rate: u32,
    channels: u16,
    samples: Vec<i16>, // interleaved
}

/// 只支持 PCM16 little-endian WAV
fn read_pcm16_wav(path: &std::path::Path) -> Result<Pcm16Wav> {
    use std::io::Read;

    let mut f = std::fs::File::open(path)?;
    let mut buf = Vec::new();
    f.read_to_end(&mut buf)?;

    // very small / invalid
    if buf.len() < 44 {
        anyhow::bail!("wav too small: {:?}", path);
    }

    // RIFF header
    if &buf[0..4] != b"RIFF" || &buf[8..12] != b"WAVE" {
        anyhow::bail!("not a RIFF/WAVE file: {:?}", path);
    }

    // find "fmt " and "data" chunks (robust scanning)
    let mut pos = 12;
    let mut fmt_found = None;
    let mut data_found = None;

    while pos + 8 <= buf.len() {
        let chunk_id = &buf[pos..pos + 4];
        let chunk_size = u32::from_le_bytes(buf[pos + 4..pos + 8].try_into().unwrap()) as usize;
        let chunk_data_start = pos + 8;
        let chunk_data_end = chunk_data_start + chunk_size;

        if chunk_data_end > buf.len() {
            break;
        }

        if chunk_id == b"fmt " {
            fmt_found = Some((chunk_data_start, chunk_size));
        } else if chunk_id == b"data" {
            data_found = Some((chunk_data_start, chunk_size));
            break; // 通常 data 在后面，找到就可以结束
        }

        // chunks are word-aligned
        pos = chunk_data_end + (chunk_size % 2);
    }

    let (fmt_start, fmt_size) = fmt_found.context("fmt chunk not found")?;
    let (data_start, data_size) = data_found.context("data chunk not found")?;

    if fmt_size < 16 {
        anyhow::bail!("fmt chunk too small");
    }

    let audio_format = u16::from_le_bytes(buf[fmt_start..fmt_start + 2].try_into().unwrap());
    let channels = u16::from_le_bytes(buf[fmt_start + 2..fmt_start + 4].try_into().unwrap());
    let sample_rate = u32::from_le_bytes(buf[fmt_start + 4..fmt_start + 8].try_into().unwrap());
    let bits_per_sample =
        u16::from_le_bytes(buf[fmt_start + 14..fmt_start + 16].try_into().unwrap());

    if audio_format != 1 {
        anyhow::bail!("only PCM supported (format=1). got {}", audio_format);
    }
    if bits_per_sample != 16 {
        anyhow::bail!("only 16-bit PCM supported. got {} bits", bits_per_sample);
    }

    let data = &buf[data_start..data_start + data_size];

    if data.len() % 2 != 0 {
        anyhow::bail!("data chunk not aligned");
    }

    let mut samples = Vec::with_capacity(data.len() / 2);
    for i in (0..data.len()).step_by(2) {
        samples.push(i16::from_le_bytes([data[i], data[i + 1]]));
    }

    Ok(Pcm16Wav {
        sample_rate,
        channels,
        samples,
    })
}

/// 把 stereo/mono 统一转换成 mono（简单平均）
/// 返回 mono samples
fn to_mono(w: &Pcm16Wav) -> Vec<i16> {
    if w.channels == 1 {
        return w.samples.clone();
    }
    if w.channels == 2 {
        let mut mono = Vec::with_capacity(w.samples.len() / 2);
        let mut i = 0;
        while i + 1 < w.samples.len() {
            let l = w.samples[i] as i32;
            let r = w.samples[i + 1] as i32;
            mono.push(((l + r) / 2) as i16);
            i += 2;
        }
        return mono;
    }
    // 超过 2 声道就取第一个声道
    let ch = w.channels as usize;
    let frames = w.samples.len() / ch;
    let mut mono = Vec::with_capacity(frames);
    for f in 0..frames {
        mono.push(w.samples[f * ch]);
    }
    mono
}

/// 混音：mix = mic * mic_gain + sys * sys_gain
/// 输出 mono PCM16 WAV（sample_rate 同 input）
fn mix_pcm16_wav(
    mic_path: &std::path::Path,
    sys_path: &std::path::Path,
    out_path: &std::path::Path,
    mic_gain: f32,
    sys_gain: f32,
) -> Result<()> {
    let mic = read_pcm16_wav(mic_path)?;
    let sys = read_pcm16_wav(sys_path)?;

    // 检查采样率一致（最小版本先不做 resample）
    if mic.sample_rate != sys.sample_rate {
        anyhow::bail!(
            "sample rate mismatch: mic={} sys={}",
            mic.sample_rate,
            sys.sample_rate
        );
    }

    let mic_mono = to_mono(&mic);
    let sys_mono = to_mono(&sys);

    let max_len = mic_mono.len().max(sys_mono.len());
    let mut mixed: Vec<i16> = Vec::with_capacity(max_len);

    for i in 0..max_len {
        let m = if i < mic_mono.len() {
            mic_mono[i] as f32
        } else {
            0.0
        };
        let s = if i < sys_mono.len() {
            sys_mono[i] as f32
        } else {
            0.0
        };

        let y = m * mic_gain + s * sys_gain;
        // 防止爆音
        let y = y.clamp(i16::MIN as f32, i16::MAX as f32);
        mixed.push(y as i16);
    }

    // 写出 WAV（复用你的 WavWriter）
    let mut w = WavWriter::create(out_path)?;
    w.init_pcm16(mic.sample_rate, 1)?;
    let mut bytes = Vec::with_capacity(mixed.len() * 2);
    for v in mixed {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    w.write_data(&bytes);
    w.finalize()?;

    Ok(())
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

            let mix_path = base.join("mix.wav");

            // 经验值：mic 通常会偏小，所以 mic_gain 可以调大
            // 你可以先用 mic_gain=3.0, sys_gain=1.0
            mix_pcm16_wav(&mic_path, &system_path, &mix_path, 3.0, 1.0)
                .map_err(|e| format!("mix failed: {e:?}"))?;

            println!("✅ mix.wav: {}", mix_path.display());

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
