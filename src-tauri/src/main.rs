#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use anyhow::{Context, Result};
use futures_util::StreamExt;
use std::{
    fs::File,
    io::{Seek, SeekFrom, Write},
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
    time::Duration as StdDuration,
};
use tauri::Emitter;

static LAST_RECORD_BASE: once_cell::sync::OnceCell<std::sync::Mutex<Option<std::path::PathBuf>>> =
    once_cell::sync::OnceCell::new();

static MIC_PCM_TX: Lazy<std::sync::Mutex<Option<std::sync::mpsc::Sender<i16>>>> =
    Lazy::new(|| std::sync::Mutex::new(None));
static SYS_PCM_TX: Lazy<std::sync::Mutex<Option<std::sync::mpsc::Sender<i16>>>> =
    Lazy::new(|| std::sync::Mutex::new(None));

use tauri::Manager;

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

mod audio;

use once_cell::sync::Lazy;
use std::sync::atomic::{AtomicBool, Ordering};
use webrtc_vad::{Vad, VadMode};

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

// VAD wrapper to allow sending to audio thread
// VAD wrapper to allow sending to audio thread
pub struct SendVad(pub Vad);
unsafe impl Send for SendVad {}
unsafe impl Sync for SendVad {}

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
fn start_mic_stream(
    writer_raw: Arc<Mutex<WavWriter>>,
    writer_asr: Arc<Mutex<WavWriter>>,
) -> Result<cpal::Stream> {
    let host = cpal::default_host();
    let dev = host
        .default_input_device()
        .context("no default input device")?;

    let cfg = dev
        .default_input_config()
        .context("no default input config")?;

    let sample_rate = cfg.sample_rate().0;
    let channels = cfg.channels() as u16;

    writer_raw
        .lock()
        .unwrap()
        .init_pcm16(sample_rate, channels)?;
    // ASR-ready track: 16kHz mono PCM16
    writer_asr.lock().unwrap().init_pcm16(16_000, 1)?;

    let mut vad_wrapper = Arc::new(Mutex::new(SendVad(Vad::new_with_rate_and_mode(
        webrtc_vad::SampleRate::Rate16kHz,
        VadMode::VeryAggressive,
    ))));
    let mut vad_buf: Vec<i16> = Vec::with_capacity(320 * 10); // buffer for VAD
    let mut speech_hold_frames = 0; // for short "hangover" after speech

    // --- Mic AGC state (shared across callbacks) ---
    // Store gain in Q8 fixed-point (gain * 256) so we can keep it in an atomic.
    use std::sync::atomic::{AtomicU32, Ordering};
    static MIC_GAIN_Q8: AtomicU32 = AtomicU32::new((50.0_f32 * 256.0_f32) as u32);

    // Tunables (safe defaults)
    // Base gain keeps your original loudness in non-call scenarios.
    // AGC will only BOOST above this when the system/WeChat suppresses the mic.
    let base_gain: f32 = 15.0; // Reduced from 50.0 to prevent noise floor boosting
    let target_rms: f32 = 0.15; // desired loudness (0..1) *when boosting* ~ -16dBFS
                                // Note: we do NOT attenuate below base_gain in this strategy.
    let max_gain: f32 = 50.0; // Reduced from 400.0. 50x is plenty (34dB).
    let smooth: f32 = 0.90; // 0.0..1.0, higher = smoother/slower gain changes
    let decay: f32 = 0.99; // Slow release for gate
    let noise_gate: f32 = 0.01; // Input RMS below this is considered noise: don't boost!
    let limiter: f32 = 0.98; // soft limiter threshold
    let rms_floor: f32 = 1.0e-5; // avoid divide-by-zero / silence spikes

    let stream_config: cpal::StreamConfig = cfg.clone().into();
    let err_fn = |err| eprintln!("mic stream error: {err}");

    match cfg.sample_format() {
        cpal::SampleFormat::F32 => {
            let w = writer_raw.clone();
            let w_asr = writer_asr.clone();
            let mut rs_phase: f32 = 0.0;
            let ratio: f32 = sample_rate as f32 / 16_000.0;
            let mut prev_mono: f32 = 0.0;
            let stream = dev.build_input_stream(
                &stream_config,
                move |data: &[f32], _| {
                    if data.is_empty() {
                        return;
                    }

                    // Compute RMS on the incoming buffer (interleaved channels).
                    let mut sum = 0.0f32;
                    for &x in data {
                        sum += x * x;
                    }
                    let rms = (sum / (data.len() as f32)).sqrt();

                    // Read current gain (Q8 -> f32)
                    let mut gain = (MIC_GAIN_Q8.load(Ordering::Relaxed) as f32) / 256.0;
                    if gain < base_gain {
                        gain = base_gain;
                    }

                    // Update gain
                    // Update gain logic moved to after VAD check
                    // But we need 'current' gain to process audio first.
                    // We will update MIC_GAIN_Q8 at the END of the loop for the NEXT buffer.

                    // Convert to PCM16 with soft limiter.
                    let mut bytes = Vec::with_capacity(data.len() * 2);
                    for &x in data {
                        let mut y = x * gain;
                        if y > limiter {
                            y = limiter;
                        } else if y < -limiter {
                            y = -limiter;
                        }
                        let v = (y * i16::MAX as f32) as i16;
                        bytes.extend_from_slice(&v.to_le_bytes());
                    }

                    w.lock().unwrap().write_data(&bytes);

                    // --- ASR track: downmix to mono, resample to 16k, then PCM16 ---
                    let ch = channels as usize;
                    let mut asr_bytes = Vec::new();
                    if ch == 0 {
                        return;
                    }
                    // Iterate frames
                    for frame_idx in 0..(data.len() / ch) {
                        let mut mono = 0.0f32;
                        for c in 0..ch {
                            mono += data[frame_idx * ch + c] as f32;
                        }
                        mono /= ch as f32;
                        mono *= gain;

                        // soft limiter
                        if mono > limiter {
                            mono = limiter;
                        } else if mono < -limiter {
                            mono = -limiter;
                        }

                        // linear resample: emit when phase crosses 1.0
                        // phase advances by 1/ratio per input sample (input_sr / 16000 = ratio)
                        rs_phase += 1.0 / ratio;
                        while rs_phase >= 1.0 {
                            let t = 1.0 - (rs_phase - 1.0);
                            let y = prev_mono + (mono - prev_mono) * t;
                            let v = (y * i16::MAX as f32) as i16;
                            asr_bytes.extend_from_slice(&v.to_le_bytes());
                            rs_phase -= 1.0;
                        }
                        prev_mono = mono;
                    }
                    if !asr_bytes.is_empty() {
                        w_asr.lock().unwrap().write_data(&asr_bytes);

                        // Send 16k mono ASR samples to mixer
                        if let Some(tx) = MIC_PCM_TX.lock().unwrap().as_ref() {
                            for ch in asr_bytes.chunks_exact(2) {
                                let v = i16::from_le_bytes([ch[0], ch[1]]);
                                let _ = tx.send(v);
                            }
                        }
                    }

                    // --- VAD & AGC Update ---
                    // 1. Append new 16kHz samples to vad_buf
                    if !asr_bytes.is_empty() {
                        for ch in asr_bytes.chunks_exact(2) {
                            let v = i16::from_le_bytes([ch[0], ch[1]]);
                            vad_buf.push(v);
                        }
                    }

                    // 2. Process VAD frames (20ms = 320 samples)
                    let mut is_speech_now = false;
                    while vad_buf.len() >= 320 {
                        let frame: Vec<i16> = vad_buf.drain(0..320).collect();
                        if let Ok(true) = vad_wrapper.lock().unwrap().0.is_voice_segment(&frame) {
                            is_speech_now = true;
                            speech_hold_frames = 20; // Hold 'speech' state for ~400ms (20 * 20ms)
                        } else {
                            if speech_hold_frames > 0 {
                                speech_hold_frames -= 1;
                            }
                        }
                    }

                    // 3. Update AGC Gain for NEXT callback
                    // Use 'is_speech_now' OR 'speech_hold_frames > 0' as "Speech Active"
                    let speech_active = is_speech_now || speech_hold_frames > 0;

                    if speech_active {
                        // Speech detected: Move gain towards target based on current RMS
                        // desired gain = target_rms / rms
                        // But rms is from the RAW input (full bandwidth).
                        // If speech is active, we trust that RMS represents speech energy.
                        if rms > rms_floor {
                            let desired = (target_rms / rms).clamp(1.0, max_gain).max(base_gain);
                            gain = (gain * smooth + desired * (1.0 - smooth))
                                .clamp(base_gain, max_gain);
                        }
                    } else {
                        // Silence: Decay gain
                        gain = gain * decay + base_gain * (1.0 - decay);
                    }
                    MIC_GAIN_Q8.store((gain * 256.0) as u32, Ordering::Relaxed);
                },
                err_fn,
                None,
            )?;
            Ok(stream)
        }
        cpal::SampleFormat::I16 => {
            let w = writer_raw.clone();
            let w_asr = writer_asr.clone();
            let mut rs_phase: f32 = 0.0;
            let ratio: f32 = sample_rate as f32 / 16_000.0;
            let mut prev_mono: f32 = 0.0;
            let mut rs_phase: f32 = 0.0;
            let ratio: f32 = sample_rate as f32 / 16_000.0;
            let mut prev_mono: f32 = 0.0;
            let stream = dev.build_input_stream(
                &stream_config,
                move |data: &[i16], _| {
                    if data.is_empty() {
                        return;
                    }

                    // RMS in normalized float domain
                    let mut sum = 0.0f32;
                    for &x in data {
                        let xf = x as f32 / i16::MAX as f32;
                        sum += xf * xf;
                    }
                    let rms = (sum / (data.len() as f32)).sqrt();

                    let mut gain = (MIC_GAIN_Q8.load(Ordering::Relaxed) as f32) / 256.0;
                    if gain < base_gain {
                        gain = base_gain;
                    }
                    if rms > noise_gate {
                        let desired = (target_rms / rms).clamp(1.0, max_gain).max(base_gain);
                        gain =
                            (gain * smooth + desired * (1.0 - smooth)).clamp(base_gain, max_gain);
                    } else {
                        gain = gain * decay + base_gain * (1.0 - decay);
                    }
                    MIC_GAIN_Q8.store((gain * 256.0) as u32, Ordering::Relaxed);

                    let mut bytes = Vec::with_capacity(data.len() * 2);
                    for &x in data {
                        let mut y = (x as f32 / i16::MAX as f32) * gain;
                        if y > limiter {
                            y = limiter;
                        } else if y < -limiter {
                            y = -limiter;
                        }
                        let v = (y * i16::MAX as f32) as i16;
                        bytes.extend_from_slice(&v.to_le_bytes());
                    }
                    w.lock().unwrap().write_data(&bytes);

                    // --- ASR track: downmix to mono, resample to 16k, then PCM16 ---
                    let ch = channels as usize;
                    let mut asr_bytes = Vec::new();
                    if ch == 0 {
                        return;
                    }
                    // Iterate frames
                    for frame_idx in 0..(data.len() / ch) {
                        let mut mono = 0.0f32;
                        for c in 0..ch {
                            mono += data[frame_idx * ch + c] as f32;
                        }
                        mono /= ch as f32;
                        mono *= gain;

                        // soft limiter
                        if mono > limiter {
                            mono = limiter;
                        } else if mono < -limiter {
                            mono = -limiter;
                        }

                        // linear resample: emit when phase crosses 1.0
                        // phase advances by 1/ratio per input sample (input_sr / 16000 = ratio)
                        rs_phase += 1.0 / ratio;
                        while rs_phase >= 1.0 {
                            let t = 1.0 - (rs_phase - 1.0);
                            let y = prev_mono + (mono - prev_mono) * t;
                            let v = (y * i16::MAX as f32) as i16;
                            asr_bytes.extend_from_slice(&v.to_le_bytes());
                            rs_phase -= 1.0;
                        }
                        prev_mono = mono;
                    }
                    if !asr_bytes.is_empty() {
                        w_asr.lock().unwrap().write_data(&asr_bytes);

                        // Send 16k mono ASR samples to mixer
                        if let Some(tx) = MIC_PCM_TX.lock().unwrap().as_ref() {
                            for ch in asr_bytes.chunks_exact(2) {
                                let v = i16::from_le_bytes([ch[0], ch[1]]);
                                let _ = tx.send(v);
                            }
                        }
                    }
                },
                err_fn,
                None,
            )?;
            Ok(stream)
        }
        cpal::SampleFormat::U16 => {
            let w = writer_raw.clone();
            let w_asr = writer_asr.clone();
            let mut rs_phase: f32 = 0.0;
            let ratio: f32 = sample_rate as f32 / 16_000.0;
            let mut prev_mono: f32 = 0.0;
            let stream = dev.build_input_stream(
                &stream_config,
                move |data: &[u16], _| {
                    if data.is_empty() {
                        return;
                    }

                    // Map u16 [0, 65535] -> float [-1, 1]
                    let mut sum = 0.0f32;
                    for &x in data {
                        let xf = (x as f32 / u16::MAX as f32) * 2.0 - 1.0;
                        sum += xf * xf;
                    }
                    let rms = (sum / (data.len() as f32)).sqrt();

                    let mut gain = (MIC_GAIN_Q8.load(Ordering::Relaxed) as f32) / 256.0;
                    if gain < base_gain {
                        gain = base_gain;
                    }
                    if rms > rms_floor {
                        let desired = (target_rms / rms).clamp(1.0, max_gain).max(base_gain);
                        gain =
                            (gain * smooth + desired * (1.0 - smooth)).clamp(base_gain, max_gain);
                        MIC_GAIN_Q8.store((gain * 256.0) as u32, Ordering::Relaxed);
                    }

                    let mut bytes = Vec::with_capacity(data.len() * 2);
                    for &x in data {
                        let mut y = ((x as f32 / u16::MAX as f32) * 2.0 - 1.0) * gain;
                        if y > limiter {
                            y = limiter;
                        } else if y < -limiter {
                            y = -limiter;
                        }
                        let v = (y * i16::MAX as f32) as i16;
                        bytes.extend_from_slice(&v.to_le_bytes());
                    }
                    w.lock().unwrap().write_data(&bytes);

                    // --- ASR track: downmix to mono, resample to 16k, then PCM16 ---
                    let ch = channels as usize;
                    let mut asr_bytes = Vec::new();
                    if ch == 0 {
                        return;
                    }
                    // Iterate frames
                    for frame_idx in 0..(data.len() / ch) {
                        let mut mono = 0.0f32;
                        for c in 0..ch {
                            mono += data[frame_idx * ch + c] as f32;
                        }
                        mono /= ch as f32;
                        mono *= gain;

                        // soft limiter
                        if mono > limiter {
                            mono = limiter;
                        } else if mono < -limiter {
                            mono = -limiter;
                        }

                        // linear resample: emit when phase crosses 1.0
                        // phase advances by 1/ratio per input sample (input_sr / 16000 = ratio)
                        rs_phase += 1.0 / ratio;
                        while rs_phase >= 1.0 {
                            let t = 1.0 - (rs_phase - 1.0);
                            let y = prev_mono + (mono - prev_mono) * t;
                            let v = (y * i16::MAX as f32) as i16;
                            asr_bytes.extend_from_slice(&v.to_le_bytes());
                            rs_phase -= 1.0;
                        }
                        prev_mono = mono;
                    }
                    if !asr_bytes.is_empty() {
                        w_asr.lock().unwrap().write_data(&asr_bytes);

                        // Send 16k mono ASR samples to mixer
                        if let Some(tx) = MIC_PCM_TX.lock().unwrap().as_ref() {
                            for ch in asr_bytes.chunks_exact(2) {
                                let v = i16::from_le_bytes([ch[0], ch[1]]);
                                let _ = tx.send(v);
                            }
                        }
                    }
                },
                err_fn,
                None,
            )?;
            Ok(stream)
        }
        other => anyhow::bail!("mic format {:?} not handled", other),
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
/// 计算 RMS（归一化到 0..1 的浮点）
fn rms_i16(samples: &[i16]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let mut sum = 0.0f64;
    for &v in samples {
        let x = v as f64 / i16::MAX as f64;
        sum += x * x;
    }
    ((sum / samples.len() as f64) as f32).sqrt()
}

/// 计算峰值（0..1）
fn peak_i16(samples: &[i16]) -> f32 {
    let mut peak = 0.0f32;
    for &v in samples {
        let x = (v as f32).abs() / i16::MAX as f32;
        if x > peak {
            peak = x;
        }
    }
    peak
}

/// 软限幅 limiter：
/// threshold: 0..1（建议 0.95）
/// knee: 0..1（建议 0.2）
///
/// 输入输出都是 -1..1
fn soft_limiter(x: f32, threshold: f32, knee: f32) -> f32 {
    let ax = x.abs();
    if ax <= threshold {
        return x;
    }

    // knee 区间：从 threshold 到 threshold + knee 渐进压缩
    let t = threshold;
    let k = knee.max(1e-6);
    let upper = (t + k).min(1.0);

    if ax >= upper {
        // 超过上限：硬夹紧
        return x.signum() * upper;
    }

    // 在 knee 内：smoothstep
    let u = (ax - t) / (upper - t); // 0..1
    let s = u * u * (3.0 - 2.0 * u); // smoothstep
    let y = t + (upper - t) * s;
    x.signum() * y
}

/// 自动 RMS 对齐混音：
/// - 读取 mic/system wav
/// - 转 mono
/// - 自动算 mic_gain，使 mic_rms ≈ sys_rms
/// - 加一点主观补偿，让 mic 稍微更突出
/// - system 可稍微降一点，避免压过人声
/// - 软限幅防爆音
///
/// 输出 mono PCM16 wav
fn mix_pcm16_wav(
    mic_path: &std::path::Path,
    sys_path: &std::path::Path,
    out_path: &std::path::Path,
) -> Result<()> {
    let mic = read_pcm16_wav(mic_path)?;
    let sys = read_pcm16_wav(sys_path)?;

    // 最小版本：采样率必须一致
    if mic.sample_rate != sys.sample_rate {
        anyhow::bail!(
            "sample rate mismatch: mic={} sys={}",
            mic.sample_rate,
            sys.sample_rate
        );
    }

    let mic_mono = to_mono(&mic);
    let sys_mono = to_mono(&sys);

    let mic_rms = rms_i16(&mic_mono);
    let sys_rms = rms_i16(&sys_mono);

    let target_mic_rms = 0.25;
    // 防止 mic 静音导致除 0
    let mut mic_gain = if mic_rms > 0.0001 {
        target_mic_rms / mic_rms
    } else {
        1.0
    };

    // 夹紧避免离谱（耳机 mic 很小会算出很大）
    mic_gain = mic_gain.clamp(1.0, 150.0);

    // 主观补偿：让 mic 稍微更突出一点（你也可以调 1.0~1.5）
    let mic_gain = mic_gain * 1.4;

    // system 略降一点（你也可以改成 1.0）
    let sys_gain = 0.75;

    // 打印调试信息（方便你确认自动 gain 是否合理）
    println!(
        "🔊 RMS: mic={:.4}, sys={:.4}, mic_gain={:.2}, sys_gain={:.2}",
        mic_rms, sys_rms, mic_gain, sys_gain
    );

    let max_len = mic_mono.len().max(sys_mono.len());
    let mut mixed: Vec<i16> = Vec::with_capacity(max_len);

    // limiter 参数
    let threshold = 0.95;
    let knee = 0.20;

    for i in 0..max_len {
        let m = if i < mic_mono.len() {
            mic_mono[i] as f32 / i16::MAX as f32
        } else {
            0.0
        };
        let s = if i < sys_mono.len() {
            sys_mono[i] as f32 / i16::MAX as f32
        } else {
            0.0
        };

        // 混音（浮点域 -1..1）
        let mut y = m * mic_gain + s * sys_gain;

        // 防爆音：软限幅
        y = soft_limiter(y, threshold, knee);

        // 转回 i16
        let out = (y * i16::MAX as f32).clamp(i16::MIN as f32, i16::MAX as f32) as i16;
        mixed.push(out);
    }

    // 输出前再打印 peak 信息
    let out_peak = peak_i16(&mixed);
    println!("🔊 mix peak = {:.3}", out_peak);

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

// fn mix_pcm16_wav(
//     mic_path: &std::path::Path,
//     sys_path: &std::path::Path,
//     out_path: &std::path::Path,
//     mic_gain: f32,
//     sys_gain: f32,
// ) -> Result<()> {
//     let mic = read_pcm16_wav(mic_path)?;
//     let sys = read_pcm16_wav(sys_path)?;

//     // 检查采样率一致（最小版本先不做 resample）
//     if mic.sample_rate != sys.sample_rate {
//         anyhow::bail!(
//             "sample rate mismatch: mic={} sys={}",
//             mic.sample_rate,
//             sys.sample_rate
//         );
//     }

//     let mic_mono = to_mono(&mic);
//     let sys_mono = to_mono(&sys);

//     let max_len = mic_mono.len().max(sys_mono.len());
//     let mut mixed: Vec<i16> = Vec::with_capacity(max_len);

//     for i in 0..max_len {
//         let m = if i < mic_mono.len() {
//             mic_mono[i] as f32
//         } else {
//             0.0
//         };
//         let s = if i < sys_mono.len() {
//             sys_mono[i] as f32
//         } else {
//             0.0
//         };

//         let y = m * mic_gain + s * sys_gain;
//         // 防止爆音
//         let y = y.clamp(i16::MIN as f32, i16::MAX as f32);
//         mixed.push(y as i16);
//     }

//     // 写出 WAV（复用你的 WavWriter）
//     let mut w = WavWriter::create(out_path)?;
//     w.init_pcm16(mic.sample_rate, 1)?;
//     let mut bytes = Vec::with_capacity(mixed.len() * 2);
//     for v in mixed {
//         bytes.extend_from_slice(&v.to_le_bytes());
//     }
//     w.write_data(&bytes);
//     w.finalize()?;

//     Ok(())
// }

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

    // 输出目录：你现在是 app_data_dir；如果你想放桌面，改这里
    let base: PathBuf = app.path().app_data_dir().map_err(|e| e.to_string())?;
    std::fs::create_dir_all(&base).map_err(|e| e.to_string())?;

    let system_path = base.join("system.wav");
    let mic_path = base.join("mic.wav");
    let mic_asr_path = base.join("mic_asr_16k_mono.wav");
    let mix_path = base.join("mix.wav");
    let mix_asr_path = base.join("mix_asr_16k_mono.wav");

    // 先创建 writer（写 wav header）
    let system_writer = Arc::new(Mutex::new(
        WavWriter::create(&system_path).map_err(|e| e.to_string())?,
    ));
    let mic_writer = Arc::new(Mutex::new(
        WavWriter::create(&mic_path).map_err(|e| e.to_string())?,
    ));
    let mic_asr_writer = Arc::new(Mutex::new(
        WavWriter::create(&mic_asr_path).map_err(|e| e.to_string())?,
    ));

    // stop flag
    let stop = Arc::new(AtomicBool::new(false));
    // realtime pcm channels
    let (mic_pcm_tx, mic_pcm_rx) = std::sync::mpsc::channel::<i16>();
    let (sys_pcm_tx, sys_pcm_rx) = std::sync::mpsc::channel::<i16>();
    let (mix_pcm_tx, mix_pcm_rx) = std::sync::mpsc::channel::<i16>();
    *MIC_PCM_TX.lock().unwrap() = Some(mic_pcm_tx.clone());
    *SYS_PCM_TX.lock().unwrap() = Some(sys_pcm_tx.clone());

    // mixer thread
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
                // If system stream is slower/empty, reuse last sample or zero?
                // reusing last sample is better than silence for tiny jitters,
                // but for ASR, silence (0) might be safer to avoid buzzing artifacts.
                // Let's degrade to 0 if queue is effectively empty to avoid stuck-tone.
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

    // asr worker thread (chunks -> /inference)
    let asr_app = app.clone();
    let stop_asr = stop.clone();
    std::thread::spawn(move || {
        let _ = realtime_inference_worker(asr_app, stop_asr, mix_pcm_rx);
    });
    let stop2 = stop.clone();

    let mic_path_t = mic_path.clone();
    let system_path_t = system_path.clone();
    let mix_path_t = mix_path.clone();
    let mix_asr_path_t = mix_asr_path.clone();

    // 开线程跑录音
    let join = std::thread::spawn(move || {
        // 1) MIC stream
        let mic_stream = match start_mic_stream(mic_writer.clone(), mic_asr_writer.clone()) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("❌ start_mic_stream failed: {e:?}");
                return;
            }
        };
        if let Err(e) = mic_stream.play() {
            eprintln!("❌ mic_stream.play failed: {e:?}");
            return;
        }

        // 2) System stream
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

        // 3) 在这个线程里跑一个 tokio current-thread runtime，不限时循环，直到 stop=true
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
            use tokio::time::{Duration, Instant};

            let mut buf: Vec<i16> = Vec::with_capacity(48000);
            let mut last_flush = Instant::now();

            // while !stop2.load(Ordering::Acquire) {
            //     match system_stream.next().await {
            //         Some(s) => {
            //             let s: f32 = s;
            //             let v = (s.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
            //             buf.push(v);
            // if let Some(tx)=SYS_PCM_TX.lock().unwrap().as_ref(){ let _=tx.send(v); }

            //             // 每 1 秒 flush 一次，避免内存一直涨
            //             if buf.len() >= 48000 || last_flush.elapsed() >= Duration::from_secs(1) {
            //                 flush_i16(system_writer2.clone(), &mut buf);
            //                 last_flush = Instant::now();
            //             }
            //         }
            //         None => break,
            //     }
            // }

            use tokio::time::timeout;

            // System Resample State
            let mut rs_phase: f32 = 0.0;
            // Assuming system audio is 48kHz (typical) or whatever `system_stream.sample_rate()` returned.
            // But here we need to know the INPUT sample rate to calc ratio.
            // `sr` variable holds it from line 887.
            let ratio = sr as f32 / 16_000.0;
            let mut prev_sample: f32 = 0.0;

            while !stop2.load(Ordering::Acquire) {
                // ✅ 最关键：不要无限等待 next()
                match timeout(Duration::from_millis(200), system_stream.next()).await {
                    Ok(Some(s)) => {
                        let s: f32 = s;
                        let v = (s.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
                        buf.push(v);

                        // --- Resample to 16kHz for ASR Mixer ---
                        // Linear interpolation
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
                            flush_i16(system_writer2.clone(), &mut buf);
                            last_flush = Instant::now();
                        }
                    }
                    Ok(None) => break, // stream ended
                    Err(_) => {
                        // timeout：没拿到数据，继续循环
                        // 这样每 200ms 都能检查 stop flag
                    }
                }
            }

            if !buf.is_empty() {
                flush_i16(system_writer2.clone(), &mut buf);
            }

            // 给 mic 回调一点时间写尾巴（可选）
            tokio::time::sleep(Duration::from_millis(50)).await;
        });

        // 4) drop streams => stop
        drop(mic_stream);

        // finalize wav（system/mic）
        let _ = system_writer.lock().unwrap().finalize();
        let _ = mic_writer.lock().unwrap().finalize();
        let _ = mic_asr_writer.lock().unwrap().finalize();

        // 5) 生成 mix（mic_gain 可调）
        if let Err(e) = mix_pcm16_wav(&mic_path_t, &system_path_t, &mix_path_t) {
            eprintln!("❌ mix failed: {e:?}");
        } else {
            println!("✅ mix.wav: {}", mix_path_t.display());

            // Convert mix.wav -> mix_asr_16k_mono.wav (Whisper.cpp-ready)
            match crate::convert_wav_to_16k_mono_pcm16(&mix_path_t, &mix_asr_path_t) {
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

    // 返回路径（让你知道文件位置）
    Ok((
        system_path.display().to_string(),
        mic_path.display().to_string(),
    ))
}

// #[tauri..:command]
// fn stop_recording() -> Result<(String, String, String), String> {
//     let mut guard = RECORDER.lock().unwrap();
//     let Some(handle) = guard.take() else {
//         return Err("recording is not running".into());
//     };

//     handle.stop.store(true, Ordering::Release);

//     // 等线程结束（录音 finalize + mix）
//     let _ = handle.join.join();

//     Ok((
//         handle.system_path.display().to_string(),
//         handle.mic_path.display().to_string(),
//         handle.mix_path.display().to_string(),
//     ))
// }

#[tauri::command]
async fn stop_recording() -> Result<(String, String, String), String> {
    let handle = {
        let mut guard = RECORDER.lock().unwrap();
        guard.take().ok_or("recording is not running")?
    };

    handle.stop.store(true, Ordering::Release);

    // ✅ 不要阻塞 IPC/UI，放到 blocking 线程池里 join
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

// fn start_demo_recording(app: tauri::AppHandle, seconds: u64) -> Result<(String, String), String> {
//     let (tx, rx) = std::sync::mpsc::channel();

//     std::thread::spawn(move || {
//         let res: Result<(String, String), String> = (|| {
//             let base: PathBuf = app.path().app_data_dir().map_err(|e| e.to_string())?;
//             std::fs::create_dir_all(&base).map_err(|e| e.to_string())?;

//             let system_path = base.join("system.wav");
//             let mic_path = base.join("mic.wav");

//             let system_writer = Arc::new(Mutex::new(
//                 WavWriter::create(&system_path).map_err(|e| e.to_string())?,
//             ));
//             let mic_writer = Arc::new(Mutex::new(
//                 WavWriter::create(&mic_path).map_err(|e| e.to_string())?,
//             ));

//             // 1) MIC stream (CPAL)
//             let mic_stream = start_mic_stream(mic_writer.clone()).map_err(|e| e.to_string())?;
//             mic_stream.play().map_err(|e| e.to_string())?;

//             // 2) System audio stream (CoreAudio Tap)
//             let mut system_stream = audio::capture::core_audio::CoreAudioCapture::new()
//                 .map_err(|e| format!("CoreAudioCapture::new failed: {e:?}"))?
//                 .stream()
//                 .map_err(|e| format!("CoreAudioCapture::stream failed: {e:?}"))?;

//             let sr = system_stream.sample_rate();
//             system_writer
//                 .lock()
//                 .unwrap()
//                 .init_pcm16(sr, 1)
//                 .map_err(|e| e.to_string())?;

//             // 3) Run async loop on a local tokio runtime (current-thread)
//             let rt = tokio::runtime::Builder::new_current_thread()
//                 .enable_time()
//                 .build()
//                 .map_err(|e| e.to_string())?;

//             let system_writer2 = system_writer.clone();

//             rt.block_on(async move {
//                 use tokio::time::{Duration, Instant};

//                 let deadline = Instant::now() + Duration::from_secs(seconds);
//                 let mut buf: Vec<i16> = Vec::with_capacity(48000);

//                 while Instant::now() < deadline {
//                     if let Some(s) = system_stream.next().await {
//                         // 强制类型，避免推断失败
//                         let s: f32 = s;

//                         let v = (s.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
//                         buf.push(v);

//                         if buf.len() >= 48000 {
//                             flush_i16(system_writer2.clone(), &mut buf);
//                         }
//                     } else {
//                         break;
//                     }
//                 }

//                 if !buf.is_empty() {
//                     flush_i16(system_writer2.clone(), &mut buf);
//                 }

//                 // 小睡一下让 CPAL 回调把最后一小段写进去（可选）
//                 tokio::time::sleep(Duration::from_millis(50)).await;
//             });

//             // 4) stop mic
//             drop(mic_stream);

//             // 5) finalize wav
//             system_writer
//                 .lock()
//                 .unwrap()
//                 .finalize()
//                 .map_err(|e| e.to_string())?;
//             mic_writer
//                 .lock()
//                 .unwrap()
//                 .finalize()
//                 .map_err(|e| e.to_string())?;

//             let mix_path = base.join("mix.wav");

//             // 经验值：mic 通常会偏小，所以 mic_gain 可以调大
//             // 你可以先用 mic_gain=3.0, sys_gain=1.0
//             mix_pcm16_wav(&mic_path, &system_path, &mix_path, 3.0, 1.0)
//                 .map_err(|e| format!("mix failed: {e:?}"))?;

//             println!("✅ mix.wav: {}", mix_path.display());

//             Ok((
//                 system_path.display().to_string(),
//                 mic_path.display().to_string(),
//             ))
//         })();

//         let _ = tx.send(res);
//     });

//     rx.recv().map_err(|e| e.to_string())?
// }

// =====================
// App entry
// =====================
//
fn set_last_record_base(p: std::path::PathBuf) {
    let m = LAST_RECORD_BASE.get_or_init(|| Mutex::new(None));
    *m.lock().unwrap() = Some(p);
}
fn get_last_record_base() -> Option<std::path::PathBuf> {
    let m = LAST_RECORD_BASE.get_or_init(|| Mutex::new(None));
    m.lock().unwrap().clone()
}

fn main() {
    tauri::Builder::default()
        .invoke_handler(tauri::generate_handler![start_recording, stop_recording])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}

fn resample_f32_to_16k(input: &[f32], in_rate: usize) -> Vec<f32> {
    if in_rate == 16_000 {
        return input.to_vec();
    }
    let ratio = in_rate as f64 / 16_000.0;
    let out_len = (input.len() as f64 / ratio).ceil() as usize;
    let mut out = Vec::with_capacity(out_len);
    for i in 0..out_len {
        let src_pos = i as f64 * ratio;
        let idx = src_pos.floor() as usize;
        let frac = (src_pos - idx as f64) as f32;
        let a = *input.get(idx).unwrap_or(&0.0);
        let b = *input.get(idx + 1).unwrap_or(&a);
        out.push(a + (b - a) * frac);
    }
    out
}

fn write_pcm16_wav_16k_mono(path: &std::path::Path, samples: &[i16]) -> std::io::Result<()> {
    use std::io::Write;
    let mut f = std::fs::File::create(path)?;
    let data_bytes = (samples.len() * 2) as u32;
    // RIFF header
    f.write_all(b"RIFF")?;
    f.write_all(&(36 + data_bytes).to_le_bytes())?;
    f.write_all(b"WAVE")?;
    // fmt chunk
    f.write_all(b"fmt ")?;
    f.write_all(&16u32.to_le_bytes())?; // PCM
    f.write_all(&1u16.to_le_bytes())?; // audio format PCM
    f.write_all(&1u16.to_le_bytes())?; // channels
    f.write_all(&16000u32.to_le_bytes())?; // sample rate
    f.write_all(&(16000u32 * 2).to_le_bytes())?; // byte rate
    f.write_all(&2u16.to_le_bytes())?; // block align
    f.write_all(&16u16.to_le_bytes())?; // bits
                                        // data chunk
    f.write_all(b"data")?;
    f.write_all(&data_bytes.to_le_bytes())?;
    for &s in samples {
        f.write_all(&s.to_le_bytes())?;
    }
    Ok(())
}

fn convert_wav_to_16k_mono_pcm16(
    input_path: &std::path::Path,
    output_path: &std::path::Path,
) -> Result<(), String> {
    // Read WAV PCM16/32float and resample/downmix to 16k mono PCM16.
    let data = std::fs::read(input_path).map_err(|e| e.to_string())?;
    if data.len() < 44 {
        return Err("wav too small".into());
    }
    // very small wav parser (PCM/IEEE float)
    let mut pos = 12;
    let mut audio_fmt = 1u16;
    let mut channels = 1u16;
    let mut sample_rate = 16000u32;
    let mut bits_per_sample = 16u16;
    let mut data_chunk: &[u8] = &[];
    while pos + 8 <= data.len() {
        let id = &data[pos..pos + 4];
        let sz = u32::from_le_bytes(data[pos + 4..pos + 8].try_into().unwrap()) as usize;
        pos += 8;
        if pos + sz > data.len() {
            break;
        }
        if id == b"fmt " && sz >= 16 {
            audio_fmt = u16::from_le_bytes(data[pos..pos + 2].try_into().unwrap());
            channels = u16::from_le_bytes(data[pos + 2..pos + 4].try_into().unwrap());
            sample_rate = u32::from_le_bytes(data[pos + 4..pos + 8].try_into().unwrap());
            bits_per_sample = u16::from_le_bytes(data[pos + 14..pos + 16].try_into().unwrap());
        } else if id == b"data" {
            data_chunk = &data[pos..pos + sz];
        }
        pos += sz + (sz % 2);
    }
    if data_chunk.is_empty() {
        return Err("no data chunk".into());
    }
    // decode to f32 mono
    let mut mono: Vec<f32> = Vec::new();
    if audio_fmt == 1 && bits_per_sample == 16 {
        let frame_bytes = (channels as usize) * 2;
        for frame in data_chunk.chunks_exact(frame_bytes) {
            let mut acc = 0f32;
            for c in 0..channels as usize {
                let s = i16::from_le_bytes(frame[c * 2..c * 2 + 2].try_into().unwrap()) as f32
                    / 32768.0;
                acc += s;
            }
            mono.push(acc / channels as f32);
        }
    } else if audio_fmt == 3 && bits_per_sample == 32 {
        let frame_bytes = (channels as usize) * 4;
        for frame in data_chunk.chunks_exact(frame_bytes) {
            let mut acc = 0f32;
            for c in 0..channels as usize {
                let s = f32::from_le_bytes(frame[c * 4..c * 4 + 4].try_into().unwrap());
                acc += s;
            }
            mono.push(acc / channels as f32);
        }
    } else {
        return Err(format!(
            "unsupported wav fmt={} bps={}",
            audio_fmt, bits_per_sample
        ));
    }
    // resample to 16k
    let out = resample_f32_to_16k(&mono, sample_rate as usize);
    // convert to i16
    let mut pcm: Vec<i16> = Vec::with_capacity(out.len());
    for &x in &out {
        let y = (x.max(-1.0).min(1.0) * 32767.0) as i16;
        pcm.push(y);
    }
    write_pcm16_wav_16k_mono(output_path, &pcm).map_err(|e| e.to_string())?;
    Ok(())
}

fn realtime_inference_worker(
    app: tauri::AppHandle,
    stop: std::sync::Arc<std::sync::atomic::AtomicBool>,
    mut rx: std::sync::mpsc::Receiver<i16>,
) -> Result<(), String> {
    use std::sync::atomic::Ordering;
    use std::time::Duration;
    let client = reqwest::blocking::Client::new();
    let server_url = "http://127.0.0.1:8178/inference";

    // Strict Dynamic VAD Parameters
    // We use WebRTC VAD (very aggressive) instead of simple RMS
    let mut vad =
        Vad::new_with_rate_and_mode(webrtc_vad::SampleRate::Rate16kHz, VadMode::VeryAggressive);
    let vad_frame_size = 320; // 20ms @ 16kHz
    let mut vad_accum: Vec<i16> = Vec::with_capacity(vad_frame_size);

    // Debounce: require consecutive speech frames to trigger
    let min_speech_frames = 2; // ~40ms

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
                                  // frame_accum removed, using vad_accum

    while !stop.load(Ordering::Relaxed) {
        match rx.recv_timeout(Duration::from_millis(100)) {
            Ok(s) => {
                buf.push(s);
                vad_accum.push(s);

                // Process VAD every 20ms
                if vad_accum.len() >= vad_frame_size {
                    let is_voice = vad.is_voice_segment(&vad_accum).unwrap_or(false);
                    vad_accum.clear();

                    if is_voice {
                        speech_run_count += 1;
                    } else {
                        speech_run_count = 0;
                    }

                    // Trigger "speaking" state only after stability
                    if speech_run_count >= min_speech_frames {
                        if !is_speaking {
                            is_speaking = true;
                            println!("🎤 VAD: Speech START");
                        }
                        silence_frames = 0;
                    } else {
                        // If we are already speaking, this counts as silence frame
                        if is_speaking {
                            silence_frames += 1;
                            // Debug log every 10 frames of silence (~200ms) to track why it's not cutting
                            if silence_frames % 10 == 0 {
                                println!("... VAD: Silence frame {}", silence_frames);
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
                    if is_speaking && silence_frames >= current_threshold {
                        if buf.len() > 1000 {
                            // 1000 samples ~ 60ms
                            send_audio_to_asr(&app, &client, server_url, &buf);
                        }
                        buf.clear();
                        is_speaking = false;
                        silence_frames = 0;
                        speech_run_count = 0;
                    }
                }

                // 2. Max Length (Force send at 5s)
                if buf.len() >= max_len_samples {
                    if is_speaking {
                        send_audio_to_asr(&app, &client, server_url, &buf);
                    }
                    buf.clear();
                    is_speaking = false;
                    silence_frames = 0;
                    speech_run_count = 0;
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
                }
            }
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {}
            Err(_) => break,
        }
    }

    // Final flush
    if !buf.is_empty() && is_speaking {
        send_audio_to_asr(&app, &client, server_url, &buf);
    }

    Ok(())
}

fn send_audio_to_asr(
    app: &tauri::AppHandle,
    client: &reqwest::blocking::Client,
    url: &str,
    samples: &[i16],
) {
    // Add 300ms silence padding to end (Post-roll) to help ASR complete the last word
    let mut padded = Vec::with_capacity(samples.len() + 4800);
    padded.extend_from_slice(samples);
    padded.resize(padded.len() + 4800, 0);

    if let Ok(tmp) = tempfile::NamedTempFile::new() {
        if let Ok(_) = write_pcm16_wav_16k_mono(tmp.path(), &padded) {
            let form = reqwest::blocking::multipart::Form::new()
                .text("response_format", "verbose_json")
                .text("language", "zh")
                .part(
                    "file",
                    reqwest::blocking::multipart::Part::file(tmp.path()).unwrap(), // safe unwrap for temp file we just wrote
                );

            if let Ok(resp) = client.post(url).multipart(form).send() {
                if let Ok(txt) = resp.text() {
                    // Parse verbose_json response
                    // Expected structure: { "text": "...", "segments": [ { "avg_logprob": -0.5, ... }, ... ] }
                    if let Ok(v) = serde_json::from_str::<serde_json::Value>(&txt) {
                        // Check logprobs first
                        let mut is_reliable = true;
                        if let Some(segments) = v.get("segments").and_then(|arr| arr.as_array()) {
                            for seg in segments {
                                if let Some(lp) = seg.get("avg_logprob").and_then(|f| f.as_f64()) {
                                    if lp < -1.0 {
                                        eprintln!("⚠️ ASR Low Confidence: logprob={:.3} < -1.0, text={:?}", lp, seg.get("text"));
                                        is_reliable = false;
                                        break;
                                    }
                                }
                            }
                        } else {
                            // If no segments found, we can't verify logprob.
                            // Depending on strategy, either trust or warn.
                            // For now, let's warn but proceed if text exists,
                            // OR assume it might be a simple json fallback (unlikely if we asked for verbose_json).
                            eprintln!("⚠️ ASR response missing segments for logprob check");
                        }

                        if is_reliable {
                            if let Some(t) = v.get("text").and_then(|x| x.as_str()) {
                                let t = t.trim();
                                if !t.is_empty() {
                                    let _ = app.emit("asr_final", t.to_string());
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
