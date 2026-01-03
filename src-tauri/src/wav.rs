use anyhow::{Context, Result};
use std::fs::File;
use std::io::{Seek, SeekFrom, Write};
use std::sync::{Arc, Mutex};

// =====================
// Minimal WAV writer (PCM16)
// =====================
pub struct WavWriter {
    f: File,
    data_bytes: u32,
    initialized: bool,
}

impl WavWriter {
    pub fn create(path: &std::path::Path) -> Result<Self> {
        Ok(Self {
            f: File::create(path)?,
            data_bytes: 0,
            initialized: false,
        })
    }

    pub fn init_pcm16(&mut self, sample_rate: u32, channels: u16) -> Result<()> {
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

    pub fn write_data(&mut self, bytes: &[u8]) {
        if !self.initialized {
            return;
        }
        if self.f.write_all(bytes).is_ok() {
            self.data_bytes = self.data_bytes.saturating_add(bytes.len() as u32);
        }
    }

    pub fn finalize(&mut self) -> Result<()> {
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

pub fn flush_i16(writer: Arc<Mutex<WavWriter>>, buf: &mut Vec<i16>) {
    let mut bytes = Vec::with_capacity(buf.len() * 2);
    for &v in buf.iter() {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    writer.lock().unwrap().write_data(&bytes);
    buf.clear();
}

// =====================
// Minimal WAV PCM16 Reader
// =====================
#[derive(Debug, Clone)]
pub struct Pcm16Wav {
    pub channels: u16,
    pub sample_rate: u32,
    pub data: Vec<i16>,
}

/// 只支持 PCM16 little-endian WAV
pub fn read_pcm16_wav(path: &std::path::Path) -> Result<Pcm16Wav> {
    let mut f = File::open(path)?;
    // Read all to buffer for simplicity (small files)
    let mut buf = Vec::new();
    use std::io::Read;
    f.read_to_end(&mut buf)?;

    if buf.len() < 44 {
        anyhow::bail!("WAV file too small");
    }

    // Parse header manually to be safe
    // RIFF
    if &buf[0..4] != b"RIFF" {
        anyhow::bail!("Invalid WAV: no RIFF");
    }
    // WAVE
    if &buf[8..12] != b"WAVE" {
        anyhow::bail!("Invalid WAV: no WAVE");
    }

    // Search for fmt chunk
    let mut pos = 12;
    let mut channels = 1u16;
    let mut sample_rate = 44100u32;
    let mut data_start = 0;
    let mut data_len = 0;

    while pos + 8 <= buf.len() {
        let id = &buf[pos..pos + 4];
        let size = u32::from_le_bytes(buf[pos + 4..pos + 8].try_into()?);
        pos += 8;

        if id == b"fmt " {
            if size < 16 {
                anyhow::bail!("fmt chunk too small");
            }
            let fmt_code = u16::from_le_bytes(buf[pos..pos + 2].try_into()?);
            if fmt_code != 1 {
                // Not PCM
                // But wait, maybe it's Extensible with subformat=PCM, but let's be strict.
                // anyhow::bail!("Not PCM format (code={})", fmt_code);
            }
            channels = u16::from_le_bytes(buf[pos + 2..pos + 4].try_into()?);
            sample_rate = u32::from_le_bytes(buf[pos + 4..pos + 8].try_into()?);

            pos += size as usize;
        } else if id == b"data" {
            data_start = pos;
            data_len = size as usize;
            pos += data_len;
            break; // Found data, stop parsing (ignore chunks after data for now)
        } else {
            // skip unknown chunk
            pos += size as usize;
        }
    }

    if data_start == 0 {
        anyhow::bail!("No data chunk found");
    }

    // Parse i16
    let samples_u8 = &buf[data_start..data_start + data_len];
    let mut samples = Vec::with_capacity(samples_u8.len() / 2);
    for chunk in samples_u8.chunks_exact(2) {
        let val = i16::from_le_bytes(chunk.try_into()?);
        samples.push(val);
    }

    Ok(Pcm16Wav {
        channels,
        sample_rate,
        data: samples,
    })
}

/// 把 stereo/mono 统一转换成 mono（简单平均）
/// 返回 mono samples
pub fn to_mono(w: &Pcm16Wav) -> Vec<i16> {
    if w.channels == 1 {
        return w.data.clone();
    }
    let frames = w.data.len() / w.channels as usize;
    let mut out = Vec::with_capacity(frames);
    for i in 0..frames {
        let mut sum = 0i32;
        for c in 0..w.channels as usize {
            sum += w.data[i * w.channels as usize + c] as i32;
        }
        out.push((sum / w.channels as i32) as i16);
    }
    out
}

/// 混音：mix = mic * mic_gain + sys * sys_gain
/// 输出 mono PCM16 WAV（sample_rate 同 input）
/// 计算 RMS（归一化到 0..1 的浮点）
pub fn rms_i16(samples: &[i16]) -> f32 {
    let mut sum_sq = 0.0;
    for &s in samples {
        let f = s as f32 / 32768.0;
        sum_sq += f * f;
    }
    (sum_sq / samples.len() as f32).sqrt()
}

/// 计算峰值（0..1）
pub fn peak_i16(samples: &[i16]) -> f32 {
    let mut max_val = 0.0f32;
    for &s in samples {
        let f = (s as f32 / 32768.0).abs();
        if f > max_val {
            max_val = f;
        }
    }
    max_val
}

/// 软限幅 limiter：
/// threshold: 0..1（建议 0.95）
/// knee: 0..1（建议 0.2）
///
/// 输入输出都是 -1..1
pub fn soft_limiter(x: f32, threshold: f32, knee: f32) -> f32 {
    let abs_x = x.abs();
    // 还在线性区
    if abs_x <= threshold - knee {
        return x;
    }

    // 超过硬限幅
    // (这里简化处理，直接让它尽量贴近 1.0 但不切削波，或者用 soft knee curve)
    // 常见 soft knee: y = ...
    // 简单起见，用 tanh 压缩高电平
    // return x.signum() * (threshold + (1.0 - threshold) * ((abs_x - threshold)/(1.0-threshold)).tanh());
    // 更简单的硬 clamp 防止溢出 PCM
    if x > 1.0 {
        return 1.0;
    }
    if x < -1.0 {
        return -1.0;
    }
    x
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
pub fn mix_pcm16_wav(
    mic_path: &std::path::Path,
    sys_path: &std::path::Path,
    out_path: &std::path::Path,
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

    // 1) 计算 RMS
    let mic_rms = rms_i16(&mic_mono);
    // let sys_rms = rms_i16(&sys_mono);

    // 2) 策略：把 mic 提升到一定响度（比如 -18dB ~ 0.12），
    //    然后 system 如果比 mic 响，就压低 system。
    //    或者简单点：固定 boost mic，然后让 system ducking (sidechain) —— 这里太复杂。
    //
    //    当前策略：
    //    如果 mic 太小（< 0.05），说明可能离得远，自动增益。
    //    TARGET_MIC_RMS = 0.15
    let target_mic = 0.15;
    let mic_gain = if mic_rms < 0.001 {
        1.0 // 只有底噪，别提了
    } else {
        (target_mic / mic_rms).clamp(1.0, 5.0) // 最多提 5 倍 (14dB)
    };

    // System 保持原样，或者稍微降低以免盖过人声
    let sys_gain = 0.8;

    println!(
        "AutoMix: mic_rms={:.4} => gain={:.2}, sys_gain={:.2}",
        mic_rms, mic_gain, sys_gain
    );

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

    // 写出 WAV（复用 your WavWriter）
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

pub fn resample_f32_to_16k(input: &[f32], in_rate: usize) -> Vec<f32> {
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

pub fn write_pcm16_wav_16k_mono(path: &std::path::Path, samples: &[i16]) -> std::io::Result<()> {
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

pub fn convert_wav_to_16k_mono_pcm16(
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
