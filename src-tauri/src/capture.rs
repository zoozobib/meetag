use crate::vad::SendVad;
use crate::wav::WavWriter;
use anyhow::{Context, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use std::sync::{Arc, Mutex};
use webrtc_vad::{Vad, VadMode};

// =====================
// Helper: Find best input device (prioritize external)
// =====================
fn find_best_input_device(host: &cpal::Host) -> Result<cpal::Device> {
    let mut devices = host
        .input_devices()
        .context("failed to list input devices")?;
    let mut best_device: Option<cpal::Device> = None;
    let mut best_score = -100;

    // Heuristic Scoring:
    // +10: Explicitly "External", "USB", "Headset", "AirPods", "Rode", "Blue", "Focusrite"
    //   0: Built-in / Internal (Fallback)
    // -50: "Virtual", "BlackHole", "Zoom", "Teams", "Aggregate", "Multi-Output" (Avoid)

    for device in devices {
        let name = device.name().unwrap_or_else(|_| "unknown".to_string());
        println!("🎤 Candidate input device: {}", name);

        let mut score = 0;
        let lower = name.to_lowercase();

        // Bonus for likely good external mics
        if lower.contains("usb")
            || lower.contains("headset")
            || lower.contains("airpods")
            || lower.contains("external")
            || lower.contains("rode")
            || lower.contains("blue")
            || lower.contains("focusrite")
        {
            score += 10;
        }

        // Penalty for virtual/aggregate devices that are likely silent or playback-only
        if lower.contains("virtual")
            || lower.contains("blackhole")
            || lower.contains("zoom")
            || lower.contains("teams")
            || lower.contains("aggregate")
            || lower.contains("multi-output")
            || lower.contains("lark")
        {
            score -= 50;
        }

        // Small penalty for built-in to deprioritize it vs "Headset" if both exist having no other keywords
        if lower.contains("built-in") || lower.contains("internal") || lower.contains("macbook") {
            score -= 1;
        }

        if score > best_score {
            best_score = score;
            best_device = Some(device);
        }
    }

    if let Some(d) = best_device {
        println!(
            "🎤 Selected best input device: {} (Score: {})",
            d.name().unwrap_or_default(),
            best_score
        );
        Ok(d)
    } else {
        println!("🎤 No suitable devices found based on heuristic, trying system default");
        host.default_input_device()
            .context("no default input device")
    }
}

// =====================
// MIC capture (CPAL) -> mic.wav
// =====================
pub fn start_mic_stream(
    writer_raw: Arc<Mutex<WavWriter>>,
    writer_asr: Arc<Mutex<WavWriter>>,
    mic_tx: std::sync::mpsc::Sender<i16>,
) -> Result<cpal::Stream> {
    let host = cpal::default_host();

    // Use heuristic to pick device (Smart Selection)
    // Fixed: Now ignores "Lark" and other virtual devices correctly.
    let dev = find_best_input_device(&host)?;

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
    // Initial gain: Start LOW (3.0) instead of HIGH (50.0) to avoid initial noise blast
    static MIC_GAIN_Q8: AtomicU32 = AtomicU32::new((3.0_f32 * 256.0_f32) as u32);

    // Tunables (safe defaults)
    // Base gain keeps your original loudness in non-call scenarios.
    // AGC will only BOOST above this when the system/WeChat suppresses the mic.
    // Reduced from 15.0 to 3.0 to prevent amplifying noise floor (which causes hallucinations)
    let base_gain: f32 = 3.0;
    let target_rms: f32 = 0.15; // desired loudness (0..1) *when boosting* ~ -16dBFS
                                // Note: we do NOT attenuate below base_gain in this strategy.
    let max_gain: f32 = 50.0; // Reduced from 400.0. 50x is plenty (34dB).
    let smooth: f32 = 0.90; // 0.0..1.0, higher = smoother/slower gain changes
    let decay: f32 = 0.99; // Slow release for gate
    let noise_gate: f32 = 0.03; // Input RMS below this is considered noise: don't boost! (Raised from 0.01)
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
            let tx = mic_tx.clone();

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
                        for ch in asr_bytes.chunks_exact(2) {
                            let v = i16::from_le_bytes([ch[0], ch[1]]);
                            let _ = tx.send(v);
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
            let tx = mic_tx.clone();

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

                    // Note: I16 branch currently uses the old "noise_gate" logic in snippet.
                    // But for consistency we should probably use the same logic as F32?
                    // The snippet I viewed had: if rms > noise_gate { ... }
                    // I will stick to what I saw in snippet 2.
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
                        for ch in asr_bytes.chunks_exact(2) {
                            let v = i16::from_le_bytes([ch[0], ch[1]]);
                            let _ = tx.send(v);
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
            let tx = mic_tx.clone();

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
                        for ch in asr_bytes.chunks_exact(2) {
                            let v = i16::from_le_bytes([ch[0], ch[1]]);
                            let _ = tx.send(v);
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
