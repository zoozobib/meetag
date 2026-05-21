use crate::wav::WavWriter;
use anyhow::{Context, Result};
use cpal::traits::{DeviceTrait, HostTrait};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

// =====================
// Helper: Find best input device (prioritize external)
// =====================
fn find_best_input_device(host: &cpal::Host) -> Result<cpal::Device> {
    let devices = host
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
            || lower.contains("meetily")
            || lower.contains("audio-tap")
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
    writer_mic_raw: Arc<Mutex<WavWriter>>,
    writer_asr: Arc<Mutex<WavWriter>>,
    mixer_tx: std::sync::mpsc::Sender<i16>,
    asr_tx: std::sync::mpsc::Sender<i16>,
    system_speaking: Arc<AtomicBool>,
) -> Result<cpal::Stream> {
    let host = cpal::default_host();

    // Use heuristic to pick device (Smart Selection)
    let dev = find_best_input_device(&host)?;

    let cfg = dev
        .default_input_config()
        .context("no default input config")?;

    let sample_rate = cfg.sample_rate().0;
    let channels = cfg.channels() as u16;

    println!(
        "🎤 [CAPTURE] Starting mic stream: Rate={}, Channels={}",
        sample_rate, channels
    );

    writer_raw
        .lock()
        .unwrap()
        .init_pcm16(sample_rate, channels)?;
    // Raw mic track: original signal before DAGC (same format as mic.wav)
    writer_mic_raw
        .lock()
        .unwrap()
        .init_pcm16(sample_rate, channels)?;
    // ASR-ready track: 16kHz mono PCM16
    writer_asr.lock().unwrap().init_pcm16(16_000, 1)?;

    // --- DAGC Initialization ---
    // Target RMS: 0.15 (~ -16dB)
    // We pass target_rms^2 because dagc expects energy/variance reference
    let target_rms = 0.15f32;
    let agc_target_energy = target_rms * target_rms;
    // Distortion factor: 0.001 (Slow/Smooth adaptation) to prevent pumping
    let agc_distortion = 0.001;

    // We create a separate AGC instance for each channel (or mix then AGC? No, usually per channel or mono)
    // For simplicity and ASR focus, we apply AGC to the *input* channels before downmix,
    // OR apply to the downmixed mono signal?
    // Applying to mono is more efficient for ASR.
    // BUT we also save `writer_raw` (multichannel).
    // User wants "optimization". If we record raw as processed, we should process all channels.
    // Let's create one AGC per channel max (e.g. up to 2).
    // Actually, handling multi-channel AGC synchronization is complex (stereo image shift).
    // Safe bet: Apply AGC *independently* to channels (ok for voice) or Link them?
    // Given most mics are mono or dual-mono:
    // Let's instantiate a vector of AGCs.
    // NOTE: dagc::MonoAgc is not Clone.

    // Gain Logging
    use std::sync::atomic::{AtomicUsize, Ordering};
    let log_counter = Arc::new(AtomicUsize::new(0));

    let stream_config: cpal::StreamConfig = cfg.clone().into();
    let err_fn = |err| eprintln!("mic stream error: {err}");

    match cfg.sample_format() {
        cpal::SampleFormat::F32 => {
            let w = writer_raw.clone();
            let w_raw = writer_mic_raw.clone();
            let w_asr = writer_asr.clone();
            let mut rs_phase: f32 = 0.0;
            let ratio: f32 = sample_rate as f32 / 16_000.0;
            let mut prev_mono: f32 = 0.0;
            let tx_mix = mixer_tx.clone();
            let tx_asr = asr_tx.clone();

            let sys_speak_f32 = system_speaking.clone();
            let log_cnt = log_counter.clone();

            // Initialize AGCs (one per channel)
            let mut agcs: Vec<dagc::MonoAgc> = (0..channels)
                .map(|_| dagc::MonoAgc::new(agc_target_energy, agc_distortion).unwrap())
                .collect();

            // Input Gate Threshold (RMS)
            // below this, we freeze AGC (don't boost silence)
            let gate_threshold_rms = 0.01;

            let stream = dev.build_input_stream(
                &stream_config,
                move |data: &[f32], _| {
                    if data.is_empty() {
                        return;
                    }

                    // 1. Process with DAGC
                    // We need a mutable buffer.
                    // Optimization: We can write directly to a reusable buffer if we had one,
                    // but allocating a vector is safe for correctness.
                    let mut processed = data.to_vec();
                    let ch = channels as usize;

                    // Simple Input Gate: Calculate RMS of the block
                    // If block is silent, freeze ALL agcs.
                    let mut sum_sq = 0.0;
                    for &x in data {
                        sum_sq += x * x;
                    }
                    let block_rms = (sum_sq / data.len() as f32).sqrt();
                    let freeze = block_rms < gate_threshold_rms;

                    // Apply AGC
                    for (_i, agc) in agcs.iter_mut().enumerate() {
                        agc.freeze_gain(freeze);

                        // Extract channel stride
                        // processed structure: [L, R, L, R...]
                        // We can't iterate easily with stride in simple loop for `process`
                        // because `process` takes `&mut [f32]`.
                        // dagc 0.1.0 `process` takes `&mut [f32]`.
                        // It iterates efficiently.
                        // We must gather channel data, process, put back?
                        // Or process sample by sample? dagc `process` loop: `for x in samples { ... }`
                        // Calling `process` on a 1-element slice is fine!
                        // It might be slightly less efficient due to function call overhead, but negligible for 10ms audio.
                    }

                    // Interleaved processing
                    for frame in processed.chunks_mut(ch) {
                        for (i, sample) in frame.iter_mut().enumerate() {
                            if i < agcs.len() {
                                let mut s_slice = [*sample];
                                agcs[i].process(&mut s_slice);
                                *sample = s_slice[0];
                            }
                        }
                    }

                    // Logging (throttle)
                    if log_cnt.fetch_add(1, Ordering::Relaxed) % 100 == 0 {
                        let g = agcs[0].gain();
                        println!(
                            "🎤 [DAGC] RMS={:.4} Gain={:.2} Frozen={}",
                            block_rms, g, freeze
                        );
                    }

                    // 2. Write DAGC-processed to mic.wav
                    let mut bytes = Vec::with_capacity(processed.len() * 2);
                    for &x in &processed {
                        let v = (x.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
                        bytes.extend_from_slice(&v.to_le_bytes());
                    }
                    w.lock().unwrap().write_data(&bytes);

                    // 2b. Write raw pre-DAGC to mic_raw.wav
                    let mut raw_bytes = Vec::with_capacity(data.len() * 2);
                    for &x in data {
                        let v = (x.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
                        raw_bytes.extend_from_slice(&v.to_le_bytes());
                    }
                    w_raw.lock().unwrap().write_data(&raw_bytes);

                    // 3. ASR Path (Downmix -> Resample -> Send)
                    // When system audio is playing, send RAW (pre-DAGC) audio to ASR.
                    // This way the Silero VAD sees echo at its natural ~0.01 RMS and won't trigger.
                    // When system is silent, send DAGC audio to help capture quiet in-room speakers.
                    let use_raw_for_asr = sys_speak_f32.load(Ordering::Relaxed);

                    let mut asr_bytes = Vec::new();
                    let mut raw_idx = 0usize; // tracks position in original `data`

                    // Iterate frames from DAGC-processed audio
                    for frame in processed.chunks(ch) {
                        // DAGC mono
                        let mut mono_dagc = 0.0f32;
                        for &s in frame {
                            mono_dagc += s;
                        }
                        mono_dagc /= ch as f32;

                        // Raw mono (from original `data`, same frame layout)
                        let mut mono_raw = 0.0f32;
                        let end = (raw_idx + ch).min(data.len());
                        for i in raw_idx..end {
                            mono_raw += data[i];
                        }
                        mono_raw /= ch as f32;
                        raw_idx += ch;

                        // Choose source based on system audio state
                        let mono = if use_raw_for_asr { mono_raw } else { mono_dagc };

                        // Linear resample to 16kHz
                        rs_phase += 1.0 / ratio;
                        while rs_phase >= 1.0 {
                            let t = 1.0 - (rs_phase - 1.0);
                            let y = prev_mono + (mono - prev_mono) * t;
                            let v = (y.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
                            asr_bytes.extend_from_slice(&v.to_le_bytes());
                            rs_phase -= 1.0;
                        }
                        prev_mono = mono;
                    }

                    if !asr_bytes.is_empty() {
                        w_asr.lock().unwrap().write_data(&asr_bytes);

                        // Send 16k mono ASR samples to mixer and ASR worker
                        for ch_bytes in asr_bytes.chunks_exact(2) {
                            let v = i16::from_le_bytes([ch_bytes[0], ch_bytes[1]]);
                            let _ = tx_mix.send(v);
                            let _ = tx_asr.send(v);
                        }
                    }
                },
                err_fn,
                None,
            )?;
            Ok(stream)
        }
        cpal::SampleFormat::I16 => {
            let w = writer_raw.clone();
            let w_raw = writer_mic_raw.clone();
            let w_asr = writer_asr.clone();
            let mut rs_phase: f32 = 0.0;
            let ratio: f32 = sample_rate as f32 / 16_000.0;
            let mut prev_mono: f32 = 0.0;
            let tx_mix = mixer_tx.clone();
            let tx_asr = asr_tx.clone();

            let log_cnt = log_counter.clone();
            let sys_speak = system_speaking.clone();

            // Initialize AGCs
            let mut agcs: Vec<dagc::MonoAgc> = (0..channels)
                .map(|_| dagc::MonoAgc::new(agc_target_energy, agc_distortion).unwrap())
                .collect();
            let gate_threshold_rms = 0.01;

            let stream = dev.build_input_stream(
                &stream_config,
                move |data: &[i16], _| {
                    if data.is_empty() {
                        return;
                    }

                    // Convert i16 -> f32 for processing
                    let mut processed_f32: Vec<f32> = Vec::with_capacity(data.len());
                    let mut sum_sq = 0.0;
                    for &x in data {
                        let f = x as f32 / i16::MAX as f32;
                        processed_f32.push(f);
                        sum_sq += f * f;
                    }

                    let block_rms = (sum_sq / data.len() as f32).sqrt();
                    let freeze = block_rms < gate_threshold_rms;

                    // Apply AGC
                    let ch = channels as usize;
                    for (_i, agc) in agcs.iter_mut().enumerate() {
                        agc.freeze_gain(freeze);
                    }

                    for frame in processed_f32.chunks_mut(ch) {
                        for (i, sample) in frame.iter_mut().enumerate() {
                            if i < agcs.len() {
                                let mut s_slice = [*sample];
                                agcs[i].process(&mut s_slice);
                                *sample = s_slice[0];
                            }
                        }
                    }

                    if log_cnt.fetch_add(1, Ordering::Relaxed) % 100 == 0 {
                        let g = agcs[0].gain();
                        println!(
                            "🎤 [DAGC-I16] RMS={:.4} Gain={:.2} Frozen={}",
                            block_rms, g, freeze
                        );
                    }

                    // Write DAGC-processed to mic.wav
                    let mut bytes = Vec::with_capacity(data.len() * 2);
                    for &x in &processed_f32 {
                        let v = (x.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
                        bytes.extend_from_slice(&v.to_le_bytes());
                    }
                    w.lock().unwrap().write_data(&bytes);

                    // Write raw pre-DAGC to mic_raw.wav (original i16 data)
                    let mut raw_bytes = Vec::with_capacity(data.len() * 2);
                    for &x in data {
                        raw_bytes.extend_from_slice(&x.to_le_bytes());
                    }
                    w_raw.lock().unwrap().write_data(&raw_bytes);

                    // ASR Path: raw when system active, DAGC when silent
                    let use_raw_for_asr = sys_speak.load(Ordering::Relaxed);
                    let mut asr_bytes = Vec::new();
                    let mut raw_idx = 0usize;

                    for frame in processed_f32.chunks(ch) {
                        let mut mono_dagc = 0.0f32;
                        for &s in frame {
                            mono_dagc += s;
                        }
                        mono_dagc /= ch as f32;

                        // Raw mono from original i16 data
                        let mut mono_raw = 0.0f32;
                        let end = (raw_idx + ch).min(data.len());
                        for i in raw_idx..end {
                            mono_raw += data[i] as f32 / i16::MAX as f32;
                        }
                        mono_raw /= ch as f32;
                        raw_idx += ch;

                        let mono = if use_raw_for_asr { mono_raw } else { mono_dagc };

                        rs_phase += 1.0 / ratio;
                        while rs_phase >= 1.0 {
                            let t = 1.0 - (rs_phase - 1.0);
                            let y = prev_mono + (mono - prev_mono) * t;
                            let v = (y.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
                            asr_bytes.extend_from_slice(&v.to_le_bytes());
                            rs_phase -= 1.0;
                        }
                        prev_mono = mono;
                    }

                    if !asr_bytes.is_empty() {
                        w_asr.lock().unwrap().write_data(&asr_bytes);
                        for ch_bytes in asr_bytes.chunks_exact(2) {
                            let v = i16::from_le_bytes([ch_bytes[0], ch_bytes[1]]);
                            let _ = tx_mix.send(v);
                            let _ = tx_asr.send(v);
                        }
                    }
                },
                err_fn,
                None,
            )?;
            Ok(stream)
        }
        cpal::SampleFormat::U16 => {
            // U16 is rare, but we must handle it.
            // Convert U16 -> f32 -> AGC -> PCM16
            let w = writer_raw.clone();
            let w_asr = writer_asr.clone();
            let mut rs_phase: f32 = 0.0;
            let ratio: f32 = sample_rate as f32 / 16_000.0;
            let mut prev_mono: f32 = 0.0;
            let tx_mix = mixer_tx.clone();
            let tx_asr = asr_tx.clone();

            let log_cnt = log_counter.clone();
            let sys_speak = system_speaking.clone();
            let mut agcs: Vec<dagc::MonoAgc> = (0..channels)
                .map(|_| dagc::MonoAgc::new(agc_target_energy, agc_distortion).unwrap())
                .collect();
            let gate_threshold_rms = 0.01;

            let stream = dev.build_input_stream(
                &stream_config,
                move |data: &[u16], _| {
                    if data.is_empty() {
                        return;
                    }

                    // U16 -> F32 [-1.0, 1.0]
                    let mut processed_f32: Vec<f32> = Vec::with_capacity(data.len());
                    let mut sum_sq = 0.0;
                    for &x in data {
                        let f = (x as f32 / u16::MAX as f32) * 2.0 - 1.0;
                        processed_f32.push(f);
                        sum_sq += f * f;
                    }

                    let block_rms = (sum_sq / data.len() as f32).sqrt();
                    let freeze = block_rms < gate_threshold_rms;
                    let ch = channels as usize;

                    for (_i, agc) in agcs.iter_mut().enumerate() {
                        agc.freeze_gain(freeze);
                    }

                    for frame in processed_f32.chunks_mut(ch) {
                        for (i, sample) in frame.iter_mut().enumerate() {
                            if i < agcs.len() {
                                let mut s_slice = [*sample];
                                agcs[i].process(&mut s_slice);
                                *sample = s_slice[0];
                            }
                        }
                    }

                    if log_cnt.fetch_add(1, Ordering::Relaxed) % 100 == 0 {
                        // let _g = agcs[0].gain();
                    }

                    // Write Back
                    let mut bytes = Vec::with_capacity(data.len() * 2);
                    for &x in &processed_f32 {
                        let v = (x.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
                        bytes.extend_from_slice(&v.to_le_bytes());
                    }
                    w.lock().unwrap().write_data(&bytes);

                    // ASR Path: raw when system active, DAGC when silent
                    let use_raw_for_asr = sys_speak.load(Ordering::Relaxed);
                    let mut asr_bytes = Vec::new();
                    let mut raw_idx = 0usize;

                    for frame in processed_f32.chunks(ch) {
                        let mut mono_dagc = 0.0f32;
                        for &s in frame {
                            mono_dagc += s;
                        }
                        mono_dagc /= ch as f32;

                        let mut mono_raw = 0.0f32;
                        let end = (raw_idx + ch).min(data.len());
                        for i in raw_idx..end {
                            mono_raw += (data[i] as f32 / u16::MAX as f32) * 2.0 - 1.0;
                        }
                        mono_raw /= ch as f32;
                        raw_idx += ch;

                        let mono = if use_raw_for_asr { mono_raw } else { mono_dagc };

                        rs_phase += 1.0 / ratio;
                        while rs_phase >= 1.0 {
                            let t = 1.0 - (rs_phase - 1.0);
                            let y = prev_mono + (mono - prev_mono) * t;
                            let v = (y.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
                            asr_bytes.extend_from_slice(&v.to_le_bytes());
                            rs_phase -= 1.0;
                        }
                        prev_mono = mono;
                    }
                    if !asr_bytes.is_empty() {
                        w_asr.lock().unwrap().write_data(&asr_bytes);
                        for ch_bytes in asr_bytes.chunks_exact(2) {
                            let v = i16::from_le_bytes([ch_bytes[0], ch_bytes[1]]);
                            let _ = tx_mix.send(v);
                            let _ = tx_asr.send(v);
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
