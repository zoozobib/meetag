// ScreenCaptureKit audio-only capture implementation for macOS
// Uses Apple's ScreenCaptureKit API to capture system audio without video

#[cfg(target_os = "macos")]
use anyhow::{Context, Result};
use futures_util::Stream;
use log::{error, info, warn};
use ringbuf::{
    traits::{Consumer, Observer, Producer, Split},
    HeapCons, HeapProd, HeapRb,
};
use std::collections::HashMap;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::task::{Context as TaskContext, Poll, Waker};

#[cfg(target_os = "macos")]
use cidre::{arc as c_arc, cm, define_obj_type, ns, objc, sc};

#[cfg(target_os = "macos")]
use cidre::sc::StreamOutput;

// Global registry for audio processors
#[cfg(target_os = "macos")]
lazy_static::lazy_static! {
    static ref PROCESSORS: Mutex<HashMap<usize, Arc<Mutex<AudioProcessor>>>> = Mutex::new(HashMap::new());
    static ref NEXT_ID: AtomicUsize = AtomicUsize::new(1);
}

/// Audio-only ScreenCaptureKit capture
#[cfg(target_os = "macos")]
pub struct ScreenCaptureKitCapture {
    sample_rate: u32,
}

/// Stream implementation for ScreenCaptureKit
#[cfg(target_os = "macos")]
pub struct ScreenCaptureKitStream {
    _stream: c_arc::R<sc::Stream>,
    _output: c_arc::R<StreamOutputHandler>,
    _processor_id: usize,
    consumer: HeapCons<f32>,
    waker_state: Arc<Mutex<WakerState>>,
    should_terminate: Arc<AtomicBool>,
    sample_rate: Arc<AtomicU32>,
}

struct WakerState {
    waker: Option<Waker>,
    has_data: bool,
}

/// Audio processor with actual processing logic
#[cfg(target_os = "macos")]
struct AudioProcessor {
    producer: HeapProd<f32>,
    waker_state: Arc<Mutex<WakerState>>,
    should_terminate: Arc<AtomicBool>,
    samples_processed: Arc<AtomicU32>,
    sample_rate: Arc<AtomicU32>,
}

/// Inner data for StreamOutputHandler
#[cfg(target_os = "macos")]
#[repr(C)]
struct AudioOutputHandlerInner {
    handler_id: usize,
}

// Define our output handler type using cidre pattern (like sc-record example)
#[cfg(target_os = "macos")]
define_obj_type!(
    pub StreamOutputHandler + sc::stream::OutputImpl,
    AudioOutputHandlerInner,
    STREAM_OUTPUT_HANDLER
);

// Implement Output trait (empty, like sc-record example)
#[cfg(target_os = "macos")]
impl sc::stream::Output for StreamOutputHandler {}

// Implement OutputImpl trait with actual method (like sc-record example)
#[cfg(target_os = "macos")]
#[objc::add_methods]
impl sc::stream::OutputImpl for StreamOutputHandler {
    extern "C" fn impl_stream_did_output_sample_buf(
        &mut self,
        _cmd: Option<&cidre::objc::Sel>,
        _stream: &sc::Stream,
        sample_buf: &mut cm::SampleBuf,
        kind: sc::OutputType,
    ) {
        if kind != sc::OutputType::Audio {
            return;
        }

        let handler_id = self.inner().handler_id;

        // Get processor from global registry
        if let Ok(processors) = PROCESSORS.lock() {
            if let Some(processor) = processors.get(&handler_id) {
                if let Ok(mut proc) = processor.lock() {
                    if let Err(e) = proc.process_audio_buffer(sample_buf) {
                        error!("Failed to process audio buffer: {}", e);
                    }
                }
            }
        }
    }
}

#[cfg(target_os = "macos")]
impl ScreenCaptureKitCapture {
    pub async fn new() -> Result<Self> {
        info!("🎙️ ScreenCaptureKit: Starting audio-only capture initialization...");

        Ok(Self { sample_rate: 48000 })
    }

    pub async fn stream(self) -> Result<ScreenCaptureKitStream> {
        info!("🎙️ ScreenCaptureKit: Creating audio stream...");

        // Get shareable content (async)
        info!("📋 ScreenCaptureKit: Fetching shareable content...");
        let content = sc::ShareableContent::current()
            .await
            .map_err(|e| anyhow::anyhow!("Failed to get shareable content: {:?}", e))?;

        let displays = content.displays();
        if displays.is_empty() {
            return Err(anyhow::anyhow!("No displays found"));
        }

        let display = &displays[0];
        info!(
            "✅ ScreenCaptureKit: Using display: {:?}",
            display.display_id()
        );

        // Create content filter with empty windows array
        let windows = ns::Array::new();
        let filter = sc::ContentFilter::with_display_excluding_windows(display, &windows);

        // Create stream configuration - AUDIO ONLY
        let mut config = sc::StreamCfg::new();

        // Audio configuration
        // SCK delivers Stereo f32 at 48000 Hz (verified via diagnostic logs)
        let requested_sample_rate = 48000i64;
        let requested_channels = 2i64;
        config.set_captures_audio(true);
        config.set_sample_rate(requested_sample_rate);
        config.set_channel_count(requested_channels);

        // ScreenCaptureKit requires valid video dimensions even for audio-only capture
        // Minimum valid resolution and lowest frame rate to minimize overhead
        config.set_width(64);
        config.set_height(64);
        config.set_minimum_frame_interval(cm::Time::new(1, 1)); // 1 fps
        config.set_shows_cursor(false);

        info!(
            "📋 SCK Config: requested_sr={}Hz, requested_ch={}, video=64x64@1fps",
            requested_sample_rate, requested_channels
        );

        // Create ring buffer for audio data
        let buffer_size = 48000 * 10; // 10 seconds at 48kHz
        let ring_buffer = HeapRb::<f32>::new(buffer_size);
        let (producer, consumer) = ring_buffer.split();

        let waker_state = Arc::new(Mutex::new(WakerState {
            waker: None,
            has_data: false,
        }));

        let should_terminate = Arc::new(AtomicBool::new(false));
        let samples_processed = Arc::new(AtomicU32::new(0));

        let detected_sample_rate = Arc::new(AtomicU32::new(48000)); // Initial guess, will be updated

        // Create audio processor
        let processor = Arc::new(Mutex::new(AudioProcessor {
            producer,
            waker_state: waker_state.clone(),
            should_terminate: should_terminate.clone(),
            samples_processed: samples_processed.clone(),
            sample_rate: detected_sample_rate.clone(),
        }));

        // Create stream
        let stream = sc::Stream::new(&filter, &config);

        // Get unique handler ID
        let handler_id = NEXT_ID.fetch_add(1, Ordering::SeqCst);

        // Create output handler using cidre pattern
        let inner = AudioOutputHandlerInner { handler_id };
        let output = StreamOutputHandler::with(inner);

        // Register processor with handler ID
        {
            let mut processors = PROCESSORS.lock().unwrap();
            processors.insert(handler_id, processor);
        }

        // Add output handler to stream
        stream
            .add_stream_output(output.as_ref(), sc::OutputType::Audio, None)
            .map_err(|e| anyhow::anyhow!("Failed to add stream output: {:?}", e))?;

        // Start capture
        info!("🎙️ ScreenCaptureKit: Starting audio capture...");
        stream
            .start()
            .await
            .map_err(|e| anyhow::anyhow!("Failed to start stream: {:?}", e))?;

        info!("✅ ScreenCaptureKit: Audio capture started successfully!");

        Ok(ScreenCaptureKitStream {
            _stream: stream,
            _output: output,
            _processor_id: handler_id,
            consumer,
            waker_state,
            should_terminate,
            sample_rate: detected_sample_rate, // Use the shared atomic that processor will update
        })
    }

    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }
}

impl AudioProcessor {
    fn process_audio_buffer(&mut self, sample_buffer: &mut cm::SampleBuf) -> Result<()> {
        // Get block buffer
        let block_buf = sample_buffer.data_buf().context("No data buffer")?;

        let data_len = block_buf.data_len();
        if data_len == 0 {
            return Ok(());
        }

        // Get audio data using safe cidre API
        let (data_slice, _total_len) = block_buf.data_ptr_at(0)?;

        if data_slice.is_empty() {
            return Ok(());
        }

        // Get duration info for sample rate calculation
        let duration = sample_buffer.duration();
        let duration_valid = duration.flags.contains(cm::TimeFlags::VALID);
        let duration_sec = if duration.scale > 0 {
            duration.value as f64 / duration.scale as f64
        } else {
            0.0
        };

        // === COMPREHENSIVE DIAGNOSTIC LOG (once) ===
        static DIAGNOSTIC_LOGGED: std::sync::Once = std::sync::Once::new();
        DIAGNOSTIC_LOGGED.call_once(|| {
            info!("╔══════════════════════════════════════════════════════════════╗");
            info!("║      SCK AUDIO FORMAT DIAGNOSTIC (FIRST PACKET)              ║");
            info!("╠══════════════════════════════════════════════════════════════╣");

            // === TRY TO GET AUTHORITATIVE FORMAT INFO FROM format_desc ===
            info!("║ AUDIO FORMAT DESCRIPTION (from CMFormatDesc):                ║");
            if let Some(format_desc) = sample_buffer.format_desc() {
                // The format_desc should give us the ASBD (AudioStreamBasicDescription)
                // which contains authoritative sample rate, channels, format info
                unsafe {
                    // Try to cast to AudioFormatDesc and get ASBD
                    let audio_desc: &cidre::cm::AudioFormatDesc = std::mem::transmute(format_desc);
                    if let Some(asbd) = audio_desc.stream_basic_desc() {
                        info!("║   ✅ ASBD FOUND - THIS IS THE AUTHORITATIVE FORMAT!");
                        info!("║   sample_rate: {} Hz", asbd.sample_rate);
                        info!("║   channels_per_frame: {}", asbd.channels_per_frame);
                        info!("║   bits_per_channel: {}", asbd.bits_per_channel);
                        info!("║   bytes_per_frame: {}", asbd.bytes_per_frame);
                        info!("║   bytes_per_packet: {}", asbd.bytes_per_packet);
                        info!("║   frames_per_packet: {}", asbd.frames_per_packet);
                        info!("║   format: {:?}", asbd.format);
                        info!("║   format_flags: {:?}", asbd.format_flags);

                        // Use cidre's built-in method
                        let is_interleaved = asbd.is_interleaved();
                        let is_native_endian = asbd.is_native_endian();
                        
                        // Parse format flags for additional info
                        let flags_raw = asbd.format_flags.0;
                        let is_float = (flags_raw & 0x1) != 0; // kAudioFormatFlagIsFloat

                        info!("║   FORMAT INFO (from ASBD methods):");
                        info!("║     is_interleaved: {} ⬅️ CRITICAL", is_interleaved);
                        info!("║     is_native_endian: {}", is_native_endian);
                        info!("║     isFloat (flag): {}", is_float);

                        if is_interleaved {
                            info!("║   � FORMAT IS INTERLEAVED [L0,R0,L1,R1...]");
                        } else {
                            info!("║   � FORMAT IS PLANAR (non-interleaved) [L0,L1...Ln,R0,R1...Rn]");
                        }
                    } else {
                        info!("║   ⚠️ No ASBD available from format_desc");
                    }
                }
            } else {
                info!("║   ⚠️ No format_desc available from sample_buffer");
            }

            info!("╠══════════════════════════════════════════════════════════════╣");
            info!("║ RAW BUFFER INFO:                                             ║");
            info!("║   data_slice.len() = {} bytes", data_slice.len());
            info!("║   data_len (block_buf) = {} bytes", data_len);

            // Show first few bytes as hex
            let preview: Vec<u8> = data_slice.iter().take(32).cloned().collect();
            info!("║   First 32 bytes (hex): {:02X?}", preview);

            // Interpret first few bytes as different formats
            if data_slice.len() >= 16 {
                // As float32
                let ptr_f32 = data_slice.as_ptr() as *const f32;
                let f32_samples: Vec<f32> = (0..4).map(|i| unsafe { *ptr_f32.add(i) }).collect();
                info!("║   As f32 (first 4): {:?}", f32_samples);

                // As int16
                let ptr_i16 = data_slice.as_ptr() as *const i16;
                let i16_samples: Vec<i16> = (0..8).map(|i| unsafe { *ptr_i16.add(i) }).collect();
                info!("║   As i16 (first 8): {:?}", i16_samples);

                // As int32
                let ptr_i32 = data_slice.as_ptr() as *const i32;
                let i32_samples: Vec<i32> = (0..4).map(|i| unsafe { *ptr_i32.add(i) }).collect();
                info!("║   As i32 (first 4): {:?}", i32_samples);
            }

            info!("╠══════════════════════════════════════════════════════════════╣");
            info!("║ DURATION METADATA:                                           ║");
            info!(
                "║   valid={}, value={}, scale={}",
                duration_valid, duration.value, duration.scale
            );
            info!("║   duration_sec = {:.6}s", duration_sec);
            info!(
                "║   duration.scale typically represents sample rate = {} Hz",
                duration.scale
            );

            info!("╠══════════════════════════════════════════════════════════════╣");
            info!("║ FRAME COUNT INTERPRETATIONS:                                 ║");

            // Calculate frame counts for different format assumptions
            let bytes = data_slice.len();

            // Mono float32
            let frames_mono_f32 = bytes / 4;
            let sr_mono_f32 = if duration_sec > 0.0 {
                frames_mono_f32 as f64 / duration_sec
            } else {
                0.0
            };
            info!(
                "║   Mono f32:   {} frames -> {:.0} Hz",
                frames_mono_f32, sr_mono_f32
            );

            // Stereo float32
            let frames_stereo_f32 = bytes / 8;
            let sr_stereo_f32 = if duration_sec > 0.0 {
                frames_stereo_f32 as f64 / duration_sec
            } else {
                0.0
            };
            info!(
                "║   Stereo f32: {} frames -> {:.0} Hz",
                frames_stereo_f32, sr_stereo_f32
            );

            // Mono int16
            let frames_mono_i16 = bytes / 2;
            let sr_mono_i16 = if duration_sec > 0.0 {
                frames_mono_i16 as f64 / duration_sec
            } else {
                0.0
            };
            info!(
                "║   Mono i16:   {} frames -> {:.0} Hz",
                frames_mono_i16, sr_mono_i16
            );

            // Stereo int16
            let frames_stereo_i16 = bytes / 4;
            let sr_stereo_i16 = if duration_sec > 0.0 {
                frames_stereo_i16 as f64 / duration_sec
            } else {
                0.0
            };
            info!(
                "║   Stereo i16: {} frames -> {:.0} Hz",
                frames_stereo_i16, sr_stereo_i16
            );

            // Mono int32
            let frames_mono_i32 = bytes / 4;
            let sr_mono_i32 = if duration_sec > 0.0 {
                frames_mono_i32 as f64 / duration_sec
            } else {
                0.0
            };
            info!(
                "║   Mono i32:   {} frames -> {:.0} Hz",
                frames_mono_i32, sr_mono_i32
            );

            // Stereo int32
            let frames_stereo_i32 = bytes / 8;
            let sr_stereo_i32 = if duration_sec > 0.0 {
                frames_stereo_i32 as f64 / duration_sec
            } else {
                0.0
            };
            info!(
                "║   Stereo i32: {} frames -> {:.0} Hz",
                frames_stereo_i32, sr_stereo_i32
            );

            info!("╠══════════════════════════════════════════════════════════════╣");
            info!(
                "║ EXPECTED: duration.scale ({}) should match calculated SR      ║",
                duration.scale
            );
            info!("║ If Stereo f32 matches duration.scale -> data is stereo f32   ║");
            info!("║ If Mono f32 matches duration.scale -> data is mono f32       ║");

            // Determine which format matches
            let scale = duration.scale as f64;
            if (sr_stereo_f32 - scale).abs() < 1000.0 {
                info!(
                    "║ ✅ MATCH: Stereo f32 ({:.0} Hz ≈ {} Hz)",
                    sr_stereo_f32, duration.scale
                );
            } else if (sr_mono_f32 - scale).abs() < 1000.0 {
                info!(
                    "║ ✅ MATCH: Mono f32 ({:.0} Hz ≈ {} Hz)",
                    sr_mono_f32, duration.scale
                );
            } else if (sr_stereo_i16 - scale).abs() < 1000.0 {
                info!(
                    "║ ✅ MATCH: Stereo i16 ({:.0} Hz ≈ {} Hz)",
                    sr_stereo_i16, duration.scale
                );
            } else if (sr_mono_i16 - scale).abs() < 1000.0 {
                info!(
                    "║ ✅ MATCH: Mono i16 ({:.0} Hz ≈ {} Hz)",
                    sr_mono_i16, duration.scale
                );
            } else {
                info!("║ ⚠️ NO CLEAR MATCH - format unknown!");
            }

            // === CRITICAL: Compare INTERLEAVED vs PLANAR sample interpretation ===
            info!("╠══════════════════════════════════════════════════════════════╣");
            info!("║ STEREO FORMAT COMPARISON (INTERLEAVED vs PLANAR):             ║");

            let data_ptr = data_slice.as_ptr() as *const f32;
            let total_f32_samples = bytes / 4;
            let all_f32 = unsafe { std::slice::from_raw_parts(data_ptr, total_f32_samples) };
            let frame_count_stereo = total_f32_samples / 2;

            // Find first non-zero samples for meaningful comparison
            let mut first_nonzero_idx = 0usize;
            for (i, &s) in all_f32.iter().enumerate() {
                if s.abs() > 0.0001 {
                    first_nonzero_idx = i;
                    break;
                }
            }

            info!("║ First non-zero sample index: {}", first_nonzero_idx);
            info!(
                "║ Total f32 samples: {}, frame_count: {}",
                total_f32_samples, frame_count_stereo
            );

            // Show raw sample values at offset for both interpretations
            let offset = first_nonzero_idx.saturating_sub(2);
            info!("╠══════════════════════════════════════════════════════════════╣");
            info!("║ RAW SAMPLES at offset {} (10 values):", offset);
            let preview: Vec<f32> = all_f32.iter().skip(offset).take(10).copied().collect();
            info!("║   {:?}", preview);

            info!("╠══════════════════════════════════════════════════════════════╣");
            info!("║ INTERPRETATION 1: INTERLEAVED [L0,R0,L1,R1...]");
            info!("║   Meaning: samples alternate left-right-left-right");
            // Show how it would be mixed to mono
            let mut interleaved_mono: Vec<f32> = Vec::new();
            for chunk in all_f32.chunks(2) {
                if chunk.len() == 2 {
                    interleaved_mono.push((chunk[0] + chunk[1]) / 2.0);
                }
            }
            let preview_start = offset / 2;
            let interleaved_preview: Vec<f32> = interleaved_mono
                .iter()
                .skip(preview_start)
                .take(5)
                .copied()
                .collect();
            info!("║   Mono mix preview: {:?}", interleaved_preview);

            // Calculate RMS for interleaved interpretation
            let interleaved_rms: f32 = (interleaved_mono.iter().map(|x| x * x).sum::<f32>()
                / interleaved_mono.len() as f32)
                .sqrt();
            info!("║   RMS: {:.6}", interleaved_rms);

            info!("╠══════════════════════════════════════════════════════════════╣");
            info!("║ INTERPRETATION 2: PLANAR [L0,L1...Ln, R0,R1...Rn]");
            info!("║   Meaning: first half = all left, second half = all right");
            // Show how it would be mixed to mono
            let left_half = &all_f32[..frame_count_stereo];
            let right_half = &all_f32[frame_count_stereo..];
            let mut planar_mono: Vec<f32> = Vec::new();
            for (l, r) in left_half.iter().zip(right_half.iter()) {
                planar_mono.push((l + r) / 2.0);
            }
            let planar_preview: Vec<f32> = planar_mono
                .iter()
                .skip(preview_start)
                .take(5)
                .copied()
                .collect();
            info!("║   Mono mix preview: {:?}", planar_preview);

            // Calculate RMS for planar interpretation
            let planar_rms: f32 =
                (planar_mono.iter().map(|x| x * x).sum::<f32>() / planar_mono.len() as f32).sqrt();
            info!("║   RMS: {:.6}", planar_rms);

            info!("╠══════════════════════════════════════════════════════════════╣");
            info!("║ INTERPRETATION 3: MONO (no mixing needed)");
            info!("║   Meaning: data is already mono, just use first half");
            let mono_preview: Vec<f32> = all_f32.iter().skip(offset).take(5).copied().collect();
            info!("║   Sample preview: {:?}", mono_preview);
            let mono_rms: f32 = (all_f32
                .iter()
                .take(frame_count_stereo)
                .map(|x| x * x)
                .sum::<f32>()
                / frame_count_stereo as f32)
                .sqrt();
            info!("║   RMS (first half): {:.6}", mono_rms);

            info!("╠══════════════════════════════════════════════════════════════╣");
            info!("║ DECISION HELPER:");
            info!("║   - If audio sounds HIGH PITCHED: wrong interpretation");
            info!("║   - Compare RMS values - they should be similar");
            info!("║   - Interleaved RMS: {:.6}", interleaved_rms);
            info!("║   - Planar RMS: {:.6}", planar_rms);
            info!("║   - Mono RMS: {:.6}", mono_rms);

            // Check for signs of planar vs interleaved
            // In planar, left_half[0] and right_half[0] should be correlated
            // In interleaved, all_f32[0] and all_f32[1] should be correlated (L and R of same time)
            let correlation_interleaved: f32 = all_f32
                .chunks(2)
                .take(100)
                .filter(|c| c.len() == 2)
                .map(|c| c[0] * c[1])
                .sum::<f32>()
                / 100.0;
            let correlation_planar: f32 = left_half
                .iter()
                .zip(right_half.iter())
                .take(100)
                .map(|(l, r)| l * r)
                .sum::<f32>()
                / 100.0;

            info!("║   Correlation (higher = more likely correct):");
            info!("║     Interleaved L*R: {:.6}", correlation_interleaved);
            info!("║     Planar L*R: {:.6}", correlation_planar);

            if correlation_interleaved > correlation_planar {
                info!("║   ➡️ LIKELY: INTERLEAVED format");
            } else if correlation_planar > correlation_interleaved {
                info!("║   ➡️ LIKELY: PLANAR format");
            } else {
                info!("║   ⚠️ UNCERTAIN - correlations are similar");
            }

            info!("╠══════════════════════════════════════════════════════════════╣");
            info!("║ CURRENT CODE USES: PLANAR format                             ║");
            info!("╚══════════════════════════════════════════════════════════════╝");
        });

        // STEREO float32 format (verified from diagnostic logs)
        // SCK delivers interleaved stereo: [L0, R0, L1, R1, ...]
        let bytes_per_sample = 4usize; // f32
        let channels = 2u16; // STEREO
        let bytes_per_frame = bytes_per_sample * channels as usize; // 8 bytes per frame
        let frame_count = data_slice.len() / bytes_per_frame;

        if frame_count == 0 {
            warn!(
                "⚠️ SCK: frame_count=0, data_slice.len()={}, skipping",
                data_slice.len()
            );
            return Ok(());
        }

        // Update sample rate from duration metadata
        if duration_valid && duration.scale > 0 && duration.value > 0 && duration_sec > 0.0001 {
            let calculated_rate = (frame_count as f64 / duration_sec).round() as u32;
            if calculated_rate >= 8000 && calculated_rate <= 192000 {
                let current = self.sample_rate.load(Ordering::Relaxed);
                if current != calculated_rate {
                    info!(
                        "🔄 SCK SR Update: {} Hz -> {} Hz (frames={}, dur={:.6}s, format=stereo_f32)",
                        current, calculated_rate, frame_count, duration_sec
                    );
                    self.sample_rate.store(calculated_rate, Ordering::Relaxed);
                }
            }
        }

        // Cast to f32 slice
        // NOTE: SCK may provide PLANAR stereo [L0,L1,L2...Ln, R0,R1,R2...Rn]
        // instead of INTERLEAVED [L0,R0,L1,R1...]
        let data_ptr = data_slice.as_ptr() as *const f32;
        let total_samples = frame_count * 2; // L and R for each frame
        let all_samples = unsafe { std::slice::from_raw_parts(data_ptr, total_samples) };

        // Try PLANAR format: first half = left channel, second half = right channel
        let left_samples = &all_samples[..frame_count];
        let right_samples = &all_samples[frame_count..];

        // Mix stereo to mono: (L + R) / 2
        let mono_samples: Vec<f32> = left_samples
            .iter()
            .zip(right_samples.iter())
            .map(|(l, r)| (l + r) / 2.0)
            .collect();

        // Push to ring buffer
        let pushed = self.producer.push_slice(&mono_samples);

        if pushed < mono_samples.len() {
            warn!(
                "Ring buffer full, dropped {} samples",
                mono_samples.len() - pushed
            );
        }

        // Update counter and wake
        if pushed > 0 {
            self.samples_processed
                .fetch_add(pushed as u32, Ordering::Relaxed);

            let mut waker_state = self.waker_state.lock().unwrap();
            waker_state.has_data = true;
            if let Some(waker) = waker_state.waker.take() {
                waker.wake();
            }
        }

        // Debug: Log first successful capture with sample preview
        static FIRST_AUDIO: std::sync::Once = std::sync::Once::new();
        FIRST_AUDIO.call_once(|| {
            let sr = self.sample_rate.load(Ordering::Relaxed);
            info!("🎵 SCK First Audio Data Pushed:");
            info!("   Stored SR: {} Hz", sr);
            info!(
                "   frame_count: {}, pushed to ring: {}",
                frame_count, pushed
            );
            info!("   RMS: {:.6}", rms(&mono_samples));

            // Show first few samples
            let sample_preview: Vec<f32> = mono_samples.iter().take(10).copied().collect();
            info!("   First 10 samples: {:?}", sample_preview);

            // Show min/max
            let min_val = mono_samples.iter().cloned().fold(f32::INFINITY, f32::min);
            let max_val = mono_samples
                .iter()
                .cloned()
                .fold(f32::NEG_INFINITY, f32::max);
            info!("   Sample range: [{:.6}, {:.6}]", min_val, max_val);
        });

        Ok(())
    }
}

// Calculate RMS for debugging
#[cfg(target_os = "macos")]
fn rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum: f32 = samples.iter().map(|s| s * s).sum();
    (sum / samples.len() as f32).sqrt()
}

#[cfg(target_os = "macos")]
impl ScreenCaptureKitStream {
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate.load(std::sync::atomic::Ordering::Relaxed)
    }
}

#[cfg(target_os = "macos")]
impl Stream for ScreenCaptureKitStream {
    type Item = f32;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut TaskContext<'_>) -> Poll<Option<Self::Item>> {
        if self.should_terminate.load(Ordering::Acquire) {
            return Poll::Ready(None);
        }

        // Try to pop a sample
        if let Some(sample) = self.consumer.try_pop() {
            // Clear has_data flag if buffer is empty
            if self.consumer.is_empty() {
                let mut waker_state = self.waker_state.lock().unwrap();
                waker_state.has_data = false;
            }
            return Poll::Ready(Some(sample));
        }

        // No data available, register waker
        let mut waker_state = self.waker_state.lock().unwrap();
        waker_state.waker = Some(cx.waker().clone());
        waker_state.has_data = false;

        Poll::Pending
    }
}

#[cfg(target_os = "macos")]
impl Drop for ScreenCaptureKitStream {
    fn drop(&mut self) {
        info!("ScreenCaptureKitStream dropped, cleaning up processor");
        self.should_terminate.store(true, Ordering::Release);

        // Remove processor from global registry
        if let Ok(mut processors) = PROCESSORS.lock() {
            processors.remove(&self._processor_id);
        }
    }
}

// Stub implementations for non-macOS platforms
#[cfg(not(target_os = "macos"))]
pub struct ScreenCaptureKitCapture;

#[cfg(not(target_os = "macos"))]
impl ScreenCaptureKitCapture {
    pub async fn new() -> Result<Self, anyhow::Error> {
        Err(anyhow::anyhow!(
            "ScreenCaptureKit is only available on macOS"
        ))
    }

    pub async fn stream(self) -> Result<ScreenCaptureKitStream, anyhow::Error> {
        Err(anyhow::anyhow!(
            "ScreenCaptureKit is only available on macOS"
        ))
    }

    pub fn sample_rate(&self) -> u32 {
        48000
    }
}

#[cfg(not(target_os = "macos"))]
pub struct ScreenCaptureKitStream;
