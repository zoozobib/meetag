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
        config.set_captures_audio(true);
        config.set_sample_rate(48000);
        config.set_channel_count(2); // Stereo, we'll convert to mono later

        // Video is disabled by default, but set minimal params
        config.set_width(1);
        config.set_height(1);

        info!("✅ ScreenCaptureKit: Configuration created (audio-only, 48kHz stereo)");

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

        // Create audio processor
        let processor = Arc::new(Mutex::new(AudioProcessor {
            producer,
            waker_state: waker_state.clone(),
            should_terminate: should_terminate.clone(),
            samples_processed: samples_processed.clone(),
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
            sample_rate: Arc::new(AtomicU32::new(self.sample_rate)),
        })
    }

    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }
}

#[cfg(target_os = "macos")]
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

        // Cast to f32 slice (assuming PCM float32 format)
        let data_ptr = data_slice.as_ptr() as *const f32;
        let sample_count = data_slice.len() / 8; // 4 bytes per f32, 2 channels

        if sample_count == 0 {
            return Ok(());
        }

        // Convert to slice
        let samples = unsafe { std::slice::from_raw_parts(data_ptr, sample_count * 2) };

        // Mix stereo to mono
        let mono_samples: Vec<f32> = samples
            .chunks(2)
            .map(|chunk| (chunk[0] + chunk.get(1).unwrap_or(&0.0)) / 2.0)
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

        // Debug: Log first successful capture
        static FIRST_AUDIO: std::sync::Once = std::sync::Once::new();
        FIRST_AUDIO.call_once(|| {
            info!("🎵 ScreenCaptureKit: First audio data received!");
            info!("   Samples: {}", mono_samples.len());
            info!("   RMS: {:.4}", rms(&mono_samples));
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
