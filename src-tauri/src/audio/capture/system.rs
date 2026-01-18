use anyhow::{Context as AnyhowContext, Result};
use cpal::traits::{DeviceTrait, HostTrait};
use futures_util::{Stream, StreamExt};
use std::pin::Pin;
use std::task::{Context, Poll};

#[cfg(target_os = "macos")]
use super::core_audio::CoreAudioCapture;
#[cfg(target_os = "macos")]
use super::sc_capture::ScreenCaptureKitCapture;
#[cfg(target_os = "macos")]
use futures_channel::mpsc;
#[cfg(target_os = "macos")]
use log::{info, warn};
#[cfg(target_os = "macos")]
use std::sync::atomic::{AtomicBool, Ordering};

#[cfg(target_os = "macos")]
static SC_KIT_FAILED: AtomicBool = AtomicBool::new(false);

/// System audio capture using ScreenCaptureKit or Core Audio tap (macOS) or CPAL (other platforms)
pub struct SystemAudioCapture {
    _host: cpal::Host,
}

impl SystemAudioCapture {
    pub fn new() -> Result<Self> {
        let host = cpal::default_host();
        Ok(Self { _host: host })
    }

    pub fn list_system_devices() -> Result<Vec<String>> {
        let host = cpal::default_host();
        let devices = host
            .output_devices()
            .map_err(|e| anyhow::anyhow!("Failed to enumerate output devices: {}", e))?;

        let mut device_names = Vec::new();
        for device in devices {
            if let Ok(name) = device.name() {
                device_names.push(name);
            }
        }

        Ok(device_names)
    }

    pub async fn start_system_audio_capture(&self) -> Result<SystemAudioStream> {
        #[cfg(target_os = "macos")]
        {
            use crate::settings::{get_audio_settings, AudioBackend};

            let audio_settings = get_audio_settings();
            let preferred = audio_settings.preferred_backend;
            let allow_fallback = audio_settings.allow_fallback;

            info!("╔══════════════════════════════════════════════════════════════╗");
            info!("║           AUDIO BACKEND SELECTION                            ║");
            info!("╠══════════════════════════════════════════════════════════════╣");
            info!("║ Preferred backend: {:?}", preferred);
            info!("║ Allow fallback: {}", allow_fallback);
            info!("╚══════════════════════════════════════════════════════════════╝");

            // Helper closures to start each backend
            let try_screencapturekit = || async {
                // Don't retry if previously failed in this session
                if SC_KIT_FAILED.load(Ordering::Relaxed) {
                    info!("⏭️ [SCK] Skipping - previously failed in this session");
                    return Err(anyhow::anyhow!("ScreenCaptureKit previously failed"));
                }

                info!("🎙️ [SCK] Attempting ScreenCaptureKit for system audio capture...");
                let capture = ScreenCaptureKitCapture::new().await.map_err(|e| {
                    warn!("⚠️ [SCK] Initialization failed: {}", e);
                    SC_KIT_FAILED.store(true, Ordering::Relaxed);
                    e
                })?;

                let sc_stream = capture.stream().await.map_err(|e| {
                    warn!("⚠️ [SCK] Stream creation failed: {}", e);
                    SC_KIT_FAILED.store(true, Ordering::Relaxed);
                    e
                })?;

                let sample_rate = sc_stream.sample_rate();
                info!(
                    "✅ [SCK] ScreenCaptureKit started successfully ({}Hz)",
                    sample_rate
                );

                // Convert to SystemAudioStream
                let (tx, rx) = mpsc::unbounded::<Vec<f32>>();
                let (drop_tx, mut drop_rx) = tokio::sync::oneshot::channel::<()>();

                tokio::spawn(async move {
                    info!("🔄 [SCK] Forwarding task started");
                    use futures_util::StreamExt;
                    let mut stream = sc_stream;
                    let mut buffer = Vec::new();
                    let chunk_size = 1024;

                    loop {
                        tokio::select! {
                            _ = &mut drop_rx => {
                                info!("🛑 [SCK] Received stop signal");
                                break;
                            }
                            maybe_sample = stream.next() => {
                                match maybe_sample {
                                    Some(sample) => {
                                        buffer.push(sample);
                                        if buffer.len() >= chunk_size {
                                            if tx.unbounded_send(buffer.clone()).is_err() {
                                                break;
                                            }
                                            buffer.clear();
                                        }
                                    }
                                    None => break,
                                }
                            }
                        }
                    }

                    if !buffer.is_empty() {
                        let _ = tx.unbounded_send(buffer);
                    }
                    info!("🛑 [SCK] Forwarding task ended");
                });

                let receiver = rx.map(futures_util::stream::iter).flatten();

                Ok(SystemAudioStream {
                    drop_tx: Some(drop_tx),
                    sample_rate,
                    receiver: Box::pin(receiver),
                    _keep_alive: None,
                })
            };

            let try_coreaudio = || {
                info!("🎵 [CoreAudio] Attempting Core Audio Process Tap...");
                let core_audio = CoreAudioCapture::new().map_err(|e| {
                    warn!("⚠️ [CoreAudio] Initialization failed: {}", e);
                    e
                })?;

                let core_audio_stream = core_audio.stream().map_err(|e| {
                    warn!("⚠️ [CoreAudio] Stream creation failed: {}", e);
                    e
                })?;

                let sample_rate = core_audio_stream.sample_rate();
                info!(
                    "✅ [CoreAudio] Core Audio started successfully ({}Hz)",
                    sample_rate
                );

                // Convert to SystemAudioStream
                let (tx, rx) = mpsc::unbounded::<Vec<f32>>();
                let (drop_tx, mut drop_rx) = tokio::sync::oneshot::channel::<()>();

                tokio::spawn(async move {
                    info!("🔄 [CoreAudio] Forwarding task started");
                    use futures_util::StreamExt;
                    let mut stream = core_audio_stream;
                    let mut buffer = Vec::new();
                    let chunk_size = 1024;

                    loop {
                        tokio::select! {
                            _ = &mut drop_rx => {
                                info!("🛑 [CoreAudio] Received stop signal");
                                break;
                            }
                            maybe_sample = stream.next() => {
                                match maybe_sample {
                                    Some(sample) => {
                                        buffer.push(sample);
                                        if buffer.len() >= chunk_size {
                                            if tx.unbounded_send(buffer.clone()).is_err() {
                                                break;
                                            }
                                            buffer.clear();
                                        }
                                    }
                                    None => break,
                                }
                            }
                        }
                    }

                    if !buffer.is_empty() {
                        let _ = tx.unbounded_send(buffer);
                    }
                    info!("🛑 [CoreAudio] Forwarding task ended");
                });

                let receiver = rx.map(futures_util::stream::iter).flatten();

                Ok(SystemAudioStream {
                    drop_tx: Some(drop_tx),
                    sample_rate,
                    receiver: Box::pin(receiver),
                    _keep_alive: None,
                })
            };

            // Try preferred backend first
            match preferred {
                AudioBackend::ScreenCaptureKit => match try_screencapturekit().await {
                    Ok(stream) => {
                        info!("✅ Using preferred backend: ScreenCaptureKit");
                        return Ok(stream);
                    }
                    Err(e) => {
                        warn!("⚠️ Preferred backend ScreenCaptureKit failed: {}", e);
                        if allow_fallback {
                            info!("📦 Fallback enabled, trying CoreAudio...");
                            match try_coreaudio() {
                                Ok(stream) => {
                                    info!("✅ Fallback to CoreAudio successful");
                                    return Ok(stream);
                                }
                                Err(e2) => {
                                    return Err(anyhow::anyhow!(
                                        "Both backends failed. SCK: {}, CoreAudio: {}",
                                        e,
                                        e2
                                    ));
                                }
                            }
                        } else {
                            info!("❌ Fallback disabled, not trying other backends");
                            return Err(e);
                        }
                    }
                },
                AudioBackend::CoreAudio => match try_coreaudio() {
                    Ok(stream) => {
                        info!("✅ Using preferred backend: CoreAudio");
                        return Ok(stream);
                    }
                    Err(e) => {
                        warn!("⚠️ Preferred backend CoreAudio failed: {}", e);
                        if allow_fallback {
                            info!("📦 Fallback enabled, trying ScreenCaptureKit...");
                            match try_screencapturekit().await {
                                Ok(stream) => {
                                    info!("✅ Fallback to ScreenCaptureKit successful");
                                    return Ok(stream);
                                }
                                Err(e2) => {
                                    return Err(anyhow::anyhow!(
                                        "Both backends failed. CoreAudio: {}, SCK: {}",
                                        e,
                                        e2
                                    ));
                                }
                            }
                        } else {
                            info!("❌ Fallback disabled, not trying other backends");
                            return Err(e);
                        }
                    }
                },
            }
        }

        #[cfg(not(target_os = "macos"))]
        {
            // For non-macOS platforms, you would implement WASAPI/ALSA loopback here
            anyhow::bail!("System audio capture not yet implemented for this platform")
        }
    }

    pub fn check_system_audio_permissions() -> bool {
        // Check if we can enumerate audio devices
        match cpal::default_host().output_devices() {
            Ok(_) => true,
            Err(_) => false,
        }
    }
}

pub struct SystemAudioStream {
    // Use Option<Sender> so we can take() the sender in drop (oneshot can only send once)
    drop_tx: Option<tokio::sync::oneshot::Sender<()>>,
    sample_rate: u32,
    receiver: Pin<Box<dyn Stream<Item = f32> + Send + Sync>>,
    _keep_alive: Option<Box<dyn std::any::Any + Send + Sync>>,
}

impl Drop for SystemAudioStream {
    fn drop(&mut self) {
        println!("🛑 [LIFECYCLE: SystemAudioStream::drop START] Sending drop signal to forwarding task...");
        if let Some(tx) = self.drop_tx.take() {
            let result = tx.send(());
            println!(
                "🛑 [LIFECYCLE: SystemAudioStream::drop] drop_tx.send result: {:?}",
                result.is_ok()
            );
        } else {
            println!("🛑 [LIFECYCLE: SystemAudioStream::drop] drop_tx already consumed!");
        }
        println!("🛑 [LIFECYCLE: SystemAudioStream::drop END] Signal sent, SystemAudioStream dropping...");
        // Note: The tokio task should now exit promptly thanks to tokio::select!
    }
}

impl Stream for SystemAudioStream {
    type Item = f32;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.receiver.as_mut().poll_next_unpin(cx)
    }
}

impl SystemAudioStream {
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }
}

/// Public interface for system audio capture
pub async fn start_system_audio_capture() -> Result<SystemAudioStream> {
    let capture = SystemAudioCapture::new()?;
    capture.start_system_audio_capture().await
}

pub fn list_system_audio_devices() -> Result<Vec<String>> {
    SystemAudioCapture::list_system_devices()
}

pub fn check_system_audio_permissions() -> bool {
    SystemAudioCapture::check_system_audio_permissions()
}
