// Audio capture implementations module

pub mod microphone;
pub mod system;

#[cfg(target_os = "macos")]
pub mod core_audio;

#[cfg(target_os = "macos")]
pub mod sc_capture;

// Re-export capture functionality
pub use system::{
    check_system_audio_permissions, list_system_audio_devices, start_system_audio_capture,
    SystemAudioCapture, SystemAudioStream,
};

#[cfg(target_os = "macos")]
pub use core_audio::{CoreAudioCapture, CoreAudioStream};

#[cfg(target_os = "macos")]
pub use sc_capture::{ScreenCaptureKitCapture, ScreenCaptureKitStream};
