// Application settings management
// Handles loading and saving settings from/to settings.json

use anyhow::{Context, Result};
use log::{info, warn};
use once_cell::sync::Lazy;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;
use std::sync::RwLock;

/// Audio capture backend options
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AudioBackend {
    /// ScreenCaptureKit - Apple's high-level API, requires screen recording permission
    #[serde(alias = "sck")]
    ScreenCaptureKit,

    /// Core Audio Process Tap - Lower level, requires aggregate device
    #[serde(alias = "core_audio")]
    CoreAudio,
}

impl Default for AudioBackend {
    fn default() -> Self {
        // Default to ScreenCaptureKit as it's more user-friendly
        AudioBackend::ScreenCaptureKit
    }
}

impl std::fmt::Display for AudioBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AudioBackend::ScreenCaptureKit => write!(f, "ScreenCaptureKit"),
            AudioBackend::CoreAudio => write!(f, "CoreAudio"),
        }
    }
}

/// ASR backend options
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AsrBackend {
    /// Whisper (whisper-rs with Metal + CoreML)
    Whisper,
    /// FunASR (SenseVoiceSmall via sherpa-rs with CPU)
    FunAsr,
}

impl Default for AsrBackend {
    fn default() -> Self {
        // Default to Whisper for now as requested, but allow config to switch
        AsrBackend::Whisper
    }
}

impl std::fmt::Display for AsrBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AsrBackend::Whisper => write!(f, "Whisper"),
            AsrBackend::FunAsr => write!(f, "FunASR"),
        }
    }
}

/// Audio-related settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioSettings {
    /// User's preferred audio capture backend
    #[serde(default)]
    pub preferred_backend: AudioBackend,

    /// Whether to allow fallback to another backend if preferred fails
    #[serde(default = "default_true")]
    pub allow_fallback: bool,

    /// VAD backend selection
    #[serde(default)]
    pub vad_backend: VadBackend,

    /// VAD threshold (0.0 - 1.0)
    #[serde(default = "default_vad_threshold")]
    pub vad_threshold: f32,
}

fn default_true() -> bool {
    true
}

/// VAD backend options
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum VadBackend {
    /// WebRTC VAD (Fast, Standard)
    WebRtc,
    /// Silero VAD (High Accuracy, via ten-vad-rs)
    Silero,
}

impl Default for VadBackend {
    fn default() -> Self {
        VadBackend::Silero
    }
}

impl std::fmt::Display for VadBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            VadBackend::WebRtc => write!(f, "WebRTC"),
            VadBackend::Silero => write!(f, "Silero (TEN)"),
        }
    }
}

fn default_vad_threshold() -> f32 {
    0.5
}

impl Default for AudioSettings {
    fn default() -> Self {
        Self {
            preferred_backend: AudioBackend::default(),
            allow_fallback: true,
            vad_backend: VadBackend::default(),
            vad_threshold: default_vad_threshold(),
        }
    }
}

/// ASR-related settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AsrSettings {
    /// Language for speech recognition
    #[serde(default = "default_language")]
    pub language: String,

    /// Selected ASR backend
    #[serde(default)]
    pub backend: AsrBackend,
}

fn default_language() -> String {
    "zh".to_string()
}

impl Default for AsrSettings {
    fn default() -> Self {
        Self {
            language: default_language(),
            backend: AsrBackend::default(),
        }
    }
}

/// LLM-related settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmSettings {
    /// Model name for summarization (e.g., "qwen3:4b")
    #[serde(default = "default_llm_model")]
    pub model: String,
}

fn default_llm_model() -> String {
    "qwen3:1.7b".to_string()
}

impl Default for LlmSettings {
    fn default() -> Self {
        Self {
            model: default_llm_model(),
        }
    }
}

/// Root settings structure
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Settings {
    /// Audio capture settings
    #[serde(default)]
    pub audio: AudioSettings,

    /// ASR settings
    #[serde(default)]
    pub asr: AsrSettings,

    /// LLM settings
    #[serde(default)]
    pub llm: LlmSettings,
}

impl Settings {
    /// Get the settings file path
    pub fn settings_path() -> PathBuf {
        // Try to use system app data directory
        // macOS: ~/Library/Application Support/com.zoozobib.rec/settings.json
        if let Some(base_dir) = dirs::data_dir() {
            let app_dir = base_dir.join("com.zoozobib.rec");
            return app_dir.join("settings.json");
        }

        // Fallback to ~/.meetily/settings.json
        let home = dirs::home_dir().unwrap_or_else(|| PathBuf::from("."));
        let settings_dir = home.join(".meetily");
        settings_dir.join("settings.json")
    }

    /// Load settings from file, returning defaults if file doesn't exist
    pub fn load() -> Self {
        let path = Self::settings_path();

        info!("📁 [SETTINGS] Loading settings from: {:?}", path);

        if !path.exists() {
            info!("📁 [SETTINGS] Settings file not found, using defaults");
            let defaults = Settings::default();

            // Try to save defaults so user has a file to edit
            if let Err(e) = defaults.save() {
                warn!("📁 [SETTINGS] Failed to save default settings: {}", e);
            } else {
                info!("📁 [SETTINGS] Created default settings file");
            }

            return defaults;
        }

        match fs::read_to_string(&path) {
            Ok(content) => match serde_json::from_str::<Settings>(&content) {
                Ok(settings) => {
                    info!("📁 [SETTINGS] Loaded settings successfully:");
                    info!(
                        "📁 [SETTINGS]   audio.preferred_backend: {}",
                        settings.audio.preferred_backend
                    );
                    info!(
                        "📁 [SETTINGS]   audio.allow_fallback: {}",
                        settings.audio.allow_fallback
                    );
                    info!("📁 [SETTINGS]   asr.language: {}", settings.asr.language);

                    // Force save to ensure file on disk is updated with new schema fields
                    // (Migration for existing users who lack new fields like vad_backend)
                    if let Err(e) = settings.save() {
                        warn!(
                            "📁 [SETTINGS] Failed to migrate/update settings file: {}",
                            e
                        );
                    }

                    settings
                }
                Err(e) => {
                    warn!("📁 [SETTINGS] Failed to parse settings file: {}", e);
                    warn!("📁 [SETTINGS] Using defaults");
                    Settings::default()
                }
            },
            Err(e) => {
                warn!("📁 [SETTINGS] Failed to read settings file: {}", e);
                warn!("📁 [SETTINGS] Using defaults");
                Settings::default()
            }
        }
    }

    /// Save settings to file
    pub fn save(&self) -> Result<()> {
        let path = Self::settings_path();

        // Ensure parent directory exists
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).context("Failed to create settings directory")?;
        }

        let content = serde_json::to_string_pretty(self).context("Failed to serialize settings")?;

        fs::write(&path, content).context("Failed to write settings file")?;

        info!("📁 [SETTINGS] Settings saved to: {:?}", path);
        Ok(())
    }

    /// Update a specific field and save
    pub fn update<F>(&mut self, updater: F) -> Result<()>
    where
        F: FnOnce(&mut Self),
    {
        updater(self);
        self.save()
    }
}

/// Global settings instance
pub static SETTINGS: Lazy<RwLock<Settings>> = Lazy::new(|| RwLock::new(Settings::load()));

/// Get current settings (read-only clone)
pub fn get_settings() -> Settings {
    SETTINGS.read().unwrap().clone()
}

/// Get audio settings
pub fn get_audio_settings() -> AudioSettings {
    SETTINGS.read().unwrap().audio.clone()
}

/// Get preferred audio backend
pub fn get_preferred_backend() -> AudioBackend {
    SETTINGS.read().unwrap().audio.preferred_backend
}

/// Check if fallback is allowed
pub fn is_fallback_allowed() -> bool {
    SETTINGS.read().unwrap().audio.allow_fallback
}

/// Update settings with a closure
pub fn update_settings<F>(updater: F) -> Result<()>
where
    F: FnOnce(&mut Settings),
{
    let mut settings = SETTINGS.write().unwrap();
    updater(&mut settings);
    settings.save()
}

/// Get LLM settings
pub fn get_llm_settings() -> LlmSettings {
    SETTINGS.read().unwrap().llm.clone()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_settings() {
        let settings = Settings::default();
        assert_eq!(
            settings.audio.preferred_backend,
            AudioBackend::ScreenCaptureKit
        );
        assert!(settings.audio.allow_fallback);
        assert_eq!(settings.asr.language, "zh");
    }

    #[test]
    fn test_serialize_deserialize() {
        let settings = Settings::default();
        let json = serde_json::to_string_pretty(&settings).unwrap();
        println!("Default settings JSON:\n{}", json);

        let parsed: Settings = serde_json::from_str(&json).unwrap();
        assert_eq!(
            parsed.audio.preferred_backend,
            settings.audio.preferred_backend
        );
    }

    #[test]
    fn test_backend_aliases() {
        // Test that aliases work
        let json = r#"{"audio": {"preferred_backend": "sck"}}"#;
        let settings: Settings = serde_json::from_str(json).unwrap();
        assert_eq!(
            settings.audio.preferred_backend,
            AudioBackend::ScreenCaptureKit
        );

        let json = r#"{"audio": {"preferred_backend": "core_audio"}}"#;
        let settings: Settings = serde_json::from_str(json).unwrap();
        assert_eq!(settings.audio.preferred_backend, AudioBackend::CoreAudio);
    }
}
