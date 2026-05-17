#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use tauri::{Listener, Manager};

mod asr;
mod audio;
mod audio_processor;
mod capture;
mod diarization;
mod funasr;
mod history;
mod llm;
mod recorder;
mod settings;
mod text_filter;
mod tray;
mod vad;
mod wav;
mod whisper;

fn main() {
    // Initialize logging to enable info!() macros in CoreAudio code
    std::env::set_var("RUST_LOG", "info");
    env_logger::init();
    log::info!("Starting application...");

    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .on_window_event(|window, event| {
            if let tauri::WindowEvent::CloseRequested { api, .. } = event {
                // For now, adhere to the user's existing logic (hide on close),
                // but this contributes to the "zombie process" feeling if not careful.
                // However, the main fix is ensuring real exit kills the child.
                window.hide().unwrap();
                api.prevent_close();
            }
        })
        .setup(|app| {
            let handle = app.handle().clone();

            // Initialize settings on startup
            // This ensures the settings file is created if it doesn't exist
            settings::get_settings();

            // Init Tray
            tray::create_tray(&handle)?;

            // Listen for Tray Events
            let h1 = handle.clone();
            handle.listen("tray-open-window", move |_| {
                if let Some(window) = h1.get_webview_window("main") {
                    let _ = window.show();
                    let _ = window.set_focus();
                }
            });

            let h2 = handle.clone();
            handle.listen("tray-record-start", move |_| {
                println!("▶ tray request: start_recording...");
                let _ = recorder::start_recording(h2.clone());
            });

            let h3 = handle.clone();
            handle.listen("tray-record-stop", move |_| {
                println!("▶ tray request: stop_recording...");
                let app_handle = h3.clone();
                tauri::async_runtime::spawn(async move {
                    let _ = recorder::stop_recording(app_handle).await;
                });
            });

            tauri::async_runtime::spawn(async move {
                let resource_path = handle
                    .path()
                    .resolve(
                        "resources/ggml-large-v3-turbo-q8_0.bin",
                        tauri::path::BaseDirectory::Resource,
                    )
                    .unwrap();

                // Initialize whisper-rs directly (no sidecar needed)
                println!("🎙️ Initializing Whisper model...");
                if let Err(e) = crate::whisper::WhisperManager::init(&resource_path) {
                    eprintln!("❌ Failed to initialize Whisper: {}", e);
                }

                // Initialize FunASR (SenseVoice) model
                let resource_base = handle
                     .path()
                     .resolve("resources", tauri::path::BaseDirectory::Resource)
                     .unwrap();
                
               println!("🚀 [MAIN] Initializing FunASR...");
                if let Err(e) = crate::funasr::SenseVoiceManager::init(&resource_base) {
                     eprintln!("⚠️ Failed to initialize FunASR: {}", e);
                     eprintln!("   (This is expected if SenseVoice model files are not downloaded)");
                }

                // Initialize Speaker Diarization pipeline (v8.0: Offline Post-Processing)
                println!("🎙️ [MAIN] Initializing Speaker Diarization pipeline...");
                let segmentation_model = resource_base.join("sherpa-onnx-pyannote-segmentation-3-0/model.onnx");
                let embedding_model = resource_base.join("speaker_embedding/3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx");
                if let Err(e) = crate::diarization::init(&segmentation_model, &embedding_model) {
                    eprintln!("⚠️ Failed to initialize DiarizationPipeline: {}", e);
                    eprintln!("   Segmentation model: {}", segmentation_model.display());
                    eprintln!("   Embedding model: {}", embedding_model.display());
                }
            });
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            recorder::start_recording,
            recorder::stop_recording,
            history::get_sessions,
            history::get_session_detail,
            llm::generate_summary,
            llm::save_summary,
            llm::get_summary,
            recorder::add_manual_transcript
        ])
        .build(tauri::generate_context!())
        .expect("error while building tauri application")
        .run(|_app_handle, event| {
            if let tauri::RunEvent::Exit = event {
                println!("🛑 Application exiting, Whisper resources will be released.");
            }
        });
}
