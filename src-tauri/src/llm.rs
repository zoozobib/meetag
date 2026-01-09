use crate::history::SessionDetail;
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::OpenOptions;
use std::io::Write;
use tauri::Manager;

// =====================
// Data Structures
// =====================

#[derive(Debug, Serialize)]
struct OllamaOptions {
    num_ctx: u32,
    temperature: f32,
}

#[derive(Debug, Serialize)]
struct OllamaRequest {
    model: String,
    stream: bool,
    options: OllamaOptions,
    prompt: String,
}

#[derive(Debug, Deserialize)]
struct OllamaResponse {
    response: String,
    // we ignore other fields like done, context, etc.
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SummaryResult {
    success: bool,
    content: Option<String>,
    error: Option<String>,
}

// =====================
// Constants / Prompt
// =====================

// =====================
// Configuration for Prompts
// =====================

#[derive(Debug, Deserialize)]
struct PromptConfig {
    #[serde(flatten)]
    templates: HashMap<String, Vec<String>>,
}

fn load_runtime_config(filename: &str) -> PromptConfig {
    // Try to read relative to CWD (usually project root during dev)
    // Path: src-tauri/src/llm_config/<filename>
    let path = format!("src-tauri/src/llm_config/{}", filename);

    // Attempt to read file
    if let Ok(content) = std::fs::read_to_string(&path) {
        if let Ok(config) = serde_json::from_str(&content) {
            return config;
        } else {
            println!("[LLM Config] JSON parse error for {}", path);
        }
    } else {
        println!("[LLM Config] Failed to read file {}, using fallback.", path);
    }

    // Fallback logic
    match filename {
        "instructions.json" => {
            let json_str = include_str!("llm_config/instructions.json");
            serde_json::from_str(json_str).expect("Failed to parse fallback instructions")
        }
        "output_formats.json" => {
            let json_str = include_str!("llm_config/output_formats.json");
            serde_json::from_str(json_str).expect("Failed to parse fallback output_formats")
        }
        _ => panic!("Unknown config file: {}", filename),
    }
}

const OLLAMA_API_URL: &str = "http://localhost:11434/api/generate";
// Default model, can be made configurable later
const DEFAULT_MODEL: &str = "qwen3:4b";

fn build_prompt(transcript_text: &str) -> String {
    // 1. Get Instructions
    let instruction_config = load_runtime_config("instructions.json");
    let instruction_lines = instruction_config
        .templates
        .get("default")
        .unwrap_or_else(|| {
            panic!("Default instruction not found!");
        });
    let instruction = instruction_lines.join("\n");

    // 2. Get Output Formats
    let format_config = load_runtime_config("output_formats.json");
    let format_lines = format_config.templates.get("default").unwrap_or_else(|| {
        panic!("Default output format not found!");
    });
    let format_req = format_lines.join("\n");

    let prompt = format!(
        "{}\n\n{}\n\n----------------\n\n【会议录音转写文本】：\n{}\n",
        instruction, format_req, transcript_text
    );

    println!("[LLM Prompt Debug] Generated Prompt:\n{}", prompt);

    prompt
}

// =====================
// Commands
// =====================

#[tauri::command]
pub async fn generate_summary(
    app: tauri::AppHandle,
    session_id: String,
) -> Result<SummaryResult, String> {
    match generate_summary_inner(&app, &session_id).await {
        Ok(content) => Ok(SummaryResult {
            success: true,
            content: Some(content),
            error: None,
        }),
        Err(e) => Ok(SummaryResult {
            success: false,
            content: None,
            error: Some(e.to_string()),
        }),
    }
}

async fn generate_summary_inner(app: &tauri::AppHandle, session_id: &str) -> Result<String> {
    // 1. Resolve paths
    let base_app_data = app
        .path()
        .app_data_dir()
        .context("failed to get app data dir")?;
    let session_dir = base_app_data.join("sessions").join(session_id);
    if !session_dir.exists() {
        anyhow::bail!("Session directory not found: {}", session_id);
    }

    // 2. Load transcript
    // We reuse logic similar to history::get_session_detail, but we just want raw text for LLM
    let transcript_path = session_dir.join("transcript.jsonl");
    if !transcript_path.exists() {
        anyhow::bail!("Transcript file not found");
    }

    let file = std::fs::File::open(&transcript_path).context("failed to open transcript")?;
    let reader = std::io::BufReader::new(file);
    use std::io::BufRead;

    let mut full_text = String::new();
    for line in reader.lines() {
        let line = line?;
        if let Ok(val) = serde_json::from_str::<serde_json::Value>(&line) {
            // Assume format: {"text": "...", "speaker": "...", "timestamp": ...}
            // We'll format it as "[Speaker] Text"
            if let Some(text) = val.get("text").and_then(|v| v.as_str()) {
                if !text.trim().is_empty() {
                    // Optional: timestamp
                    // let ts = val.get("timestamp_start").and_then(|v| v.as_f64()).unwrap_or(0.0);
                    // let spk = val.get("speaker").and_then(|v| v.as_str()).unwrap_or("Unknown");
                    // full_text.push_str(&format!("[{:.1}s] {}: {}\n", ts, spk, text));
                    full_text.push_str(text);
                    full_text.push('\n');
                }
            }
        }
    }

    if full_text.trim().is_empty() {
        anyhow::bail!("Transcript is empty");
    }

    // 3. Call Ollama
    let client = reqwest::Client::new();
    let prompt = build_prompt(&full_text);

    let request_body = OllamaRequest {
        model: DEFAULT_MODEL.to_string(),
        stream: false,
        options: OllamaOptions {
            num_ctx: 4096,
            temperature: 0.3,
        },
        prompt,
    };

    let res = client
        .post(OLLAMA_API_URL)
        .json(&request_body)
        .send()
        .await
        .context("Failed to connect to Ollama. Is it running at localhost:11434?")?;

    if !res.status().is_success() {
        anyhow::bail!("Ollama API error: {}", res.status());
    }

    let ollama_res: OllamaResponse = res
        .json()
        .await
        .context("Failed to parse Ollama response")?;
    let summary = ollama_res.response;

    // 4. Save to summary.md
    let summary_path = session_dir.join("summary.md");
    let mut f = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(&summary_path)
        .context("Failed to open summary.md for writing")?;

    f.write_all(summary.as_bytes())?;

    Ok(summary)
}

#[tauri::command]
pub fn save_summary(
    app: tauri::AppHandle,
    session_id: String,
    content: String,
) -> Result<(), String> {
    let base_app_data = app.path().app_data_dir().map_err(|e| e.to_string())?;
    let session_dir = base_app_data.join("sessions").join(session_id);
    let summary_path = session_dir.join("summary.md");

    let mut f = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(summary_path)
        .map_err(|e| e.to_string())?;

    f.write_all(content.as_bytes()).map_err(|e| e.to_string())?;
    Ok(())
}

#[tauri::command]
pub fn get_summary(app: tauri::AppHandle, session_id: String) -> Result<String, String> {
    let base_app_data = app.path().app_data_dir().map_err(|e| e.to_string())?;
    let session_dir = base_app_data.join("sessions").join(session_id);
    let summary_path = session_dir.join("summary.md");

    if !summary_path.exists() {
        return Ok("".to_string());
    }

    std::fs::read_to_string(summary_path).map_err(|e| e.to_string())
}
