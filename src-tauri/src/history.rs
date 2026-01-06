use std::fs::File;
use tauri::Manager;

#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct SessionInfo {
    id: String,
    date: String,
    preview: String,
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct SessionDetail {
    id: String,
    transcript: Vec<serde_json::Value>,
}

#[tauri::command]
pub fn get_sessions(app: tauri::AppHandle) -> Result<Vec<SessionInfo>, String> {
    let base_app_data = app.path().app_data_dir().map_err(|e| e.to_string())?;
    let sessions_dir = base_app_data.join("sessions");

    if !sessions_dir.exists() {
        return Ok(vec![]);
    }

    let mut sessions = Vec::new();
    let entries = std::fs::read_dir(sessions_dir).map_err(|e| e.to_string())?;

    for entry in entries {
        let entry = entry.map_err(|e| e.to_string())?;
        let path = entry.path();
        if path.is_dir() {
            if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                // Name is YYYY-MM-DD_HH-mm-ss
                // We format it a bit nicer for UI
                let date_str = name.replace("_", " ");
                sessions.push(SessionInfo {
                    id: name.to_string(),
                    date: date_str,
                    preview: "Click to view transcript".to_string(),
                });
            }
        }
    }
    // Sort desc (newest first)
    sessions.sort_by(|a, b| b.id.cmp(&a.id));
    Ok(sessions)
}

#[tauri::command]
pub fn get_session_detail(app: tauri::AppHandle, id: String) -> Result<SessionDetail, String> {
    // Basic security check
    if id.contains("..") || id.contains("/") || id.contains("\\") {
        return Err("Invalid session ID".into());
    }

    let base_app_data = app.path().app_data_dir().map_err(|e| e.to_string())?;
    let session_dir = base_app_data.join("sessions").join(&id);
    let transcript_path = session_dir.join("transcript.jsonl");

    let mut transcript = Vec::new();
    if transcript_path.exists() {
        let file = File::open(transcript_path).map_err(|e| e.to_string())?;
        let reader = std::io::BufReader::new(file);
        use std::io::BufRead;
        for line in reader.lines() {
            let line = line.map_err(|e| e.to_string())?;
            if let Ok(val) = serde_json::from_str::<serde_json::Value>(&line) {
                transcript.push(val);
            }
        }
    }

    Ok(SessionDetail { id, transcript })
}
