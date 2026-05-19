//! Hallucination filter — detects and discards ASR artifacts.
//!
//! Word lists are loaded from `resources/hallucination_filter.json` at startup.
//! If the file is missing, a compiled-in default is used as fallback.

use once_cell::sync::Lazy;
use regex::Regex;
use serde::Deserialize;
use std::collections::HashSet;

/// JSON schema for the external filter word list
#[derive(Deserialize)]
struct FilterData {
    substrings: Vec<String>,
    exact: Vec<String>,
}

/// Compiled-in fallback (minimal) in case the JSON is missing
const FALLBACK_JSON: &str = include_str!("../resources/hallucination_filter.json");

static FILTER: Lazy<FilterData> = Lazy::new(|| {
    // Try loading from the resource directory at runtime
    if let Ok(exe) = std::env::current_exe() {
        // In a bundled .app: Contents/MacOS/app → Contents/Resources/resources/
        let candidates = [
            exe.parent()
                .and_then(|p| p.parent())
                .map(|p| p.join("Resources/resources/hallucination_filter.json")),
            // Dev mode: alongside the binary
            exe.parent()
                .map(|p| p.join("resources/hallucination_filter.json")),
        ];
        for candidate in candidates.iter().flatten() {
            if candidate.exists() {
                if let Ok(data) = std::fs::read_to_string(candidate) {
                    if let Ok(filter) = serde_json::from_str::<FilterData>(&data) {
                        println!(
                            "✅ [TEXT_FILTER] Loaded {} substrings + {} exact from {}",
                            filter.substrings.len(),
                            filter.exact.len(),
                            candidate.display()
                        );
                        return filter;
                    }
                }
            }
        }
    }

    // Fallback to compiled-in copy
    println!("⚠️ [TEXT_FILTER] Using compiled-in fallback word lists");
    serde_json::from_str(FALLBACK_JSON).expect("compiled-in fallback JSON must be valid")
});

/// Pre-built HashSet for O(1) exact lookups
static EXACT_SET: Lazy<HashSet<String>> = Lazy::new(|| {
    FILTER.exact.iter().cloned().collect()
});

/// Regex for patterns like "The..." or "Okay..."
static HALLUCINATION_REGEX: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"^(The|Yeah|Okay|Yes|No|Oh|Ah|Um|So)\.+$").unwrap());

/// Checks if the text is a known hallucination.
pub fn is_hallucination(text: &str) -> bool {
    let t = text.trim();

    // 1. Check exact matches (O(1) with HashSet)
    if EXACT_SET.contains(t) {
        return true;
    }

    // 2. Check regex patterns (The..., Yeah...)
    if HALLUCINATION_REGEX.is_match(t) {
        return true;
    }

    let lower_text = t.to_lowercase();

    // 3. Check substrings (Subtitle garbage)
    for phrase in FILTER.substrings.iter() {
        if lower_text.contains(phrase.as_str()) {
            return true;
        }
    }

    // 4. Check for punctuation-only text
    let unique_chars: HashSet<char> = t.chars().collect();
    if unique_chars.iter().all(|c| !c.is_alphanumeric()) {
        return true;
    }

    // 5. Check for very short nonsensical output (e.g. "a.")
    if t.len() <= 3 && (t.ends_with('.') || t.ends_with('。')) {
        if t.len() == 2 && t.chars().next().unwrap().is_alphabetic() {
            return true; // "X."
        }
    }

    // 6. Excessive Repetition Check
    if has_excessive_repetition(&lower_text) {
        return true;
    }

    false
}

fn has_excessive_repetition(text: &str) -> bool {
    let n = text.chars().count();
    if n < 4 {
        return false;
    }

    let chars: Vec<char> = text.chars().collect();

    for pat_len in 1..=n / 3 {
        let mut max_count = 0;
        let mut i = 0;

        while i + pat_len <= n {
            let pattern = &chars[i..i + pat_len];
            let mut j = i + pat_len;
            let mut current_repeats = 1;

            while j + pat_len <= n {
                if &chars[j..j + pat_len] == pattern {
                    current_repeats += 1;
                    j += pat_len;
                } else {
                    break;
                }
            }

            if current_repeats > max_count {
                max_count = current_repeats;
            }

            if current_repeats > 1 {
                i = j;
            } else {
                i += 1;
            }
        }

        let repeated_chars = max_count * pat_len;
        let proportion = repeated_chars as f32 / n as f32;
        
        let threshold = if pat_len == 1 { 6 } else { 4 };
        if max_count >= threshold && proportion > 0.8 {
            return true;
        }
    }

    false
}
