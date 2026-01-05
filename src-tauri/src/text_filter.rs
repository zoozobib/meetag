use std::collections::HashSet;

pub fn is_hallucination(text: &str) -> bool {
    let lower_text = text.trim().to_lowercase();

    // 1. Empty check
    if lower_text.is_empty() {
        return true;
    }

    // 2. Blacklist Filtering
    let blacklist = [
        "thank you for watching",
        "subtitle",
        "copyright",
        "...",
        // Subtitle specific hallucinations
        "by the author",
        "subtitles",
        "captioned by",
        "字幕製作",
        "字幕:",
        "j chong",
        "謝謝收看",
        "謝謝大家收看",
        "下次再見",
    ];

    for phrase in blacklist {
        if lower_text.contains(phrase) {
            println!("🛑 Filtered Hallucination (Blacklist): {:?}", text);
            return true;
        }
    }

    // 3. Punctuation only check
    let unique_chars: HashSet<char> = lower_text.chars().collect();
    if unique_chars.iter().all(|c| !c.is_alphanumeric()) {
        println!("🛑 Filtered Hallucination (Punctuation Only): {:?}", text);
        return true;
    }

    // 4. Repetition Detection
    // Check for 3+ consecutive repetitions of the same word/phrase
    // Simple heuristic: split by space, check sliding window of 3
    // Also handle Chinese characters repetition if needed, but let's start with general split.
    // For Chinese, "测试测试测试" is one string usually.
    // We can check for repeated substrings.

    if has_excessive_repetition(&lower_text) {
        println!("🛑 Filtered Hallucination (Repetition): {:?}", text);
        return true;
    }

    false
}

fn has_excessive_repetition(text: &str) -> bool {
    // Check for repeated sequences like "abcabcabc" (length >= 2, count >= 3)
    let n = text.chars().count();
    if n < 4 {
        return false;
    }

    let chars: Vec<char> = text.chars().collect();

    // Check pattern lengths from 1 to n/3
    for pat_len in 1..=n / 3 {
        let mut count = 0;
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

            // Optimization: skip the repeats we just found
            if current_repeats > 1 {
                i = j;
            } else {
                i += 1;
            }
        }

        // Threshold: 3 repeats for longer patterns, 4 for single chars
        let threshold = if pat_len == 1 { 4 } else { 3 };

        if max_count >= threshold {
            return true;
        }
    }

    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_blacklist() {
        assert!(is_hallucination("Thank you for watching"));
        assert!(!is_hallucination("Hello world"));
    }

    #[test]
    fn test_punctuation() {
        assert!(is_hallucination("..."));
        assert!(is_hallucination("?!"));
        assert!(!is_hallucination("Hi!"));
    }

    #[test]
    fn test_repetition() {
        assert!(is_hallucination("测试测试测试测试"));
        assert!(is_hallucination("abcabcabc"));
        assert!(!is_hallucination("abcabc")); // 2 is fine
        assert!(!is_hallucination("This is a test"));
    }
}
