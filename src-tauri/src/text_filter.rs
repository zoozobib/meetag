use lazy_static::lazy_static;
use regex::Regex;
use std::collections::HashSet;

lazy_static! {
    /// List of phrases that trigger hallucination detection if the text CONTAINS them (case-insensitive)
    static ref HALLUCINATION_SUBSTRINGS: Vec<&'static str> = vec![
        "thank you for watching",
        "subtitle",
        "copyright",
        "by the author",
        "subtitles",
        "captioned by",
        "字幕製作",
        "字幕:",
        "j chong",
        "謝謝收看",
        "謝謝大家收看",
        "下次再見",
        "謝謝觀看",
        "谢谢观看",
        "謝謝大家",
        "谢谢大家",
        "下集見",
        "下集见",
        "字幕君",
        "中文字幕",
        "请使用规范的书面语进行转写",
        "转写。",
        "..."
    ];

    /// List of exact phrases that are known hallucinations (exact match only)
    static ref HALLUCINATION_EXACT: Vec<&'static str> = vec![
        "The.",
        "I.",
        "You.",
        "He.",
        "She.",
        "It.",
        "We.",
        "They.",
        "Okay.",
        "Yeah.",
        "Yes.",
        "No.",
        "Oh.",
        "Ah.",
        "Hmm.",
        "Um.",
        "Uh.",
        "So.",
        "And.",
        "But.",
        "Or.",
        "If.",
        "When.",
        "Where.",
        "Why.",
        "How.",
        "Who.",
        "What.",
        "Thank.",
        "Thanks.",
        "Hello.",
        "Hi.",
        "Bye.",
        "Good.",
        "Bad.",
        "Right.",
        "Wrong.",
        "True.",
        "False.",
        "Maybe.",
        "Perhaps.",
        "Sure.",
        "Fine.",
        "Well.",
        "Now.",
        "Then.",
        "Here.",
        "There.",
        "This.",
        "That.",
        "These.",
        "Those.",
        "One.",
        "Two.",
        "Three.",
        "Four.",
        "Five.",
        "Six.",
        "Seven.",
        "Eight.",
        "Nine.",
        "Ten.",
        "To.",
        "For.",
        "Of.",
        "In.",
        "On.",
        "At.",
        "By.",
        "With.",
        "From.",
        "About.",
        "As.",
        "Like.",
        "Up.",
        "Down.",
        "Out.",
        "Over.",
        "Under.",
        "Again.",
        "Always.",
        "Never.",
        "Sometimes.",
        "Often.",
        "Usually.",
        "Really.",
        "Very.",
        "Too.",
        "Quite.",
        "Just.",
        "Only.",
        "Even.",
        "Still.",
        "Yet.",
        "Already.",
        "Almost.",
        "Nearly.",
        "Enough.",
        "More.",
        "Less.",
        "Most.",
        "Least.",
        "Best.",
        "Worst.",
        "Better.",
        "Worse.",
        "Great.",
        "Excellent.",
        "Wonderful.",
        "Amazing.",
        "Awesome.",
        "Beautiful.",
        "Nice.",
        "Cool.",
        "Fun.",
        "Interesting.",
        "Boring.",
        "Tired.",
        "Busy.",
        "Happy.",
        "Sad.",
        "Angry.",
        "Scared.",
        "Surprised.",
        "Excited.",
        "Nervous.",
        "Worried.",
        "Confused.",
        "Proud.",
        "Ashamed.",
        "Guilty.",
        "Jealous.",
        "Envious.",
        "Lonely.",
        "Loved.",
        "Hated.",
        "Hope.",
        "Wish.",
        "Want.",
        "Need.",
        "Love.",
        "Hate.",
        "Think.",
        "Know.",
        "Understand.",
        "Believe.",
        "Feel.",
        "See.",
        "Hear.",
        "Smell.",
        "Taste.",
        "Touch.",
        "Do.",
        "Make.",
        "Get.",
        "Give.",
        "Go.",
        "Come.",
        "Take.",
        "Put.",
        "Say.",
        "Tell.",
        "Ask.",
        "Answer.",
        "Call.",
        "Talk.",
        "Speak.",
        "Write.",
        "Read.",
        "Study.",
        "Learn.",
        "Teach.",
        "Buy.",
        "Sell.",
        "Pay.",
        "Cost.",
        "Work.",
        "Play.",
        "Rest.",
        "Sleep.",
        "Eat.",
        "Drink.",
        "Run.",
        "Walk.",
        "Drive.",
        "Fly.",
        "Ride.",
        "Swim.",
        "Jump.",
        "Dance.",
        "Sing.",
        "Laugh.",
        "Cry.",
        "Smile.",
        "Frown.",
        "Look.",
        "Watch.",
        "Listen.",
        "Wait.",
        "Stop.",
        "Start.",
        "Begin.",
        "Finish.",
        "End.",
        "Open.",
        "Close.",
        "Win.",
        "Lose.",
        "Help.",
        "Save.",
        "Kill.",
        "Die.",
        "Live.",
        "Born.",
        "Grow.",
        "Change.",
        "Stay.",
        "Leave.",
        "Meet.",
        "Visit.",
        "Travel.",
        "Move.",
        "Return.",
        "Arrive.",
        "Depart.",
        "Enter.",
        "Exit.",
        "Join.",
        "Quit.",
        "Add.",
        "Subtract.",
        "Multiply.",
        "Divide.",
        "Cut.",
        "Copy.",
        "Paste.",
        "Print.",
        "Delete.",
        "Search.",
        "Find.",
        "Replace.",
        "Undo.",
        "Redo.",
        "Select.",
        "Cancel.",
        "Apply.",
        "OK.",
        "Off.",
        "High.",
        "Low.",
        "Medium.",
        "Large.",
        "Small.",
        "Full.",
        "Empty.",
        "Old.",
        "Young.",
        "Rich.",
        "Poor.",
        "Strong.",
        "Weak.",
        "Fast.",
        "Slow.",
        "Hot.",
        "Cold.",
        "Warm.",
        "Dry.",
        "Wet.",
        "Hard.",
        "Soft.",
        "Rough.",
        "Smooth.",
        "Sharp.",
        "Dull.",
        "Bright.",
        "Dark.",
        "Light.",
        "Heavy.",
        "Clean.",
        "Dirty.",
        "Neat.",
        "Messy.",
        "Safe.",
        "Dangerous.",
        "Quiet.",
        "Noisy.",
        "Silent.",
        "Loud.",
        "Sweet.",
        "Sour.",
        "Bitter.",
        "Salty.",
        "Spicy.",
        "Fresh.",
        "Stale.",
        "Real.",
        "Fake.",
        "Same.",
        "Different.",
        "Similar.",
        "Opposite.",
        "Equal.",
        "Unequal.",
        "Free.",
        "Available.",
        "Unavailable.",
        "Closed.",
        "Public.",
        "Private.",
        "General.",
        "Specific.",
        "Common.",
        "Rare.",
        "Usual.",
        "Unusual.",
        "Normal.",
        "Abnormal.",
        "Strange.",
        "Weird.",
        "Funny.",
        "Serious.",
        "Important.",
        "Unimportant.",
        "Necessary.",
        "Unnecessary.",
        "Useful.",
        "Useless.",
        "Helpful.",
        "Unhelpful.",
        "Kind.",
        "Cruel.",
        "Polite.",
        "Rude.",
        "Friendly.",
        "Hostile.",
        "Honest.",
        "Dishonest.",
        "Loyal.",
        "Disloyal.",
        "Brave.",
        "Cowardly.",
        "Smart.",
        "Stupid.",
        "Wise.",
        "Foolish.",
        "Clever.",
        "Clumsy.",
        "Lucky.",
        "Unlucky.",
        "Healthy.",
        "Sick.",
        "Ugly.",
        "Used.",
        "Ting.",
        "Com.",
        "Hpe.",
        "Co.",
        "De.",
        "S.",
        "F.",
        "lish.",
    ];

    /// Regex for patterns like "The..." or "Okay..."
    static ref HALLUCINATION_REGEX: Regex = Regex::new(r"^(The|Yeah|Okay|Yes|No|Oh|Ah|Um|So)\.+$").unwrap();
}

/// Checks if the text is a known hallucination.
pub fn is_hallucination(text: &str) -> bool {
    let t = text.trim();

    // 1. Check exact matches
    if HALLUCINATION_EXACT.contains(&t) {
        return true;
    }

    // 2. Check regex patterns (The..., Yeah...)
    if HALLUCINATION_REGEX.is_match(t) {
        return true;
    }

    let lower_text = t.to_lowercase();

    // 3. Check substrings (Subtitle garbage)
    for phrase in HALLUCINATION_SUBSTRINGS.iter() {
        if lower_text.contains(phrase) {
            return true;
        }
    }

    // 4. Check for specific repetitive characters (e.g. ".....")
    // Also punctuation only check from previous version
    let unique_chars: HashSet<char> = t.chars().collect();
    if unique_chars.iter().all(|c| !c.is_alphanumeric()) {
        return true;
    }

    // 5. Check for very short nonsensical output (e.g. "a.")
    if t.len() <= 3 && (t.ends_with('.') || t.ends_with('。')) {
        // Allow "No." / "Hi." but block single letters "I." "A."
        if t.len() == 2 && t.chars().next().unwrap().is_alphabetic() {
            return true; // "X."
        }
    }

    // 6. Excessive Repetition Check (Restored)
    if has_excessive_repetition(&lower_text) {
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

            if current_repeats > 1 {
                i = j;
            } else {
                i += 1;
            }
        }

        // Threshold: 6 repeats for single chars, 4 for longer patterns
        let threshold = if pat_len == 1 { 6 } else { 4 };

        if max_count >= threshold {
            return true;
        }
    }

    false
}
