use anyhow::Result;
use sherpa_rs::sense_voice::{SenseVoiceConfig, SenseVoiceRecognizer};

fn main() -> Result<()> {
    // 0. 检查命令行参数
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        println!("Usage: cargo run --example sensevoice_demo <path_to_wav_file>");
        println!("Example: cargo run --example sensevoice_demo test_audio.wav");
        return Ok(());
    }
    let wav_path = &args[1];

    println!("🚀 Initializing SenseVoiceSmall model...");
    println!("📦 Assuming model files are in 'resources/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/'");

    // 1. 配置 SenseVoice 模型路径
    // 注意：需要用户先下载这些文件到对应目录
    // 下载地址: https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2
    let config = SenseVoiceConfig {
        model: "resources/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/model.int8.onnx"
            .into(),
        tokens: "resources/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/tokens.txt".into(),
        language: "".to_string(), // Empty for auto-detect or default
        use_itn: true,
        provider: None,
        num_threads: Some(4),
        debug: false,
    };

    // 2. 创建识别器 (Recognizer)
    let mut recognizer = SenseVoiceRecognizer::new(config).map_err(|e| anyhow::anyhow!("{}", e))?;
    println!("✅ Model loaded successfully!");

    // 3. 读取音频文件
    // SenseVoice 要求 16kHz 采样率
    println!("🎤 Reading audio file: {}", wav_path);
    let mut reader = hound::WavReader::open(wav_path).expect("Failed to open WAV file");
    let spec = reader.spec();

    if spec.sample_rate != 16000 {
        eprintln!(
            "⚠️ Warning: Sample rate is {} Hz. SenseVoice expects 16000 Hz.",
            spec.sample_rate
        );
        eprintln!("⚠️ The validation might fail or produce garbage if not resampled.");
    }

    let samples: Vec<i16> = reader.samples::<i16>().filter_map(Result::ok).collect();
    // 转换为 float32 并归一化 [-1, 1]
    let samples_f32: Vec<f32> = samples.iter().map(|&s| s as f32 / 32768.0).collect();

    println!("⏳ Transcribing ({} samples)...", samples_f32.len());
    let start = std::time::Instant::now();

    // 4. 执行推理
    let result = recognizer.transcribe(16000, &samples_f32);

    let duration = start.elapsed();
    let audio_duration = samples_f32.len() as f32 / 16000.0;

    println!("\n========================================");
    println!("📝 Transcription Result:");
    println!("========================================");
    println!("{}", result.text);
    println!("========================================");

    println!(
        "⏱️  Inference Time: {:.2?} (Audio Duration: {:.2}s)",
        duration, audio_duration
    );
    println!(
        "⚡ Real-time Factor: {:.2}x",
        audio_duration / duration.as_secs_f32()
    );

    Ok(())
}
