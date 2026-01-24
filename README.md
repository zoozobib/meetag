# Meetily Rec (Tauri Edition)

<div align="center">

**一个专注在 macOS (Apple Silicon) 上运行的高性能会议记录与总结工具**

[English](./README_EN.md) | [简体中文](./README.md)

</div>

## 🙏 致谢

本项目核心代码参考并致敬开源项目 **[meetily](https://github.com/meetily/meetily)**。感谢原作者的无私分享与卓越贡献！本项目在此基础上进行了 Tauri 移植与特定功能增强。

## 🏗️ 架构与功能

本项目基于 **Tauri v2** + **Rust** 构建，利用 macOS 原生能力实现极致的性能与体验。

### 核心功能
*   **🔊 系统级录音**: 支持 **ScreenCaptureKit** (macOS 12.3+) 高性能内录，及 CoreAudio 传统录制。
*   **🎙️ 混合录制**: 自动混音系统声音（会议内容）与麦克风声音（你的发言），完美还原会议全貌。
*   **🤖 双引擎语音识别 (ASR)**:
    *   **SenseVoice (FunASR)**: ⚡️ **极速**、高精度的中文识别，支持 Int8 量化，纯 CPU 推理也如闪电般迅速。
    *   **Whisper**: 经典的 OpenAI 模型支持，适合多语言场景。
*   **📝 智能摘要**: 集成 **Ollama** 本地大模型，一键生成会议纪要、Action Items。
*   **🧠 智能过滤**: 内置抗幻觉（Anti-Hallucination）过滤器，消除静音段的 "The." 等噪音干扰。
*   **📦 零配置分发**: 解决了 macOS 严苛的动态库打包问题，下载即用。

---

## 🛠️ 本地部署与开发

> ⚠️ **注意**: 本项目目前 **仅支持 Apple Silicon (M1/M2/M3)** 架构的 macOS 设备。

### 1. 环境准备
*   Rust (最新的 stable 版本)
*   Node.js & pnpm
*   Tauri CLI (`cargo install tauri-cli`)

### 2. 克隆项目
```bash
git clone https://github.com/your-repo/rec.git
cd rec
pnpm install
```

### 3. 下载模型文件 (关键!)
由于模型文件体积较大，未包含在 git 仓库中。你需要手动下载并放置到指定目录：

#### A. 语音识别模型 (ASR)
请将模型放入 `src-tauri/resources/` 目录：

1.  **SenseVoice (推荐)**
    *   下载地址: [Sherpa-onnx SenseVoice Int8](https://github.com/k2-fsa/sherpa-onnx/releases) (找 `sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17`)
    *   路径: `src-tauri/resources/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/`
        *   `model.int8.onnx`
        *   `tokens.txt`

2.  **Whisper (可选)**
    *   下载地址: [HuggingFace ggml-base](https://huggingface.co/ggerganov/whisper.cpp)
    *   路径: `src-tauri/resources/ggml-small.bin` (或其他规格)

#### B. 动态库 (Dependencies)
确保 `src-tauri/` 根目录下存在以下库文件（已通过脚本自动处理，但手动开发需注意）：
*   `libonnxruntime.1.17.1.dylib`
*   `libsherpa-onnx-c-api.dylib`

#### C. 摘要模型 (LLM)
请安装 [Ollama](https://ollama.com/) 并拉取你喜欢的模型：
```bash
ollama run qwen2.5:7b  # 推荐使用通义千问或其他中文能力强的模型
```

### 4. 运行与构建

**开发模式 (Debug)**:
*会自动修复库路径并打开 App*
```bash
cd src-tauri
./build.sh --debug
```

**发布构建 (Release)**:
*生成可分发的 .dmg*
```bash
cd src-tauri
./build.sh
```

---

## ⚙️ 配置文件 (settings.json)

配置文件位于 `~/Library/Application Support/com.zoozobib.rec/settings.json` (或 `~/.meetily/settings.json`)。

```json
{
  "audio": {
    // 音频捕获后端: "sck" (ScreenCaptureKit) 或 "core_audio"
    "preferred_backend": "sck",
    // 允许自动降级
    "allow_fallback": true
  },
  "asr": {
    // 识别语言
    "language": "zh",
    // ASR 引擎: "FunAsr" (SenseVoice) 或 "Whisper"
    "backend": "FunAsr"
  },
  "llm": {
      // Ollama 的 API 地址
      "host": "http://localhost:11434",
      // 使用的模型名称
      "model": "qwen2.5:7b"
  }
}
```

## 📄 开源协议

MIT License. 完全开源。
