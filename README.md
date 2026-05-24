# live-translator

实时语音识别+翻译，纯本地运行。Whisper small（ASR）+ Marian OPUS-MT（NMT），基于 candle-transformers（纯 Rust，CPU）。

## 快速开始

```shell
# 1. 下载模型（~1.2 GB）
./scripts/setup-models.sh

# 2. 编译
./scripts/build.sh

# 3. 启动（抓系统声音，英译中）
./target/release/live-translator --quiet --auto-start --audio-device monitor
```

## 模型

```
models/
├── whisper-small/        # 923 MB, Whisper small (candle safetensors)
│   ├── model.safetensors
│   ├── config.json
│   └── vocab.json
└── opus-mt-en-zh/       # 296 MB, Marian OPUS-MT 英→中 (candle safetensors)
    ├── model.safetensors
    ├── config.json
    └── vocab.json
```

## 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--asr-model` | `models/whisper-small` | Whisper 模型路径 |
| `--translate-model` | `models/opus-mt-en-zh` | 翻译模型路径 |
| `--audio-device` | 空 | 音频设备名，填 `monitor` 抓系统声音 |
| `--auto-start` | — | 启动后自动开始监听 |
| `--auto-src` | `en` | 源语言 |
| `--auto-tgt` | `zh` | 目标语言 |
| `--quiet` | — | 纯净模式：只输出 "原文\t翻译"，无调试日志 |
| `--vad-threshold` | `0.005` | VAD 能量阈值 |
| `--list-devices` | — | 列出音频设备后退出 |
| `--test-wav` | — | 测试模式：识别 WAV 文件后退出 |

## 使用方法

### 纯净模式（只输出识别+翻译）

```shell
./target/release/live-translator --quiet --auto-start --audio-device monitor
```

输出格式：
```
These chocolate eggs	这些巧克力蛋
Children will do Easter egg hunts	孩子们会做复活节彩蛋狩猎
```

### 调试模式（看详细日志）

```shell
./target/release/live-translator --auto-start --audio-device monitor
```

### 测试 WAV 文件

```shell
./target/release/live-translator --quiet --test-wav audio.wav
```

## 协议

stdin 接受 JSON 命令（无需 `--auto-start` 时使用）：

```json
{"type":"start","src":"en","tgt":"zh"}
{"type":"stop"}
{"type":"quit"}
```

| 语言 | 代码 |
|------|------|
| 英语 | `en` |
| 中文 | `zh` |
| 日语 | `ja` |
| 韩语 | `ko` |

stdout 输出 JSON 事件：

```json
{"type":"partial","text":"hello","lang":"en","confidence":-0.85}
{"type":"final","text":"hello world","translated":"你好世界","lang":"en","confidence":-0.11}
{"type":"status","state":"listening"}
```

## 性能（whisper-small，CPU）

| 阶段 | 耗时 |
|------|------|
| 音频采集 + VAD | < 10ms |
| ASR（10s 音频段） | ~6-9s |
| 翻译 | ~0.5s |

ASR 与翻译在独立线程并行，互不阻塞。

## 架构文档

详见 [ARCHITECTURE.md](ARCHITECTURE.md)。
