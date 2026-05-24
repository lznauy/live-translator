# 设计文档

## 技术栈

| 组件 | 库 | 说明 |
|------|-----|------|
| 音频捕获 | cpal + pw-cat fallback | 抓 PipeWire/PulseAudio 系统声音，16kHz mono |
| VAD | 自实现 | 能量阈值检测（RMS > 0.005），切语音段 |
| ASR | candle-transformers | Whisper small ~923MB，支持 en/zh/ja/ko |
| NMT | candle-transformers | Marian OPUS-MT en→zh ~300MB |

## 架构

```
系统声音 (monitor)
    │ 16kHz mono, 512-sample chunks
    ▼
audio.rs ──→ vad.rs ──→ asr.rs (Whisper) ──→ translator.rs (Marian)
                            │                        │
                        partial 预览           后台线程并行翻译
                            │                        │
                            └────────┬───────────────┘
                                     ▼
                              stdout JSON / stderr 日志
```

## 文件结构

```
src/
├── main.rs         # 入口，CLI，stdin/stdout 线程
├── protocol.rs     # JSON 命令/事件定义
├── audio.rs        # 音频采集，RMS 归一化
├── vad.rs          # 能量阈值 VAD，语音段切分
├── asr.rs          # Whisper 模型加载、Mel 频谱、编码解码
├── translator.rs   # Marian 翻译模型
├── pipeline.rs     # 主循环，状态机，线程协调
└── log.rs          # --quiet 纯净输出模式
```

## Pipeline 状态机

```
Idle ──Start──→ Listening ──VAD 检测到说话──→ Speaking
  ↑                    ↑                          │
  └───Stop─────────────┘                VAD 检测到静音 / 超 10s
                                              │
                                              ▼
                                    切出 segment → ASR → 翻译 → 输出 Final
                                              │
                                         speech_buffer 清理 → Listening
```

## 解码策略

- **partial**：每 500ms-1s，取 speech_buffer 最近 5 秒送 ASR，提供实时字幕预览
- **final**：VAD 切出的完整 segment（最长 10s）送 ASR，结果送翻译线程
- 低置信度（avg_logprob < -3.0）的 partial 结果被过滤
- Whisper 格式 artifact（`(laughs)` 等）自动清理

## 并行设计

- ASR 和 NMT 跑在独立线程，通过 channel 通信
- 翻译 segment_1 的同时，ASR 处理 segment_2
- 语言切换时：set_language() 免重载 ASR 模型，清空翻译队列后重载翻译模型
