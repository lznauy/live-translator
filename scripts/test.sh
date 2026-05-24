#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL_DIR="${MODEL_DIR:-./models}"
BIN="${BIN:-./target/release/live-translator}"
ASR_MODEL="${ASR_MODEL:-$MODEL_DIR/whisper-tiny}"
MARIAN_MODEL="${MARIAN_MODEL:-$MODEL_DIR/opus-mt-en-zh}"

if ! [ -x "$BIN" ]; then
    echo "二进制不存在，先运行 scripts/build.sh"
    exit 1
fi

echo "==> 检查模型文件..."
missing=""
for f in "$ASR_MODEL/model.safetensors" "$MARIAN_MODEL/model.safetensors"; do
    if [ -f "$f" ]; then
        echo "  OK  $f"
    else
        echo "  !! $f 缺失"
        missing=1
    fi
done
if [ -n "$missing" ]; then
    echo "模型不完整，先运行 scripts/setup-models.sh 下载"
    exit 1
fi

if [ $# -gt 0 ] && [ "$1" = "daemon" ]; then
    echo "==> 启动 live-translator 后台进程..."
    FIFO="/tmp/live-translator-stdin.$$"
    mkfifo "$FIFO"
    cleanup() { rm -f "$FIFO"; }
    trap cleanup EXIT

    # Start binary first (its stdin open blocks until we open the write end)
    "$BIN" \
        --asr-model "$ASR_MODEL" \
        --translate-model "$MARIAN_MODEL" \
        < "$FIFO" &
    PID=$!

    # Open write end (unblocks binary's stdin open) and keep open so
    # transient echo writes don't cause EOF on the reader side.
    exec 3>"$FIFO"

    if ! kill -0 $PID 2>/dev/null; then
        echo "进程启动失败"
        exec 3>&-
        exit 1
    fi
    echo "PID=$PID"

    # Command sits in pipe buffer; processed when stdin thread starts reading
    echo '{"type":"start","src":"en","tgt":"zh"}' >&3
    echo "已发送 start 命令，模型加载完成后自动生效"
    echo ""
    echo "发送更多命令:"
    echo "  echo '{\"type\":\"quit\"}' > $FIFO"
    echo "停止进程:"
    echo "  kill $PID"

    wait $PID || true
    exec 3>&-

else
    echo "==> 交互测试..."
    echo "    JSON 命令示例:"
    echo '    {"type":"start","src":"en","tgt":"zh"}'
    echo '    {"type":"stop"}'
    echo '    {"type":"quit"}'
    echo "    JSON stdout → /dev/null，人类可读日志 → stderr"
    echo ""
    "$BIN" \
        --asr-model "$ASR_MODEL" \
        --translate-model "$MARIAN_MODEL" \
        1>/dev/null
fi
