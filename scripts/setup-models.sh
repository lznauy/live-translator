#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="${1:-./models}"
HF_MIRROR="${HF_MIRROR:-https://hf-mirror.com}"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log()  { echo -e "${GREEN}[OK]${NC} $*"; }
warn() { echo -e "${YELLOW}[..]${NC} $*"; }
err()  { echo -e "${RED}[!!]${NC} $*"; exit 1; }

command -v wget >/dev/null || command -v curl >/dev/null || err "需要 wget 或 curl"
mkdir -p "$MODEL_DIR"

download() {
    local url="$1" dest="$2"
    if [ -f "$dest" ]; then
        warn "已存在，跳过: $dest"
        return
    fi
    log "下载: $(basename "$dest")"
    if command -v wget >/dev/null; then
        wget -q --show-progress "$url" -O "$dest"
    else
        curl -L --progress-bar "$url" -o "$dest"
    fi
}

download_hf() {
    local repo="$1" dest_dir="$2"
    local base="$HF_MIRROR/$repo/resolve/main"
    mkdir -p "$dest_dir"

    if [ -f "$dest_dir/model.safetensors" ] && [ -f "$dest_dir/config.json" ]; then
        warn "已存在，跳过: $dest_dir"
        return
    fi

    log "下载 $repo → $dest_dir"

    # Always download config + sp model (small files)
    for f in config.json vocab.json tokenizer_config.json source.spm; do
        local url="$base/$f"
        if command -v wget >/dev/null; then
            wget -q --show-progress "$url" -O "$dest_dir/$f" 2>/dev/null || true
        else
            curl -L --progress-bar "$url" -o "$dest_dir/$f" 2>/dev/null || true
        fi
    done

    # Try safetensors first, fall back to pytorch → convert
    local model_url="$base/model.safetensors"
    local model_dest="$dest_dir/model.safetensors"
    local downloaded=false

    if command -v wget >/dev/null; then
        wget -q --show-progress "$model_url" -O "$model_dest" 2>/dev/null && downloaded=true || true
    else
        curl -L --progress-bar "$model_url" -o "$model_dest" 2>/dev/null && downloaded=true || true
    fi

    if $downloaded && [ -s "$model_dest" ]; then
        log "model.safetensors 就绪"
    else
        # Download PyTorch format and convert
        rm -f "$model_dest"
        local pt_dest="$dest_dir/pytorch_model.bin"
        log "safetensors 不存在，下载 pytorch_model.bin 并转换..."
        if command -v wget >/dev/null; then
            wget -q --show-progress "$base/pytorch_model.bin" -O "$pt_dest" || err "下载失败"
        else
            curl -L --progress-bar "$base/pytorch_model.bin" -o "$pt_dest" || err "下载失败"
        fi
        log "转换 PyTorch → safetensors..."
        nix-shell -p python3 python3Packages.torch python3Packages.safetensors --run "
python3 -c \"
import torch
from safetensors.torch import save_file
state = torch.load('$pt_dest', map_location='cpu')
save_file(state, '$model_dest')
print('converted')
\"" || err "转换失败"
        rm -f "$pt_dest"
        log "转换完成"
    fi

    if ! [ -s "$model_dest" ]; then
        err "下载失败: $model_dest 为空"
    fi
    log "完成: $repo"
}

# ── 1. Whisper base (~277MB) ────────────────────────────────
download_hf "openai/whisper-base" "$MODEL_DIR/whisper-base"

# ── 2. Marian OPUS-MT en→zh (~300MB) ────────────────────────
download_hf "zhijian12345/marian-finetuned-kde4-en-to-zh_CN" "$MODEL_DIR/opus-mt-en-zh"

# ── 验证 ───────────────────────────────────────────────────
echo ""
log "检查模型文件..."

check() {
    if [ -f "$1" ]; then
        local size=$(du -h "$1" | cut -f1)
        log "  $1 ($size)"
    else
        err "  $1 缺失!"
    fi
}

check "$MODEL_DIR/whisper-base/model.safetensors"
check "$MODEL_DIR/whisper-base/config.json"
check "$MODEL_DIR/whisper-base/vocab.json"
check "$MODEL_DIR/whisper-base/tokenizer.json"
check "$MODEL_DIR/opus-mt-en-zh/model.safetensors"
check "$MODEL_DIR/opus-mt-en-zh/config.json"
check "$MODEL_DIR/opus-mt-en-zh/vocab.json"
check "$MODEL_DIR/opus-mt-en-zh/tokenizer_config.json"

echo ""
log "全部模型就绪"
du -sh "$MODEL_DIR"/*/
