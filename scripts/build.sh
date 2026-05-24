#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

echo "==> 构建 release 版本..."

nix-shell -p cmake pkg-config alsa-lib rustc cargo clang openssl sentencepiece --run '
  LIBCLANG_PATH="$(nix eval --raw nixpkgs#clang.cc.lib)/lib"
  export LIBCLANG_PATH
  cargo build --release
'

echo ""
echo "==> 构建完成: target/release/live-translator"
