#!/usr/bin/env bash
# Assemble a self-contained deploy folder: Docker build context and Hugging Face Space in one.
#
#   deploy/make_space.sh [OUT_DIR] [CHECKPOINT]
#
# OUT_DIR    default $FNO_DATA_DIR/space   (must not exist yet, or be empty)
# CHECKPOINT default $FNO_DATA_DIR/checkpoints/fno_darcy.pt
# $FNO_DATA_DIR defaults to ~/Desktop/QUEST-data, so nothing large lands in the repo.
set -euo pipefail
cd "$(dirname "$0")/.."

DATA_DIR=${FNO_DATA_DIR:-$HOME/Desktop/QUEST-data}
OUT=${1:-$DATA_DIR/space}
CKPT=${2:-$DATA_DIR/checkpoints/fno_darcy.pt}

if [ ! -f "$CKPT" ]; then
  echo "error: no checkpoint at $CKPT" >&2
  exit 1
fi
if [ -e "$OUT" ] && [ -n "$(ls -A "$OUT")" ]; then
  echo "error: $OUT is not empty; remove it or pass another folder" >&2
  exit 1
fi

mkdir -p "$OUT/model"
cp deploy/Dockerfile "$OUT/Dockerfile"
cp deploy/requirements.txt "$OUT/requirements.txt"
cp deploy/space_README.md "$OUT/README.md"
cp "$CKPT" "$OUT/model/fno_darcy.pt"
rsync -a --exclude __pycache__ app src "$OUT/"
echo "assembled $OUT ($(du -sh "$OUT" | cut -f1)) with checkpoint $CKPT"
