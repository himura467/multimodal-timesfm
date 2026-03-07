#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"; pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
TIME_MMD_DIR="$REPO_DIR/data/Time-MMD"

if [[ -d "$TIME_MMD_DIR" ]]; then
  echo "Time-MMD dataset already exists at $TIME_MMD_DIR, skipping clone."
else
  mkdir -p "$REPO_DIR/data"
  git clone --depth 1 https://github.com/AdityaLab/Time-MMD.git "$TIME_MMD_DIR"
  echo "Time-MMD dataset cloned to $TIME_MMD_DIR."
fi

uv run "$SCRIPT_DIR/split_time_mmd_datasets.py" \
  --data-path "$TIME_MMD_DIR" \
  --train-ratio 0.8 \
  --val-ratio 0.1
