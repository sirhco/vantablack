#!/usr/bin/env bash
# compare-gemma-output.sh — Phase 19e gate.
#
# Runs vantablack against the user's downloaded Gemma 4 E2B .litertlm
# and diffs the output against the reference captured from Google's
# `litert-lm` runtime (tests/golden/gemma4-e2b-reference.txt).
#
# Status: SCAFFOLDING ONLY. Will fail today — vantablack cannot run a
# .litertlm forward path until roadmap items 19c (tensor mapping) and
# 19d (int2/int4 kernels) ship.

set -euo pipefail

MODEL="${MODEL:-$HOME/.cache/huggingface/hub/litertlm-models/gemma-4-E2B-it.litertlm}"
PROMPT="${PROMPT:-Once upon a time}"
MAX_TOKENS="${MAX_TOKENS:-256}"
SEED="${SEED:-42}"
# Reference command used to generate the golden:
#   uvx litert-lm run <model> --backend cpu --seed 42 --prompt "Once upon a time"
# Sampler is handled INSIDE the TFLite graph (model has no sampler_params).

REFERENCE="tests/golden/gemma4-e2b-reference.txt"
ACTUAL="/tmp/vantablack-gemma4-e2b-actual.txt"

if [[ ! -f "$MODEL" ]]; then
  echo "model not found: $MODEL" >&2
  echo "download via: huggingface-cli download litert-community/gemma-4-E2B-it-litert-lm --local-dir ~/.cache/huggingface/hub/litertlm-models/" >&2
  exit 1
fi

if [[ ! -f "$REFERENCE" ]]; then
  echo "reference not found: $REFERENCE" >&2
  exit 1
fi

zig build -Doptimize=ReleaseFast -Dmetal=true

echo "running vantablack prompt $MAX_TOKENS \"$PROMPT\" --seed $SEED"
./zig-out/bin/vantablack "$MODEL" prompt "$MAX_TOKENS" "$PROMPT" \
  --seed "$SEED" --temp 0 > "$ACTUAL" || {
  echo "vantablack failed — expected until Phase 19c + 19d land" >&2
  exit 2
}

if diff -u "$REFERENCE" "$ACTUAL"; then
  echo "PASS: output matches reference"
else
  echo "FAIL: output diverges from reference" >&2
  exit 3
fi
