#!/usr/bin/env bash
# Phase 5: process → features → baseline train → advanced train → predict → evaluate → submit.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

run_step() {
    local name="$1"
    shift
    echo "--- Running: $name ---"
    local start end
    start=$(date +%s)
    "$@"
    end=$(date +%s)
    echo "--- $name completed in $((end - start))s ---"
    echo ""
}

echo "=== WiDS Pipeline (Phase 5 — advanced models) ==="
echo "Start: $(date)"
echo ""

run_step "process" python -m src.data.process
run_step "features" python -m src.features.build
run_step "train_baselines" python -m src.models.train
run_step "train_advanced" python -m src.models.train_advanced
run_step "predict" python -m src.models.predict
run_step "evaluate" python -m src.models.evaluate
run_step "submit" python -m src.submission.format

echo "=== Pipeline complete (check models/phase5_best_model.txt for default submit model) ==="
echo "End: $(date)"
