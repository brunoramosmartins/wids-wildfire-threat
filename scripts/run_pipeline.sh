#!/usr/bin/env bash
set -euo pipefail

echo "=== WiDS Wildfire Pipeline ==="
echo "Start: $(date)"
echo ""

STEPS=("process" "features" "train" "predict" "evaluate" "submit")

for step in "${STEPS[@]}"; do
    echo "--- Running: make $step ---"
    STEP_START=$(date +%s)
    make "$step"
    STEP_END=$(date +%s)
    echo "--- $step completed in $((STEP_END - STEP_START))s ---"
    echo ""
done

echo "=== Pipeline complete ==="
echo "End: $(date)"
