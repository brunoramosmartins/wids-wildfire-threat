#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# CREATE GITHUB LABELS
# ============================================================================
# This script creates all project labels for the WiDS 2026 Wildfire
# Time-to-Threat Prediction repository.
#
# Prerequisites:
#   gh auth status  (must be authenticated)
#
# Usage:
#   export REPO_NAME="wids-wildfire-threat"
#   chmod +x create_labels.sh
#   ./create_labels.sh
# ============================================================================

REPO_NAME="${REPO_NAME:-wids-wildfire-threat}"
GH_USER=$(gh api user --jq '.login')
REPO="${GH_USER}/${REPO_NAME}"

echo "============================================"
echo "  Creating Labels for: ${REPO}"
echo "============================================"

# Helper function: create label (skip if exists)
create_label() {
  local name="$1"
  local color="$2"
  local description="$3"

  if gh label create "${name}" \
    --repo "${REPO}" \
    --color "${color}" \
    --description "${description}" 2>/dev/null; then
    echo "  ✅ ${name}"
  else
    echo "  ⏭️  ${name} (already exists)"
  fi
}

# --- DELETE DEFAULT LABELS ---
echo ""
echo ">>> Removing default GitHub labels..."
for default_label in "bug" "documentation" "duplicate" "enhancement" \
  "good first issue" "help wanted" "invalid" "question" "wontfix"; do
  gh label delete "${default_label}" --repo "${REPO}" --yes 2>/dev/null || true
done
echo "  ✅ Default labels removed"

# --- TYPE LABELS ---
echo ""
echo ">>> Creating type labels..."
create_label "type: feature"   "0E8A16" "New functionality"
create_label "type: bug"       "D73A4A" "Something broken"
create_label "type: test"      "FBCA04" "Test additions"
create_label "type: docs"      "0075CA" "Documentation"
create_label "type: refactor"  "D4C5F9" "Code improvement"
create_label "type: chore"     "EDEDED" "Maintenance"

# --- LAYER LABELS ---
echo ""
echo ">>> Creating layer labels..."
create_label "layer: infra"          "BFD4F2" "Repo, CI, environment"
create_label "layer: data"           "C2E0C6" "Data processing"
create_label "layer: eda"            "FEF2C0" "Exploratory analysis"
create_label "layer: features"       "D7BDE2" "Feature engineering"
create_label "layer: ml"             "AED6F1" "ML pipeline"
create_label "layer: submission"     "F9D0C4" "Kaggle submission"
create_label "layer: observability"  "E6CCB2" "Logging, monitoring, health"
create_label "layer: business"       "EB984E" "Business logic and docs"

# --- PRIORITY LABELS ---
echo ""
echo ">>> Creating priority labels..."
create_label "priority: critical"  "B60205" "Blocks other work"
create_label "priority: high"      "D93F0B" "Important"
create_label "priority: medium"    "FBCA04" "Normal priority"
create_label "priority: low"       "0E8A16" "Nice to have"

# --- PHASE LABELS ---
echo ""
echo ">>> Creating phase labels..."
create_label "phase: 0"  "C5DEF5" "Phase 0 — Setup & Alignment"
create_label "phase: 1"  "C5DEF5" "Phase 1 — EDA & Data Understanding"
create_label "phase: 2"  "C5DEF5" "Phase 2 — Data Processing Pipeline"
create_label "phase: 3"  "C5DEF5" "Phase 3 — Feature Engineering"
create_label "phase: 4"  "C5DEF5" "Phase 4 — Baseline Model"
create_label "phase: 5"  "C5DEF5" "Phase 5 — Advanced Modeling"
create_label "phase: 6"  "C5DEF5" "Phase 6 — Optimization & Ensembles"
create_label "phase: 7"  "C5DEF5" "Phase 7 — Observability"
create_label "phase: 8"  "C5DEF5" "Phase 8 — Portfolio Polish"

echo ""
echo "============================================"
echo "  ✅ All labels created (27 total)"
echo "============================================"
