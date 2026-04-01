#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# CREATE GITHUB MILESTONES
# ============================================================================
# This script creates all project milestones for the WiDS 2026 Wildfire
# Time-to-Threat Prediction repository.
#
# Prerequisites:
#   gh auth status  (must be authenticated)
#
# Usage:
#   export REPO_NAME="wids-wildfire-threat"
#   chmod +x create_milestones.sh
#   ./create_milestones.sh
# ============================================================================

REPO_NAME="${REPO_NAME:-wids-wildfire-threat}"
GH_USER=$(gh api user --jq '.login')
REPO="${GH_USER}/${REPO_NAME}"

echo "============================================"
echo "  Creating Milestones for: ${REPO}"
echo "============================================"

# Helper function: create milestone (skip if exists)
create_milestone() {
  local title="$1"
  local description="$2"

  if gh api "repos/${REPO}/milestones" \
    --method POST \
    --field "title=${title}" \
    --field "description=${description}" \
    --field "state=open" \
    --silent 2>/dev/null; then
    echo "  ✅ ${title}"
  else
    echo "  ⏭️  ${title} (already exists or error)"
  fi
}

echo ""
echo ">>> Creating milestones..."

create_milestone \
  "Phase 0 — Setup & Alignment" \
  "Initialize repository, environment, download Kaggle data, document problem statement. Tag: v0.1-setup"

create_milestone \
  "Phase 1 — EDA & Data Understanding" \
  "Systematic EDA: overview, geospatial, survival analysis. Data dictionary. Tag: v0.2-eda"

create_milestone \
  "Phase 2 — Data Processing Pipeline" \
  "Reproducible data cleaning pipeline with schema validation and quality checks. Tag: v0.3-processing"

create_milestone \
  "Phase 3 — Feature Engineering" \
  "Feature engineering: geospatial, temporal, weather, infrastructure. Feature catalog. Tag: v0.4-features"

create_milestone \
  "Phase 4 — Baseline Model" \
  "Baseline models, evaluation framework, MLflow setup, first Kaggle submission. Tag: v0.5-baseline. Release: Yes"

create_milestone \
  "Phase 5 — Advanced Modeling" \
  "Survival analysis (Cox PH, RSF, GBS), gradient boosting (XGB, LGB), feature selection. Tag: v0.6-advanced. Release: Yes"

create_milestone \
  "Phase 6 — Optimization & Ensembles" \
  "Hyperparameter tuning, ensemble methods (stacking, blending), error analysis, final submission. Tag: v0.7-optimized"

create_milestone \
  "Phase 7 — Observability" \
  "Structured logging, data quality monitoring, feature drift detection, pipeline health checks. Tag: v0.8-observability"

create_milestone \
  "Phase 8 — Portfolio Polish" \
  "Portfolio-grade README, runbook, experiment log, CHANGELOG, final pipeline orchestration. Tag: v1.0.0. Release: Yes"

echo ""
echo "============================================"
echo "  ✅ All milestones created (9 total)"
echo "============================================"
echo ""
echo "  Verify at: https://github.com/${REPO}/milestones"
echo "============================================"
