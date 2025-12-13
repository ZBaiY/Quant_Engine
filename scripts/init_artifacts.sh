#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------
# Quant Engine – Professional Artifact Layout Initializer
# ------------------------------------------------------------

ROOT_DIR="$(pwd)"
ARTIFACTS_DIR="${ROOT_DIR}/artifacts"

echo "[init] Creating artifact directories..."

# -----------------------
# Run-scoped artifacts
# -----------------------
mkdir -p "${ARTIFACTS_DIR}/runs"

# -----------------------
# Portfolio / accounting
# -----------------------
mkdir -p "${ARTIFACTS_DIR}/reports/portfolio/daily"
mkdir -p "${ARTIFACTS_DIR}/reports/portfolio/monthly"
mkdir -p "${ARTIFACTS_DIR}/reports/portfolio/final"

# -----------------------
# Operational logs (optional sink)
# -----------------------
mkdir -p "${ARTIFACTS_DIR}/ops"

# -----------------------
# Git hygiene (important)
# -----------------------
touch "${ARTIFACTS_DIR}/.gitkeep"
touch "${ARTIFACTS_DIR}/runs/.gitkeep"
touch "${ARTIFACTS_DIR}/reports/.gitkeep"
touch "${ARTIFACTS_DIR}/ops/.gitkeep"

echo "[init] Artifact directory structure created successfully."