#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(pwd)"
TEST_ROOT="$ROOT_DIR/tests"

echo "Project root:  $ROOT_DIR"
echo "Tests root:    $TEST_ROOT"
echo

# ---------------------------------------------------------------------
# Helper: create file with a simple pytest placeholder if missing
# ---------------------------------------------------------------------
create_if_missing () {
    local file_path="$1"
    if [ ! -f "$file_path" ]; then
        echo "Creating $file_path"
        cat > "$file_path" <<EOF_INNER
import pytest

# TODO: replace this placeholder with real tests.

def test_placeholder():
    assert True
EOF_INNER
    else
        echo "Skip existing $file_path"
    fi
}

# Helper: create __init__.py if missing
create_init () {
    local file_path="$1"
    if [ ! -f "$file_path" ]; then
        echo "Creating $file_path"
        echo "" > "$file_path"
    else
        echo "Skip existing $file_path"
    fi
}

# ---------------------------------------------------------------------
# 1. Base directories
# ---------------------------------------------------------------------
mkdir -p "$TEST_ROOT"
mkdir -p "$TEST_ROOT/unit"
mkdir -p "$TEST_ROOT/unit/contracts"
mkdir -p "$TEST_ROOT/unit/contracts/features"
mkdir -p "$TEST_ROOT/unit/features"
mkdir -p "$TEST_ROOT/unit/features/ohlcv"
mkdir -p "$TEST_ROOT/unit/features/orderbook"
mkdir -p "$TEST_ROOT/unit/features/options"
mkdir -p "$TEST_ROOT/unit/features/iv_surface"
mkdir -p "$TEST_ROOT/unit/features/sentiment"

# ---------------------------------------------------------------------
# 2. __init__.py for all packages
# ---------------------------------------------------------------------
create_init "$TEST_ROOT/__init__.py"
create_init "$TEST_ROOT/unit/__init__.py"
create_init "$TEST_ROOT/unit/contracts/__init__.py"
create_init "$TEST_ROOT/unit/contracts/features/__init__.py"
create_init "$TEST_ROOT/unit/features/__init__.py"
create_init "$TEST_ROOT/unit/features/ohlcv/__init__.py"
create_init "$TEST_ROOT/unit/features/orderbook/__init__.py"
create_init "$TEST_ROOT/unit/features/options/__init__.py"
create_init "$TEST_ROOT/unit/features/iv_surface/__init__.py"
create_init "$TEST_ROOT/unit/features/sentiment/__init__.py"

# ---------------------------------------------------------------------
# 3. Contract-level feature tests
# ---------------------------------------------------------------------
create_if_missing "$TEST_ROOT/unit/contracts/features/test_feature_channel_base_contract.py"
create_if_missing "$TEST_ROOT/unit/contracts/features/test_feature_loader_contract.py"
create_if_missing "$TEST_ROOT/unit/contracts/features/test_feature_extractor_contract.py"

# ---------------------------------------------------------------------
# 4. Pipeline-level feature tests
# ---------------------------------------------------------------------
create_if_missing "$TEST_ROOT/unit/features/test_feature_extractor_pipeline.py"

# ---------------------------------------------------------------------
# 5. OHLCV feature tests
# ---------------------------------------------------------------------
create_if_missing "$TEST_ROOT/unit/features/ohlcv/test_rsi_feature.py"
create_if_missing "$TEST_ROOT/unit/features/ohlcv/test_macd_feature.py"
create_if_missing "$TEST_ROOT/unit/features/ohlcv/test_vol_feature.py"

# ---------------------------------------------------------------------
# 6. Orderbook feature tests
# ---------------------------------------------------------------------
create_if_missing "$TEST_ROOT/unit/features/orderbook/test_spread_feature.py"
create_if_missing "$TEST_ROOT/unit/features/orderbook/test_imbalance_feature.py"

# ---------------------------------------------------------------------
# 7. Options feature tests
# ---------------------------------------------------------------------
create_if_missing "$TEST_ROOT/unit/features/options/test_iv_point_feature.py"
create_if_missing "$TEST_ROOT/unit/features/options/test_smile_skew_features.py"

# ---------------------------------------------------------------------
# 8. IV surface feature tests
# ---------------------------------------------------------------------
create_if_missing "$TEST_ROOT/unit/features/iv_surface/test_iv_surface_features.py"

# ---------------------------------------------------------------------
# 9. Sentiment feature tests
# ---------------------------------------------------------------------
create_if_missing "$TEST_ROOT/unit/features/sentiment/test_sentiment_score_features.py"
create_if_missing "$TEST_ROOT/unit/features/sentiment/test_event_agg_features.py"

echo
echo "Feature test skeleton created."
