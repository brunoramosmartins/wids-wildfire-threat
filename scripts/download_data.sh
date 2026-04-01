#!/usr/bin/env bash
set -euo pipefail

COMPETITION="WiDSWorldWide_GlobalDathon26"
DATA_DIR="data/raw"

echo "=== Downloading Kaggle dataset ==="

if ! command -v kaggle &> /dev/null; then
    echo "ERROR: kaggle CLI not found. Install with: pip install kaggle"
    exit 1
fi

mkdir -p "$DATA_DIR"

echo "Downloading competition data..."
kaggle competitions download -c "$COMPETITION" -p "$DATA_DIR"

echo "Extracting files..."
cd "$DATA_DIR"
for f in *.zip; do
    [ -f "$f" ] && unzip -o "$f" && rm "$f"
done

echo ""
echo "=== Download complete ==="
echo "Files in $DATA_DIR:"
ls -la
