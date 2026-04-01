#!/usr/bin/env bash
set -euo pipefail

VENV_DIR=".venv"

echo "=== Setting up Python environment ==="

if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
else
    echo "Virtual environment already exists."
fi

echo "Activating virtual environment..."
source "$VENV_DIR/bin/activate" 2>/dev/null || source "$VENV_DIR/Scripts/activate"

echo "Upgrading pip..."
pip install --upgrade pip

echo "Installing dependencies..."
pip install -r requirements.txt

echo "Setting up pre-commit hooks..."
pre-commit install

echo ""
echo "=== Setup complete ==="
echo "Activate the environment with:"
echo "  source $VENV_DIR/bin/activate  (Linux/macOS)"
echo "  source $VENV_DIR/Scripts/activate  (Windows/Git Bash)"
