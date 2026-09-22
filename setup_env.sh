#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "===================================================================="
echo "🥭 Setting up Python Virtual Environment for Mango Orchard AI"
echo "===================================================================="

VENV_DIR="$SCRIPT_DIR/.venv"

if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment at $VENV_DIR..."
    python3 -m venv "$VENV_DIR"
else
    echo "Virtual environment already exists at $VENV_DIR."
fi

echo "Upgrading pip and installing core ML and processing dependencies..."
"$VENV_DIR/bin/python" -m pip install --upgrade pip setuptools wheel

# Install core dependencies for Phase 1 & Phase 2
"$VENV_DIR/bin/python" -m pip install \
    numpy \
    pillow \
    scikit-learn \
    requests \
    timm \
    torch \
    torchvision \
    opencv-python-headless \
    ultralytics \
    kaggle \
    roboflow

echo "===================================================================="
echo "✅ Environment setup complete! Activate with: source .venv/bin/activate"
echo "===================================================================="
