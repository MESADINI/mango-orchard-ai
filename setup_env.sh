#!/usr/bin/env bash
# Creates backend/venv and installs the backend + ML + test dependencies.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/backend/venv"

if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment at $VENV_DIR..."
    python3 -m venv "$VENV_DIR" || python -m venv "$VENV_DIR"
fi

if [ -x "$VENV_DIR/bin/python" ]; then PY="$VENV_DIR/bin/python"; else PY="$VENV_DIR/Scripts/python.exe"; fi

"$PY" -m pip install --upgrade pip
"$PY" -m pip install -r "$SCRIPT_DIR/backend/requirements-dev.txt"
# Optional: dataset download tools (Kaggle / Roboflow)
# "$PY" -m pip install kaggle roboflow

echo "Done. Activate with: source backend/venv/bin/activate  (Windows: backend\venv\Scripts\activate)"
