#!/usr/bin/env bash
# Setup script for the document conversion tool.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$REPO_ROOT/.venv"
REQUIRED_PYTHON_MIN=10

find_python() {
    for cmd in python3.14 python3.13 python3.12 python3.11 python3.10 python3; do
        if command -v "$cmd" &>/dev/null; then
            ver=$("$cmd" -c "import sys; print(sys.version_info.minor)" 2>/dev/null || echo "0")
            if [ "$ver" -ge "$REQUIRED_PYTHON_MIN" ]; then
                echo "$cmd"
                return
            fi
        fi
    done
    echo ""
}

echo "=== Document Conversion Tool Setup ==="
echo ""

PYTHON=$(find_python)
if [ -z "$PYTHON" ]; then
    echo "ERROR: Python >=3.10 not found."
    exit 1
fi
echo "Python: $($PYTHON --version) ($PYTHON)"

if [ ! -d "$VENV_DIR" ]; then
    echo "Creating venv at $VENV_DIR ..."
    $PYTHON -m venv "$VENV_DIR"
fi

echo "Installing dependencies ..."
"$VENV_DIR/bin/pip" install --quiet --upgrade pip
"$VENV_DIR/bin/pip" install --quiet \
    'markitdown[all]==0.1.5' \
    'PyMuPDF==1.27.2.2' \
    'pymupdf-layout==1.27.2.2' \
    'pymupdf4llm==1.27.2.2' \
    'pillow==10.4.0' \
    'mlx>=0.12.0' \
    'mlx-lm>=0.12.0' \
    'mlx-vlm==0.5.1' \
    'openpyxl' \
    'python-pptx' \
    'msoffcrypto-tool'

echo "Setup complete."
echo "Usage:"
echo "  .venv/bin/python3 convert_to_md.py [path]"
echo ""
