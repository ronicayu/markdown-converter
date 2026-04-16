#!/usr/bin/env bash
# Setup script for the document conversion tool.
# Creates a Python venv, installs all dependencies, and checks LibreOffice.
# Image processing (--describe-images / --extract-image-text) uses mlx-vlm on Apple Silicon.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV_DIR="$REPO_ROOT/.venv"
REQUIRED_PYTHON_MIN=10  # markitdown requires Python >=3.10
SOFFICE="/Applications/LibreOffice.app/Contents/MacOS/soffice"

# --- Find a suitable Python >=3.10 ---
find_python() {
    for cmd in python3.14 python3.13 python3.12 python3.11 python3.10 python3; do
        if command -v "$cmd" &>/dev/null; then
            ver=$("$cmd" -c "import sys; print(sys.version_info.minor)")
            if [ "$ver" -ge "$REQUIRED_PYTHON_MIN" ]; then
                echo "$cmd"
                return
            fi
        fi
    done
    echo ""
}

echo "=== tn-raw conversion tool setup ==="
echo ""

# --- Python ---
PYTHON=$(find_python)
if [ -z "$PYTHON" ]; then
    echo "ERROR: Python >=3.10 not found."
    echo "Install via: brew install python@3.13"
    exit 1
fi
echo "Python: $($PYTHON --version) ($PYTHON)"

# --- Virtual environment ---
if [ -d "$VENV_DIR" ]; then
    echo "Venv:   $VENV_DIR (exists)"
else
    echo "Creating venv at $VENV_DIR ..."
    $PYTHON -m venv "$VENV_DIR"
    echo "Venv:   $VENV_DIR (created)"
fi

# --- Install dependencies ---
echo "Installing Python dependencies ..."
"$VENV_DIR/bin/pip" install --quiet --upgrade pip
"$VENV_DIR/bin/pip" install --quiet \
    'markitdown[all]==0.1.5' \
    'PyMuPDF==1.27.2.2' \
    'pymupdf-layout==1.27.2.2' \
    'pymupdf4llm==1.27.2.2' \
    'pillow==10.4.0' \
    'mlx>=0.12.0' \
    'mlx-lm>=0.12.0' \
    'mlx-vlm==0.5.1'
echo "Deps:   markitdown $("$VENV_DIR/bin/pip" show markitdown | grep Version | cut -d' ' -f2), pymupdf4llm $("$VENV_DIR/bin/pip" show pymupdf4llm | grep Version | cut -d' ' -f2), mlx-vlm $("$VENV_DIR/bin/pip" show mlx-vlm | grep Version | cut -d' ' -f2)"

# --- LibreOffice ---
if [ -x "$SOFFICE" ]; then
    echo "Libre:  $SOFFICE (found)"
else
    echo ""
    echo "WARNING: LibreOffice not found at $SOFFICE"
    echo "  .docx files will convert fine, but .doc files need LibreOffice."
    echo "  Install via: brew install --cask libreoffice"
fi

# --- mlx-vlm (for image processing) ---
echo ""
echo "--- mlx-vlm (for --describe-images / --extract-image-text) ---"
echo "Vision: mlx-vlm installed (in-process on Apple Silicon)"
echo "  Model: Qwen2.5-VL-7B-Instruct-4bit (auto-downloads ~2-3GB on first use)"
echo "  First run may take 1-2 min to download and cache the model"

echo ""
echo "Setup complete. Usage:"
echo "  .venv/bin/python3 convert_scripts/convert_to_md.py                          # convert all"
echo "  .venv/bin/python3 convert_scripts/convert_to_md.py --dry-run               # preview"
echo "  .venv/bin/python3 convert_scripts/convert_to_md.py --force                 # reconvert"
echo "  .venv/bin/python3 convert_scripts/convert_to_md.py --describe-images       # + image alt text (mlx-vlm)"
echo "  .venv/bin/python3 convert_scripts/convert_to_md.py --extract-image-text    # + image text sidecar (mlx-vlm)"
echo ""
