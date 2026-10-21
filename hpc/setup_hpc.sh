#!/bin/bash
# Build the venv on an NHR@FAU cluster. Run on a LOGIN node (this only
# installs; it does not compute).
#
#   bash hpc/setup_hpc.sh
#
# Deliberately not pinned to a module name: NHR renames them. The script
# looks for a python module, falls back to the system python, and tells you
# what it picked.
set -euo pipefail

VENV="${VENV:-$HOME/projects/.venv-p2s}"

echo "=== host ==="
hostname
echo

echo "=== picking a python ==="
if command -v module >/dev/null 2>&1; then
  module load python 2>/dev/null || \
  module load python/3.12-conda 2>/dev/null || \
  module load python3 2>/dev/null || \
  echo "no python module loaded; using system python"
  module list 2>&1 | head -20 || true
fi

PY=$(command -v python3 || command -v python)
echo "using: $PY  ($($PY --version 2>&1))"
MAJ=$($PY -c 'import sys;print(sys.version_info[0])')
MIN=$($PY -c 'import sys;print(sys.version_info[1])')
if [ "$MAJ" -lt 3 ] || { [ "$MAJ" -eq 3 ] && [ "$MIN" -lt 10 ]; }; then
  echo "need python >= 3.10, got $MAJ.$MIN"; exit 1
fi
echo

echo "=== venv: $VENV ==="
if [ ! -d "$VENV" ]; then
  "$PY" -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
export PYTHONNOUSERSITE=1        # or ~/.local shadows the venv
python -m pip install -q --upgrade pip wheel
echo

echo "=== pip cache to woody (the hpc quota is nearly full) ==="
WOODY="/home/woody/iwso/$USER"
if [ -d "$WOODY" ]; then
  export PIP_CACHE_DIR="$WOODY/.cache/pip"
  mkdir -p "$PIP_CACHE_DIR"
  echo "PIP_CACHE_DIR=$PIP_CACHE_DIR"
else
  echo "no woody dir found; leaving pip cache in \$HOME"
fi
echo

echo "=== installing ==="
# --ignore-installed so a half-finished earlier run (Ctrl-C) cannot leave a
# package registered-but-absent; --no-input so it never blocks on a prompt.
python -m pip install --ignore-installed --no-input -r requirements.txt
echo

echo "=== GPU wheel for pose estimation (tinygpu only) ==="
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
  echo "GPU present. If onnxruntime is CPU-only, swap it:"
  echo "  pip uninstall -y onnxruntime && pip install onnxruntime-gpu"
else
  echo "no GPU on this node (expected on a woody login node)"
fi
echo

echo "=== verify ==="
python hpc/verify_hpc.py || true
echo
echo "Activate it later with:"
echo "  source $VENV/bin/activate && export PYTHONNOUSERSITE=1"
