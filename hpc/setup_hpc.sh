#!/bin/bash -l
# Build the venv on an NHR@FAU cluster. Run on a LOGIN node (this only
# installs, it does not compute).
#
#   bash -l hpc/setup_hpc.sh            <- the -l matters, see below
#
# WHY `-l`: `module` is a shell FUNCTION sourced from /etc/profile.d, not a
# binary on PATH. In a non-login shell it does not exist, so `command -v
# module` returns false, no python module gets loaded, and the script falls
# through to the system python. On woody that is 3.6.8 and nothing works.
# The shebang says `bash -l`, and the script also sources the module init
# itself, so either invocation now works.
#
# If you already have a working venv from earlier (e.g.
# ~/projects/Pose2Sim/.venv), you do not need this script. Point VENV at it:
#   VENV=$HOME/projects/Pose2Sim/.venv bash -l hpc/setup_hpc.sh
set -euo pipefail

VENV="${VENV:-$HOME/projects/.venv-p2s}"
MIN_MINOR=10

echo "=== host ==="
hostname
echo

echo "=== making the module command available ==="
if ! type -t module >/dev/null 2>&1; then
  for init in /etc/profile.d/modules.sh \
              /etc/profile.d/lmod.sh \
              /etc/profile.d/z00_lmod.sh \
              /usr/share/Modules/init/bash \
              /usr/share/lmod/lmod/init/bash; do
    if [ -r "$init" ]; then
      # shellcheck disable=SC1090
      source "$init" && echo "sourced $init" && break
    fi
  done
fi
if type -t module >/dev/null 2>&1; then
  echo "module: available ($(type -t module))"
else
  echo "module: NOT available; will only try the system python"
fi
echo

py_ok() {   # $1 = interpreter path; true if >= 3.$MIN_MINOR
  "$1" -c "import sys; sys.exit(0 if sys.version_info[:2] >= (3,$MIN_MINOR) else 1)" \
    2>/dev/null
}

echo "=== picking a python >= 3.$MIN_MINOR ==="
PY=""
if type -t module >/dev/null 2>&1; then
  echo "available python modules:"
  module -t avail python 2>&1 | grep -i python | sed 's/^/  /' || true
  echo
  # newest first; -V sorts version strings sensibly
  while read -r m; do
    [ -z "$m" ] && continue
    module load "$m" >/dev/null 2>&1 || continue
    cand=$(command -v python3 2>/dev/null || true)
    if [ -n "$cand" ] && py_ok "$cand"; then
      PY="$cand"
      echo "loaded module: $m"
      break
    fi
    module unload "$m" >/dev/null 2>&1 || true
  done < <(module -t avail python 2>&1 \
           | grep -i '^python' | sed 's#/*$##' | sort -Vr)
fi

if [ -z "$PY" ]; then
  for cand in python3.13 python3.12 python3.11 python3.10 python3; do
    p=$(command -v "$cand" 2>/dev/null || true)
    if [ -n "$p" ] && py_ok "$p"; then PY="$p"; break; fi
  done
fi

if [ -z "$PY" ]; then
  echo
  echo "FAILED: no python >= 3.$MIN_MINOR found."
  echo "System python is: $(python3 --version 2>&1)"
  echo
  echo "Next steps:"
  echo "  1. module avail            # look for python/anaconda/conda entries"
  echo "  2. module load <name>      # then re-run this script"
  echo "  3. Or reuse an existing venv:"
  echo "       VENV=\$HOME/projects/Pose2Sim/.venv bash -l hpc/setup_hpc.sh"
  exit 1
fi
echo "using: $PY  ($("$PY" --version 2>&1))"
echo

echo "=== venv: $VENV ==="
if [ -x "$VENV/bin/python" ]; then
  echo "exists already; reusing it"
else
  "$PY" -m venv "$VENV"
  echo "created"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
export PYTHONNOUSERSITE=1          # or ~/.local shadows the venv
echo "venv python: $(python --version 2>&1)"
python -m pip install -q --upgrade pip wheel
echo

echo "=== pip cache to woody (the hpc quota is nearly full) ==="
WOODY="/home/woody/iwso/${USER:-$(id -un)}"
if [ -d "$WOODY" ]; then
  export PIP_CACHE_DIR="$WOODY/.cache/pip"
  mkdir -p "$PIP_CACHE_DIR"
  echo "PIP_CACHE_DIR=$PIP_CACHE_DIR"
else
  echo "no woody dir found; leaving the pip cache in \$HOME"
fi
echo

echo "=== installing ==="
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# --ignore-installed so a half-finished earlier run (Ctrl-C) cannot leave a
# package registered-but-absent; --no-input so it never blocks on a prompt.
python -m pip install --ignore-installed --no-input -r "$REPO_DIR/requirements.txt"
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
python "$REPO_DIR/hpc/verify_hpc.py" || true
echo
echo "Activate it later with:"
echo "  source $VENV/bin/activate && export PYTHONNOUSERSITE=1"
