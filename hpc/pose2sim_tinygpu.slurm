#!/bin/bash -l
#SBATCH --job-name=p2s
#SBATCH --partition=rtx3080
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=%x_%j.out
#SBATCH --export=NONE
# --partition is REQUIRED. Omit it and Slurm says "Requested node
# configuration is not available", which reads like a hardware problem and
# is not one.
# --export=NONE then `unset SLURM_EXPORT_ENV` is the NHR-documented pattern:
# it stops your login shell leaking into the job, which is how a stale
# PYTHONPATH silently changes which packages the job imports.

unset SLURM_EXPORT_ENV
set -euo pipefail

PROJ="${1:?usage: sbatch pose2sim_tinygpu.slurm <project_dir>}"
REPO="${REPO:-$HOME/projects/therapy-pose2sim}"
VENV="${VENV:-$HOME/projects/.venv-p2s}"

module load python 2>/dev/null || true
source "$VENV/bin/activate"
export PYTHONNOUSERSITE=1          # ~/.local must not shadow the venv

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Fail in 3 seconds, not 40 minutes.
python -u "$REPO/hpc/verify_hpc.py"

cd "$PROJ"
python -u "$REPO/scripts/run_pose2sim.py" "$PROJ"

echo "=== validation ==="
TRC=$(ls -t "$PROJ"/pose-3d/*_filt_*.trc 2>/dev/null | head -1 \
      || ls -t "$PROJ"/pose-3d/*.trc | head -1)
python -u "$REPO/validation/eval_body.py" "$TRC"
