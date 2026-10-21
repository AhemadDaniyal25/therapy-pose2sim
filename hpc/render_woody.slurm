#!/bin/bash -l
#SBATCH --job-name=render
#SBATCH --partition=work
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --output=%x_%j.out
#SBATCH --export=NONE
# Woody is the CPU throughput cluster. No --gres here: there are no GPUs.
#
# WHY THIS EXISTS: the GPU logs from 19 Oct show the SOMA jobs at 5-16% GPU
# utilisation over ~15-57 minutes. The fitting is GPU work but the software
# rasteriser is numpy, so most of the wall time was CPU-bound. A job that
# only renders belongs on woody, where the queue is shorter and you are not
# holding an RTX3080 at 5%.
#
# Check your own numbers before trusting that: the GPU utilization block at
# the bottom of every NHR job log tells you whether you asked for the right
# machine.

unset SLURM_EXPORT_ENV
set -euo pipefail

TAG="${1:?usage: sbatch render_woody.slurm <project_tag> [start] [nframes]}"
START="${2:-240}"
NF="${3:-450}"
REPO="${REPO:-$HOME/projects/therapy-pose2sim}"
VENV="${VENV:-$HOME/projects/.venv-p2s}"

module load python 2>/dev/null || true
source "$VENV/bin/activate"
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"

python -u "$REPO/hpc/verify_hpc.py" || true   # torch/soma are optional here

# Outputs go to woody, not to $HOME. $HOME is /home/hpc/... and was at
# 86.8G of a 104.9G soft quota on 19 Oct. When that quota is exceeded, cp
# and sed fail SILENTLY: a 0-byte backup, "sed: couldn't flush". Writing
# large outputs to $HOME is how you lose an afternoon.
export OUT_DIR="/home/woody/iwso/$USER/animation"
mkdir -p "$OUT_DIR"

python -u "$REPO/animation/soma_video.py" "$TAG" "$START" "$NF"

echo "=== outputs (note the path: /home/woody, NOT ~) ==="
ls -lh "$OUT_DIR"
