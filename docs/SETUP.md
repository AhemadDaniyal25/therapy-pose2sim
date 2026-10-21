# Setup: what runs where

Short version: **most of this runs on Windows. The animation does not, and
nothing runs without the calibration files, which are not in this repo.**

---

## What runs on Windows

| part | Windows? | notes |
|---|---|---|
| `scripts/make_calib_v2.py` | yes | needs `all_cameras.npz` (see below) |
| `scripts/split_main.py` | yes | needs `ffmpeg` on PATH |
| `scripts/run_pose2sim.py` | yes | CPU works, slowly, see below |
| `scripts/find_axes.py`, `overlay_full.py` | yes | |
| `validation/eval_body.py` | yes | pure numpy, runs on anything |
| `validation/covisibility.py` | yes | needs the raw ChArUco clips |
| `validation/face3d_both.py` | yes | needs `face_landmarker.task` |
| `animation/soma_video.py` | **no** | needs CUDA + SOMA/MHR assets |
| `animation/local_fit.py` | probably not | needs `py-soma-x` |

**Pose2Sim itself is fine on Windows.** It runs pose estimation through
RTMLib/ONNX Runtime, so it works on CPU, just slowly. Rough expectation:
`Whole_body` (RTMW, 133 keypoints) on CPU is on the order of 1–3 s per frame
per camera, so a 300-frame two-camera segment is tens of minutes, not
seconds. For GPU, install `onnxruntime-gpu` with a matching CUDA toolkit.
The cluster is for throughput, not because the code needs Linux.

**The animation is the Linux/HPC-only part.** `py-soma-x` is NVIDIA's SOMA
and the MHR (Meta Human Rig) assets are separately gated (I got them
working on TinyGPU and have not tried a Windows install. Treat it as
cluster-only until someone proves otherwise. This is also the part the
supervisor said not to spend much time on, so it's the right thing to
deprioritise.

**Paths are hardcoded to the cluster layout.** Every script has
`Path.home / "projects/..."` near the top, which on Windows resolves to
`C:\Users\<you>\projects\...`. Either edit the constants at the top of each
script, or mirror the layout. This is a known wart, not a design, flagged
so at least it is written down.

---

## Install

Not one command, but close to it.

```bash
python -m venv.venv
.venv\Scripts\activate          # Windows
# source.venv/bin/activate     # Linux

pip install -r requirements.txt
```

Three things `pip` will **not** get you:

1. **ffmpeg**, needed by `split_main.py`. Windows: `winget install
   ffmpeg`, or download a build and put it on PATH. Check with
   `ffmpeg -version`.
2. **The MediaPipe model file**, `face_landmarker.task`, ~3.8 MB, from the
   MediaPipe Face Landmarker model page. Put it where
   `validation/face3d_both.py`'s `MODEL` constant points
   (`~/models/face_landmarker.task` by default). MediaPipe does not bundle
   it.
3). **`py-soma-x` and the MHR assets** (animation only, gated, cluster only.
   Skip unless you're doing animation.

Pose2Sim pulls a fair amount behind it (torch, onnxruntime, rtmlib). First
pose-estimation run also downloads the RTMW weights (~400 MB) to a cache, so
the first run needs network access and takes a while before it looks like
it's doing anything.

The version pins in `requirements.txt` matter, in particular
`scipy==1.11.4` with `numpy==1.26.4`. A newer scipy needs numpy ≥ 2.0 and
the mismatch surfaces as `AttributeError: np.long` from inside Pose2Sim. On
the cluster also set `PYTHONNOUSERSITE=1` so `~/.local` can't shadow the
venv.

Sanity check after installing:

```bash
python -c "import cv2; print(cv2.__version__); print(hasattr(cv2,'aruco'))"
python -c "import mediapipe, rtoml, yaml; print('ok')"
python -c "from Pose2Sim import Pose2Sim; print('pose2sim ok')"
```

`cv2.aruco` must be `True`. That needs `opencv-contrib-python`, and
`covisibility.py` is useless without it.

---

## The calibration files are NOT in this repo

This will block him immediately, so:

The repo contains the **code that builds and audits** calibration, and
`docs/CALIBRATION_MAP.md` which documents every file and how much to trust
it. It does **not** contain the matrices themselves. `.gitignore` excludes
`*.npz`, and the generated TOMLs live inside the Pose2Sim project
directories, not here.

To actually run anything he needs, from you:

| file | where it is now | size |
|---|---|---|
| `all_cameras.npz`, **the one that matters**, K/D per camera + R/T per pair | `therapy-room-pose-pipeline/calibration/` on the cluster; Ali's originals in `F:\TherapyRoom\Raw\Calibration\reference\` | small, KB |
| `calibration_v2_reference.toml` | `pose2sim_therapy_demo/Calibration/` on the cluster | KB |
| `calibration_BROKEN_backup.toml` | same | KB, worth sending as the counter-example |
| raw videos | cluster + `F:\TherapyRoom\Raw\` | GB |

**Send him `all_cameras.npz` directly**, it's a few KB and everything else
regenerates from it via `scripts/make_calib_v2.py`. It's camera geometry, not
patient data, so there's no privacy issue in sharing it.

Send the broken TOML too. Reading a calibration that is obviously wrong
(fx = 8712 on a 1080-px image, k1 = −31.8, principal points pinned to the
exact image centre) next to one that is right is the fastest way to learn
what to look for, and `make_calib_v2.py` now refuses to write a file with
those properties.

The raw videos are too large to hand over casually and are patient footage.
That needs whatever consent/handling rule the lab already has, not a
file transfer decision made by us.

A good way to add the calibration without breaking `.gitignore`:

```bash
mkdir calibration
cp /path/to/all_cameras.npz calibration/
git add -f calibration/all_cameras.npz        # -f overrides the *.npz rule
```

---

## Minimum he can do with no calibration and no videos

Still useful, and worth saying so he doesn't feel blocked:

- read `docs/NOTES_FOR_REVIEWER.md` and push back on the four open questions
- read the scripts, which carry the reasoning and the mistakes in their
  docstrings
- `pip install -r requirements.txt` and confirm the sanity checks pass
- `python -c "import ast,glob; [ast.parse(open(f).read) for f in
  glob.glob('**/*.py',recursive=True)]"`, confirms nothing is broken

Everything past that needs `all_cameras.npz` at minimum.
