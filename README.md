# therapy-pose2sim

3D human pose estimation from three calibrated cameras in a therapy room,
using Pose2Sim, plus multi-camera 3D face keypoint fusion.

The focus of this repo is the pose and face reconstruction and its
validation. There is also an animation layer under `animation/`, which was
explicitly deprioritised by the supervisor and is exploratory: treat it as a
side experiment, not a result.

This repo holds the work that is **not** in
[`therapy-room-pose-pipeline`](https://github.com/AhemadDaniyal25/therapy-room-pose-pipeline)
(my own YOLO + RTMPose COCO-17 implementation) or
[`face-landmark-pipeline`](https://github.com/AhemadDaniyal25/face-landmark-pipeline)
(single-camera MediaPipe 2D landmarks). Read those two first. The camera
geometry and calibration provenance are documented there.

Status: 2026-10-21. Woody and TinyGPU are reachable again.
Run `python hpc/verify_hpc.py` on the cluster before submitting anything;
it checks every dependency and exits non-zero if something is missing.

---

## Headline results

| | 3-camera | mainL + left | **mainR + right** |
|---|---|---|---|
| mean reprojection | 15.8 px | 16.2 px | **3.5 px** |
| cameras excluded | 84% / 17% / 0% | 0% | **0%** |
| frames reconstructed | 214 | 300 | 297 |
| sync offset found | 136 / 113 fr | 2 fr | 6 fr |
| sync correlation | 0.59 / 0.52 | 0.92 | 0.92 |
| bones in anatomical range | n/a | 6/10 | **8/10** |
| bone-length robust CV | n/a | 0.092 | **0.011** |

| 3D face fusion (MediaPipe 478 landmarks) | Person A (mainL) | **Person B (mainR)** |
|---|---|---|
| detection, main half | 50% (154/305) | 85% (260/305) |
| detection, side camera | 98% (297/302) | 100% (302/302) |
| frames fused | 147 | 259 |
| reprojection median | 12.35 px | **1.46 px** |
| reprojection p95 | 25.08 px | 7.19 px |
| biocular width | 7.52 cm | **8.53 cm** |
| biocular CV | 0.073 | **0.043** |

Adult biocular reference 8.8–9.8 cm; CV < 0.05 counts as stable.
Bone-length criterion (set in advance, from `therapy-room-pose-pipeline`):
CV < 0.08 pass, > 0.15 fail, between = marginal.

---

## The two findings that matter

**1. The supplied Pose2Sim `calibration.toml` was non-physical.**
`cam_1` had fx = 8712 px on a 1080-px-wide image, a ~7° horizontal field of
view, i.e. a telephoto lens on a side camera in a small room. Distortion
coefficients were `k1 = 9.09` and `k1 = -31.8`; all principal points were
pinned to the exact image centre (959.5 / 539.5), meaning they were never
estimated. Translations were ~1000× larger than the reference calibration,
which is why the first runs reported a ~41,900 mm metric error.

Replaced with `Calibration/calibration_v2_reference.toml`, converted from
`therapy-room-pose-pipeline/calibration/all_cameras.npz` (see
`scripts/make_calib_v2.py`). That fixed the metric scale.

**2. The cameras were not synchronised, and that was the real problem.**
`Pose2Sim.synchronization` measured **136 frames (4.5 s)** between main and
left, and **113 frames (3.8 s)** between main and right. Every earlier run had
`do_synchronization = False`, so triangulation was intersecting rays cast at
different instants. Before sync: 31.1 px, 21 of 22 markers NaN. After sync:
16.9 px, 22 of 22 markers valid.

Fixing the calibration alone made reprojection slightly *worse* (25.7 → 31.1 px)
while fixing the units. Synchronisation is what fixed the reconstruction. We
never ran broken-calibration-plus-sync, so the two effects are not fully
separated.

**3. Splitting the wide main camera is what produced the best result.**
Main is 1920×1080 and contains both people; each side camera is framed on one
person. Cropping main into two 960×1080 halves and pairing each half with its
adjacent side camera gives one person per project, which removes the
person-association ambiguity entirely: 3.5 px, 0% camera exclusion, bone CV
0.011. The pairing (left half with left camera) was verified, not assumed.
Pose2Sim measured 2- and 6-frame offsets at 0.92 correlation for the split
projects, versus 136/113 at 0.52–0.59 for the unsplit three-camera case.

Cropping moves the principal point: left half keeps `cx = 898.6`, right half
becomes `cx = 898.6 − 960 = −61.4`. Negative is correct, the optical centre
sits outside the cropped frame.

---

## Repo layout

```
Config_reference.toml     the Pose2Sim settings that mattered, with why
scripts/
  make_calib_v2.py        all_cameras.npz -> Pose2Sim TOML, with sanity gates
  split_main.py           crop main into halves + per-half calibration
  run_pose2sim.py         the 5 stages in order, with a preflight check
  find_axes.py            brute-force the TRC -> camera axis mapping
  overlay_full.py         3D reprojected onto footage (body + hands + face)
validation/
  eval_body.py            bone-length plausibility + robust CV
  covisibility.py         is the ChArUco board ever in two cameras at once?
  face3d_both.py          MediaPipe 478 landmarks -> two-view 3D + metrics
animation/
  soma_video.py           MHR parametric body fitted to the TRC, rasterised
  local_fit.py            CPU-only joint-mapping ablation, no cluster needed
hpc/
  setup_hpc.sh            build the venv on a login node
  verify_hpc.py           dependency self-check, run before every job
  pose2sim_tinygpu.slurm  pose estimation + triangulation (GPU)
  render_woody.slurm      animation render (CPU; the GPU sat at 5-16%)
docs/
  SETUP.md                Windows vs cluster, install, the calibration gap
  CALIBRATION_MAP.md      every calibration file and whether to trust it
  RUNBOOK.md              how to process a new recording
  PROJECT_LOG.md          dated experiment log, numbers from logs
  NOTES_FOR_REVIEWER.md   brief for someone picking this up -- start here
```

The scripts carry their reasoning in the module docstrings, including the
mistakes. That is deliberate: the error messages are the expensive part of
this project and a bare script loses them.

---

## Config

Pose2Sim `Config.toml`, line 39:

```toml
pose_model = 'Whole_body'    # was 'Body_with_feet'
```

`Whole_body` loads **RTMW** (`rtmw-dw-x-l_simcc-cocktail14_270e-256x192`) and
produces **COCO_133**: 17 body + 6 feet + 68 face + 42 hand keypoints. This is
how fingers and face keypoints get into Pose2Sim. There is no separate model
to add. Tracking is `sports2d`, `match_by = keypoints`.

Note `multi_person = true` made things worse on this data: 23.0 px and cam2
excluded 100% of the time, versus 16.9 px in single-person mode. Reverted.

Also note `tracked_keypoint = 'Neck'` is not available in COCO_133; Pose2Sim
silently falls back to `Nose`.

---

## Known limitations

- **Extrinsics are not independently verified.** `all_cameras.npz` was imported
  by `ali_repo/scripts/import_reference_calibration.py` from an external
  reference codebase, not computed from this footage. Its `rms` field holds
  `mean_translation_residual` **in metres** despite the key name: 0.636 m
  main-left on a 2.413 m baseline (26.4%) and 0.572 m main-right on 3.066 m
  (18.6%). Do not quote those numbers as pixels.
- **They also cannot be recomputed from the available footage.**
  `validation/covisibility.py` scanned 28:00–37:30 at 3 fps across all three
  cameras: the board was visible at ≥12 corners in **0 frames** for main+left
  and **6 frames** for main+right. The board was shown to one camera at a time.
  Intrinsics *were* independently reproduced with a held-out 80/20 split
  (main 0.350 px, left 0.594 px, right 0.869 px), so intrinsics are sound and
  extrinsics are the weak link.
- **Left/right asymmetry is systematic.** Person A is worse than Person B on
  every independent measure, body 16.2 vs 3.5 px, face 12.35 vs 1.46 px,
  bone CV 0.092 vs 0.011. Two independent methods agreeing points at the
  main↔left extrinsic calibration rather than at either method.
- **Reconstruction is slightly under-scaled.** Several bones sit 1–2 cm below
  their anatomical floor and both biocular widths are below the adult range.
  Consistent with the extrinsic translation error.
- **Texture projection does not work.** The TRC is not in camera coordinates.
    raw TRC joints project to v = 3322–9641 on a 1080-px-tall frame. Recovering
  the camera pose in TRC space (PnP against the 2D pose JSONs) is the missing
  step. `textured 0/18056 vertices` in the animation logs is this, not a
  sampling bug.
- Validated on ~10 s segments, not the full 43.7 min recording.

---

## What a new recording needs

Before any therapy footage, record a calibration clip with the **ChArUco board
visible to two cameras simultaneously**, ~2 minutes per overlapping pair. The
previous protocol showed the board to one camera at a time, which is why
extrinsics cannot be rebuilt. Print the board as large as possible, A3 at room
distance was too small for the main camera to resolve.

New recording = new camera positions = existing calibration invalid. See
`docs/RUNBOOK.md`.
