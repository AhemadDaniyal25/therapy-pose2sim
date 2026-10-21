# Project log

Append-only. Newest last. Numbers are from logs, not memory.

---

**Calibration audit.** Inspected the Pose2Sim demo's `calibration.toml`.
cam_1 fx = 8712.8 px on a 1080-px-wide image (~7° horizontal FOV); cam_2
k1 = 9.09; cam_1 k1 = −31.8; all three principal points at exactly the image
centre; translations ~1000× the reference values. Non-physical. Pose2Sim was
reporting a ~41,900 mm metric error. Backed up as
`calibration_BROKEN_backup.toml`, kept as evidence, not deleted.

**Independent intrinsics check.** Re-estimated from the ChArUco clips
(28:00–37:30) with an 80/20 held-out split, 10 fps sampling, sharpness and
spatial-coverage filtering. Held-out RMS: main 0.350 px, left 0.594 px,
right 0.869 px. Intrinsics confirmed good.

Two bugs found on the way: `pyyaml` missing → `common.py` silently fell back
to a nonexistent relative path and detected 0 frames with no error message;
and the left/right baselines were swapped in `recalibrate.py` (corrected to
main 0.8623, left 0.9270, right 0.5513). `FPS = 30.0` verified by ffprobe.

**Extrinsic provenance, unit error found.** Read
`ali_repo/scripts/import_reference_calibration.py`:

```python
residual = float(d["mean_translation_residual"])
np.savez(..., rms=np.float64(residual))
```

The npz key `rms` holds a **translation residual in metres**, not a pixel
RMS. So main-left = 0.636 m on a 2.413 m baseline (26.4%) and main-right =
0.572 m on 3.066 m (18.6%). I had called these "sub-pixel", that was wrong
and is corrected everywhere in this repo. Same file gave the camera name
mapping: cam0_main→main, cam1_doctor→**right**, cam2_patient→**left**.

**Can we recalibrate extrinsics?** No. `validation/covisibility.py` scanned
28:00–37:30 at 3 fps with a 2× upscale retry: board visible at ≥12 corners
in 0 frames (main+left), 6 frames (main+right), 0 frames (left+right). The
board was shown to one camera at a time. Needs a new recording.

**Calibration swap alone made things worse, in a useful way.** Replacing the
broken TOML took reprojection from 25.7 → 31.1 px while fixing the metric
scale. 21 of 22 markers were still NaN. So the calibration was not the whole
story.

**Found the real problem: synchronisation.** `do_synchronization = false` in
every config we inherited. Turning it on, Pose2Sim measured **136 frames
(4.5 s)** main↔left and **113 frames (3.8 s)** main↔right (the cameras were
started by hand. Triangulation had been intersecting rays cast at different
instants). After sync: **16.9 px, 22/22 markers valid.** (We never ran
broken-calibration + sync, so the two effects aren't fully separated.)

**`multi_person = true` is worse here.** 23.0 px vs 16.9 px, cam2 excluded
100% of the time, and it still only tracked one person. Reverted.

**Overlay debugging, two lessons.** First overlay wrote 0 skeletons onto a
normal-looking video: I divided by 1000 assuming millimetres, but **TRC is
in metres**. Then the axis convention was wrong. Rather than guess, wrote
`find_axes.py` to brute-force all 48 permutation/sign combinations against
the detected 2D keypoints: `perm=(2,0,1) signs=(+1,+1,+1)` won at **24.0 px**
against a runner-up at 203.1 px. An 8.5× gap means unambiguous.

**Split the main camera. Best result in the project.** Main (1920×1080)
frames both people; each side camera frames one. Cropping main into two
960×1080 halves and pairing each with its adjacent side camera removes the
person-association problem entirely.

| | 3-camera | mainL | mainR |
|---|---|---|---|
| reprojection | 15.8 px | 16.2 px | **3.5 px** |
| excluded | 84/17/0% | 0% | 0% |
| sync offset | 136/113 fr | 2 fr | 6 fr |
| correlation | 0.59/0.52 | 0.92 | 0.92 |

Note the crop shifts the principal point: right half cx = 898.6 − 960 =
**−61.4**), which is correct (the optical centre is outside the crop.

**Body validation.** `eval_body.py`, thresholds declared before looking
(CV < 0.08 pass, > 0.15 fail). mainR: 8/10 bones in anatomical range, median
robust CV **0.011**. mainL: 6/10, CV 0.092. Bones 1–2 cm below their
anatomical floor with low CV = consistent but under-scaled, i.e. an
extrinsic error rather than noise.

**Face: the crop trick.** MediaPipe finds nothing on full frames at room
distance (~40 px face). Took the head box from the COCO_133 face keypoints
Pose2Sim already detected, cropped, upscaled (main ×5, side ×2), ran
MediaPipe on the crop, mapped back. Detection went from ~0% to: right
299/301, left 301/301, main 112/301 on the first pass; 85% / 50% on the main
halves after refinement.

**Face fusion results.** Offset searched over [−12, +12] minimising
reprojection (agreed with Pose2Sim's estimate, independent confirmation).

| | Person A (mainL) | Person B (mainR) |
|---|---|---|
| frames fused | 147 | 259 |
| reproj median | 12.35 px | **1.46 px** |
| reproj p95 | 25.08 px | 7.19 px |
| biocular width | 7.52 cm | **8.53 cm** |
| biocular CV | 0.073 | **0.043** |

Adult biocular 8.8–9.8 cm, so both under-scaled, Person A badly. Validated
on biocular width specifically because it's anatomically fixed and was not
optimised for.

**The Person A / Person B asymmetry is systematic.** Body fitting and face
fusion are independent methods; both put Person A at 4–8× worse. Shared
input is the main↔left extrinsic. That's the diagnosis.

**Animation. SMPL was unavailable** (smpl.is.tue.mpg.de down). Used NVIDIA SOMA
wrapping MHR (Meta Human Rig): 45 identity coeffs, 68 scale params, 77
joints, 18,056 vertices, 36,108 faces.

**Spiders.** First fits produced, verbatim, "spiders and monsters with huge
fingers." Cause: 60 of the 68 `scale_params` are per-bone scales *including
every finger joint*, and fingers aren't in a body fit, so Adam stretched
them freely. Froze `scale_params`: error rose **3.3 → 11.4 cm** with correct
human proportions. The 3.3 cm was fitting noise with anatomy as the
currency; the trade is right.

Also added: rotation clamp `MAXROT = 1.75` rad (~100°), anatomical
bone-length priors, temporal smoothness, `ic.clamp_(-2, 2)`. Final:
`anat_p2s_mainR` 11.4 cm, `anat_p2s_mainL` 10.6 cm.

**Rendering: matplotlib was the problem.** `Poly3DCollection` has no
z-buffer. It sorts whole polygons, so 18k vertices render as a flat blob
with faces punching through. Replaced with a small software rasteriser:
per-pixel depth buffer, barycentric interpolation, smooth vertex normals,
Lambert + specular + rim. That's the difference between "blob" and "body".

Also: up axis must be **detected** (median Nose minus median mid-hip), take
the largest-magnitude axis and its sign). A hardcoded `elev` gave a tumbling
figure.

**Texture projection: abandoned, with a reason.** `tex_debug.py` showed raw
TRC projects to u ∈ [−1109, 2348], v ∈ [3322, 9641] on a 960×1080 frame,
z ∈ [0.32, 0.85] m, **0 of 131 keypoints inside the frame**. The TRC is not
in camera coordinates. The missing step is PnP against the 2D pose JSONs to
recover camera pose in TRC space. `textured 0/18056 vertices (0%)` in the
logs is this, not a sampling bug.

**Face in the animation.** Body TRC and face JSON were in different
coordinate frames, face invisible. `align_face.py` found `perm=(1,2,0)
signs=(+1,+1,+1)`, 2.7 cm centroid-to-Nose. Then the face was still a blob
because a head ellipsoid was being drawn over the 854-triangle mesh; removed
it and added a zoomed inset panel.

**2026-10-20, SOMA joint-mapping ablation (local CPU, cluster down).**
Identical TRC, frames and iterations; only the mapping changed:

| variant | final marker RMSE |
|---|---|
| 17 joints, no Hips / no Chest | 17.8 cm |
| + `Chest`→`Neck` | **13.6 cm** |
| + `Hips`→`Hip` | 17.0 cm |

`Chest`→`Neck` is a 24% improvement, keep it. `Hips`→`Hip` is **actively
harmful**: SOMA's `Hips` is the kinematic root of the rig, Pose2Sim's `Hip`
is the mid-hip surface marker, roughly 10 cm apart. Pinning them drags the
root and therefore the whole chain. A name match is not a semantic match,
and an ablation is how you find out). (This also explained an earlier
"mapped 17 joints" when 19 were expected, neither name resolved.)

**Cluster quota incident.** `/home/hpc` (100 G soft) exceeded. It broke `cp`
and `sed` **silently**, a 0-byte backup file and `sed: couldn't flush`.
Cleared pip cache, rsynced `multi_camera_pose` to `/home/woody` with a
symlink back, freed ~22 G. Lesson: large outputs go to `/home/woody`
(1000 G) from the start.

**Other cluster lessons, all paid for in wasted jobs.** Slurm needs
`#SBATCH --partition=...` explicitly or you get "Requested node
configuration is not available". `PYTHONNOUSERSITE=1` or `~/.local` packages
shadow the venv (a scipy 1.18.0 against numpy 1.26.4 gave `AttributeError:
np.long`; fixed with scipy 1.11.4 in the venv). `python -u` or logs stay
empty). And a `NameError: name 'SKIN' is not defined` (defined at line 166,
used at line 132, cost two 40-minute GPU jobs; a patch attempt then failed
silently because `python` wasn't on PATH without activating the venv.

**2026-10-20, repo created.** Cluster down for maintenance. Assembled this
repo from session logs so the work can be reviewed and handed over without
cluster access. Scripts are the versions that produced the numbers above;
`validation/face3d_both.py` and `scripts/split_main.py` were tidied for
readability during assembly and have not been re-executed since, re-run
them before trusting them on new data.


**2026-10-21. Baselines were swapped in my own notes.** `verify_hpc.py` read
them straight from the npz: main-left 2.413 m, main-right 3.066 m. I had
written them the other way round, which put the relative errors at 21% and
24% instead of 26.4% and 18.6%. The correction matters because it reverses
which pair is worse: main-left is, and Person A (the left camera's subject)
is the one that comes out worse on body reprojection, bone CV, face
reprojection and biocular width. The numbers now point the same direction as
the evidence instead of mildly against it.

Same class of mistake as the swapped baselines found in `recalibrate.py`
earlier. Left/right got transposed twice in this project, in two different
places. Worth stating the convention once and checking against it rather
than re-deriving it each time: cam1_doctor is RIGHT, cam2_patient is LEFT.
