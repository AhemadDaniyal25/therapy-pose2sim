# Brief for whoever is picking this up

Written for a colleague joining the project, 2026-10-20. The cluster
(NHR@FAU TinyGPU, account `iwso240h`) is down, so nobody can run anything
right now. This repo is the code plus the reasoning, so you can read and
critique it before we next have hardware.

You can paste this file plus the scripts into your own Claude session and ask
it to work through them with you. The questions at the bottom are the ones I
most want a second opinion on.

---

## What the project is

A therapy room with multiple cameras. We want 3D body pose and 3D face
keypoints for the two people in the room (therapist and patient), accurate
enough for the collaborative-therapy department to use, plus visuals good
enough to show them.

## The three repos, and which are current

| repo | what | status |
|---|---|---|
| `therapy-room-pose-pipeline` | my own pipeline: YOLO person detection + RTMPose COCO-17, plus all the calibration work | **current**. The calibration validation lives here and is the most reliable part of the project |
| `face-landmark-pipeline` | single-camera MediaPipe Face Landmarker, 478 landmarks, 2D + stability metrics | **current but superseded** for the multi-camera work. The 2D extraction and its metrics are still the base of what we do |
| `therapy-pose2sim` (this one) | Pose2Sim, multi-camera 3D face fusion, animation | **current**. The newest work |

Not replaced, layered. The first repo's calibration is the input to this
one. The second repo's MediaPipe extraction is the input to the face fusion
here, with one change, documented below, which is that we run MediaPipe on
upscaled head crops rather than full frames.

Note `face-landmark-pipeline`'s README says explicitly: *"The current
repository does NOT contain a validated camera-fusion algorithm."* That was
true when written. `validation/face3d_both.py` in this repo is that fusion
step, and it is the downstream step the README suggests.

## What to read, in order

1. `docs/CALIBRATION_MAP.md`, **start here.** Everything is downstream of
   calibration, and ours has a specific, documented weakness.
2. `README.md`, results table and limitations.
3). `docs/RUNBOOK.md` (how a new recording gets processed.
4. `scripts/run_pose2sim.py` then `scripts/split_main.py`. The two things
   that fixed the reconstruction.
5. `validation/eval_body.py`, how we judge whether a 3D reconstruction is
   real rather than just self-consistent.
6. `animation/` only if you care; the supervisor has deprioritised it.

## What to study on Pose2Sim

Pose2Sim is a 6-stage pipeline and the interesting content is in what each
stage can silently get wrong:

1. **poseEstimation**, runs RTMPose/RTMW per camera, writes OpenPose-format
   JSON per frame. `pose_model` decides the keypoint set; `'Whole_body'`
   gives COCO_133 (17 body + 6 feet + 68 face + 42 hands), which is how
   fingers and face keypoints get in. There is no separate model to add.
2. **synchronization**, cross-correlates filtered keypoint signals between
   cameras to find frame offsets. *This is the stage that was turned off and
   it was our single biggest bug.* Understand that it reports a correlation
   as well as an offset, and that a big offset at low correlation means it
   failed, not that the cameras were far apart.
3. **personAssociation**, matches detections across cameras. Fails when one
   camera sees two people and others see one; our fix was to crop rather
   than to tune it.
4). **triangulation** (DLT across views, with per-camera exclusion when a
   view disagrees. The "excluded %" in the log is the number to watch.
5. **filtering**, Butterworth, writes `*_filt_butterworth.trc`.
6. Output is TRC (OpenSim marker format) in **metres**.

Worth understanding properly:

- **Why reprojection error is a weak validator with two views.** It's the
  quantity triangulation minimises. Two views + a wrong baseline gives you a
  low reprojection error and a wrongly-sized person. This is the single most
  important conceptual point in the project, and it's why
  `eval_body.py`/biocular width exist.
- **The difference between intrinsics and extrinsics calibration**, and why
  one can be verified from our footage and the other can't (co-visibility).
- `P = K[R|T]`, main camera as world origin, and what cropping does to the
  principal point.

Good background: the Pose2Sim paper (Pagnon et al., *Sensors* 2021/2022),
and the Pose2Sim GitHub docs on `Config.toml`. For the alternatives we
looked at and rejected for this setup: EasyMocap, Anipose, FreeMoCap, and
the learned multi-view methods (VoxelPose, PlaneSweepPose), the learned
ones need training data with ground truth we don't have.

## If someone asks "which 2D model is better and how do you know"

That was a supervisor question, so here's the answer in the form I'd give it.

We compared `Body_with_feet` (RTMPose), HALPE-26) against `Whole_body` (RTMW
`rtmw-dw-x-l_simcc-cocktail14`, COCO_133). `Whole_body` is what we use, for
a reason that isn't accuracy:

- `Body_with_feet` has 26 keypoints, no face, no hands. It is faster.
- `Whole_body` has 133, including the 68 face points and 42 hand points.
  Fingers and face keypoints are the deliverables the supervisor asked for,
  so the keypoint set decides it, switching models is how you "add
  fingers", there's no separate finger stage.

Honest caveat if pressed: we did **not** run a controlled accuracy
comparison between the two on this data. The choice was driven by required
keypoints. A body-only comparison on identical frames would be a reasonable
small experiment and we haven't done it.

What we *did* measure, with numbers: synchronisation on/off (31.1 → 16.9 px),
`multi_person` true/false (23.0 → 16.9 px), and cam2 excluded 100% in
multi-person mode), and three-camera vs split-camera (15.8 → 3.5 px).

## Things in here that are wrong or unfinished

Being explicit so you don't have to discover them:

- **Extrinsics have ~20–25% translation error** and cannot be recomputed
  from existing footage. `CALIBRATION_MAP.md` has the full account. This is
  the project's main limitation.
- **I reported the extrinsic residuals as pixels and they are metres.** The
  npz key is called `rms` but holds `mean_translation_residual`. The
  `therapy-room-pose-pipeline` README still says "Extrinsic residuals:
  main→left 0.5718, main→right 0.6364" without units, which reads as pixels
, **that needs fixing**, along with adding the co-visibility finding to
  its Limitations section.
- **Texture projection does not work.** The TRC is not in camera
  coordinates; raw TRC joints project to v = 3322–9641 on a 1080-px frame.
  The missing step is recovering camera pose in TRC space via PnP against
  the 2D pose JSONs. `textured 0/18056 vertices` in the animation logs is
  this.
- **`common.py` in the older repo has no `calibration_clips`** although
  `paths.yaml` defines the key. Known, unfixed. Related: if `pyyaml` isn't
  installed, `common.py` silently falls back to a nonexistent relative path
  and calibration "finds 0 frames" with no error, that cost a day.
- **Everything is validated on ~10 s segments**, not the full 43.7 min
  recording.
- `tracked_keypoint = 'Neck'` is set in some configs and doesn't exist in
  COCO_133; Pose2Sim silently uses `Nose`.

## Questions I'd like a second opinion on

1. Is there any way to recover usable extrinsics from 6 co-visible frames
   plus known intrinsics plus the two people's motion, e.g. bundle
   adjustment using the humans themselves as the calibration target? My
   read is no, not to metric accuracy, but I'd like to be argued out of it.
2. Could we constrain the scale using a known anatomical measurement of one
   of the two people (measure their actual biocular width or forearm length
   with a tape) and rescale the whole reconstruction? That feels like it
   should work and would turn a 20% error into something defensible.
3. Is the PnP route to texture projection worth doing, or is it a distraction
   given the supervisor said not to spend much time on animation?
4. Anything obviously wrong in `eval_body.py`'s use of robust CV as the
   stability metric.
