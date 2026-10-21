# Runbook: processing a new recording

Read `CALIBRATION_MAP.md` first.

> **New recording = new camera positions = existing calibration is invalid.**
> Even if nothing looks like it moved. A camera nudged by 2 cm changes the
> extrinsics and silently degrades every metric number. Do not reuse
> `all_cameras.npz` across sessions.

## 0. Before the recording

- [ ] ChArUco board printed **A2 or larger**, mounted rigid and flat
- [ ] Calibration clip A: board filling the frame of each camera in turn,
      varied angles/distances, **intrinsics**
- [ ] Calibration clip B: board walked slowly through each **overlapping
      pair**, visible to two cameras at once, ~2 min per pair.
      **extrinsics**. This is the step the previous session skipped.
- [ ] One sync marker at the start of every take (clap), flash, or visible
      timecode) (turns synchronisation from an estimation problem into a
      lookup
- [ ] Note which physical camera wrote which file. Do it at the time; the
      filenames will not tell you later (`cam1_doctor` is the *right*
      camera).

## 1. Frame rate

Confirm, don't assume:

```bash
ffprobe -v error -select_streams v:0 \
  -show_entries stream=r_frame_rate,nb_frames,width,height \
  -of default=nw=1 cam0_main.mp4
```

Everything in this project ran at **30 fps**, verified by ffprobe. The
pipeline processes **every frame** at full resolution, no decimation.
Calibration re-estimation samples at 10 fps (enough views, less redundancy);
`covisibility.py` scans at 3 fps because it only counts detections.

## 2. Calibration

```bash
# intrinsics + extrinsics, with a held-out split
python therapy-room-pose-pipeline/validation/recalibrate.py

# did the board ever appear in two cameras at once?
python validation/covisibility.py
```

Gate: held-out intrinsic RMS < 1.0 px per camera, **and** ≥ 30 well-spread
co-visible frames per pair. If co-visibility fails, stop and re-record the
calibration clip. Do not proceed and hope. That is how this project spent
weeks on a 20% baseline error.

Then convert for Pose2Sim:

```bash
python scripts/make_calib_v2.py
```

It refuses to write a TOML with implausible fx or k1, which is the check
that would have caught the original broken file immediately.

## 3. Split the wide camera

If one camera frames two people and the others frame one each:

```bash
python scripts/split_main.py
```

Creates `p2s_mainL/` and `p2s_mainR/`, each with cropped video and a
calibration whose principal point is shifted for the crop. One person per
project, which is where the 3.5 px result came from.

## 4. Pose2Sim

`Config.toml` settings that matter:

```toml
pose_model          = 'Whole_body'   # RTMW, COCO_133: body+feet+face+hands
multi_person        = false          # true was measurably worse here
do_synchronization  = true           # NEVER skip this
tracked_keypoint    = 'Nose'         # 'Neck' isn't in COCO_133
```

```bash
python scripts/run_pose2sim.py ~/projects/p2s_mainR
```

The preflight refuses to start on an implausible calibration or with
synchronisation off.

**Read the synchronisation log.** You want a small offset at high
correlation (we got 2 and 6 frames at 0.92). A large offset at low
correlation (0.5–0.6) means it latched onto the wrong person or the wrong
motion. It does not mean your cameras were really 4.5 s apart.

Gate: mean reprojection < 10 px, and 0% camera exclusion.

## 5. Visual check, not optional

The supervisor's standing rule: *for everything we say, bring an annotated
video so results can be evaluated.*

```bash
python scripts/find_axes.py   ~/projects/p2s_mainR mainR   # confirm mapping
python scripts/overlay_full.py ~/projects/p2s_mainR mainR 240 300
```

`find_axes.py` must show the winner beating the runner-up by **> 2×**. If it
doesn't, the mapping is ambiguous and something upstream is wrong.

If the overlay looks like an unannotated video: TRC is in **metres** (not
mm), and check the axis mapping. Those two accounted for every "0 skeletons
drawn" we had.

## 6. Quantitative validation

```bash
python validation/eval_body.py ~/projects/p2s_mainR/pose-3d/*_filt_*.trc
```

Thresholds are declared in the script, before seeing results: robust
CV < 0.08 pass, > 0.15 fail. Also check bones against anatomical ranges, 
low CV with bones out of range means a *consistent but mis-scaled*
reconstruction, i.e. an extrinsic error, not noise.

## 7. Face

```bash
python validation/face3d_both.py ~/projects/p2s_mainR mainR right
```

Validate on **biocular width** (8.8–9.8 cm adult, CV < 0.05). It's
anatomically fixed and wasn't optimised for, so unlike reprojection error it
can actually fail.

## 8. Animation (optional, low priority)

Supervisor: *focus more on the body and do not spend much time on it.*

```bash
python animation/soma_video.py p2s_mainR 240 450     # GPU, ~40 min
python animation/local_fit.py chest <trc>            # CPU, fit quality only
```

Two things in `soma_video.py` must stay as they are: `scale_params` frozen
(otherwise finger scales blow up and you get spiders), and no `Hips`→`Hip`
mapping (different semantics), measurably worse (17.0 vs 13.6 cm).

## Cluster notes

- Never compute on the login node.
- `#SBATCH --partition=a100` (or `rtx3080`/`v100`) is **required**, 
  omitting it gives "Requested node configuration is not available".
- `PYTHONNOUSERSITE=1` in the job script, or `~/.local` packages shadow the
  venv and you get things like `AttributeError: np.long` from a scipy built
  against a different numpy.
- `python -u` or `nohup` logs stay empty until the job ends.
- Watch the `/home/hpc` quota (100 G soft). When it's exceeded, `cp` and
  `sed` fail *silently*. A 0-byte backup, `sed: couldn't flush`. Large
  outputs belong in `/home/woody` (1000 G), symlinked back if a script
  expects them under `~`.
