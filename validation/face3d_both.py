#!/usr/bin/env python3
"""
Multi-camera 3D face keypoints:  MediaPipe 478 landmarks in two views ->
triangulated 3D face, with metrics.

This is supervisor task "Implement multi-camera face keypoint detection".

Why not use Pose2Sim's own face keypoints
-----------------------------------------
Pose2Sim with pose_model = 'Whole_body' does give you 68 face keypoints as
part of COCO_133, and they triangulate fine.  But 68 sparse points is a jaw
outline and some eye corners -- it is not a face you can animate or measure
expression from.  MediaPipe Face Landmarker gives 478 with consistent
indexing, including iris.  So: Pose2Sim for the body, MediaPipe for the face,
fused in the same calibrated 3D space.

The one trick that makes it work
--------------------------------
At room distance the face is ~40 px wide in a 1920x1080 frame and MediaPipe
finds nothing.  So we do not run MediaPipe on the full frame.  We take the
head bounding box from the COCO_133 face keypoints (indices 23-91) that
Pose2Sim already detected, crop it, upscale it, and run MediaPipe on the
crop, then map the landmarks back to full-frame pixels.  Upscale factors are
different per camera because the main camera is much further away:

    main half:  x5      side camera:  x2

Detection rates with the crop+upscale (vs essentially 0% on full frames):
    main half  85% (260/305) for Person B,  50% (154/305) for Person A
    side cam  100% (302/302) / 98% (297/302)

Synchronisation
---------------
Rather than trusting the Pose2Sim sync offset, we search offsets in
[-12, +12] frames and keep the one that minimises reprojection error.  It
agreed with Pose2Sim (small offsets), which is a useful independent check.

Results
-------
    Person B (mainR + right):  259 frames fused, median 1.46 px, p95 7.19 px,
                               biocular width 8.53 cm, CV 0.043
    Person A (mainL + left):   147 frames fused, median 12.35 px, p95 25.08,
                               biocular width 7.52 cm, CV 0.073

Biocular (outer-eye-corner) width is the validation metric, because it is
anatomically fixed and was not optimised for.  Adult range is 8.8-9.8 cm, so
Person B at 8.53 cm is slightly under-scaled and Person A at 7.52 cm clearly
is -- consistent with the extrinsic translation error documented in
docs/CALIBRATION_MAP.md.  CV < 0.05 means the reconstruction is stable
frame-to-frame.

Usage:  python face3d_both.py <project_dir> <mainX_name> <side_name>
"""

import sys, json
from pathlib import Path
import numpy as np
import cv2
import rtoml
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

OUT_ROOT = Path("/home/woody/iwso/iwso240h/face3d_v2")
MODEL = Path.home() / "models/face_landmarker.task"

FACE_KP = slice(23, 91)        # COCO_133 face block, used only for the crop
# MediaPipe canonical outer eye corners
L_OUTER, R_OUTER = 33, 263
UPSCALE = {"main": 5.0, "side": 2.0}
OFFSETS = range(-12, 13)


def landmarker():
    opts = vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(MODEL)),
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.2,   # low: crops are small and soft
        min_face_presence_confidence=0.2)
    return vision.FaceLandmarker.create_from_options(opts)


def head_box(kp2d, W, H, pad=0.45):
    """Bounding box of the COCO_133 face keypoints, padded.  Returns None if
    too few confident points -- better to skip a frame than to crop noise."""
    pts = kp2d[FACE_KP]
    pts = pts[pts[:, 2] > 0.3][:, :2]
    if len(pts) < 20:
        return None
    lo, hi = pts.min(0), pts.max(0)
    c, s = (lo + hi) / 2, (hi - lo).max() * (1 + pad)
    x0 = int(max(0, c[0] - s / 2)); y0 = int(max(0, c[1] - s / 2))
    x1 = int(min(W, c[0] + s / 2)); y1 = int(min(H, c[1] + s / 2))
    if x1 - x0 < 24 or y1 - y0 < 24:
        return None
    return x0, y0, x1, y1


def run_camera(video: Path, kp2d, scale, lm, tag):
    """-> {frame: (478,2) full-frame pixels}"""
    cap = cv2.VideoCapture(str(video))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    res, tried = {}, 0
    for f in range(len(kp2d)):
        ok, frame = cap.read()
        if not ok:
            break
        bb = head_box(kp2d[f], W, H)
        if bb is None:
            continue
        tried += 1
        x0, y0, x1, y1 = bb
        crop = frame[y0:y1, x0:x1]
        crop = cv2.resize(crop, None, fx=scale, fy=scale,
                          interpolation=cv2.INTER_CUBIC)
        img = mp.Image(image_format=mp.ImageFormat.SRGB,
                       data=cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        out = lm.detect(img)
        if not out.face_landmarks:
            continue
        ch, cw = crop.shape[:2]
        pts = np.array([[p.x * cw / scale + x0, p.y * ch / scale + y0]
                        for p in out.face_landmarks[0]])
        res[f] = pts
    cap.release()
    print(f"  {tag}: {len(res)}/{tried} crops gave landmarks "
          f"({100*len(res)/max(tried,1):.0f}%)")
    return res


def triangulate(P1, P2, x1, x2):
    X = cv2.triangulatePoints(P1, P2, x1.T.astype(float), x2.T.astype(float))
    return (X[:3] / X[3]).T


def main():
    proj = Path(sys.argv[1]).resolve()
    mainname, sidename = sys.argv[2], sys.argv[3]
    tag = proj.name

    cal = rtoml.load(proj / "Calibration" / "calibration.toml")
    cams = {}
    for k, v in cal.items():
        if k == "metadata":
            continue
        K = np.array(v["matrix"], float)
        R = cv2.Rodrigues(np.array(v["rotation"], float))[0]
        T = np.array(v["translation"], float).reshape(3, 1)
        cams[v["name"]] = dict(K=K, P=K @ np.hstack([R, T]),
                               D=np.array(v["distortions"], float))

    # 2D keypoints that Pose2Sim already produced, per camera
    sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
    from find_axes import load_2d                             # noqa: E402
    NF = 320                      # frames to process; bump for a full run
    kp_main = load_2d(proj, mainname, NF)
    kp_side = load_2d(proj, sidename, NF)

    lm = landmarker()
    print(f"{tag}: running MediaPipe on head crops")
    f_main = run_camera(next((proj / "videos").glob(f"{mainname}.*")),
                        kp_main, UPSCALE["main"], lm, mainname)
    f_side = run_camera(next((proj / "videos").glob(f"{sidename}.*")),
                        kp_side, UPSCALE["side"], lm, sidename)

    P1, P2 = cams[mainname]["P"], cams[sidename]["P"]
    K1, K2 = cams[mainname]["K"], cams[sidename]["K"]

    # offset search: pick the shift that minimises reprojection
    best = None
    for off in OFFSETS:
        common = [f for f in f_main if (f + off) in f_side]
        if len(common) < 30:
            continue
        errs = []
        for f in common[:60]:
            x1, x2 = f_main[f], f_side[f + off]
            X = triangulate(P1, P2, x1, x2)
            for P, K, x in ((P1, K1, x1), (P2, K2, x2)):
                h = (P @ np.hstack([X, np.ones((len(X), 1))]).T)
                uv = (h[:2] / h[2]).T
                errs.append(np.linalg.norm(uv - x, axis=1))
        m = float(np.median(np.concatenate(errs)))
        if best is None or m < best[1]:
            best = (off, m, len(common))
    if best is None:
        sys.exit("no offset gave >= 30 common frames")
    off, _, ncommon = best
    print(f"  best offset {off:+d} frames, {ncommon} common frames")

    frames, pts3d, allerr, biocular = [], [], [], []
    for f in sorted(f_main):
        if (f + off) not in f_side:
            continue
        x1, x2 = f_main[f], f_side[f + off]
        X = triangulate(P1, P2, x1, x2)
        e = []
        for P, x in ((P1, x1), (P2, x2)):
            h = P @ np.hstack([X, np.ones((len(X), 1))]).T
            e.append(np.linalg.norm((h[:2] / h[2]).T - x, axis=1))
        frames.append(f)
        pts3d.append(X)
        allerr.append(np.concatenate(e))
        biocular.append(float(np.linalg.norm(X[L_OUTER] - X[R_OUTER])))

    E = np.concatenate(allerr)
    bw = np.array(biocular)
    med_bw = float(np.median(bw))
    cv_bw = float(np.std(bw) / med_bw)

    print(f"\n  frames fused        {len(frames)}")
    print(f"  reproj median       {np.median(E):.2f} px")
    print(f"  reproj p95          {np.percentile(E,95):.2f} px")
    print(f"  biocular width      {med_bw*100:.2f} cm   (adult 8.8-9.8)")
    print(f"  biocular CV         {cv_bw:.3f}            (stable if < 0.05)")
    if med_bw < 0.088:
        print("  -> under-scaled; consistent with extrinsic translation error")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT_ROOT / f"face3d_{tag}.npz",
                        frames=np.array(frames),
                        points=np.stack(pts3d), offset=off)
    (OUT_ROOT / f"face3d_{tag}_metrics.json").write_text(json.dumps(dict(
        project=tag, frames=len(frames), offset=off,
        reproj_median_px=float(np.median(E)),
        reproj_p95_px=float(np.percentile(E, 95)),
        biocular_m=med_bw, biocular_cv=cv_bw), indent=2))
    print(f"  wrote {OUT_ROOT}/face3d_{tag}.npz")


if __name__ == "__main__":
    main()
