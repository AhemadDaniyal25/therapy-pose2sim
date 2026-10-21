#!/usr/bin/env python3
"""
Split the wide main camera into two halves and build a calibration for each.

This is the step that produced the best reconstruction in the whole project
(3.5 px, 0% camera exclusion, bone-length CV 0.011).

Reasoning
---------
main is 1920x1080 and frames BOTH people.  Each side camera frames ONE person.
In a three-camera project Pose2Sim therefore has to decide which main-camera
detection belongs with which side-camera detection, and it gets it wrong --
you see the skeleton jump between people mid-sequence, and `cam2 excluded
100%` in the triangulation log.

Cropping main into two 960x1080 halves gives one person per project, and the
association problem disappears:

    p2s_mainL  =  main[:, 0:960]      +  left camera    -> Person A
    p2s_mainR  =  main[:, 960:1920]   +  right camera   -> Person B

The pairing is not assumed.  Pose2Sim's own synchronisation measured 2- and
6-frame offsets at 0.92 correlation for these pairs, versus 136 and 113
frames at 0.52-0.59 for the unsplit case.  High correlation at a small offset
is the evidence that the two views contain the same person.

Principal point
---------------
A crop translates the image, so cx must move with it:

    left  half:  cx stays 898.6
    right half:  cx becomes 898.6 - 960 = -61.4

Negative cx is correct and intentional -- the lens optical centre lies
outside the right-hand crop.  Do not "fix" it.  fx, fy, cy, and all
distortion coefficients are unchanged: cropping does not alter the lens.
"""

import shutil
import subprocess
from pathlib import Path
import numpy as np
import cv2
import rtoml

SRC_VIDEO = Path.home() / "projects/pose2sim_therapy_demo/videos/main.mp4"
NPZ       = Path.home() / "projects/therapy-room-pose-pipeline/calibration/all_cameras.npz"
SIDE      = {"L": "left", "R": "right"}
ROOT      = Path.home() / "projects"

HALF_W, FULL_H = 960, 1080


def _probe(path: Path):
    """Replaces ffprobe. cv2 is already a hard dependency, ffprobe is not,
    and there is no ffmpeg module on woody."""
    cap = cv2.VideoCapture(str(path))
    info = dict(w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                fps=float(cap.get(cv2.CAP_PROP_FPS)),
                n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    cap.release()
    return info


def crop_ffmpeg(side: str, out: Path) -> bool:
    """Preferred when ffmpeg exists: faster and better quality. Returns
    False if ffmpeg is unavailable so the caller can fall back."""
    if shutil.which("ffmpeg") is None:
        return False
    x = 0 if side == "L" else HALF_W
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-i", str(SRC_VIDEO),
           "-filter:v", f"crop={HALF_W}:{FULL_H}:{x}:0",
           "-c:v", "libx264", "-crf", "18", "-preset", "medium",
           "-an", str(out)]
    subprocess.run(cmd, check=True)
    return True


def crop_cv2(side: str, out: Path):
    """Fallback with no external binaries. Slower and re-encodes with
    whatever codec cv2 has (usually mp4v), which is fine for pose
    estimation: RTMPose sees decoded frames either way."""
    x0 = 0 if side == "L" else HALF_W
    out.parent.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(SRC_VIDEO))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    if w < x0 + HALF_W:
        raise SystemExit(f"source is {w}px wide, cannot crop at x={x0}")
    vw = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"),
                         fps, (HALF_W, FULL_H))
    if not vw.isOpened():
        raise SystemExit(f"cv2 could not open a writer for {out}")
    n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        vw.write(frame[0:FULL_H, x0:x0 + HALF_W])
        n += 1
        if n % 500 == 0:
            print(f"  {n} frames", flush=True)
    cap.release(); vw.release()
    if n == 0:
        raise SystemExit(f"wrote 0 frames from {SRC_VIDEO}")
    print(f"  cv2 wrote {n} frames")


def crop(side: str, out: Path):
    if not SRC_VIDEO.exists():
        raise SystemExit(f"source video not found: {SRC_VIDEO}")
    src = _probe(SRC_VIDEO)
    print(f"source: {src['w']}x{src['h']} @ {src['fps']:.3f} fps, "
          f"{src['n']} frames")
    if src["w"] != 2 * HALF_W:
        print(f"  WARNING expected width {2*HALF_W}, got {src['w']}")

    if crop_ffmpeg(side, out):
        print("  (used ffmpeg)")
    else:
        print("  ffmpeg not on PATH, using cv2")
        crop_cv2(side, out)

    # verify, do not trust
    got = _probe(out)
    assert (got["w"], got["h"]) == (HALF_W, FULL_H), \
        f"{out.name}: got {got['w']}x{got['h']}"
    if got["n"] and src["n"] and abs(got["n"] - src["n"]) > 2:
        print(f"  WARNING frame count {got['n']} != source {src['n']}")
    print(f"{out.name}: {got['w']}x{got['h']}, {got['n']} frames, "
          f"{got['fps']:.3f} fps")


def calib(side: str, out_toml: Path):
    npz = np.load(NPZ)
    name = SIDE[side]

    K_main = npz["K_main"].copy()
    if side == "R":
        K_main[0, 2] -= HALF_W          # the whole point of this script
    D_main = npz["D_main"].ravel()

    R = npz[f"R_main_{name}"]
    T = npz[f"T_main_{name}"]
    K_side = npz[f"K_{name}"]
    D_side = npz[f"D_{name}"].ravel()

    entries = [
        ("cam0", f"main{side}", (HALF_W, FULL_H), K_main, D_main,
         np.eye(3), np.zeros((3, 1))),
        ("cam1", name, (1080, 1920), K_side, D_side, R, T),
    ]

    lines = []
    for key, nm, (w, h), K, D, Rm, Tm in entries:
        rvec = cv2.Rodrigues(np.asarray(Rm, float))[0].ravel()
        tvec = np.asarray(Tm, float).ravel()
        lines += [
            f"[{key}]",
            f'name = "{nm}"',
            f"size = [ {float(w)}, {float(h)},]",
            ("matrix = [ "
             f"[ {float(K[0,0])}, 0.0, {float(K[0,2])},], "
             f"[ 0.0, {float(K[1,1])}, {float(K[1,2])},], "
             "[ 0.0, 0.0, 1.0,],]"),
            "distortions = [ " + ", ".join(f"{float(d)}" for d in D[:4]) + ",]",
            "rotation = [ " + ", ".join(f"{float(r)}" for r in rvec) + ",]",
            "translation = [ " + ", ".join(f"{float(t)}" for t in tvec) + ",]",
            "fisheye = false",
            "",
        ]
        print(f"  {nm}: cx={float(K[0,2]):.1f} fx={float(K[0,0]):.1f}")
    lines += ["[metadata]", "adjusted = false", "error = 0.0", ""]

    out_toml.parent.mkdir(parents=True, exist_ok=True)
    out_toml.write_text("\n".join(lines))
    rtoml.load(out_toml)             # parse check
    print(f"  wrote {out_toml}")


if __name__ == "__main__":
    for side in ("L", "R"):
        proj = ROOT / f"p2s_main{side}"
        print(f"=== p2s_main{side} ===")
        crop(side, proj / "videos" / f"main{side}.mp4")
        # the side camera video is copied in unchanged
        src = Path.home() / f"projects/pose2sim_therapy_demo/videos/{SIDE[side]}.mp4"
        dst = proj / "videos" / f"{SIDE[side]}.mp4"
        if not dst.exists():
            dst.write_bytes(src.read_bytes())
        calib(side, proj / "Calibration" / "calibration.toml")

    print("\nNow copy Config.toml into each project, set")
    print("    pose_model = 'Whole_body'")
    print("    do_synchronization = true")
    print("    multi_person = false")
    print("and run the Pose2Sim stages in order.")
