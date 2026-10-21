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


def crop(side: str, out: Path):
    """ffmpeg crop. -c:v libx264 -crf 18 keeps it visually lossless enough
    for pose estimation while staying small."""
    x = 0 if side == "L" else HALF_W
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-i", str(SRC_VIDEO),
        "-filter:v", f"crop={HALF_W}:{FULL_H}:{x}:0",
        "-c:v", "libx264", "-crf", "18", "-preset", "medium",
        "-an", str(out),
    ]
    subprocess.run(cmd, check=True)
    # verify, do not trust
    cap = cv2.VideoCapture(str(out))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    assert (w, h) == (HALF_W, FULL_H), f"{out.name}: got {w}x{h}"
    print(f"{out.name}: {w}x{h}, {n} frames")


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
