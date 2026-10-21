#!/usr/bin/env python3
"""
Can we recompute the extrinsics from the footage we already have?

Answer for this dataset: NO.  Run this before promising anyone a
recalibration.

Stereo extrinsics require the calibration target to be visible in TWO
cameras AT THE SAME INSTANT.  Intrinsics do not -- each camera can be
calibrated alone, which is why our intrinsics are solid (held-out 80/20
split: main 0.350 px, left 0.594 px, right 0.869 px) while the extrinsics
are inherited from an external import we cannot verify.

This script scans the calibration window across all three cameras and counts
how often the ChArUco board is detected with >= MIN_CORNERS corners in each
camera, and how often in each PAIR simultaneously.

Result on the therapy-room recording (28:00-37:30, 3 fps):

    main  + left   ->   0 co-visible frames
    main  + right  ->   6 co-visible frames
    left  + right  ->   0 co-visible frames

Six frames is not a calibration.  The board was walked around and shown to
one camera at a time, which is correct for intrinsics and useless for
extrinsics.  This is the single thing to fix in the next recording session.

Board
-----
DICT_4X4_50, 7x5 squares, 50 mm square / 37 mm marker -> 24 interior corners.
Printed at A3 it was too small for the main camera to resolve at room
distance; hence the 2x upscale retry below, and hence "print it bigger" in
the runbook.
"""

from pathlib import Path
from itertools import combinations
import numpy as np
import cv2

VIDEOS = {
    "main":  Path.home() / "projects/raw/cam0_main.mp4",
    "left":  Path.home() / "projects/raw/cam2_patient.mp4",
    "right": Path.home() / "projects/raw/cam1_doctor.mp4",
}
START_S, END_S, SAMPLE_FPS = 28 * 60, 37 * 60 + 30, 3.0
MIN_CORNERS = 12

SQUARES_X, SQUARES_Y = 7, 5
SQUARE_LEN, MARKER_LEN = 0.050, 0.037

adict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
board = cv2.aruco.CharucoBoard((SQUARES_X, SQUARES_Y),
                               SQUARE_LEN, MARKER_LEN, adict)
det = cv2.aruco.CharucoDetector(board)


def detect(gray):
    """Returns corner count.  Retries at 2x because the printed board was
    too small for the wide main camera -- upscaling genuinely recovers
    detections, it is not cargo cult."""
    for scale in (1.0, 2.0):
        g = gray if scale == 1.0 else cv2.resize(
            gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        cc, cids, _, _ = det.detectBoard(g)
        if cids is not None and len(cids) >= MIN_CORNERS:
            return len(cids)
    return 0


def scan(path: Path):
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    step = max(1, int(round(fps / SAMPLE_FPS)))
    hits = {}
    f = int(START_S * fps)
    end = int(END_S * fps)
    while f < end:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, frame = cap.read()
        if not ok:
            break
        n = detect(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        if n:
            hits[round(f / fps, 2)] = n
        f += step
    cap.release()
    return hits


if __name__ == "__main__":
    all_hits = {}
    for name, path in VIDEOS.items():
        if not path.exists():
            print(f"{name}: MISSING {path}")
            continue
        h = scan(path)
        all_hits[name] = h
        print(f"{name:>6}: board detected in {len(h)} sampled frames"
              + (f", median {int(np.median(list(h.values())))} corners"
                 if h else ""))

    print()
    tol = 1.0 / SAMPLE_FPS / 2          # half a sample interval
    for a, b in combinations(all_hits, 2):
        ta = np.array(sorted(all_hits[a]))
        tb = np.array(sorted(all_hits[b]))
        n = 0
        for t in ta:
            if len(tb) and np.min(np.abs(tb - t)) <= tol:
                n += 1
        print(f"{a:>6} + {b:<6} -> {n} co-visible frames")
        if n < 30:
            print(f"         {n} is not enough for stereo extrinsics "
                  "(want >= 30 well-spread views)")
