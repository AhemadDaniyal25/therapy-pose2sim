#!/usr/bin/env python3
"""
Reproject the triangulated 3D skeleton back onto the source footage.

This is the deliverable the supervisor asks for every time:  "for everything
we say, bring an annotated video so results can be evaluated."  A number in
a log is not evidence; a skeleton that tracks the person for 300 frames is.

Two things cost us a lot of time here, both written down so nobody repeats
them.

1. UNITS.  Pose2Sim TRC files are in METRES.  I divided by 1000 assuming
   millimetres and every joint landed behind the camera, so the overlay
   wrote 0 skeletons onto a perfectly normal-looking video.  If your overlay
   is "just the original video with no annotations", this is the first thing
   to check.

2. AXIS CONVENTION.  The TRC axis order is not the camera axis order, and it
   is not documented in a way you can rely on.  Rather than guess, run
   scripts/find_axes.py, which brute-forces all 48 permutation/sign
   combinations against the detected 2D keypoints and reports mean
   reprojection error for each.  On this data the winner was

       perm = (2, 0, 1),  signs = (+1, +1, +1)   ->  24.0 px
       runner-up                                 ->  203.1 px

   An order-of-magnitude gap means the answer is unambiguous.  That mapping
   is hardcoded below as the X = R @ [[P[2]],[P[0]],[P[1]]] line.

Usage:  python overlay_full.py <project_dir> <camera_name> [start] [count]
"""

import sys
from pathlib import Path
import numpy as np
import cv2
import rtoml

# ---------------------------------------------------------------- COCO_133
# Pose2Sim 'Whole_body' = RTMW -> COCO_133:
#   0-16   body
#   17-22  feet
#   23-90  face (68)
#   91-132 hands (21 left + 21 right)
BODY = [(5,7),(7,9),(6,8),(8,10),(5,6),(5,11),(6,12),(11,12),
        (11,13),(13,15),(12,14),(14,16),(0,5),(0,6),
        (15,17),(15,18),(15,19),(16,20),(16,21),(16,22)]

def _hand(base):
    # 21-point MediaPipe-style hand topology
    e = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
         (0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),(15,16),
         (0,17),(17,18),(18,19),(19,20),(5,9),(9,13),(13,17)]
    return [(base+a, base+b) for a, b in e]

HAND = _hand(91) + _hand(112)

# 68-point face contours, offset to index 23
def _chain(a, b):   return [(i, i+1) for i in range(a, b)]
def _loop(a, b):    return _chain(a, b) + [(b, a)]
FACE = ([(23+a, 23+b) for a, b in _chain(0, 16)]      # jaw
      + [(23+a, 23+b) for a, b in _chain(17, 21)]     # brows
      + [(23+a, 23+b) for a, b in _chain(22, 26)]
      + [(23+a, 23+b) for a, b in _chain(27, 30)]     # nose bridge
      + [(23+a, 23+b) for a, b in _chain(31, 35)]
      + [(23+a, 23+b) for a, b in _loop(36, 41)]      # eyes
      + [(23+a, 23+b) for a, b in _loop(42, 47)]
      + [(23+a, 23+b) for a, b in _loop(48, 59)]      # outer lips
      + [(23+a, 23+b) for a, b in _loop(60, 67)])     # inner lips

GROUPS = [(BODY, (0, 255, 0), 3),          # body thick green
          (HAND, (0, 200, 255), 2),        # hands amber
          (FACE, (255, 150, 0), 1)]        # face thin blue


def read_trc(path):
    """TRC: 5 header lines, then frame / time / X1 Y1 Z1 X2 Y2 Z2 ...
    Values are metres.  Missing markers appear as blanks or nan."""
    lines = Path(path).read_text().splitlines()
    rows = []
    for ln in lines[6:]:
        parts = ln.split("\t")
        if len(parts) < 5:
            continue
        vals = []
        for p in parts[2:]:
            p = p.strip()
            vals.append(float(p) if p not in ("", "nan", "NaN") else np.nan)
        n = len(vals) // 3
        rows.append(np.array(vals[: n * 3]).reshape(n, 3))
    return np.stack(rows)                   # (frames, markers, 3) in METRES


def main():
    proj = Path(sys.argv[1]).resolve()
    camname = sys.argv[2]
    start = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    count = int(sys.argv[4]) if len(sys.argv) > 4 else 300

    cal = rtoml.load(proj / "Calibration" / "calibration.toml")
    cam = next(v for k, v in cal.items()
               if k != "metadata" and v["name"] == camname)
    K = np.array(cam["matrix"], float)
    D = np.array(cam["distortions"], float)
    R = cv2.Rodrigues(np.array(cam["rotation"], float))[0]
    T = np.array(cam["translation"], float).reshape(3, 1)
    W, H = int(cam["size"][0]), int(cam["size"][1])

    trc = sorted((proj / "pose-3d").glob("*_filt_*.trc")) \
       or sorted((proj / "pose-3d").glob("*.trc"))
    if not trc:
        sys.exit("no TRC in pose-3d/")
    P3 = read_trc(trc[0])
    print(f"{trc[0].name}: {P3.shape[0]} frames x {P3.shape[1]} markers")

    vid = next((proj / "videos").glob(f"{camname}.*"))
    cap = cv2.VideoCapture(str(vid))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    out = cv2.VideoWriter(str(proj / f"overlay_{camname}.mp4"),
                          cv2.VideoWriter_fourcc(*"mp4v"), 30, (W, H))

    def proj_pt(p):
        """3D TRC point (metres) -> pixel, or None if not projectable."""
        if not np.all(np.isfinite(p)):
            return None
        # axis mapping established empirically by find_axes.py
        X = R @ np.array([[p[2]], [p[0]], [p[1]]], float) + T
        if X[2, 0] < 0.2:                   # behind, or implausibly close
            return None
        q = K @ X
        u, v = q[0, 0] / q[2, 0], q[1, 0] / q[2, 0]
        if not (-W < u < 2 * W and -H < v < 2 * H):
            return None
        return int(round(u)), int(round(v))

    drawn = 0
    for i in range(count):
        ok, frame = cap.read()
        if not ok:
            break
        fi = start + i
        if fi >= len(P3):
            break
        pts = [proj_pt(p) for p in P3[fi]]
        nseg = 0
        for edges, colour, thick in GROUPS:
            for a, b in edges:
                if a < len(pts) and b < len(pts) and pts[a] and pts[b]:
                    cv2.line(frame, pts[a], pts[b], colour, thick,
                             cv2.LINE_AA)
                    nseg += 1
        for p in pts[:23]:
            if p:
                cv2.circle(frame, p, 4, (0, 0, 255), -1, cv2.LINE_AA)
        if nseg:
            drawn += 1
        cv2.putText(frame, f"f{fi}  {nseg} seg", (12, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        out.write(frame)

    cap.release(); out.release()
    print(f"overlay_{camname}.mp4: {drawn} frames with a skeleton")
    if drawn == 0:
        print("0 frames drawn -> check units (TRC is METRES) and the axis "
              "mapping (run find_axes.py)")


if __name__ == "__main__":
    main()
