#!/usr/bin/env python3
"""
Brute-force the TRC -> camera axis mapping.

Problem
-------
A Pose2Sim TRC gives you 3D coordinates in metres, but the axis ORDER and
SIGNS relative to the camera frame are not something you should guess.  Get
it wrong and the projection lands behind the camera or thousands of pixels
off-screen, and your overlay silently writes an un-annotated video.

Approach
--------
There are 6 permutations x 8 sign combinations = 48 candidate mappings.  For
each one, project the 3D body joints into a camera and measure mean distance
to the 2D keypoints that the pose stage already detected in that camera.  The
correct mapping wins by an order of magnitude; if it does not, something else
is wrong (wrong calibration, wrong frame alignment) and you should stop
rather than pick the least-bad option.

Result on this data
-------------------
    perm = (2, 0, 1)  signs = (+1, +1, +1)   ->   24.0 px
    runner-up                                ->  203.1 px

8.5x gap, so unambiguous.

Usage:  python find_axes.py <project_dir> <camera_name>
"""

import sys, json, itertools
from pathlib import Path
import numpy as np
import cv2
import rtoml

sys.path.insert(0, str(Path(__file__).parent))
from overlay_full import read_trc          # noqa: E402

BODY_IDX = list(range(17))                 # body joints only; face/hands are
                                           # too small to drive this


def load_2d(proj: Path, camname: str, nframes: int):
    """Pose2Sim writes one OpenPose-style JSON per frame per camera."""
    cands = sorted(proj.glob(f"pose*/**/*{camname}*"))
    folder = next((c for c in cands if c.is_dir()), None)
    if folder is None:
        sys.exit(f"no pose JSON folder matching {camname}")
    files = sorted(folder.glob("*.json"))
    print(f"{folder.name}: {len(files)} json")
    out = np.full((nframes, 133, 3), np.nan)
    for i, f in enumerate(files[:nframes]):
        d = json.loads(f.read_text())
        if not d.get("people"):
            continue
        # largest-confidence person
        best = max(d["people"],
                   key=lambda p: np.nansum(p["pose_keypoints_2d"][2::3]))
        kp = np.array(best["pose_keypoints_2d"], float).reshape(-1, 3)
        out[i, : len(kp)] = kp
    return out


def main():
    proj = Path(sys.argv[1]).resolve()
    camname = sys.argv[2]

    cal = rtoml.load(proj / "Calibration" / "calibration.toml")
    cam = next(v for k, v in cal.items()
               if k != "metadata" and v["name"] == camname)
    K = np.array(cam["matrix"], float)
    R = cv2.Rodrigues(np.array(cam["rotation"], float))[0]
    T = np.array(cam["translation"], float).reshape(3, 1)

    trc = sorted((proj / "pose-3d").glob("*_filt_*.trc")) \
       or sorted((proj / "pose-3d").glob("*.trc"))
    P3 = read_trc(trc[0])
    kp2 = load_2d(proj, camname, len(P3))

    results = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1, -1), repeat=3):
            errs = []
            for f in range(len(P3)):
                for j in BODY_IDX:
                    p, q = P3[f, j], kp2[f, j]
                    if not (np.all(np.isfinite(p)) and q[2] > 0.3):
                        continue
                    v = np.array([[signs[0] * p[perm[0]]],
                                  [signs[1] * p[perm[1]]],
                                  [signs[2] * p[perm[2]]]], float)
                    X = R @ v + T
                    if X[2, 0] < 0.2:
                        continue
                    h = K @ X
                    errs.append(np.hypot(h[0, 0] / h[2, 0] - q[0],
                                         h[1, 0] / h[2, 0] - q[1]))
            if len(errs) > 50:
                results.append((float(np.mean(errs)), perm, signs, len(errs)))

    results.sort()
    print(f"\n{'mean px':>10}  perm        signs        n")
    for e, perm, signs, n in results[:8]:
        print(f"{e:10.1f}  {perm}  {signs}  {n}")

    if len(results) < 2:
        sys.exit("not enough valid candidates -- check calibration")
    best, second = results[0][0], results[1][0]
    print(f"\nwinner {results[0][1]} {results[0][2]} at {best:.1f} px, "
          f"{second/best:.1f}x better than runner-up")
    if second / best < 2.0:
        print("WARNING gap < 2x -- the mapping is NOT unambiguous. "
              "Do not proceed; something else is wrong.")


if __name__ == "__main__":
    main()
