#!/usr/bin/env python3
"""
all_cameras.npz  ->  Pose2Sim Calibration/calibration.toml

Why this exists
---------------
The calibration.toml that shipped with the Pose2Sim demo project was
non-physical:  cam_1 had fx = 8712 px on a 1080-px-wide image (~7 deg
horizontal FOV), k1 = 9.09 and -31.8, every principal point pinned to the
exact image centre, and translations ~1000x too large.  Running Pose2Sim
against it reported a ~41,900 mm metric error.

This script rebuilds the TOML from the reference calibration npz, which has
plausible intrinsics (reproduced independently with a held-out 80/20 split:
main 0.350 px, left 0.594 px, right 0.869 px).

Caveat you must pass on
-----------------------
The *extrinsics* in that npz were imported from an external codebase, not
computed from this footage.  The npz key is called `rms` but it holds
`mean_translation_residual` IN METRES (0.636 main-left, 0.572 main-right).
See docs/CALIBRATION_MAP.md.

Gotcha
------
Pose2Sim reads the TOML with rtoml, which chokes on numpy scalar reprs
("np.float64(1234.5)").  Everything written below must be a plain Python
float.  Hence the float() casts and the .tolist() calls.
"""

from pathlib import Path
import numpy as np
import cv2

NPZ = Path.home() / "projects/therapy-room-pose-pipeline/calibration/all_cameras.npz"
OUT = Path.home() / "projects/pose2sim_therapy_demo/Calibration/calibration.toml"

npz = np.load(NPZ)
print("npz keys:", list(npz.keys()))

# main camera defines the world frame:  P = K [I | 0]
# cam1 = left, cam2 = right, each with R,T relative to main.
cams = [
    ("cam0", "main",  (1920, 1080), np.eye(3),            np.zeros((3, 1))),
    ("cam1", "left",  (1080, 1920), npz["R_main_left"],   npz["T_main_left"]),
    ("cam2", "right", (1080, 1920), npz["R_main_right"],  npz["T_main_right"]),
]

lines = []
for key, name, (w, h), R, T in cams:
    K = npz[f"K_{name}"]
    D = npz[f"D_{name}"].ravel()

    # sanity: a webcam in a small room has fx roughly between 0.5*w and 3*w
    fx, fy = float(K[0, 0]), float(K[1, 1])
    if not (0.4 * w < fx < 4.0 * w):
        raise SystemExit(f"{name}: fx={fx:.1f} implausible for width {w}")
    if abs(float(D[0])) > 1.5:
        raise SystemExit(f"{name}: k1={D[0]:.3f} implausible")

    # axis-angle, because Pose2Sim stores rotation as a 3-vector
    rvec = cv2.Rodrigues(np.asarray(R, dtype=float))[0].ravel()
    tvec = np.asarray(T, dtype=float).ravel()

    lines.append(f"[{key}]")
    lines.append(f'name = "{name}"')
    lines.append(f"size = [ {float(w)}, {float(h)},]")
    lines.append(
        "matrix = [ "
        f"[ {fx}, 0.0, {float(K[0,2])},], "
        f"[ 0.0, {fy}, {float(K[1,2])},], "
        "[ 0.0, 0.0, 1.0,],]"
    )
    lines.append("distortions = [ " + ", ".join(f"{float(d)}" for d in D[:4]) + ",]")
    lines.append("rotation = [ " + ", ".join(f"{float(r)}" for r in rvec) + ",]")
    lines.append("translation = [ " + ", ".join(f"{float(t)}" for t in tvec) + ",]")
    lines.append("fisheye = false")
    lines.append("")
    print(f"{name}: fx={fx:.1f} fy={fy:.1f} c=({K[0,2]:.1f},{K[1,2]:.1f}) "
          f"k1={D[0]:+.4f} |T|={np.linalg.norm(tvec):.3f} m")

lines.append("[metadata]")
lines.append('adjusted = false')
lines.append('error = 0.0')
lines.append("")

OUT.parent.mkdir(parents=True, exist_ok=True)
if OUT.exists():
    bak = OUT.with_name("calibration_BROKEN_backup.toml")
    if not bak.exists():
        bak.write_text(OUT.read_text())
        print(f"backed up existing -> {bak.name}")

OUT.write_text("\n".join(lines))
print(f"wrote {OUT}")

# verify it parses the way Pose2Sim will read it
import rtoml
d = rtoml.load(Path(OUT))          # Path, not str -- a str is parsed AS toml
print("rtoml OK, cameras:", [k for k in d if k != "metadata"])
