#!/usr/bin/env python3
"""
Self-check. Run this on the cluster before submitting any job.

    python hpc/verify_hpc.py
    python hpc/verify_hpc.py --calib /path/to/all_cameras.npz

Prints one line per dependency with OK / MISSING / BROKEN and, where it
matters, the version. Exits 1 if anything required is missing, so it can
go at the top of a Slurm script and fail the job in 3 seconds instead of
40 minutes.

This separates "my environment is wrong" from "my code is wrong".
Most of the time lost on this project was the first pretending to be the
second: a missing pyyaml that made calibration silently find 0 frames, a
scipy/numpy mismatch that surfaced as AttributeError deep inside Pose2Sim,
a NameError that only showed up 40 minutes into a GPU job.
"""

import argparse
import importlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

OK, WARN, BAD = "OK     ", "WARN   ", "MISSING"
fails = []
warns = []


def line(status, name, detail=""):
    print(f"  [{status}] {name:<28} {detail}")
    if status == BAD:
        fails.append(name)
    elif status == WARN:
        warns.append(name)


def check_import(mod, attr=None, required=True, note=""):
    try:
        m = importlib.import_module(mod)
    except Exception as e:
        line(BAD if required else WARN, mod, f"{type(e).__name__}: {e}")
        return None
    ver = getattr(m, "__version__", "")
    if attr and not hasattr(m, attr):
        line(BAD if required else WARN, mod, f"{ver} but no .{attr} {note}")
        return m
    line(OK, mod, f"{ver} {note}".strip())
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", type=Path, default=None,
                    help="all_cameras.npz to validate")
    ap.add_argument("--trc", type=Path, default=None,
                    help="a TRC file to validate")
    args = ap.parse_args()

    print("\n--- interpreter -------------------------------------------")
    line(OK, "python", sys.version.split()[0])
    line(OK, "executable", sys.executable)
    in_venv = sys.prefix != sys.base_prefix
    line(OK if in_venv else WARN, "virtualenv",
         "active" if in_venv else "NOT in a venv")
    nous = os.environ.get("PYTHONNOUSERSITE")
    line(OK if nous else WARN, "PYTHONNOUSERSITE",
         nous or "unset: ~/.local can shadow the venv")

    print("\n--- core --------------------------------------------------")
    np = check_import("numpy")
    sp = check_import("scipy")
    if np is not None and sp is not None:
        nv = tuple(int(x) for x in np.__version__.split(".")[:2])
        sv = tuple(int(x) for x in sp.__version__.split(".")[:2])
        # scipy >= 1.14 is built against numpy 2.x; the mismatch shows up as
        # AttributeError: np.long from inside Pose2Sim, not as an ImportError
        if sv >= (1, 14) and nv < (2, 0):
            line(BAD, "scipy/numpy ABI",
                 f"scipy {sp.__version__} needs numpy>=2.0, got {np.__version__}")
        else:
            line(OK, "scipy/numpy ABI", "compatible")

    cv = check_import("cv2")
    if cv is not None:
        has_aruco = hasattr(cv, "aruco")
        line(OK if has_aruco else BAD, "cv2.aruco",
             "present" if has_aruco
             else "need opencv-contrib-python for ChArUco")
        if has_aruco:
            has_charuco = hasattr(cv.aruco, "CharucoDetector")
            line(OK if has_charuco else WARN, "cv2.aruco.CharucoDetector",
                 "present" if has_charuco
                 else "old API; covisibility.py needs opencv>=4.7")

    check_import("rtoml")
    check_import("yaml", note="(pyyaml; absent = silent 0-frame calibration)")
    check_import("pandas")
    check_import("matplotlib", required=False)

    print("\n--- pose2sim ----------------------------------------------")
    try:
        from Pose2Sim import Pose2Sim as P
        missing = [s for s in ("poseEstimation", "synchronization",
                               "personAssociation", "triangulation",
                               "filtering") if not hasattr(P, s)]
        line(OK if not missing else BAD, "Pose2Sim",
             "all 5 stages" if not missing else f"missing {missing}")
    except Exception as e:
        line(BAD, "Pose2Sim", f"{type(e).__name__}: {e}")

    ort = check_import("onnxruntime", required=False)
    if ort is not None:
        provs = ort.get_available_providers()
        gpu = any("CUDA" in p or "Tensorrt" in p for p in provs)
        line(OK if gpu else WARN, "onnxruntime providers",
             ",".join(p.replace("ExecutionProvider", "") for p in provs)
             + ("" if gpu else "  <- CPU only, pose estimation will be slow"))

    print("\n--- face --------------------------------------------------")
    mp = check_import("mediapipe", required=False)
    if mp is not None:
        cand = [Path.home() / "models/face_landmarker.task",
                Path("models/face_landmarker.task"),
                Path("face_landmarker.task")]
        found = next((p for p in cand if p.exists()), None)
        if found:
            line(OK, "face_landmarker.task",
                 f"{found}  ({found.stat().st_size/1e6:.1f} MB)")
        else:
            line(BAD, "face_landmarker.task",
                 "not found; download it, pip does not ship it")

    print("\n--- animation (optional) ----------------------------------")
    torch = check_import("torch", required=False)
    if torch is not None:
        if torch.cuda.is_available():
            line(OK, "CUDA", f"{torch.cuda.get_device_name(0)} "
                              f"x{torch.cuda.device_count()}")
        else:
            line(WARN, "CUDA", "unavailable: CPU only (fine on woody, "
                               "pointless on tinygpu)")
    check_import("soma", required=False, note="(py-soma-x, MHR assets)")

    print("\n--- external binaries -------------------------------------")
    for exe, why, req in (("ffmpeg", "split_main.py", True),
                          ("ffprobe", "frame-rate checks", True),
                          ("sbatch", "job submission", False)):
        p = shutil.which(exe)
        if p:
            try:
                v = subprocess.run([exe, "-version"], capture_output=True,
                                   text=True, timeout=10
                                   ).stdout.splitlines()[0][:52]
            except Exception:
                v = p
            line(OK, exe, v)
        else:
            line(BAD if req else WARN, exe, f"not on PATH (needed for {why})")

    print("\n--- filesystem --------------------------------------------")
    woody = Path("/home/woody/iwso/iwso240h")
    if woody.exists():
        line(OK, "woody space", str(woody))
    else:
        line(WARN, "woody space", f"{woody} not visible from this node")
    try:
        import shutil as sh
        for p in (Path.home(), woody):
            if p.exists():
                t = sh.disk_usage(p)
                line(OK, f"space {p.name}",
                     f"{t.used/1e9:.0f}G used / {t.total/1e9:.0f}G")
    except Exception as e:
        line(WARN, "disk usage", str(e))

    if args.calib:
        print("\n--- calibration -------------------------------------------")
        if not args.calib.exists():
            line(BAD, "npz", f"{args.calib} not found")
        else:
            import numpy as _np
            d = _np.load(args.calib)
            need = ["K_main", "K_left", "K_right", "D_main", "D_left",
                    "D_right", "R_main_left", "T_main_left",
                    "R_main_right", "T_main_right"]
            miss = [k for k in need if k not in d]
            line(OK if not miss else BAD, "npz keys",
                 "all present" if not miss else f"missing {miss}")
            for cam, w in (("main", 1920), ("left", 1080), ("right", 1080)):
                if f"K_{cam}" not in d:
                    continue
                K = d[f"K_{cam}"]
                k1 = float(d[f"D_{cam}"].ravel()[0])
                fx = float(K[0, 0])
                plausible = 0.4 * w < fx < 4.0 * w and abs(k1) < 1.5
                line(OK if plausible else BAD, f"K_{cam}",
                     f"fx={fx:.1f} (w={w}) cx={float(K[0,2]):.1f} k1={k1:+.4f}"
                     + ("" if plausible else "   <- implausible"))
            for pair in ("left", "right"):
                if f"T_main_{pair}" not in d:
                    continue
                t = float(_np.linalg.norm(d[f"T_main_{pair}"]))
                line(OK if 0.3 < t < 10 else WARN, f"baseline main-{pair}",
                     f"{t:.3f} m")

    if args.trc:
        print("\n--- TRC ---------------------------------------------------")
        if not args.trc.exists():
            line(BAD, "trc", f"{args.trc} not found")
        else:
            import numpy as _np
            lines_ = args.trc.read_text().splitlines()
            rows = [l for l in lines_[6:] if len(l.split("\t")) > 4]
            line(OK, "trc rows", f"{len(rows)}")
            vals = []
            for l in rows[: min(200, len(rows))]:
                for p in l.split("\t")[2:]:
                    p = p.strip()
                    if p and p.lower() != "nan":
                        vals.append(abs(float(p)))
            if vals:
                mx = max(vals)
                # metres: a person fits in a few units. millimetres: thousands.
                unit = "metres" if mx < 50 else "MILLIMETRES?"
                line(OK if mx < 50 else WARN, "trc units",
                     f"max |coord| = {mx:.2f} -> {unit}")

    print("\n-----------------------------------------------------------")
    if fails:
        print(f"FAIL: {len(fails)} required item(s) missing: "
              + ", ".join(fails))
        print("See hpc/setup_hpc.sh and docs/SETUP.md")
        sys.exit(1)
    print("All required checks passed."
          + (f"  {len(warns)} warning(s): " + ", ".join(warns) if warns else ""))
    sys.exit(0)


if __name__ == "__main__":
    main()
