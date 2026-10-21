#!/usr/bin/env python3
"""
Bone-length plausibility: the one body metric that is independent of
calibration.

Why reprojection error is not enough
------------------------------------
With only two views, triangulation minimises reprojection error by
construction.  A systematically wrong baseline can therefore give you a
beautiful 3 px reprojection and a person who is 85% of their real size.  You
need a metric that does not come from the same optimisation.

Bone length is that metric, in two independent ways:

  (a) ABSOLUTE -- compare each segment to anthropometric ranges.  Catches
      scale error.
  (b) RELATIVE -- coefficient of variation of each bone across frames.  A
      bone is rigid, so CV measures reconstruction noise and has nothing to
      do with whether the scale is right.

Thresholds are declared BEFORE looking at the numbers, so this is a test and
not a story told afterwards:

    CV < 0.08   pass
    0.08-0.15   marginal
    CV > 0.15   fail

Results on this data
--------------------
    p2s_mainR (Person B):  8/10 bones in anatomical range, median CV 0.011
    p2s_mainL (Person A):  6/10 bones in anatomical range, median CV 0.092

Person A is marginal on both counts.  Person B is good.  The same asymmetry
shows up in the face fusion (1.46 px vs 12.35 px), which is what points the
finger at the main-left extrinsic calibration rather than at the method.

Usage:  python eval_body.py <trc_file> [--json out.json]
"""

import sys, json
from pathlib import Path
import numpy as np

# Pose2Sim 'Whole_body' marker order (first 23 of COCO_133)
NAMES = ["Nose","LEye","REye","LEar","REar","LShoulder","RShoulder",
         "LElbow","RElbow","LWrist","RWrist","LHip","RHip","LKnee","RKnee",
         "LAnkle","RAnkle","LBigToe","LSmallToe","LHeel","RBigToe",
         "RSmallToe","RHeel"]

# (name, a, b, lo_m, hi_m)  adult ranges, Winter / NASA-STD-3000
BONES = [
    ("L upper arm", "LShoulder", "LElbow", 0.26, 0.36),
    ("R upper arm", "RShoulder", "RElbow", 0.26, 0.36),
    ("L forearm",   "LElbow",    "LWrist", 0.22, 0.31),
    ("R forearm",   "RElbow",    "RWrist", 0.22, 0.31),
    ("L thigh",     "LHip",      "LKnee",  0.35, 0.48),
    ("R thigh",     "RHip",      "RKnee",  0.35, 0.48),
    ("L shank",     "LKnee",     "LAnkle", 0.33, 0.45),
    ("R shank",     "RKnee",     "RAnkle", 0.33, 0.45),
    ("shoulders",   "LShoulder", "RShoulder", 0.30, 0.44),
    ("hips",        "LHip",      "RHip",   0.20, 0.32),
]

CV_PASS, CV_FAIL = 0.08, 0.15


def read_trc(path):
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
        rows.append(np.array(vals[: (len(vals)//3)*3]).reshape(-1, 3))
    return np.stack(rows)


def robust_cv(x):
    """median-absolute-deviation based CV: a few bad frames should not
    dominate the verdict, and they do with a plain std/mean."""
    x = x[np.isfinite(x)]
    if len(x) < 10:
        return np.nan
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return float(1.4826 * mad / med) if med > 1e-6 else np.nan


def main():
    trc = Path(sys.argv[1])
    P = read_trc(trc)
    idx = {n: i for i, n in enumerate(NAMES) if i < P.shape[1]}
    print(f"{trc.name}: {P.shape[0]} frames, {P.shape[1]} markers\n")

    nvalid = int(np.isfinite(P[:, :23, 0]).all(1).sum())
    print(f"frames with all 23 body markers: {nvalid}/{P.shape[0]}\n")

    print(f"{'bone':<14}{'median':>9}{'range':>14}{'in?':>5}{'robCV':>9}"
          f"{'verdict':>10}")
    print("-" * 61)

    out, in_range, cvs = [], 0, []
    for label, a, b, lo, hi in BONES:
        if a not in idx or b not in idx:
            print(f"{label:<14}  marker missing")
            continue
        d = np.linalg.norm(P[:, idx[a]] - P[:, idx[b]], axis=1)
        d = d[np.isfinite(d)]
        if len(d) < 10:
            print(f"{label:<14}  too few valid frames")
            continue
        med = float(np.median(d))
        cv = robust_cv(d)
        ok = lo <= med <= hi
        in_range += ok
        cvs.append(cv)
        verdict = ("pass" if cv < CV_PASS else
                   "fail" if cv > CV_FAIL else "marginal")
        print(f"{label:<14}{med:9.3f}{f'{lo:.2f}-{hi:.2f}':>14}"
              f"{'yes' if ok else 'NO':>5}{cv:9.3f}{verdict:>10}")
        out.append(dict(bone=label, median_m=med, lo=lo, hi=hi,
                        in_range=bool(ok), robust_cv=cv, verdict=verdict))

    med_cv = float(np.nanmedian(cvs))
    print("-" * 61)
    print(f"\n{in_range}/{len(BONES)} bones within anatomical range")
    print(f"median robust CV = {med_cv:.3f}  "
          f"({'pass' if med_cv < CV_PASS else 'fail' if med_cv > CV_FAIL else 'marginal'})")

    if in_range < len(BONES) and med_cv < CV_PASS:
        print("\nLow CV with bones out of range = consistent but mis-scaled "
              "reconstruction. That is an extrinsic (baseline) error, not "
              "noise.")

    if "--json" in sys.argv:
        p = Path(sys.argv[sys.argv.index("--json") + 1])
        p.write_text(json.dumps(dict(trc=str(trc), frames=int(P.shape[0]),
                                     bones=out, median_robust_cv=med_cv,
                                     in_range=in_range), indent=2))
        print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
