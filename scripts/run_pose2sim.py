#!/usr/bin/env python3
"""
Drive the Pose2Sim stages in order, with the two settings that mattered.

Run from inside the project directory (the one containing Config.toml,
Calibration/ and videos/).

Stage order is not arbitrary -- each stage consumes the previous stage's
output on disk:

    poseEstimation      videos/            -> pose/*.json      (RTMW, COCO_133)
    synchronization     pose/              -> pose-sync/       <-- do not skip
    personAssociation   pose-sync/         -> pose-associated/
    triangulation       pose-associated/   -> pose-3d/*.trc
    filtering           pose-3d/           -> pose-3d/*_filt_butterworth.trc

What we got wrong for weeks
---------------------------
`do_synchronization = false` was the default in the config we inherited.
The cameras were started by hand, and the real offsets were 136 frames
(4.5 s) main-left and 113 frames (3.8 s) main-right.  Triangulation was
intersecting rays cast at different instants, which is why 21 of 22 markers
came out NaN and reprojection sat at 31 px.  Turning synchronisation on took
it to 16.9 px with all 22 markers valid.

Check the synchronisation log every run.  You want a small offset at high
correlation.  Large offset + low correlation (0.5-0.6) means it has latched
onto the wrong person or the wrong motion, not that the cameras really were
4.5 s apart.
"""

import sys
from pathlib import Path
from Pose2Sim import Pose2Sim

STAGES = [
    ("poseEstimation",    Pose2Sim.poseEstimation),
    ("synchronization",   Pose2Sim.synchronization),
    ("personAssociation", Pose2Sim.personAssociation),
    ("triangulation",     Pose2Sim.triangulation),
    ("filtering",         Pose2Sim.filtering),
]


def preflight(proj: Path):
    """Fail loudly before burning GPU hours."""
    import rtoml
    cfg = proj / "Config.toml"
    cal = proj / "Calibration" / "calibration.toml"
    for p in (cfg, cal):
        if not p.exists():
            sys.exit(f"missing {p}")

    c = rtoml.load(cfg)
    pose = c.get("pose", {})
    if pose.get("pose_model") != "Whole_body":
        print(f"WARNING pose_model = {pose.get('pose_model')!r}; "
              "'Whole_body' is what gives you face + fingers")
    sync = c.get("synchronization", {})
    if not c.get("project", {}).get("do_synchronization", False) \
       and not sync.get("do_synchronization", False):
        print("WARNING do_synchronization is off -- this was our biggest bug")

    k = rtoml.load(cal)
    for name, cam in k.items():
        if name == "metadata":
            continue
        w = cam["size"][0]
        fx = cam["matrix"][0][0]
        cx = cam["matrix"][0][2]
        k1 = cam["distortions"][0]
        if not (0.4 * w < fx < 4.0 * w):
            sys.exit(f"{name}: fx={fx:.1f} implausible for width {w:.0f}")
        if abs(k1) > 1.5:
            sys.exit(f"{name}: k1={k1:.3f} implausible")
        if abs(cx - (w - 1) / 2) < 0.05 and w > 100:
            print(f"NOTE {name}: cx is exactly the image centre -- "
                  "principal point may never have been estimated")
    print("preflight OK\n")


def main():
    proj = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path.cwd()
    only = sys.argv[2:] or None
    preflight(proj)

    import os
    os.chdir(proj)
    for name, fn in STAGES:
        if only and name not in only:
            continue
        print(f"\n{'='*60}\n{name}\n{'='*60}", flush=True)
        fn()

    trcs = sorted((proj / "pose-3d").glob("*.trc"))
    print("\nTRC files:")
    for t in trcs:
        print(f"  {t.name}  {t.stat().st_size/1024:.0f} kB")
    if not trcs:
        sys.exit("no TRC produced -- read the triangulation log")


if __name__ == "__main__":
    main()
