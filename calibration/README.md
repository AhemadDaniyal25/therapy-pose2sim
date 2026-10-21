# Put the calibration files here

This folder is intentionally empty in the repo. `.gitignore` excludes
`*.npz` and `*.toml` bulk, so add them explicitly:

    cp /path/to/all_cameras.npz calibration/
    git add -f calibration/all_cameras.npz

## What goes here

| file | contents | get it from |
|---|---|---|
| `all_cameras.npz` | `K_main`, `K_left`, `K_right`, `D_*`, `R_main_left`, `T_main_left`, `R_main_right`, `T_main_right` | `therapy-room-pose-pipeline/calibration/` on the cluster, or `F:\TherapyRoom\Raw\Calibration\reference\` |
| `calibration_v2_reference.toml` | the good Pose2Sim calibration, generated from the npz | `pose2sim_therapy_demo/Calibration/` on the cluster, or regenerate with `scripts/make_calib_v2.py` |
| `calibration_BROKEN_backup.toml` | the original, non-physical one | same folder, keep it, it is the counter-example |

`all_cameras.npz` is the only one you strictly need; everything else
regenerates from it.

Read `../docs/CALIBRATION_MAP.md` before trusting any of it. Summary:
intrinsics are verified (held-out RMS 0.350 / 0.594 / 0.869 px), extrinsics
are inherited from an external import with ~20-25% translation error and
cannot be recomputed from the footage we have.
