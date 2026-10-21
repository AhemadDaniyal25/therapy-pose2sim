#!/usr/bin/env python3
"""
CPU-only joint-mapping ablation.  Runs on a laptop, no cluster needed.

Purpose: isolate which SOMA-joint -> Pose2Sim-marker mappings actually help,
instead of adding mappings because they sound right.  Run it three ways and
compare final marker RMSE on identical input:

    python local_fit.py base      # 17 joints, no Hips, no Chest
    python local_fit.py chest     # + Chest -> Neck
    python local_fit.py hips      # + Hips  -> Hip

Measured, 2026-10-20, local CPU (torch 2.12.1+cpu), same TRC and frames:

    base    17.8 cm
    chest   13.6 cm      <-- 24% better, keep it
    hips    17.0 cm      <-- worse than chest, drop it

Why hips hurts: SOMA's `Hips` is the kinematic ROOT of the rig.  Pose2Sim's
`Hip` is the mid-hip surface marker.  They are about 10 cm apart, so pinning
one to the other drags the root -- and therefore the entire kinematic chain
-- off by that amount.  The lesson generalises: a name match is not a
semantic match, and the way to find out is an ablation, not an argument.

This also doubles as the "can we do anything while the cluster is down"
script.  It will not render a video in reasonable time on CPU; it reports
the fit quality, which is the part the ablation needs.
"""

import sys
from pathlib import Path
import numpy as np
import torch
from soma import SOMALayer

VARIANT = sys.argv[1] if len(sys.argv) > 1 else "chest"
TRC     = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("pose-3d/demo.trc")
START, NF, ITERS = 240, 120, 250        # 120 frames is enough to rank variants

dev = "cpu"
ND, NJ = 45, 77
MAXROT = 1.75

BASE = [("LeftArm","LShoulder"),("LeftForeArm","LElbow"),("LeftHand","LWrist"),
        ("RightArm","RShoulder"),("RightForeArm","RElbow"),("RightHand","RWrist"),
        ("LeftLeg","LHip"),("LeftShin","LKnee"),("LeftFoot","LAnkle"),
        ("RightLeg","RHip"),("RightShin","RKnee"),("RightFoot","RAnkle"),
        ("Head","Nose"),("LeftHandMiddle1","LMiddle1"),("RightHandMiddle1","RMiddle1"),
        ("LeftToeBase","LBigToe"),("RightToeBase","RBigToe")]
EXTRA = {"base": [], "chest": [("Chest","Neck")],
         "hips": [("Chest","Neck"), ("Hips","Hip")]}
MAP = BASE + EXTRA[VARIANT]


def read_trc(path):
    lines = Path(path).read_text().splitlines()
    names = [n for n in lines[3].split("\t") if n.strip()][2:]
    rows = {}
    for ln in lines[6:]:
        parts = ln.split("\t")
        if len(parts) < 5:
            continue
        f = int(float(parts[0]))
        vals = [float(p) if p.strip() not in ("", "nan", "NaN") else np.nan
                for p in parts[2:]]
        rows[f] = np.array(vals[: (len(vals)//3)*3]).reshape(-1, 3)
    return names, rows


names, body = read_trc(TRC)
idx = {n: i for i, n in enumerate(names)}
frames = sorted(f for f in body if START <= f < START + NF)

A = np.array([body[f] for f in frames])
d = (np.nanmedian(A[:, idx["Nose"]], 0)
     - np.nanmedian((A[:, idx["RHip"]] + A[:, idx["LHip"]]) / 2, 0))
ua = int(np.argmax(np.abs(d))); us = float(np.sign(d[ua]))
rest = [a for a in range(3) if a != ua]
spn = [np.nanpercentile(A[:, :, a], 98) - np.nanpercentile(A[:, :, a], 2)
       for a in rest]
xa = rest[int(np.argmax(spn))]; za = rest[1 - int(np.argmax(spn))]

def rm(P):
    P = np.asarray(P, float); Q = np.empty_like(P)
    Q[..., 0] = P[..., xa]; Q[..., 1] = -P[..., ua]*us; Q[..., 2] = P[..., za]
    return Q

b = SOMALayer(identity_model_type="mhr", device=dev)
b.prepare_identity(torch.zeros(1, ND), scale_params=torch.zeros(1, 68))
JN = None
for n in dir(b.public_rig_view()):
    v = getattr(b.public_rig_view(), n, None)
    if isinstance(v, (list, tuple)) and v and isinstance(v[0], str) \
       and 60 < len(v) < 120:
        JN = list(v)

ji = [JN.index(a) for a, c in MAP if a in JN and c in idx]
ti = [idx[c]      for a, c in MAP if a in JN and c in idx]
unmapped = [(a, c) for a, c in MAP if not (a in JN and c in idx)]
print(f"variant={VARIANT}  mapped {len(ji)}/{len(MAP)} joints")
if unmapped:
    print(f"  NOT mapped: {unmapped}")   # this is how we caught Hips/Chest
                                         # silently not resolving at all

T = torch.tensor(np.stack([rm(body[f]) for f in frames])[:, ti, :],
                 dtype=torch.float32)
msk = torch.isfinite(T).all(-1)
B = len(frames)

def joints_of(out):
    for n in ["joints","public_joints","joint_positions","transforms_world"]:
        v = getattr(out, n, None)
        if torch.is_tensor(v):
            if v.ndim == 3 and v.shape[-1] == 3: return v
            if v.ndim == 4 and v.shape[-2:] == (4, 4): return v[..., :3, 3]
    return None

ax  = torch.zeros(B, NJ, 3, requires_grad=True)
tr  = T[:, 0].clone().detach().requires_grad_(True)
ic  = torch.zeros(1, ND, requires_grad=True)
sp0 = torch.zeros(1, 68)                      # frozen: no bone stretching
opt = torch.optim.Adam([{"params": [ax, tr], "lr": 0.05},
                        {"params": [ic],     "lr": 0.01}])

for it in range(ITERS):
    opt.zero_grad()
    n = ax.norm(dim=-1, keepdim=True)
    a = ax * torch.clamp(MAXROT / n.clamp_min(1e-6), max=1.0)
    b.prepare_identity(ic.expand(1, ND), scale_params=sp0)
    J = joints_of(b(axis_angle=a, translation=tr))
    err = ((J[:, ji, :] - T) ** 2).sum(-1)
    loss = err[msk].mean() \
         + 1e-3 * ((a[1:] - a[:-1]) ** 2).mean() \
         + 3e-3 * (a ** 2).mean() + 1e-2 * (ic ** 2).mean()
    loss.backward(); opt.step()
    with torch.no_grad():
        ic.clamp_(-2.0, 2.0)
    if it % 25 == 0:
        print(f"  it {it:3d}  RMSE {float(err[msk].mean().sqrt())*100:6.2f} cm",
              flush=True)

print(f"\nvariant={VARIANT}  final RMSE "
      f"{float(err[msk].mean().sqrt())*100:.2f} cm")
