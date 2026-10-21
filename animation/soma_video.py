#!/usr/bin/env python3
"""
Fit a parametric human body (MHR, via NVIDIA SOMA) to the Pose2Sim TRC and
render it.

Context
-------
Supervisor task: "Continue the animation for now; focus more on the body and
do not spend much time on it."  So this is deliberately a body-first,
good-enough-to-show pipeline, not a research contribution.

SMPL was unavailable (smpl.is.tue.mpg.de down), so we used SOMA's MHR
(Meta Human Rig): 45 identity coefficients, 68 scale parameters, 77 joints,
18,056 vertices, 36,108 triangles.  Fitting is Adam on axis-angle joint
rotations + root translation + identity, against the 3D marker positions.

THE BUG THAT MATTERS -- read this before you touch anything
-----------------------------------------------------------
1. FREEZE scale_params.  Of the 68 scale parameters, 60 are per-bone scales,
   including every finger joint.  Fingers do not appear in a COCO_133 body
   fit, so they are unconstrained and Adam stretches them to whatever
   marginally reduces the loss.  The result was, verbatim, "spiders and
   monsters with huge fingers."  Freezing scale_params raised the fit error
   from 3.3 cm to 11.4 cm and produced a body with correct human
   proportions.  That is the right trade: the 3.3 cm number was fitting
   noise with anatomy as the currency.

2. Do NOT map SOMA "Hips" to Pose2Sim "Hip".  They are different things.
   SOMA's Hips is the kinematic root; Pose2Sim's Hip is the mid-hip surface
   marker, roughly 10 cm away.  Mapping them pins the root to the wrong
   place and pulls the whole skeleton.  Ablation on identical input:

       17 joints, no Hips/Chest     17.8 cm   (baseline)
       + Chest->Neck                13.6 cm   <-- keep this
       + Hips->Hip                  17.0 cm   <-- actively harmful

3. Clamp rotations.  MAXROT = 1.75 rad (~100 deg) per joint.  Without it the
   optimiser finds poses that fit the markers through anatomically
   impossible joint angles, and the render looks like a figure turning
   inside out.

4. Up axis must be DETECTED, not hardcoded.  Take the median Nose position
   minus the median mid-hip position; the axis with the largest magnitude is
   up, and its sign tells you the direction.  A hardcoded elev gave us a
   tumbling figure.

Rendering
---------
matplotlib's Poly3DCollection has no z-buffer -- it sorts whole polygons, so
an 18k-vertex mesh renders as a flat blob with faces punching through each
other.  Hence the small software rasteriser at the bottom: per-pixel depth
buffer, barycentric interpolation, smooth vertex normals, Lambert + specular
+ rim light.  It is ~40 lines and it is the difference between "blob" and
"body".

What does NOT work
------------------
Projecting video texture onto the vertices.  The TRC is not in camera
coordinates: raw TRC joints project to v = 3322-9641 on a 1080-px frame, and
0 of 131 keypoints land inside the image.  Recovering the camera pose in TRC
space (PnP from the 2D pose JSONs against the 3D markers) is the missing
step and is not implemented.  `textured 0/18056 vertices (0%)` in the logs
is this, not a sampling bug.  Flat skin colour is used instead.

Usage:  python soma_video.py <project_tag> <start_frame> <n_frames>
"""

import sys
from pathlib import Path
import numpy as np
import torch
from scipy.spatial import cKDTree
import cv2

from soma import SOMALayer           # py-soma-x

dev = "cuda" if torch.cuda.is_available() else "cpu"
ND, NJ = 45, 77
W, H = 1080, 1440
ITERS = 300
MAXROT = 1.75                        # rad, ~100 deg
SKIN = np.array([.78, .63, .54])

TAG   = sys.argv[1]
START = int(sys.argv[2]) if len(sys.argv) > 2 else 240
NF    = int(sys.argv[3]) if len(sys.argv) > 3 else 450

TRC  = Path.home() / f"projects/{TAG}/pose-3d"
OUT  = Path("/home/woody/iwso/iwso240h/anim")
OUT.mkdir(parents=True, exist_ok=True)

# SOMA joint  ->  Pose2Sim marker.  Hips deliberately absent (see note 2).
MAP = [("LeftArm","LShoulder"),("LeftForeArm","LElbow"),("LeftHand","LWrist"),
       ("RightArm","RShoulder"),("RightForeArm","RElbow"),("RightHand","RWrist"),
       ("LeftLeg","LHip"),("LeftShin","LKnee"),("LeftFoot","LAnkle"),
       ("RightLeg","RHip"),("RightShin","RKnee"),("RightFoot","RAnkle"),
       ("Head","Nose"),("LeftHandMiddle1","LMiddle1"),("RightHandMiddle1","RMiddle1"),
       ("LeftToeBase","LBigToe"),("RightToeBase","RBigToe"),
       ("Chest","Neck")]

# anatomical bone priors, metres (Winter).  Used as a soft loss term so the
# fit cannot buy marker accuracy with an impossible skeleton.
BONE_T = [(("LeftArm","LeftForeArm"), 0.254), (("LeftForeArm","LeftHand"), 0.256),
          (("RightArm","RightForeArm"), 0.254), (("RightForeArm","RightHand"), 0.256),
          (("LeftLeg","LeftShin"), 0.343), (("RightLeg","RightShin"), 0.329),
          (("LeftShin","LeftFoot"), 0.411), (("RightShin","RightFoot"), 0.398),
          (("LeftArm","RightArm"), 0.355)]

# ---------------------------------------------------------------- load TRC
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

trcfile = sorted(TRC.glob("*_filt_*.trc")) or sorted(TRC.glob("*.trc"))
names, body = read_trc(trcfile[0])
idx = {n: i for i, n in enumerate(names)}

frames = sorted(f for f in body if START <= f < START + NF)

# ------------------------------------------------- detect the up axis
A = np.array([body[f] for f in frames])
d = (np.nanmedian(A[:, idx["Nose"]], 0)
     - np.nanmedian((A[:, idx["RHip"]] + A[:, idx["LHip"]]) / 2, 0))
ua = int(np.argmax(np.abs(d))); us = float(np.sign(d[ua]))
rest = [a for a in range(3) if a != ua]
spn = [np.nanpercentile(A[:, :, a], 98) - np.nanpercentile(A[:, :, a], 2)
       for a in rest]
xa = rest[int(np.argmax(spn))]; za = rest[1 - int(np.argmax(spn))]

def rm(P):
    """TRC axes -> render axes (y up, screen-space x widest)."""
    P = np.asarray(P, float); Q = np.empty_like(P)
    Q[..., 0] = P[..., xa]; Q[..., 1] = -P[..., ua] * us; Q[..., 2] = P[..., za]
    return Q

print(f"body {len(frames)} frames | up={ua}({us:+.0f})", flush=True)

# ---------------------------------------------------------------- SOMA body
b = SOMALayer(identity_model_type="mhr", device=dev)
b.prepare_identity(torch.zeros(1, ND, device=dev),
                   scale_params=torch.zeros(1, 68, device=dev))

JN = None
for n in dir(b.public_rig_view()):
    v = getattr(b.public_rig_view(), n, None)
    if isinstance(v, (list, tuple)) and v and isinstance(v[0], str) \
       and 60 < len(v) < 120:
        JN = list(v)
FACES = b.faces.detach().cpu().numpy().astype(np.int32)

ji = [JN.index(a) for a, c in MAP if a in JN and c in idx]
ti = [idx[c]      for a, c in MAP if a in JN and c in idx]
print(f"mapped {len(ji)} joints | faces {FACES.shape}", flush=True)
if len(ji) < 18:
    missing = [(a, c) for a, c in MAP
               if not (a in JN and c in idx)]
    print(f"NOT mapped: {missing}", flush=True)

sel = frames
T = torch.tensor(np.stack([rm(body[f]) for f in sel])[:, ti, :],
                 dtype=torch.float32, device=dev)
msk = torch.isfinite(T).all(-1)
B = len(sel)

def joints_of(out):
    for n in ["joints", "public_joints", "joint_positions",
              "skeleton_state", "transforms_world"]:
        v = getattr(out, n, None)
        if torch.is_tensor(v):
            if v.ndim == 3 and v.shape[-1] == 3:
                return v
            if v.ndim == 4 and v.shape[-2:] == (4, 4):
                return v[..., :3, 3]
    return None

def verts_of(out):
    for n in ["vertices", "verts", "v", "public_vertices"]:
        v = getattr(out, n, None)
        if torch.is_tensor(v) and v.ndim == 3 and v.shape[-1] == 3:
            return v
    return None

# ---------------------------------------------------------------- optimise
ax  = torch.zeros(B, NJ, 3, device=dev, requires_grad=True)
tr  = T[:, 0].clone().detach().requires_grad_(True)
ic  = torch.zeros(1, ND, device=dev, requires_grad=True)
sp0 = torch.zeros(1, 68, device=dev)          # FROZEN -- see note 1

opt = torch.optim.Adam([{"params": [ax, tr], "lr": 0.05},
                        {"params": [ic],     "lr": 0.01}])

def clamp_rot(a):
    n = a.norm(dim=-1, keepdim=True)
    return a * torch.clamp(MAXROT / n.clamp_min(1e-6), max=1.0)

jn_i = {n: i for i, n in enumerate(JN)}
bone_pairs = [(jn_i[a], jn_i[c], L) for (a, c), L in BONE_T
              if a in jn_i and c in jn_i]

for it in range(ITERS):
    opt.zero_grad()
    a = clamp_rot(ax)
    b.prepare_identity(ic.expand(1, ND), scale_params=sp0)
    out = b(axis_angle=a, translation=tr)
    J = joints_of(out)

    # marker term
    pred = J[:, ji, :]
    err = ((pred - T) ** 2).sum(-1)
    loss = err[msk].mean()

    # bone-length prior
    for i, j, L in bone_pairs:
        d = (J[:, i] - J[:, j]).norm(dim=-1)
        loss = loss + 0.05 * ((d - L) ** 2).mean()

    # temporal smoothness
    if B > 1:
        loss = loss + 1e-3 * ((a[1:] - a[:-1]) ** 2).mean() \
                    + 1e-3 * ((tr[1:] - tr[:-1]) ** 2).mean()

    # regularisation
    loss = loss + 3e-3 * (a ** 2).mean() + 1e-2 * (ic ** 2).mean()

    loss.backward()
    opt.step()
    with torch.no_grad():
        ic.clamp_(-2.0, 2.0)
    if it % 20 == 0 or it == ITERS - 1:
        rmse = float(err[msk].mean().sqrt()) * 100
        print(f"  it {it:3d}  marker RMSE {rmse:6.2f} cm", flush=True)

with torch.no_grad():
    b.prepare_identity(ic.expand(1, ND), scale_params=sp0)
    out = b(axis_angle=clamp_rot(ax), translation=tr)
    V = verts_of(out).cpu().numpy()
    J = joints_of(out).cpu().numpy()
final = float(np.sqrt(np.nanmean(np.sum(
    (J[:, ji, :] - T.cpu().numpy()) ** 2, -1)))) * 100
print(f"final marker RMSE {final:.2f} cm", flush=True)

# -------------------------------------------- software rasteriser
VC = np.tile(SKIN, (V.shape[1], 1))

ctr = np.array([np.median(V[..., 0]), np.median(V[..., 1]), 0.])
ext = np.percentile(np.abs(V[..., :2].reshape(-1, 2) - ctr[:2]), 99.5)
cam = ctr + np.array([0, 0, -ext * 3.0]); foc = H * 0.95

vid = cv2.VideoWriter(str(OUT / f"soma_{TAG}.mp4"),
                      cv2.VideoWriter_fourcc(*"mp4v"), 30, (W, H))

LIGHT = np.array([0.4, -0.7, -0.6]); LIGHT /= np.linalg.norm(LIGHT)

for k in range(B):
    Vc = V[k] - cam

    # smooth vertex normals: accumulate face normals per vertex
    N = np.zeros_like(Vc); tri = Vc[FACES]
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    for c in range(3):
        np.add.at(N, FACES[:, c], fn)
    N /= np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-9)

    z = Vc[:, 2]
    u = foc * Vc[:, 0] / np.maximum(z, 1e-6) + W / 2
    v = foc * Vc[:, 1] / np.maximum(z, 1e-6) + H / 2

    img = np.zeros((H, W, 3), np.float32); img[:] = np.array([.045, .055, .075])
    zbuf = np.full((H, W), np.inf, np.float32)

    # per-vertex shading, interpolated barycentrically per triangle
    lam = np.clip(N @ (-LIGHT), 0, 1)
    spec = np.clip(N @ (-LIGHT), 0, 1) ** 24
    rim = np.clip(1.0 - np.abs(N[:, 2]), 0, 1) ** 2
    shade = (0.22 + 0.78 * lam)[:, None] * VC \
          + 0.35 * spec[:, None] + 0.18 * rim[:, None] * np.array([.5, .6, .8])

    order = np.argsort(-z[FACES].mean(1))          # coarse back-to-front
    for t in order:
        a, bb, c = FACES[t]
        if z[a] <= 0 or z[bb] <= 0 or z[c] <= 0:
            continue
        xs = np.array([u[a], u[bb], u[c]]); ys = np.array([v[a], v[bb], v[c]])
        x0, x1 = int(max(0, xs.min())), int(min(W - 1, xs.max()))
        y0, y1 = int(max(0, ys.min())), int(min(H - 1, ys.max()))
        if x1 <= x0 or y1 <= y0:
            continue
        gx, gy = np.meshgrid(np.arange(x0, x1 + 1), np.arange(y0, y1 + 1))
        d = ((ys[1]-ys[2])*(xs[0]-xs[2]) + (xs[2]-xs[1])*(ys[0]-ys[2]))
        if abs(d) < 1e-9:
            continue
        w0 = ((ys[1]-ys[2])*(gx-xs[2]) + (xs[2]-xs[1])*(gy-ys[2])) / d
        w1 = ((ys[2]-ys[0])*(gx-xs[2]) + (xs[0]-xs[2])*(gy-ys[2])) / d
        w2 = 1 - w0 - w1
        inside = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
        if not inside.any():
            continue
        zi = w0*z[a] + w1*z[bb] + w2*z[c]
        sub = zbuf[y0:y1+1, x0:x1+1]
        better = inside & (zi < sub)
        if not better.any():
            continue
        col = (w0[..., None]*shade[a] + w1[..., None]*shade[bb]
               + w2[..., None]*shade[c])
        reg = img[y0:y1+1, x0:x1+1]
        reg[better] = col[better]
        sub[better] = zi[better]

    frame = np.clip(img ** (1/2.2), 0, 1)
    frame = (frame * 255).astype(np.uint8)
    cv2.putText(frame, f"{TAG}  f{sel[k]}  RMSE {final:.1f} cm",
                (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (230, 230, 230), 2)
    vid.write(frame)
    if k % 50 == 0:
        print(f"  rendered {k}/{B}", flush=True)

vid.release()
print(f"wrote {OUT}/soma_{TAG}.mp4", flush=True)
