"""
Ball tracking (offline, full frame rate).

1. Candidates per frame: yellow (HSV) AND foreground vs median background AND compact blob.
2. Static-clutter suppression: grid cells that fire in many frames are background objects.
3. Tracklet linking: constant-velocity prediction, nearest candidate within a gate.
4. Trajectory selection: keep moving tracklets, resolve temporal overlaps by score.
5. Short gaps filled by polynomial interpolation.

Output CSV: frame, t_s, u, v, size_px, src (det|interp)
"""
import csv, json, os, pickle, cv2, numpy as np
from common import run_args, open_video, prep
from ball.ball_cands import candidates
from camera_model import camera_from_calib, ball_3d

a = run_args()
a.video, a.calib, a.bg, a.start, a.dur, a.out = a.cfg["video"], "calib.json", "bg.png", a.cfg["start"], a.cfg["dur"], "ball_track.csv"
a.cache = f"ball_candidates_{a.start}_{a.dur}.pkl"   # delete to recompute
cam = camera_from_calib(a.calib)

C = json.load(open(a.calib))
H = cv2.findHomography(np.float32(C["world_m"]), np.float32(C["image_px"]))[0]
bg = cv2.imread(a.bg)
# image-space gate: ball must lie ABOVE some floor point of our playing area (court +2 m)
gate = cv2.perspectiveTransform(np.float32([[[-2, -2], [11, -2], [11, 20], [-2, 20]]]), H)[0]
gate_mask = np.zeros(bg.shape[:2], np.uint8)
cv2.fillPoly(gate_mask, [gate.astype(np.int32)], 255)
col_has_floor_below = np.zeros_like(gate_mask)
for u in range(gate_mask.shape[1]):              # pixel column u: rows above the lowest gate pixel
    ys = np.where(gate_mask[:, u])[0]
    if len(ys): col_has_floor_below[:ys.max(), u] = 255
col_has_floor_below[:120] = 0                    # ceiling lights / roof

cap, fps = open_video(a.video)
FR = 59.94 / fps                       # per-frame motion thresholds were tuned at 60 fps
f0 = int(a.start * fps); N = int(a.dur * fps)
cache = a.cache or f"ballcands_{a.start}_{a.dur}.pkl"
if os.path.exists(cache):
    cands = pickle.load(open(cache, "rb"))
else:
    cap.set(cv2.CAP_PROP_POS_FRAMES, f0); cands = []
    for k in range(N):
        ok, f = cap.read()
        if not ok: break
        cands.append(candidates(prep(f), bg))
        if k % 600 == 0: print("candidates", k, "/", N, flush=True)
    pickle.dump(cands, open(cache, "wb"))
# 3-D gate: apparent size -> depth -> world position must be over/near OUR court and airborne-plausible
def in_play(c):
    if not col_has_floor_below[int(c[1]), int(c[0])]: return False
    x, y, z = ball_3d(cam, c[0], c[1], c[2])
    return -3 <= x <= 12 and -4 <= y <= 22 and -1 <= z <= 10
cands = [[(*c[:3], c[3]) for c in cs if in_play(c)] for cs in cands]
N = len(cands)

# ---- static clutter: 12 px cells that fire in >8 % of frames ----
CELL = 12; hits = {}
for cs in cands:
    for key in {(int(u) // CELL, int(v) // CELL) for u, v, *_ in cs}: hits[key] = hits.get(key, 0) + 1
static = {k for k, n in hits.items() if n > 0.08 * N}
def is_static(u, v):
    cu, cv_ = int(u) // CELL, int(v) // CELL
    return any((cu + i, cv_ + j) in static for i in (-1, 0, 1) for j in (-1, 0, 1))
cands = [[c for c in cs if not is_static(c[0], c[1])] for cs in cands]

# ---- tracklet linking ----
MAXJ = 50 * FR     # px/frame gate around prediction
tracks, active = [], []
for k, cs in enumerate(cands):
    used = set(); still = []
    for t in sorted(active, key=lambda t: -len(t["pts"])):
        (k1, u1, v1, _), vel = t["pts"][-1], t["vel"]
        gap = k - k1
        pu, pv = u1 + vel[0] * gap, v1 + vel[1] * gap
        best, bd = None, MAXJ * (1 + 0.5 * (gap - 1))
        for i, (u, v, s, ar) in enumerate(cs):
            if i in used: continue
            d = np.hypot(u - pu, v - pv)
            if d < bd: best, bd = i, d
        if best is not None:
            u, v, s, ar = cs[best]; used.add(best)
            nv = ((u - u1) / gap, (v - v1) / gap)
            t["vel"] = nv if len(t["pts"]) == 1 else (0.5 * nv[0] + 0.5 * vel[0], 0.5 * nv[1] + 0.5 * vel[1])
            t["pts"].append((k, u, v, s)); still.append(t)
        elif gap <= max(2, round(6 / FR)): still.append(t)
        else: tracks.append(t)
    for i, (u, v, s, ar) in enumerate(cs):
        if i not in used: still.append({"pts": [(k, u, v, s)], "vel": (0.0, 0.0)})
    active = still
tracks += active

def score(t):
    p = np.array([q[1:3] for q in t["pts"]])
    if len(p) < 2: return 0
    st = np.hypot(*np.diff(p, axis=0).T)
    if np.mean(st < 1.0 * FR) > 0.3: return 0                       # mostly still -> clutter
    if np.max(st) > 3 * np.median(st) + 30: return 0           # teleporting -> mis-link
    return len(p) * min(1.0, st.sum() / (3 * FR * len(p)))          # long AND moving
good = [t for t in tracks if len(t["pts"]) >= max(4, round(8 / FR)) and score(t) > 4 / FR]
good.sort(key=score, reverse=True)
owner = np.full(N, -1)
chosen = []
for t in good:                                    # greedy, non-overlapping in time
    ks = [q[0] for q in t["pts"]]
    if np.mean(owner[ks] >= 0) > 0.2: continue
    for q in t["pts"]:
        if owner[q[0]] < 0: owner[q[0]] = len(chosen)
    chosen.append(t)
pos = {}
for ti, t in enumerate(chosen):
    for k, u, v, s in t["pts"]:
        if owner[k] == ti: pos[k] = (u, v, s, "det")
# ---- fill gaps <= 8 frames inside a tracklet span with quadratic fit ----
ks = sorted(pos)
for k0, k1 in zip(ks, ks[1:]):
    if 1 < k1 - k0 <= max(3, round(8 / FR)):
        nb = [k for k in ks if k0 - 6 <= k <= k1 + 6]
        T = np.array(nb); U = np.array([pos[k][0] for k in nb]); V = np.array([pos[k][1] for k in nb])
        pu, pv = np.polyfit(T, U, min(2, len(T) - 1)), np.polyfit(T, V, min(2, len(T) - 1))
        for k in range(k0 + 1, k1):
            pos[k] = (float(np.polyval(pu, k)), float(np.polyval(pv, k)), pos[k0][2], "interp")

with open(a.out, "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["frame", "t_s", "u", "v", "size_px", "src"])
    for k in sorted(pos):
        u, v, s, src = pos[k]
        w.writerow([f0 + k, round((f0 + k) / fps, 4), round(u, 1), round(v, 1), s, src])
print(f"static cells {len(static)}, tracklets {len(tracks)}, kept {len(chosen)}, "
      f"ball found in {len(pos)}/{N} frames ({100 * len(pos) / N:.0f}%)")
