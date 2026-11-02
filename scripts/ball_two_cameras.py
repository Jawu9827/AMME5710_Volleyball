"""Detect the ball in both synced cameras and triangulate it in 3D (x, y on the court + height).

Usage:
    python scripts/ball_two_cameras.py data/game2_A.mp4 data/game2_B_sdr.mp4 \
        --calib-a data/annotations/calib_A.json --calib-b data/annotations/calib_B.json \
        --sync data/annotations/sync_game2_A_game2_B_sdr.json --start 30 --duration 10
    --show                watch the result while it renders (q quit, space pause)
    --persons-every 0     skip the YOLO player boxes (faster, more false candidates)

Pipeline (no learned ball detector; see ball_detect.py for steps 1-2):
  1. Candidates in each camera: moving blobs (three-frame difference) of plausible ball size,
     shape and colour.
  2. Tracklets in each camera: candidates linked over time (constant velocity, Hungarian).
  3. Camera models from the floor calibrations (camera_3d.py): focal length, position, and a
     projection matrix for each camera.
  4. Cross-camera matching: for every pair of tracklets (one per camera) that overlap in time,
     camera B is interpolated to the exact time of each camera A frame (t_B = t_A + offset) and
     the pair is triangulated. The pair is the ball if
       - the two cameras agree (rays meet: reprojection error below MAX_REPROJ) on 10+ frames
         and on 80% of the frames where both tracklets exist,
       - the 3D path is inside the hall and moves (median speed > 2 m/s),
       - the blob size matches a 21 cm ball at the triangulated distance (0.6-2.5x). This is the
         strongest test: two unrelated blobs triangulate to a wrong distance, so their size is off.
     Pairs are accepted best first (most agreeing frames, most ball-coloured); one ball at a time.
  5. Sub-frame sync: the audio offset is good to about 1 frame (0.3-0.7 m of ball travel), so it
     is refined to the millisecond on the accepted pairs (lowest median reprojection error),
     and step 4 is repeated with the refined offset.
  6. Every frame of an accepted pair is triangulated (linear DLT). Frames where the rays are
     nearly parallel are flagged "3d weak": the cameras face each other, so this happens when
     the ball is near camera height on the line between them; near the floor the geometry is good.

Outputs (results/ball/):
    <A>_<B>_ball_from<start>.mp4   both views (circle = detected ball, cross = 3D point
                                   reprojected, dot = its shadow on the floor), top view
                                   (trail coloured by height) and side view (height, with net)
    <A>_<B>_ball_from<start>.csv   per camera-A frame: 2D positions, x, y, z, reprojection
                                   error, ray angle, status
    <A>_<B>_ball_from<start>_trajectory.png   x, y and height against time
"""
import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from calibrate_camera import COURT_LINES, draw_court, ffmpeg_exe  # noqa: E402
from camera_3d import load_camera, triangulate  # noqa: E402
from ball_detect import detect_video, link_tracklets, score_2d, tracklet_features  # noqa: E402

BALL_R = 0.105          # m, volleyball radius
MIN_ANGLE = 3.0         # deg; rays closer to parallel than this give weak depth
MAX_REPROJ = 12.0       # px at 1920 wide; worse means the cameras disagree about the ball
TRAIL_S = 1.5
REFINE_STEP = 0.002     # s
MIN_PAIR_FRAMES = 10    # frames on which both cameras must agree (0.33 s at 30 fps)
MIN_AGREE = 0.8         # ... out of the frames where both tracklets have the ball
MIN_SPEED = 2.0         # m/s, median 3D speed (amateur passes and sets can be ~4 m/s)
SIZE_RATIO = (0.6, 2.5)   # measured / expected image radius at the triangulated distance
MAX_TRACKS = 3000       # safety cap on tracklets per camera considered for matching


# ---------------------------------------------------------------- matching across cameras
class Track:
    """One camera's tracklet with lookup by frame and linear interpolation in time."""

    def __init__(self, pts, fps, scale, step=1):
        self.pts, self.fps, self.step = pts, fps, step
        self.by_k = {p["k"]: p for p in pts}
        self.ks = np.array(sorted(self.by_k))
        self.f = tracklet_features(pts, scale, step)
        self.t0, self.t1 = self.f["k0"] / fps, self.f["k1"] / fps

    def at_frame(self, k):
        p = self.by_k.get(k)
        return None if p is None else np.array([p["u"], p["v"]])

    def at_time(self, t):
        """Position at time t (this camera's clock), linear between the neighbouring detections
        (at most ~2/30 s apart)."""
        f = t * self.fps
        i = int(np.searchsorted(self.ks, f))
        if i < len(self.ks) and abs(self.ks[i] - f) < 1e-6:
            p = self.by_k[int(self.ks[i])]
            return np.array([p["u"], p["v"]])
        if i == 0 or i == len(self.ks):
            return None
        ka, kb = int(self.ks[i - 1]), int(self.ks[i])
        if (kb - ka) / self.fps > 2.1 / 30:
            return None
        w = (f - ka) / (kb - ka)
        pa, pb = self.by_k[ka], self.by_k[kb]
        return np.array([(1 - w) * pa["u"] + w * pb["u"], (1 - w) * pa["v"] + w * pb["v"]])

    def radius_at(self, k):
        p = self.by_k.get(k)
        return None if p is None else p["r"]


def build_tracks(frames, fps, scale, min_frames=MIN_PAIR_FRAMES, step=1):
    """Tracklets long enough to give min_frames agreeing frames (a pair needs that many;
    min_frames counts camera A frames, so a tracklet needs the same span of time)."""
    need = max(3, int(np.ceil((min_frames - 2) * (fps / step) / 30.0)))
    tracks = [Track(t, fps, scale, step) for t in link_tracklets(frames, scale, step=step) if len(t) >= need]
    tracks = [t for t in tracks if t.f["speed"] >= 1.0]
    tracks.sort(key=lambda t: -score_2d(t.f))
    if len(tracks) > MAX_TRACKS:
        print(f"  note: {len(tracks)} tracklets, keeping the {MAX_TRACKS} most ball-like")
    return tracks[:MAX_TRACKS]


def evaluate_pair(ta, tb, cams, fa, offset, max_reproj, min_frames=MIN_PAIR_FRAMES):
    """Triangulate a pair of tracklets on their common frames. Returns stats dict or None."""
    ks = [k for k in range(ta.f["k0"], ta.f["k1"] + 1)
          if k in ta.by_k and tb.t0 - 0.05 <= k / fa + offset <= tb.t1 + 0.05]
    if len(ks) < min_frames:
        return None
    good, both = [], 0
    for k in ks:
        ub = tb.at_time(k / fa + offset)
        if ub is None:
            continue
        both += 1
        ua = ta.at_frame(k)
        X, err, ang = triangulate([cams[0], cams[1]], [ua, ub])
        if max(err) <= max_reproj and -4 <= X[0] <= 13 and -8 <= X[1] <= 26 and -0.3 <= X[2] <= 15:
            good.append((k, X, max(err), ang, ua, ub))
    if len(good) < min_frames or len(good) < MIN_AGREE * both:
        return None
    K = np.array([g[0] for g in good])
    X = np.array([g[1] for g in good])
    step = np.diff(K)
    speed = np.linalg.norm(np.diff(X, axis=0), axis=1) / (step / fa)
    speed = float(np.median(speed[step <= 2])) if np.any(step <= 2) else 0.0
    ratios = []
    for k, Xk in zip(K, X):
        depth = (cams[0].R @ Xk + cams[0].t)[2]
        ratios.append(ta.radius_at(k) / (cams[0].K[0, 0] * BALL_R / depth))
    size = float(np.median(ratios))
    return dict(frames=good, n=len(good), reproj=float(np.median([g[2] for g in good])), speed=speed,
                size=size, colour=0.5 * (ta.f["colour"] + tb.f["colour"]), k0=int(K[0]), k1=int(K[-1]))


def match_pairs(tracks_a, tracks_b, cams, fa, offset, max_reproj, min_frames=MIN_PAIR_FRAMES, verbose=True):
    """Accepted (track A, track B, stats) pairs, best first, one ball at a time."""
    cands = []
    for ta in tracks_a:
        for tb in tracks_b:
            if tb.t1 < ta.t0 + offset - 0.05 or tb.t0 > ta.t1 + offset + 0.05:
                continue
            st = evaluate_pair(ta, tb, cams, fa, offset, max_reproj, min_frames)
            if st is None:
                continue
            if st["speed"] < MIN_SPEED or not SIZE_RATIO[0] <= st["size"] <= SIZE_RATIO[1]:
                continue
            st["score"] = st["n"] * (0.5 + st["colour"]) * (1 - 0.5 * min(ta.f["person"], tb.f["person"]))
            cands.append((ta, tb, st))
    cands.sort(key=lambda c: -c[2]["score"])
    # one ball at a time: a pair is rejected if its frames overlap an accepted pair's frames.
    # (A tracklet may appear in several pairs at different times: at 60 fps one camera's
    # tracklet can span several of the other camera's shorter ones.)
    accepted, taken = [], set()
    for ta, tb, st in cands:
        frames = {g[0] for g in st["frames"]}
        if len(frames & taken) > 2:
            continue
        accepted.append((ta, tb, st))
        taken |= frames
    if verbose:
        print(f"Cross-camera matching: {len(cands)} tracklet pairs consistent in 3D, {len(accepted)} accepted")
    return accepted


def refine_offset(accepted, cams, fa, offset, window):
    """Sub-frame sync from the ball: offset with the lowest median reprojection error."""
    def cost(o):
        e = []
        for ta, tb, st in accepted:
            for k in sorted(ta.by_k):
                ub = tb.at_time(k / fa + o)
                if ub is not None:
                    _, err, ang = triangulate([cams[0], cams[1]], [ta.at_frame(k), ub])
                    if ang >= MIN_ANGLE:
                        e.append(max(err))
        return (float(np.median(e)) if len(e) >= 15 else np.inf), len(e)
    deltas = np.arange(-window, window + 1e-9, REFINE_STEP)
    costs = [cost(offset + d)[0] for d in deltas]
    c0, n0 = cost(offset)
    k = int(np.argmin(costs))
    if not np.isfinite(costs[k]):
        print(f"Sync refinement: too few matched frames ({n0}); keeping the audio offset {offset:+.3f} s")
        return offset
    if costs[k] < 0.8 * c0 and 0 < k < len(deltas) - 1:
        print(f"Sync refinement: offset {offset:+.3f} -> {offset + deltas[k]:+.3f} s ({1000 * deltas[k]:+.0f} ms); "
              f"median reprojection error {c0:.1f} -> {costs[k]:.1f} px over {n0} frames")
        return offset + deltas[k]
    print(f"Sync refinement: no clear improvement (best {1000 * deltas[k]:+.0f} ms, {costs[k]:.1f} vs {c0:.1f} px); "
          f"keeping the audio offset {offset:+.3f} s")
    return offset


def build_rows(accepted, cams, fa, ia0, n_frames, offset, max_reproj):
    """One row per camera-A frame with the ball's 2D positions and 3D position."""
    rows = [dict(t_a=(ia0 + j) / fa, a=None, a_state="lost", b=None, b_measured=False, X=None,
                 reproj=None, angle=None, status="none") for j in range(n_frames)]
    for ta, tb, st in accepted:
        for k in range(ta.f["k0"], ta.f["k1"] + 1):
            j = k - ia0
            if not 0 <= j < n_frames:
                continue
            r = rows[j]
            ua, ub = ta.at_frame(k), tb.at_time(k / fa + offset)
            if ua is not None:
                r.update(a=ua, a_state="measured")
            if ub is not None:
                r.update(b=ub, b_measured=True)
            if ua is not None and ub is not None:
                X, err, ang = triangulate([cams[0], cams[1]], [ua, ub])
                r.update(reproj=max(err), angle=ang)
                if max(err) <= max_reproj:
                    r.update(X=X, status="3d" if ang >= MIN_ANGLE else "3d weak")
                else:
                    r["status"] = "inconsistent"
            elif ua is not None or ub is not None:
                r["status"] = "one camera"
        # camera B's part of the track outside camera A's (B only)
        for j, r in enumerate(rows):
            if r["b"] is None and tb.t0 <= r["t_a"] + offset <= tb.t1:
                ub = tb.at_time(r["t_a"] + offset)
                if ub is not None and r["a"] is None:
                    r.update(b=ub, b_measured=True, status="one camera")
    return rows


# ---------------------------------------------------------------- drawing
def height_colour(z):
    v = np.uint8([[[int(np.clip(z / 6.0, 0, 1) * 255)]]])
    return tuple(int(c) for c in cv2.applyColorMap(v, cv2.COLORMAP_TURBO)[0, 0])


def draw_view(img, cal, cam, rows, k, side, t_label):
    v = draw_court(img, cal, colour=(255, 0, 255))
    s = img.shape[1] / 1920
    key = "a" if side == "A" else "b"
    t_now = rows[k]["t_a"]
    pts = [r[key] for r in rows[max(0, k - 45):k + 1] if r[key] is not None and t_now - r["t_a"] <= TRAIL_S]
    for p0, p1 in zip(pts, pts[1:]):
        cv2.line(v, tuple(np.int32(p0)), tuple(np.int32(p1)), (255, 255, 0), max(1, int(2 * s)), cv2.LINE_AA)
    r = rows[k]
    if r[key] is not None:
        measured = r["a_state"] == "measured" if side == "A" else r["b_measured"]
        cv2.circle(v, tuple(np.int32(r[key])), int(14 * s) + 2, (255, 220, 0) if measured else (0, 170, 255),
                   max(2, int(3 * s)), cv2.LINE_AA)
    if r["X"] is not None:
        q = cam.project(r["X"])[0]
        cv2.drawMarker(v, tuple(np.int32(q)), (255, 0, 255), cv2.MARKER_CROSS, int(30 * s) + 4,
                       max(2, int(3 * s)))
        g = cam.project([r["X"][0], r["X"][1], 0.0])[0]              # shadow point on the floor
        cv2.line(v, tuple(np.int32(q)), tuple(np.int32(g)), (255, 0, 255), 1, cv2.LINE_AA)
        cv2.circle(v, tuple(np.int32(g)), int(6 * s) + 2, (255, 0, 255), -1)
    k2 = img.shape[1] / 1280
    cv2.putText(v, t_label, (12, int(40 * k2)), cv2.FONT_HERSHEY_SIMPLEX, k2, (0, 0, 0), int(6 * k2))
    cv2.putText(v, t_label, (12, int(40 * k2)), cv2.FONT_HERSHEY_SIMPLEX, k2, (255, 255, 255), int(2 * k2))
    return v


def court_panels(height, rows, k, net_h):
    """Top-down map (x across, y along) and a side view (height z against y), sharing the y axis."""
    my = 3.0
    ppm = height / (18 + 2 * my)
    P = lambda x, y: (int(round((x + 1.5) * ppm)), int(round((y + my) * ppm)))
    w_top = int(round(12 * ppm)) // 2 * 2
    top = np.full((height, w_top, 3), (90, 60, 40), np.uint8)
    cv2.rectangle(top, P(0, 0), P(9, 18), (170, 140, 110), -1)
    for a, b in COURT_LINES:
        cv2.line(top, P(*a), P(*b), (0, 230, 255), 2, cv2.LINE_AA)
    cv2.line(top, P(-1, 9), P(10, 9), (255, 255, 255), 3, cv2.LINE_AA)
    zmax = 7.0
    S = lambda z, y: (int(round((z + 0.5) * ppm)), int(round((y + my) * ppm)))
    w_side = int(round((zmax + 1) * ppm)) // 2 * 2
    side = np.full((height, w_side, 3), (60, 45, 35), np.uint8)
    cv2.line(side, S(0, 0), S(0, 18), (170, 140, 110), 3)                    # floor
    cv2.line(side, S(net_h - 1, 9), S(net_h, 9), (255, 255, 255), 3)         # net band
    for z in range(1, int(zmax) + 1):
        cv2.line(side, S(z, -my), S(z, 18 + my), (80, 65, 55), 1)
        cv2.putText(side, f"{z}m", (S(z, 0)[0] - 8, int(0.6 * my * ppm)), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                    (200, 200, 200), 1, cv2.LINE_AA)
    t_now = rows[k]["t_a"]
    trail = [r for r in rows[:k + 1] if r["X"] is not None and t_now - r["t_a"] <= TRAIL_S]
    for r0, r1 in zip(trail, trail[1:]):
        if r1["t_a"] - r0["t_a"] > 0.2:
            continue
        col = height_colour(r1["X"][2])
        cv2.line(top, P(*r0["X"][:2]), P(*r1["X"][:2]), col, 2, cv2.LINE_AA)
        cv2.line(side, S(r0["X"][2], r0["X"][1]), S(r1["X"][2], r1["X"][1]), col, 2, cv2.LINE_AA)
    r = rows[k]
    if r["X"] is not None:
        x, y, z = r["X"]
        weak = r["status"] == "3d weak"
        for img, c in ((top, P(x, y)), (side, S(z, y))):
            cv2.circle(img, c, 8, height_colour(z), -1 if not weak else 2, cv2.LINE_AA)
            cv2.circle(img, c, 10, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(top, f"height {z:.1f} m" + (" (weak)" if weak else ""), (6, height - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(top, "top view", (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(side, "side view", (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return np.hstack([top, side])


def save_plot(rows, path, offset):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = np.array([r["t_a"] for r in rows])
    fig, ax = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    for status, mk, lab in (("3d", "o", "3D"), ("3d weak", "x", "3D, weak ray angle")):
        sel = [i for i, r in enumerate(rows) if r["status"] == status]
        if sel:
            X = np.array([rows[i]["X"] for i in sel])
            for j, name in enumerate(("x across (m)", "y along (m)", "height z (m)")):
                ax[j].plot(t[sel], X[:, j], mk, ms=3, label=lab)
                ax[j].set_ylabel(name)
    ax[1].axhline(9, color="k", lw=0.8)
    ax[2].axhline(BALL_R, color="k", lw=0.8)
    ax[0].legend(loc="upper right", fontsize=8)
    ax[2].set_xlabel(f"time in camera A (s); camera B = A {offset:+.3f} s")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser()
    p.add_argument("video_a")
    p.add_argument("video_b")
    p.add_argument("--calib-a", required=True)
    p.add_argument("--calib-b", required=True)
    p.add_argument("--sync", required=True)
    p.add_argument("--start", type=float, default=0.0, help="start time in camera A (s)")
    p.add_argument("--duration", type=float, default=10.0)
    p.add_argument("--persons-every", type=int, default=3, help="YOLO player boxes every N frames (0 = off)")
    p.add_argument("--min-frames", type=int, default=MIN_PAIR_FRAMES,
                   help="frames both cameras must agree on (lower finds shorter flights, more false balls)")
    p.add_argument("--refine", type=float, default=0.15, help="sync refinement window, +/- s (0 = off)")
    p.add_argument("--net-height", type=float, default=2.43)
    p.add_argument("--width", type=int, default=960)
    p.add_argument("--show", action="store_true")
    args = p.parse_args()

    offset = float(json.loads(Path(args.sync).read_text())["offset_s"])
    cams, cals = [], []
    for name, path in (("A", args.calib_a), ("B", args.calib_b)):
        cam, info, d = load_camera(path)
        cams.append(cam)
        cals.append(cam.cal)
        x, y, z = info["position_m"]
        print(f"Camera {name}: f = {info['f_px']:.0f} px ({info['hfov_deg']:.0f} deg wide), position "
              f"x {x:.1f} m, y {y:.1f} m, height {z:.2f} m")
    if args.start + offset < 0:
        raise SystemExit(f"Camera B has not started at A = {args.start} s; use --start {-offset:.1f} or later")

    # 1-2. candidates and tracklets in each camera (B over the same real time, with a margin)
    t0, t1 = args.start, args.start + args.duration
    fa, ia0, frames_a, size_a, _, step_a = detect_video(args.video_a, cams[0], t0, t1, args.persons_every, label="A")
    m = args.refine + 0.1
    fb, ib0, frames_b, size_b, _, step_b = detect_video(args.video_b, cams[1], t0 + offset - m, t1 + offset + m,
                                                        args.persons_every, label="B")
    sa, sb = size_a[0] / 1920, size_b[0] / 1920
    tracks_a = build_tracks(frames_a, fa, sa, args.min_frames, step_a)
    tracks_b = build_tracks(frames_b, fb, sb, args.min_frames, step_b)
    print(f"Tracklets long enough to match (>= {max(3, args.min_frames - 2)} frames): camera A {len(tracks_a)}, "
          f"camera B {len(tracks_b)}")
    max_reproj = MAX_REPROJ * sa

    # 4-5. match across cameras with a loose threshold, refine the sync, match again
    accepted = match_pairs(tracks_a, tracks_b, cams, fa, offset, 2 * max_reproj, args.min_frames, verbose=False)
    if args.refine > 0 and accepted:
        offset = refine_offset(accepted, cams, fa, offset, args.refine)
    accepted = match_pairs(tracks_a, tracks_b, cams, fa, offset, max_reproj, args.min_frames)
    rows = build_rows(accepted, cams, fa, ia0, len(frames_a) * step_a, offset, max_reproj)
    n = len(rows)
    count = {s: sum(r["status"] == s for r in rows) for s in ("3d", "3d weak", "inconsistent", "one camera", "none")}
    good = [r["reproj"] for r in rows if r["status"] in ("3d", "3d weak")]
    print(f"Frames: {n}.  3D: {count['3d']} ({100 * count['3d'] / max(n, 1):.0f}%), weak geometry: "
          f"{count['3d weak']}, cameras disagree: {count['inconsistent']}, one camera only: "
          f"{count['one camera']}, no ball: {count['none']}")
    if good:
        print(f"Reprojection error of 3D points: median {np.median(good):.1f} px")

    stem = f"{Path(args.video_a).stem}_{Path(args.video_b).stem}_ball_from{args.start:g}"
    out_dir = Path("results/ball")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"{stem}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time_a_s", "a_u", "a_v", "a_state", "b_u", "b_v", "b_measured",
                    "x_m", "y_m", "z_m", "reproj_px", "ray_angle_deg", "status"])
        for r in rows:
            a = r["a"] if r["a"] is not None else (None, None)
            b = r["b"] if r["b"] is not None else (None, None)
            X = r["X"] if r["X"] is not None else (None, None, None)
            fmt = lambda v, d=2: "" if v is None else round(float(v), d)
            w.writerow([round(r["t_a"], 4), fmt(a[0], 1), fmt(a[1], 1), r["a_state"], fmt(b[0], 1), fmt(b[1], 1),
                        int(r["b_measured"]), fmt(X[0], 3), fmt(X[1], 3), fmt(X[2], 3), fmt(r["reproj"], 1),
                        fmt(r["angle"], 1), r["status"]])
    save_plot(rows, out_dir / f"{stem}_trajectory.png", offset)

    # 4. render: decode both videos again, B at the nearest frame to t_A + offset
    ca, cb = cv2.VideoCapture(args.video_a), cv2.VideoCapture(args.video_b)
    for _ in range(ia0):
        ca.grab()
    ib, last_b, ff, paused = 0, None, None, False
    for k, r in enumerate(rows):
        ok, a = ca.read()
        if not ok:
            break
        target = int(round((r["t_a"] + offset) * fb))
        while ib < target and cb.grab():
            ib += 1
        if ib == target:
            okb, fr = cb.read()
            if okb:
                last_b, ib = fr, ib + 1
        if last_b is None:
            continue
        views = []
        for img, cal, cam, side, tl in ((a, cals[0], cams[0], "A", f"camera A  t={r['t_a']:6.2f}s"),
                                         (last_b, cals[1], cams[1], "B", f"camera B  t={r['t_a'] + offset:6.2f}s")):
            v = draw_view(img, cal, cam, rows, k, side, tl)
            vh = int(round(args.width * img.shape[0] / img.shape[1]))
            views.append(cv2.resize(v, (args.width, vh)))
        h = max(v.shape[0] for v in views)
        h += h % 2
        views = [cv2.copyMakeBorder(v, 0, h - v.shape[0], 0, 0, cv2.BORDER_CONSTANT) for v in views]
        frame = np.hstack(views + [court_panels(h, rows, k, args.net_height)])
        if ff is None:
            ff = subprocess.Popen([ffmpeg_exe(), "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
                                   "-s", f"{frame.shape[1]}x{frame.shape[0]}", "-r", f"{fa:.5f}", "-i", "-",
                                   "-c:v", "libx264", "-preset", "veryfast", "-crf", "21", "-pix_fmt", "yuv420p",
                                   str(out_dir / f"{stem}.mp4")], stdin=subprocess.PIPE)
        ff.stdin.write(frame.tobytes())
        if args.show:
            cv2.imshow("ball in 3D (q quit, space pause)", frame)
            key = cv2.waitKey(0 if paused else 1) & 0xFF
            if key == ord("q"):
                break
            if key == ord(" "):
                paused = not paused
    if ff:
        ff.stdin.close()
        ff.wait()
    ca.release(); cb.release()
    cv2.destroyAllWindows()
    print(f"Saved {out_dir / (stem + '.mp4')}, .csv and _trajectory.png")


if __name__ == "__main__":
    main()