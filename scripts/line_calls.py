"""Find ball bounces in the 3D track and call them in or out.

Usage (after ball_two_cameras.py):
    python scripts/line_calls.py results/ball/game2_A_game2_B_sdr_ball_from30.json
    --contact-radius 0.03   radius (m) of the ball's contact patch with the floor
    --calib-sigma 0.05      position error (m) from the calibration, added to the fit error

1. Bounce candidates: in each continuous stretch of 3D samples, the low points (height below
   Z_LOW) where the ball stops going down. A stretch that ends while descending near the floor
   (the ball was lost at the bounce) also counts.
2. Contact point: the samples before the low point (descending) and after it (ascending) are
   each fitted with a ballistic arc: x and y linear in time, height z = z0 + vz t - g t^2 / 2
   with g = 9.81 m/s^2 fixed, weighted least squares (weight sin^2 of the angle between the
   two cameras' rays, since the triangulation error grows as 1 / sin(angle)). The contact time is where the arc's
   height equals the ball radius (0.105 m): the later root for the descending arc, the earlier
   root for the ascending one. The two estimates are averaged (weighted by their precision).
   Between frames the ball travels 0.3-0.7 m, so this is far more accurate than taking the
   lowest sample. If neither arc has 3 samples, the bounce is skipped.
3. Uncertainty: the fit's covariance (from the residuals) gives the standard deviation of the
   contact point in x and in y separately - the cameras face each other along the court, so
   the along-court (y) error is larger. The call uses the error perpendicular to the line
   being judged (x for sidelines, y for end lines), plus the calibration error (--calib-sigma)
   in quadrature.
4. Call (FIVB rule: the ball is in if it touches the court, lines included; the 9 x 18 m
   court is measured to the outer edges of the lines): signed distance d from the contact
   point to the court boundary (positive inside). IN if d >= -contact radius (the patch reaches
   the line), else OUT. The call is marked "close" when the margin is under 2 sigma.

Outputs (results/calls/):
    <stem>_calls.csv         one row per bounce
    <stem>_calls.png         court diagram (contact points with 2-sigma circles) + height plot
    <stem>_bounce<N>.jpg     replay: both cameras at the contact frame, zoomed on the contact
                             point, with the court lines and the ball's contact patch drawn
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from calibrate_camera import read_frame  # noqa: E402
from camera_3d import load_camera  # noqa: E402

G = 9.81
BALL_R = 0.105
Z_LOW = 0.45         # m; a low point below this height is a bounce candidate
MAX_GAP = 0.2        # s; a gap longer than this splits the 3D track into separate stretches
ARC_WINDOW = 0.4     # s of samples used on each side of the bounce
MAX_FIT_RMS = 0.25   # m; arcs that fit worse than this are not ballistic (e.g. a player touch)
COURT = (0.0, 9.0, 0.0, 18.0)


def load_track(csv_path):
    t, X, weak, ang = [], [], [], []
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            if r["status"] in ("3d", "3d weak"):
                t.append(float(r["time_a_s"]))
                X.append([float(r["x_m"]), float(r["y_m"]), float(r["z_m"])])
                weak.append(r["status"] == "3d weak")
                ang.append(float(r["ray_angle_deg"]))
    return np.array(t), np.array(X).reshape(-1, 3), np.array(weak, bool), np.array(ang)


def fit_arc(t, X, t_ref, ang=None):
    """Weighted ballistic fit around t_ref. Returns params (3 x 2: [x0, vx], [y0, vy], [z0, vz]),
    the parameter covariance scale matrix, and the weighted RMS residual (m).

    Weights: the triangulation error grows like 1 / sin(angle between the two rays), so each
    sample is weighted by sin^2(angle) (normalised). Low samples, where the rays cross at a
    wider angle, count more - which is also where the contact point is."""
    s = t - t_ref
    w = np.ones_like(s) if ang is None else np.sin(np.radians(np.clip(ang, 1, 90))) ** 2
    w = w / w.mean()
    A = np.c_[np.ones_like(s), s]
    Aw = A * np.sqrt(w)[:, None]
    params, res = [], []
    for j in range(3):
        y = X[:, j] + (0.5 * G * s ** 2 if j == 2 else 0.0)
        p, *_ = np.linalg.lstsq(Aw, y * np.sqrt(w), rcond=None)
        params.append(p)
        res.append((y - A @ p) * np.sqrt(w))
    dof = max(1, len(s) - 2)
    rms_xyz = np.array([np.sqrt(np.sum(r ** 2) / dof) for r in res])   # per axis
    rms = float(np.sqrt(np.mean(rms_xyz ** 2)))
    return np.array(params), np.linalg.inv(Aw.T @ Aw), (rms, rms_xyz)


def arc_contact(params, cov, rms, t_ref, descending, t_samples):
    """Time (relative), x, y and per-axis sigma (x, y) of where the arc's height equals the
    ball radius. The errors differ by direction: the cameras face each other along the court,
    so the along-court (y) position is less certain than the across-court (x) position."""
    rms, rms_xyz = rms
    z0, vz = params[2]
    disc = vz ** 2 + 2 * G * (z0 - BALL_R)
    if disc < 0:
        return None                                   # arc never comes down to the floor
    r1, r2 = (vz - np.sqrt(disc)) / G, (vz + np.sqrt(disc)) / G
    s = max(r1, r2) if descending else min(r1, r2)
    s_rel = s
    # the contact must be close to the samples: just after the descent / just before the ascent
    if descending and not -0.02 <= s - (t_samples.max() - t_ref) <= 0.15:
        return None
    if not descending and not -0.15 <= s - (t_samples.min() - t_ref) <= 0.02:
        return None
    a = np.array([1.0, s])
    xy = np.array([params[0] @ a, params[1] @ a])
    sigma = rms_xyz[:2] * float(np.sqrt(a @ cov @ a))  # std of x(s) and y(s) from the fit
    return s_rel, xy, sigma


def find_bounces(t, X, weak, ang):
    """List of bounce dicts from the 3D track."""
    if len(t) < 3:
        return []
    cuts = np.where(np.diff(t) > MAX_GAP)[0] + 1
    bounces = []
    for seg in np.split(np.arange(len(t)), cuts):
        if len(seg) < 3:
            continue
        z = X[seg, 2]
        cand = []
        for i in range(len(seg)):
            lo, hi = max(0, i - 3), min(len(seg), i + 4)
            if z[i] <= Z_LOW and z[i] == z[lo:hi].min():
                cand.append(i)
        # stretch ending while still descending close to the floor: the ball was lost at contact
        if z[-1] <= Z_LOW + 0.3 and z[-1] < z[-2] and (len(seg) - 1) not in cand:
            cand.append(len(seg) - 1)
        # keep one candidate per low region
        merged = []
        for i in cand:
            if merged and t[seg[i]] - t[seg[merged[-1]]] < 0.15:
                if z[i] < z[merged[-1]]:
                    merged[-1] = i
            else:
                merged.append(i)
        for i in merged:
            b = contact_point(t[seg], X[seg], i, ang[seg])
            if b is not None:
                b["weak"] = bool(np.any(weak[seg][max(0, i - 3):i + 4]))
                bounces.append(b)
    return bounces


def contact_point(t, X, i, ang):
    tm = t[i]
    zmin = X[i, 2]
    before = (t < tm) & (t >= tm - ARC_WINDOW)
    after = (t > tm) & (t <= tm + ARC_WINDOW)
    # the low sample itself joins the side whose arc it fits better (it may be just after contact)
    best = None
    for with_low in ("before", "after"):
        bm, am = before.copy(), after.copy()
        (bm if with_low == "before" else am)[i] = True
        ests = []
        for mask, desc in ((bm, True), (am, False)):
            if mask.sum() < 3:
                continue
            p, cov, rms = fit_arc(t[mask], X[mask], tm, ang[mask])
            if rms[0] > MAX_FIT_RMS:
                continue
            c = arc_contact(p, cov, rms, tm, desc, t[mask])
            if c is not None:
                ests.append((c, rms[0], int(mask.sum()), desc))
        if ests:
            score = np.mean([e[1] for e in ests]) / len(ests)
            if best is None or score < best[0]:
                best = (score, ests)
    if best is None:
        return None
    ests = best[1]
    W = np.array([1 / np.maximum(e[0][2], 1e-3) ** 2 for e in ests])     # per-axis weights
    xy = sum(wi * e[0][1] for wi, e in zip(W, ests)) / W.sum(axis=0)
    wt = W.mean(axis=1)
    s = sum(wi * e[0][0] for wi, e in zip(wt, ests)) / wt.sum()
    sigma = 1 / np.sqrt(W.sum(axis=0))
    if len(ests) == 2:                                   # disagreement between the arcs adds error
        sigma = np.hypot(sigma, 0.5 * np.abs(ests[0][0][1] - ests[1][0][1]))
    return dict(t=float(tm + s), x=float(xy[0]), y=float(xy[1]), z_low=float(zmin),
                sigma_x=float(sigma[0]), sigma_y=float(sigma[1]),
                arcs="+".join("down" if e[3] else "up" for e in ests),
                n=sum(e[2] for e in ests))


def boundary_distance(x, y):
    """Signed distance (m) from (x, y) to the court boundary (positive inside) and the line."""
    x0, x1, y0, y1 = COURT
    inside = x0 <= x <= x1 and y0 <= y <= y1
    names = {"x0": "sideline x = 0", "x1": "sideline x = 9", "y0": "end line y = 0", "y1": "end line y = 18"}
    if inside:
        d = {"x0": x - x0, "x1": x1 - x, "y0": y - y0, "y1": y1 - y}
        k = min(d, key=d.get)
        return d[k], names[k]
    dx = max(x0 - x, 0, x - x1)
    dy = max(y0 - y, 0, y - y1)
    k = ("x0" if x < x0 else "x1") if dx >= dy else ("y0" if y < y0 else "y1")
    return -float(np.hypot(dx, dy)), names[k]


def make_call(b, contact_r, calib_sigma):
    d, line = boundary_distance(b["x"], b["y"])
    # the error that matters is the one perpendicular to the line being judged
    s_fit = b["sigma_x"] if line.startswith("sideline") else b["sigma_y"]
    sigma = float(np.hypot(s_fit, calib_sigma))
    margin = d + contact_r                            # > 0: the contact patch reaches the court
    call = "IN" if margin >= 0 else "OUT"
    close = abs(margin) < 2 * sigma
    side = "camera B's half" if b["y"] < 9 else "camera A's half"
    b.update(distance_m=d, line=line, sigma=sigma, call=call, close=close, side=side)
    return b


# ---------------------------------------------------------------- figures
def plot_calls(bounces, t, X, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Rectangle
    fig = plt.figure(figsize=(11, 6.5))
    ax = fig.add_axes([0.04, 0.08, 0.32, 0.86])
    ax.add_patch(Rectangle((0, 0), 9, 18, fc="#c9a77c", ec="none"))
    for a, b in [((0, 0), (9, 0)), ((0, 6), (9, 6)), ((0, 12), (9, 12)), ((0, 18), (9, 18)),
                 ((0, 0), (0, 18)), ((9, 0), (9, 18))]:
        ax.plot([a[0], b[0]], [a[1], b[1]], color="#e8c400", lw=2)
    ax.plot([-1, 10], [9, 9], color="k", lw=3)
    ax.plot(X[:, 0], X[:, 1], ".", ms=2, color="#888888")
    for n, b in enumerate(bounces, 1):
        col = "#1a9850" if b["call"] == "IN" else "#d73027"
        ax.add_patch(Circle((b["x"], b["y"]), 2 * b["sigma"], fc=col, alpha=0.25, ec=col))
        ax.plot(b["x"], b["y"], "o", color=col, ms=6)
        ax.annotate(f"{n}", (b["x"], b["y"]), xytext=(6, 4), textcoords="offset points", fontsize=9)
    ax.set_xlim(-3, 12)
    ax.set_ylim(21, -3)
    ax.set_aspect("equal")
    ax.set_xlabel("x across (m)")
    ax.set_ylabel("y along (m); camera B end at top, camera A end at bottom")
    ax.set_title("contact points (circle = 2 sigma)", fontsize=10)
    ax2 = fig.add_axes([0.45, 0.55, 0.52, 0.38])
    ax2.plot(t, X[:, 2], ".", ms=3, color="#4575b4")
    for n, b in enumerate(bounces, 1):
        ax2.axvline(b["t"], color="#1a9850" if b["call"] == "IN" else "#d73027", lw=1)
        ax2.text(b["t"], ax2.get_ylim()[1] if ax2.get_ylim()[1] > 0 else 3, f" {n}", fontsize=9, va="top")
    ax2.axhline(BALL_R, color="k", lw=0.8)
    ax2.set_xlabel("time in camera A (s)")
    ax2.set_ylabel("ball height (m)")
    ax3 = fig.add_axes([0.45, 0.06, 0.52, 0.36])
    ax3.axis("off")
    lines = [f"{'#':>2}  {'time':>7}  {'x':>6}  {'y':>6}  {'call':<10} {'to line':>8}  {'+/- 2s':>7}  line"]
    for n, b in enumerate(bounces, 1):
        call = b["call"] + (" close" if b["close"] else "")
        lines.append(f"{n:>2}  {b['t']:7.2f}  {b['x']:6.2f}  {b['y']:6.2f}  {call:<10} "
                     f"{100 * b['distance_m']:+7.0f}cm  {200 * b['sigma']:6.0f}cm  {b['line']}")
    ax3.text(0, 1, "\n".join(lines), family="monospace", fontsize=9, va="top")
    fig.suptitle(title, fontsize=11)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def replay_image(b, cams, videos, offset, path, contact_r, zoom_m=1.6):
    """Both cameras at the contact time, zoomed on the contact point."""
    panels = []
    for cam, video, t in ((cams[0], videos[0], b["t"]), (cams[1], videos[1], b["t"] + offset)):
        img = read_frame(video, t)
        if img is None:
            continue
        s = np.linspace(0, 1, 80)[:, None]
        for (a0, a1) in [((0, 0), (9, 0)), ((0, 18), (9, 18)), ((0, 0), (0, 18)), ((9, 0), (9, 18)),
                         ((0, 6), (9, 6)), ((0, 9), (9, 9)), ((0, 12), (9, 12))]:
            pts = cam.project(np.c_[np.asarray(a0) * (1 - s) + np.asarray(a1) * s, np.zeros(80)])
            cv2.polylines(img, [np.int32(np.round(pts))], False, (255, 0, 255), 1, cv2.LINE_AA)
        ang = np.linspace(0, 2 * np.pi, 60)
        col = (0, 200, 0) if b["call"] == "IN" else (0, 0, 255)
        patch = cam.project(np.c_[b["x"] + contact_r * np.cos(ang), b["y"] + contact_r * np.sin(ang), np.zeros(60)])
        cv2.polylines(img, [np.int32(np.round(patch))], True, col, 2, cv2.LINE_AA)
        two_s = cam.project(np.c_[b["x"] + 2 * b["sigma"] * np.cos(ang), b["y"] + 2 * b["sigma"] * np.sin(ang),
                                  np.zeros(60)])
        cv2.polylines(img, [np.int32(np.round(two_s))], True, (255, 255, 255), 1, cv2.LINE_AA)
        ball = cam.project(np.c_[b["x"] + BALL_R * np.cos(ang), np.full(60, b["y"]), BALL_R + BALL_R * np.sin(ang)])
        cv2.polylines(img, [np.int32(np.round(ball))], True, (0, 255, 255), 1, cv2.LINE_AA)
        # crop a square of about zoom_m metres around the contact point
        c = cam.project([b["x"], b["y"], 0.0])[0]
        e = cam.project([b["x"] + zoom_m / 2, b["y"], 0.0])[0]
        half = int(np.clip(np.linalg.norm(e - c), 40, 400))
        x0, y0 = int(c[0] - half), int(c[1] - half)
        crop = cv2.copyMakeBorder(img, half, half, half, half, cv2.BORDER_CONSTANT)[y0 + half:y0 + 3 * half,
                                                                                    x0 + half:x0 + 3 * half]
        crop = cv2.resize(crop, (480, 480), interpolation=cv2.INTER_CUBIC)
        small = cv2.resize(img, (480 * img.shape[1] // img.shape[0] // 2 * 2 // 2, 240))
        panels.append((crop, small))
    if not panels:
        return
    top = np.hstack([p[0] for p in panels])
    bottom = np.hstack([cv2.resize(p[1], (480, 270)) for p in panels])
    out = np.vstack([top, bottom])
    txt = (f"{b['call']}{' (close)' if b['close'] else ''}  {100 * b['distance_m']:+.0f} cm from {b['line']} "
           f"(+/- {200 * b['sigma']:.0f} cm)  t={b['t']:.2f}s")
    cv2.putText(out, txt, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(out, txt, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(out, "camera A", (10, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(out, "camera B", (490, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.imwrite(str(path), out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run_json", help="the .json written by ball_two_cameras.py")
    p.add_argument("--contact-radius", type=float, default=0.03)
    p.add_argument("--calib-sigma", type=float, default=0.05)
    p.add_argument("--no-replays", action="store_true")
    args = p.parse_args()
    run = json.loads(Path(args.run_json).read_text())
    t, X, weak, ang = load_track(run["csv"])
    if len(t) == 0:
        raise SystemExit("No 3D ball samples in " + run["csv"])
    bounces = [make_call(b, args.contact_radius, args.calib_sigma) for b in find_bounces(t, X, weak, ang)]
    stem = Path(run["csv"]).stem
    out = Path("results/calls")
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f"{stem}_calls.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["bounce", "time_a_s", "x_m", "y_m", "call", "close", "distance_to_line_m", "nearest_line",
                    "sigma_m", "half", "arcs_used", "samples", "lowest_sample_z_m", "weak_geometry"])
        for n, b in enumerate(bounces, 1):
            w.writerow([n, round(b["t"], 3), round(b["x"], 3), round(b["y"], 3), b["call"], int(b["close"]),
                        round(b["distance_m"], 3), b["line"], round(b["sigma"], 3), b["side"], b["arcs"], b["n"],
                        round(b["z_low"], 3), int(b["weak"])])
    print(f"{len(bounces)} bounce(s) found in {len(t)} 3D samples:")
    for n, b in enumerate(bounces, 1):
        print(f"  {n}. t = {b['t']:6.2f} s  ({b['x']:5.2f}, {b['y']:5.2f}) m  {b['call']:<3}"
              f"{' (close)' if b['close'] else '        '}  {100 * b['distance_m']:+5.0f} cm from {b['line']}, "
              f"+/- {200 * b['sigma']:.0f} cm (2 sigma), {b['side']}, arcs: {b['arcs']}")
    plot_calls(bounces, t, X, out / f"{stem}_calls.png", f"{stem}: line calls")
    if not args.no_replays and bounces:
        cams = [load_camera(run["calib_a"])[0], load_camera(run["calib_b"])[0]]
        for n, b in enumerate(bounces, 1):
            replay_image(b, cams, (run["video_a"], run["video_b"]), run["offset_s"], out / f"{stem}_bounce{n}.jpg",
                         args.contact_radius)
    print(f"Saved {out / (stem + '_calls.csv')}, _calls.png" + ("" if args.no_replays else " and _bounce<N>.jpg"))


if __name__ == "__main__":
    main()
