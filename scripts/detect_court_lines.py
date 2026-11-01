"""Detect the yellow volleyball boundary lines (near end line + both sidelines) automatically.

Usage:
    python scripts/detect_court_lines.py frames/game2_A/t0060.00.png
    python scripts/detect_court_lines.py data/game2_A.mp4 --time 60

Pipeline (one frame):
  0. Colour model: the floor colour is measured in a strip at the bottom centre of the frame
     (court floor in front of the camera), and the thresholds are set relative to it. This
     copes with different exposure and with HDR video (iPhone HLG), whose colours come out
     washed out (floor saturation ~22 instead of ~67) when decoded without HDR handling.
  1. Floor mask: pixels near the floor's hue, largest region, holes filled.
  2. Line colour mask: yellow pixels (HSV) on or next to the floor; the walls/ceiling are
     excluded by the floor mask, small blobs (skin, shoes) by size/shape filtering.
  3. Probabilistic Hough transform on the mask -> line segments, grouped by angle into
     near end line (near-horizontal, lowest), left sideline and right sideline.
  4. RANSAC line fit for each group on the mask pixels near its segments (players
     standing on the line are outliers), then a least-squares refit on the inliers.
  5. Near corners = end line x sidelines.
Output: results/court_lines/<name>.json (line equations, corners, sample points along each
line for calibration) and <name>_debug.jpg (mask, segments, fitted lines).
"""
import argparse
import json
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np

# Colour thresholds (OpenCV HSV: H 0-179). Saturation/brightness limits are fractions of the
# floor colour measured in each frame; hues are fixed. Measured on our footage:
#   yellow lines hue 26-36 (both cameras), orange lines 9-20 (HDR shifts them up), floor ~100.
LINE_HUE = (22, 42)          # yellow, excluding orange
FLOOR_HUE_TOL = 12           # floor = within this of the measured floor hue
FLOOR_SAT_FRAC, FLOOR_VAL_FRAC = 0.4, 0.4
LINE_SAT_FRAC, LINE_VAL_FRAC = 0.5, 0.6
MIN_SAT = 20                 # absolute floor for line saturation (below this, hue is noise)
MIN_SEG_FRAC = 0.04          # Hough segments at least this x image width
RANSAC_TOL_FRAC = 0.002      # inlier distance (x image width)


def ffmpeg_exe():
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        raise SystemExit("ffmpeg not found. Run:  pip install imageio-ffmpeg")


def load_image(source, t):
    if Path(source).suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp"):
        return cv2.imread(str(source))
    cmd = [ffmpeg_exe(), "-v", "error", "-ss", f"{t:.3f}", "-i", str(source), "-frames:v", "1",
           "-f", "image2pipe", "-vcodec", "png", "-"]
    data = subprocess.run(cmd, capture_output=True).stdout
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR) if data else None


def colour_model(hsv):
    """Floor colour from the bottom-centre strip -> thresholds for floor and line masks."""
    h, w = hsv.shape[:2]
    strip = hsv[int(0.75 * h):int(0.95 * h), int(0.3 * w):int(0.7 * w)].reshape(-1, 3)
    fh, fs, fv = np.median(strip, axis=0)
    if not 70 <= fh <= 130:
        print(f"  warning: bottom-centre colour (hue {fh:.0f}) doesn't look like a blue floor; "
              "using default thresholds")
        fh, fs, fv = 100.0, 60.0, 170.0
    return dict(floor_hue=float(fh), floor_sat=float(fs), floor_val=float(fv),
                floor_min_sat=max(8.0, FLOOR_SAT_FRAC * fs), floor_min_val=FLOOR_VAL_FRAC * fv,
                line_min_sat=max(MIN_SAT, LINE_SAT_FRAC * fs), line_min_val=LINE_VAL_FRAC * fv)


def floor_mask(hsv, cm):
    h, w = hsv.shape[:2]
    dh = np.abs(hsv[..., 0].astype(np.int16) - cm["floor_hue"])
    m = ((dh <= FLOOR_HUE_TOL) & (hsv[..., 1] >= cm["floor_min_sat"]) &
         (hsv[..., 2] >= cm["floor_min_val"])).astype(np.uint8)
    m[: int(0.35 * h)] = 0                                   # floor is in the lower part
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    # Other sports' lines (orange, black) cut the floor into pieces. Close first so thin
    # lines are bridged, then keep every large floor region (not just the biggest one).
    k = max(15, w // 60) | 1
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    if n <= 1:
        raise ValueError("no floor-coloured region found")
    areas = st[1:, cv2.CC_STAT_AREA]
    keep = np.r_[False, areas >= max(0.2 * areas.max(), 0.01 * h * w)]
    m = keep[lab].astype(np.uint8)
    return cv2.dilate(m, np.ones((7, 7), np.uint8))


def line_mask(img, hsv, floor, cm):
    m = ((hsv[..., 0] >= LINE_HUE[0]) & (hsv[..., 0] <= LINE_HUE[1]) &
         (hsv[..., 1] >= cm["line_min_sat"]) & (hsv[..., 2] >= cm["line_min_val"])).astype(np.uint8)
    m &= floor
    # Remove compact blobs (skin, shoes); keep thin elongated pieces
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    keep = np.zeros(n, bool)
    for i in range(1, n):
        x, y, bw, bh, area = st[i]
        length = np.hypot(bw, bh)
        fill = area / max(1, bw * bh)
        keep[i] = length >= 0.02 * img.shape[1] and (fill < 0.5 or min(bw, bh) <= 6)
    return keep[lab].astype(np.uint8)


def ransac_line(pts, tol, iters=400, seed=0):
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(iters):
        a, b = pts[rng.choice(len(pts), 2, replace=False)]
        d = b - a
        L = np.hypot(*d)
        if L < 1e-6:
            continue
        nrm = np.array([-d[1], d[0]]) / L
        inl = np.abs((pts - a) @ nrm) < tol
        if best is None or inl.sum() > best.sum():
            best = inl
    p = pts[best]
    c = p.mean(axis=0)
    _, _, vt = np.linalg.svd(p - c)
    return c, vt[0], best


def intersect(l1, l2):
    (p1, d1), (p2, d2) = l1, l2
    t = np.linalg.solve(np.array([d1, -d2]).T, p2 - p1)
    return p1 + t[0] * d1


def detect(img):
    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    cm = colour_model(hsv)
    print(f"  floor colour: hue {cm['floor_hue']:.0f}, saturation {cm['floor_sat']:.0f}, "
          f"brightness {cm['floor_val']:.0f}  -> line saturation >= {cm['line_min_sat']:.0f}"
          + ("   (low saturation: HDR or washed-out video)" if cm["floor_sat"] < 35 else ""))
    floor = floor_mask(hsv, cm)
    mask = line_mask(img, hsv, floor, cm)
    segs = cv2.HoughLinesP(mask * 255, 1, np.pi / 360, threshold=max(30, w // 40),
                           minLineLength=int(MIN_SEG_FRAC * w), maxLineGap=int(0.01 * w))
    segs = [] if segs is None else segs.reshape(-1, 4)
    groups = {"near end line": [], "left sideline": [], "right sideline": []}
    for x1, y1, x2, y2 in segs:
        if x2 < x1:
            x1, y1, x2, y2 = x2, y2, x1, y1
        ang = np.degrees(np.arctan2(y2 - y1, x2 - x1))      # image y points down
        if abs(ang) < 8:
            groups["near end line"].append((x1, y1, x2, y2))
        elif -80 < ang < -8:                                  # rising to the right: left sideline
            groups["left sideline"].append((x1, y1, x2, y2))
        elif 8 < ang < 80:                                    # falling to the right: right sideline
            groups["right sideline"].append((x1, y1, x2, y2))
    # For the end line keep only the lowest cluster of near-horizontal segments
    if groups["near end line"]:
        ys = np.array([(s[1] + s[3]) / 2 for s in groups["near end line"]])
        groups["near end line"] = [s for s, y in zip(groups["near end line"], ys) if y > ys.max() - 0.02 * h]

    ys, xs = np.nonzero(mask)
    allpts = np.c_[xs, ys].astype(float)
    tol = max(1.5, RANSAC_TOL_FRAC * w)
    lines = {}
    for name, ss in groups.items():
        if not ss:
            continue
        # mask pixels close to any segment of this group
        near = np.zeros(len(allpts), bool)
        for x1, y1, x2, y2 in ss:
            d = np.array([x2 - x1, y2 - y1], float)
            L = np.hypot(*d)
            nrm = np.array([-d[1], d[0]]) / L
            t = (allpts - (x1, y1)) @ (d / L)
            near |= (np.abs((allpts - (x1, y1)) @ nrm) < 6) & (t > -0.05 * w) & (t < L + 0.05 * w)
        pts = allpts[near]
        if len(pts) < 20:
            continue
        c, d, inl = ransac_line(pts, tol)
        lines[name] = dict(point=c, direction=d, inliers=pts[inl])
    return floor, mask, segs, groups, lines


def main():
    p = argparse.ArgumentParser()
    p.add_argument("source", help="image, or video with --time")
    p.add_argument("--time", type=float, default=0.0)
    p.add_argument("--samples", type=int, default=8, help="sample points saved per line")
    args = p.parse_args()

    img = load_image(args.source, args.time)
    if img is None:
        raise SystemExit(f"Could not read {args.source}")
    h, w = img.shape[:2]
    floor, mask, segs, groups, lines = detect(img)

    print(f"{len(segs)} Hough segments; groups: " +
          ", ".join(f"{k} {len(v)}" for k, v in groups.items()))
    corners = {}
    for side in ("left", "right"):
        if "near end line" in lines and f"{side} sideline" in lines:
            L1, L2 = lines["near end line"], lines[f"{side} sideline"]
            corners[f"near-{side} corner"] = intersect((L1["point"], L1["direction"]),
                                                      (L2["point"], L2["direction"]))
    for name, L in lines.items():
        print(f"  {name:15s}: {len(L['inliers'])} inlier pixels")
    for name, c in corners.items():
        print(f"  {name:18s}: ({c[0]:.1f}, {c[1]:.1f})")
    missing = [k for k in groups if k not in lines]
    if missing:
        print(f"  NOT FOUND: {', '.join(missing)}  -> check the debug image / colour thresholds")

    # Sample points along each line from the ACTUAL line pixels (not the fitted straight line):
    # split the inliers into bins along the line and take the median pixel of each bin.
    # These are real measurements, so they carry real noise and follow any lens bending.
    samples = {}
    for name, L in lines.items():
        q = L["inliers"]
        t = (q - L["point"]) @ L["direction"]
        edges = np.linspace(np.percentile(t, 2), np.percentile(t, 98), args.samples + 1)
        pts = []
        for a, b in zip(edges[:-1], edges[1:]):
            sel = (t >= a) & (t < b)
            if sel.sum() >= 5:
                pts.append(np.median(q[sel], axis=0).tolist())
        samples[name] = pts

    # Debug figure: mask | segments + fitted lines
    vis = img.copy()
    vis[mask > 0] = (0, 0, 255)
    fit = img.copy()
    colours = {"near end line": (0, 255, 255), "left sideline": (255, 0, 255), "right sideline": (255, 255, 0)}
    for name, ss in groups.items():
        for x1, y1, x2, y2 in ss:
            cv2.line(vis, (int(x1), int(y1)), (int(x2), int(y2)), colours[name], 2)
    for name, L in lines.items():
        a, b = L["point"] - 3000 * L["direction"], L["point"] + 3000 * L["direction"]
        cv2.line(fit, tuple(map(int, a)), tuple(map(int, b)), colours[name], max(1, w // 960), cv2.LINE_AA)
        for q in samples[name]:
            cv2.circle(fit, (int(q[0]), int(q[1])), max(3, w // 400), colours[name], -1)
    for c in corners.values():
        cv2.drawMarker(fit, (int(c[0]), int(c[1])), (0, 0, 255), cv2.MARKER_CROSS, max(20, w // 60), 2)
    debug = np.vstack([vis, fit])
    debug = cv2.resize(debug, (min(w, 1600), int(min(w, 1600) * debug.shape[0] / w)))

    is_video = Path(args.source).suffix.lower() not in (".png", ".jpg", ".jpeg", ".bmp")
    stem = Path(args.source).stem + (f"_t{args.time:g}" if is_video else "")
    out_dir = Path("results/court_lines")
    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_dir / f"{stem}_debug.jpg"), debug)
    data = dict(source=str(args.source), time_s=args.time if is_video else None, image_size=[w, h],
                lines={k: dict(point=v["point"].tolist(), direction=v["direction"].tolist(),
                               inlier_pixels=int(len(v["inliers"])), samples=samples[k])
                       for k, v in lines.items()},
                corners={k: v.tolist() for k, v in corners.items()})
    (out_dir / f"{stem}.json").write_text(json.dumps(data, indent=1))
    print(f"Saved {out_dir / (stem + '.json')} and _debug.jpg")


if __name__ == "__main__":
    main()
