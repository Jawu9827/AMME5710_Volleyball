"""Automatic court detection: find the court outline and compute the homography.

Usage:
    python scripts/court_detect.py frames/volleyballmatch/frame_000300.jpg
    python scripts/court_detect.py frames/volleyballmatch/frame_000300.jpg \
        --gt data/annotations/volleyballmatch_frame_000300_clicks.npy --gt-points near-half

Pipeline
    1. Free-zone colour model: sample the floor colour just outside the near end
       line (bottom strip of the frame) and mark every pixel of that hue
       (HSV hue distance + saturation/value floor), then clean with an opening.
    2. Boundary scanning: march along image rows/columns from inside the court
       until a run of free-zone pixels is hit. This gives candidate points on the
       near end line, far end line and both sidelines.
    3. RANSAC line fit per edge, so players standing on or over the lines
       (outliers) don't bend the fit; least squares on the inliers.
    4. Intersect the four lines -> four court corners -> homography to the
       9 m x 18 m court model (lines count as part of the court, so the outer
       edge of the boundary lines is exactly the court edge).
    5. Check: project the centre and attack lines (never used in the fit) back
       into the image, and compare with hand-clicked ground truth if given.

Outputs (next to the image):
    <name>_auto_H.npy        image -> court metres
    <name>_auto_corners.npy  4x2 corners: far-left, far-right, near-right, near-left
    <name>_auto_debug.jpg    free-zone mask, scanned points, fitted lines, overlay
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

# ---- Court model (metres): x across the 9 m width, y along the 18 m length ----
COURT_CORNERS = np.float32([[0, 0], [9, 0], [9, 18], [0, 18]])
COURT_LINES = {
    "far end": ((0, 0), (9, 0)),
    "far attack": ((0, 6), (9, 6)),
    "centre": ((0, 9), (9, 9)),
    "near attack": ((0, 12), (9, 12)),
    "near end": ((0, 18), (9, 18)),
    "left side": ((0, 0), (0, 18)),
    "right side": ((9, 0), (9, 18)),
}
GT_POINT_SETS = {  # must match manual_homography.py
    "near-half": np.float32([[0, 12], [9, 12], [9, 18], [0, 18]]),
    "full": COURT_CORNERS,
}

# ---- Tuning parameters (fractions of image size so they work at any resolution) ----
HUE_TOL = 12            # OpenCV hue units (0-179)
MIN_SAT, MIN_VAL = 70, 70
RUN_FRAC = 0.012        # a free-zone run this long (x image width) counts as "outside"
FAR_RUN_PX = 3          # the free zone beyond the far end line is thin in the image,
                        # and the net's bottom cable can cross it, so use a short run
SCAN_STEP_FRAC = 0.005  # spacing between scan lines
RANSAC_TOL_FRAC = 0.004 # inlier distance threshold (x image height)
RANSAC_ITERS = 500


def free_zone_mask(img):
    """Pixels with the same colour as the floor outside the court."""
    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    strip = hsv[int(h * 0.965):int(h * 0.985), int(w * 0.1):int(w * 0.9)].reshape(-1, 3)
    strip = strip[(strip[:, 1] > MIN_SAT) & (strip[:, 2] > MIN_VAL)]
    if len(strip) == 0:
        raise SystemExit("Bottom strip has no saturated colour: is the near free zone in view?")
    hue0 = float(np.median(strip[:, 0]))
    dh = np.abs(hsv[..., 0].astype(np.int16) - hue0)
    dh = np.minimum(dh, 180 - dh)
    mask = (dh < HUE_TOL) & (hsv[..., 1] > MIN_SAT) & (hsv[..., 2] > MIN_VAL)
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return mask.astype(bool), hue0


def first_run(line, run):
    """Index of the first position where `run` consecutive True values start, or None."""
    if len(line) < run:
        return None
    c = np.convolve(line.astype(np.int32), np.ones(run, np.int32), mode="valid")
    idx = np.flatnonzero(c == run)
    return int(idx[0]) if len(idx) else None


def ransac_line(pts, tol, iters=RANSAC_ITERS, seed=0):
    """Fit a 2D line to points with RANSAC. Returns (point, unit direction, inlier mask)."""
    rng = np.random.default_rng(seed)
    pts = np.asarray(pts, np.float64)
    best = None
    for _ in range(iters):
        a, b = pts[rng.choice(len(pts), 2, replace=False)]
        d = b - a
        if np.hypot(*d) < 1e-6:
            continue
        n = np.array([-d[1], d[0]]) / np.hypot(*d)
        inl = np.abs((pts - a) @ n) < tol
        if best is None or inl.sum() > best.sum():
            best = inl
    # Least-squares refit on inliers (total least squares via SVD)
    p = pts[best]
    c = p.mean(axis=0)
    _, _, vt = np.linalg.svd(p - c)
    return c, vt[0], best


def intersect(l1, l2):
    (p1, d1), (p2, d2) = l1, l2
    A = np.array([d1, -d2]).T
    t = np.linalg.solve(A, p2 - p1)
    return p1 + t[0] * d1


def detect_court(img):
    h, w = img.shape[:2]
    blue, hue0 = free_zone_mask(img)
    run = max(3, int(RUN_FRAC * w))
    step = max(1, int(SCAN_STEP_FRAC * w))
    tol = max(1.5, RANSAC_TOL_FRAC * h)
    cols = np.arange(int(0.12 * w), int(0.88 * w), step)

    # 1. Near end line: from the bottom, pass through the free zone, then find the court.
    near_pts = []
    for x in cols:
        col = blue[::-1, x]                      # bottom -> top
        s = first_run(col, 3)
        if s is None:
            continue
        e = first_run(~col[s:], run)             # first long non-free-zone run
        if e is not None:
            near_pts.append((x, h - 1 - (s + e) + 0.5))
    near = ransac_line(near_pts, tol)

    def near_y(x):
        p, d = near[0], near[1]
        return p[1] + (x - p[0]) * d[1] / d[0]

    # 2. Far end line: from just inside the near end line, go up until the free zone.
    far_pts = []
    for x in cols:
        y0 = int(near_y(x) - 0.04 * h)
        if y0 <= 0 or blue[y0, x]:
            continue
        col = blue[y0::-1, x]                    # y0 -> top
        s = first_run(col, FAR_RUN_PX)
        if s is not None:
            far_pts.append((x, y0 - s + 0.5))
    far = ransac_line(far_pts, tol)

    def far_y(x):
        p, d = far[0], far[1]
        return p[1] + (x - p[0]) * d[1] / d[0]

    # 3. Sidelines: rows between the end lines, scan outwards from the centre column.
    left_pts, right_pts = [], []
    xc = w // 2
    y_lo, y_hi = int(far_y(xc) + 0.02 * h), int(near_y(xc) - 0.01 * h)
    for y in range(y_lo, y_hi, step):
        if blue[y, xc]:
            continue
        s = first_run(blue[y, xc::-1], run)      # centre -> left
        if s is not None:
            left_pts.append((xc - s + 0.5, y))
        s = first_run(blue[y, xc:], run)         # centre -> right
        if s is not None:
            right_pts.append((xc + s - 0.5, y))
    left = ransac_line(left_pts, tol)
    right = ransac_line(right_pts, tol)

    lines = {"far": far, "near": near, "left": left, "right": right}
    L = {k: (v[0], v[1]) for k, v in lines.items()}
    corners = np.float32([
        intersect(L["far"], L["left"]), intersect(L["far"], L["right"]),
        intersect(L["near"], L["right"]), intersect(L["near"], L["left"]),
    ])
    H = cv2.getPerspectiveTransform(corners, COURT_CORNERS)
    scans = {"far": far_pts, "near": near_pts, "left": left_pts, "right": right_pts}
    return H, corners, blue, scans, lines, hue0


def project(H_inv, pts_m):
    return cv2.perspectiveTransform(np.float32(pts_m).reshape(-1, 1, 2), H_inv).reshape(-1, 2)


def compare_to_gt(H, gt_clicks, gt_court):
    H_inv = np.linalg.inv(H)
    pred_px = project(H_inv, gt_court)
    err_px = np.linalg.norm(pred_px - gt_clicks, axis=1)
    pred_m = cv2.perspectiveTransform(np.float32(gt_clicks).reshape(-1, 1, 2), H).reshape(-1, 2)
    err_m = np.linalg.norm(pred_m - gt_court, axis=1)
    return err_px, err_m


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    parser.add_argument("--gt", help="clicks .npy saved by manual_homography.py")
    parser.add_argument("--gt-points", choices=GT_POINT_SETS, default="near-half")
    args = parser.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        raise SystemExit(f"Could not read {args.image}")
    H, corners, blue, scans, lines, hue0 = detect_court(img)
    H_inv = np.linalg.inv(H)

    print(f"Free-zone hue: {hue0:.0f}")
    for k in ("far", "near", "left", "right"):
        print(f"{k:>5} edge: {lines[k][2].sum():4d}/{len(scans[k]):4d} scan points are RANSAC inliers")
    names = ["far-left", "far-right", "near-right", "near-left"]
    for n, c in zip(names, corners):
        print(f"{n:>10} corner: ({c[0]:7.1f}, {c[1]:7.1f}) px")

    if args.gt:
        gt = np.load(args.gt)
        err_px, err_m = compare_to_gt(H, gt, GT_POINT_SETS[args.gt_points])
        print(f"\nAgainst hand-clicked ground truth ({args.gt_points}):")
        for i, (ep, em) in enumerate(zip(err_px, err_m), 1):
            print(f"  point {i}: {ep:5.1f} px   {em * 100:5.1f} cm on court")
        print(f"  mean:    {err_px.mean():5.1f} px   {err_m.mean() * 100:5.1f} cm")

    # ---- Debug figure ----
    h, w = img.shape[:2]
    mask_vis = cv2.cvtColor(blue.astype(np.uint8) * 255, cv2.COLOR_GRAY2BGR)
    scan_vis = img.copy()
    colours = {"far": (255, 0, 255), "near": (0, 255, 255), "left": (255, 128, 0), "right": (0, 128, 255)}
    for k, pts in scans.items():
        inl = lines[k][2]
        for (x, y), ok in zip(pts, inl):
            cv2.circle(scan_vis, (int(x), int(y)), 3, colours[k] if ok else (0, 0, 255), -1)
    overlay = img.copy()
    for name, (a, b) in COURT_LINES.items():
        pa, pb = project(H_inv, [a, b])
        colour = (0, 255, 0) if name in ("far end", "near end", "left side", "right side") else (0, 200, 255)
        cv2.line(overlay, tuple(map(int, pa)), tuple(map(int, pb)), colour, 2)
    for c in corners:
        cv2.circle(overlay, tuple(map(int, c)), 6, (0, 0, 255), -1)
    panel = np.vstack([np.hstack([mask_vis, scan_vis]), np.hstack([overlay, img])])
    panel = cv2.resize(panel, None, fx=0.5, fy=0.5)

    stem = Path(args.image).with_suffix("")
    np.save(f"{stem}_auto_H.npy", H)
    np.save(f"{stem}_auto_corners.npy", corners)
    cv2.imwrite(f"{stem}_auto_debug.jpg", panel)
    cv2.imwrite(f"{stem}_auto_overlay.jpg", overlay)
    print(f"\nSaved {stem}_auto_H.npy, _auto_corners.npy, _auto_debug.jpg, _auto_overlay.jpg")


if __name__ == "__main__":
    main()
