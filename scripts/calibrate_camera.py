"""Calibrate one fixed camera by clicking court points and lines, then check it across the video.

Usage (yellow-line court, recommended): detect lines first, then click only what's left
    python scripts/detect_court_lines.py data/game2_A.mp4 --time 60
    python scripts/calibrate_camera.py data/game2_A.mp4 --camera A --end near --time 60 \
        --auto-lines results/court_lines/game2_A_t60.json
    (the detected end line, sidelines and near corners are used automatically; press s to
     skip the near corners and click the two centre-line points under the net, PLUS at
     least one more pair at a different distance, e.g. the attack-line ends or the far
     corners: without it nothing independently checks the court's length, and the
     accuracy estimate cannot detect a misclicked centre line)

Usage (clicking everything):
    python scripts/calibrate_camera.py data/Game1Trim.mp4  --camera A --end near --time 60
    python scripts/calibrate_camera.py data/Game1Trim2.mp4 --camera B --end far  --time 60
    python scripts/calibrate_camera.py data/Game1Trim2.mp4 --camera B --end far --time 60 \
        --points-json data/annotations/calib_B.json        (refit saved clicks, no clicking)

--end says which end line the camera is behind, in the shared court frame:
    near: behind the y = 18 end line;   far: behind the y = 0 end line.
Names are always from the camera's own point of view ("near-left" = the corner nearest
the camera on its left); the script converts them, so both cameras use the same 9 x 18 m
court coordinates.

Clicking happens in two stages. A magnifier (top-right, 5x, full resolution) follows the
mouse: put its crosshair on the centre of the line and click.
  1. Intersections (4+ needed). Keys: s skip one you can't see, u undo, i/j/k/l nudge the
     last click by 1 px, Enter go to stage 2.
  2. Points ALONG lines (optional but strongly recommended). For each named line, click 3-6
     points anywhere on it, spread out, avoiding players. n next line, b previous line,
     u undo, Enter finish. Each point adds a constraint, so a few imperfect clicks average
     out, and lines can be used where their intersections are hidden or at the frame edge.

Fit: a homography (court -> image) by robust least squares on the pixel distances of the
intersections and of the line points from their lines. An optional radial lens-distortion
term k1 is also fitted; with --distortion auto (default) it is kept only if it improves
the leave-one-out error.

Outputs: data/annotations/calib_<camera>.json (commit this), plus in results/calib/
    calib_<camera>_overlay.jpg   court drawn on the calibration frame, clicks marked
    calib_<camera>_check.jpg     court on frames through the video, with tripod shift
Leave-one-out error: each intersection is predicted by a fit that did not use it, and
the distance from its true court position is reported. This is the honest accuracy number.

Other scripts: `from calibrate_camera import load_calibration` then
    cal = load_calibration("data/annotations/calib_A.json")
    court_xy = cal.image_to_court([[u, v], ...])     # pixels -> metres (floor points only)
"""
import argparse
import json
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares

# Court lines are 5 cm wide and lie INSIDE the 9 x 18 m court (whose size is measured to the
# outer edges). Clicks and detected lines are on the CENTRE of the painted line, so they are
# modelled at the line centres: boundary lines 2.5 cm inside the edge, attack lines with their
# rear edge 3 m from the centre-line axis (centre 2.975 m from it), centre line on the axis.
# The court frame itself (0-9, 0-18) stays the outer boundary, which is what in/out is judged on.
LINE_W = 0.05
_h = LINE_W / 2
X_L, X_R = _h, 9 - _h                       # sideline centres
Y_FAR, Y_NEAR = _h, 18 - _h                 # end line centres (camera's own view)
Y_ATT_FAR, Y_ATT_NEAR = 9 - 3 + _h, 9 + 3 - _h
# Intersections in the camera's own view; court coords for a camera behind y = 18.
POINTS = [
    ("near-left corner", (X_L, Y_NEAR)), ("near-right corner", (X_R, Y_NEAR)),
    ("near attack line / left sideline", (X_L, Y_ATT_NEAR)), ("near attack line / right sideline", (X_R, Y_ATT_NEAR)),
    ("centre line / left sideline (under net)", (X_L, 9)), ("centre line / right sideline (under net)", (X_R, 9)),
    ("far attack line / left sideline", (X_L, Y_ATT_FAR)), ("far attack line / right sideline", (X_R, Y_ATT_FAR)),
    ("far-left corner", (X_L, Y_FAR)), ("far-right corner", (X_R, Y_FAR)),
]
# Lines in the camera's own view: (name, axis, value) meaning "axis = value" (line centre)
LINES = [
    ("near end line", "y", Y_NEAR), ("near attack line", "y", Y_ATT_NEAR), ("centre line (under net)", "y", 9),
    ("left sideline", "x", X_L), ("right sideline", "x", X_R),
    ("far attack line", "y", Y_ATT_FAR), ("far end line", "y", Y_FAR),
]
COURT_LINES = [((0, 0), (9, 0)), ((0, 6), (9, 6)), ((0, 9), (9, 9)), ((0, 12), (9, 12)),
               ((0, 18), (9, 18)), ((0, 0), (0, 18)), ((9, 0), (9, 18))]
MAX_DISPLAY = 1400
ZOOM, ZOOM_HALF = 5, 30
SHIFT_OK_PX = 2.0
ROBUST_PX = 3.0          # residuals beyond ~this many px are down-weighted (likely misclicks)
UNCONSTRAINED_CM = 300   # a held-out error beyond this means the fit had nothing to go on there


# ---------------------------------------------------------------- court geometry
def to_shared(xy, end):
    x, y = xy
    return (x, y) if end == "near" else (9 - x, 18 - y)


def line_to_shared(axis, value, end):
    if end == "near":
        return axis, value
    return axis, (9 - value) if axis == "x" else (18 - value)


class CourtCalibration:
    """Court (metres) <-> image (pixels) for one camera: homography + optional radial distortion."""

    def __init__(self, H_c2i, k1=0.0, centre=(0.0, 0.0), norm=1.0):
        self.H_c2i = np.asarray(H_c2i, float)
        self.H_i2c = np.linalg.inv(self.H_c2i)
        self.k1, self.c, self.norm = float(k1), np.asarray(centre, float), float(norm)

    # radial distortion about the image centre: p_d = c + (p_u - c)(1 + k1 r^2)
    def distort(self, pu):
        d = np.asarray(pu, float) - self.c
        r2 = np.sum(d ** 2, axis=-1, keepdims=True) / self.norm ** 2
        return self.c + d * (1 + self.k1 * r2)

    def undistort(self, pd):
        pd = np.asarray(pd, float)
        pu = pd.copy()
        for _ in range(8):
            r2 = np.sum((pu - self.c) ** 2, axis=-1, keepdims=True) / self.norm ** 2
            pu = self.c + (pd - self.c) / (1 + self.k1 * r2)
        return pu

    def court_to_image(self, xy):
        xy = np.asarray(xy, float).reshape(-1, 2)
        h = np.c_[xy, np.ones(len(xy))] @ self.H_c2i.T
        return self.distort(h[:, :2] / h[:, 2:3])

    def image_to_court(self, uv):
        pu = self.undistort(np.asarray(uv, float).reshape(-1, 2))
        h = np.c_[pu, np.ones(len(pu))] @ self.H_i2c.T
        return h[:, :2] / h[:, 2:3]

    def to_json(self):
        return dict(H_court_to_image=self.H_c2i.tolist(), H_image_to_court=self.H_i2c.tolist(),
                    distortion=dict(k1=self.k1, cx=float(self.c[0]), cy=float(self.c[1]), norm=self.norm),
                    note="image_to_court: undistort the pixel, then apply H_image_to_court")


def load_calibration(path):
    d = json.loads(Path(path).read_text())
    dist = d.get("distortion", {"k1": 0.0, "cx": 0.0, "cy": 0.0, "norm": 1.0})
    return CourtCalibration(d["H_court_to_image"], dist["k1"], (dist["cx"], dist["cy"]), dist["norm"])


# ---------------------------------------------------------------- fitting
def _residuals(params, pts_img, pts_court, line_img, line_def, centre, norm, use_k1):
    H = np.append(params[:8], 1.0).reshape(3, 3)
    cal = CourtCalibration(H, params[8] if use_k1 else 0.0, centre, norm)
    res = []
    if len(pts_img):
        res.append((cal.court_to_image(pts_court) - pts_img).ravel())
    if len(line_img):
        pu = cal.undistort(line_img)
        hp = np.c_[pu, np.ones(len(pu))]
        # Court line a*x + b*y + c = 0  ->  image line  l_img = H^-T l_court
        Hit = np.linalg.inv(H).T
        d = []
        for (axis, v), p in zip(line_def, hp):
            lc = np.array([1.0, 0.0, -v]) if axis == "x" else np.array([0.0, 1.0, -v])
            li = Hit @ lc
            d.append((li @ p) / np.hypot(li[0], li[1]))
        res.append(np.array(d))
    return np.concatenate(res)


def fit_model(pts_img, pts_court, line_img, line_def, size, use_k1):
    pts_img, pts_court = np.asarray(pts_img, float).reshape(-1, 2), np.asarray(pts_court, float).reshape(-1, 2)
    line_img = np.asarray(line_img, float).reshape(-1, 2)
    H0, _ = cv2.findHomography(np.float32(pts_court), np.float32(pts_img), 0)
    H0 = H0 / H0[2, 2]
    x0 = np.r_[H0.ravel()[:8], 0.0] if use_k1 else H0.ravel()[:8]
    centre, norm = (size[0] / 2, size[1] / 2), size[0] / 2
    n_res = 2 * len(pts_img) + len(line_img)
    if n_res <= len(x0):          # exactly determined: nothing to refine
        return CourtCalibration(H0, 0.0, centre, norm)
    sol = least_squares(_residuals, x0, loss="soft_l1", f_scale=ROBUST_PX, x_scale="jac",
                        args=(pts_img, pts_court, line_img, line_def, centre, norm, use_k1))
    H = np.append(sol.x[:8], 1.0).reshape(3, 3)
    return CourtCalibration(H, sol.x[8] if use_k1 else 0.0, centre, norm)


def evaluate(pts_img, pts_court, line_img, line_def, size, use_k1):
    cal = fit_model(pts_img, pts_court, line_img, line_def, size, use_k1)
    pts_img, pts_court = np.asarray(pts_img, float), np.asarray(pts_court, float)
    err_px = np.linalg.norm(cal.court_to_image(pts_court) - pts_img, axis=1)
    err_cm = np.linalg.norm(cal.image_to_court(pts_img) - pts_court, axis=1) * 100
    line_px = np.abs(_residuals(np.r_[(cal.H_c2i / cal.H_c2i[2, 2]).ravel()[:8], cal.k1], np.empty((0, 2)),
                                np.empty((0, 2)), line_img, line_def, cal.c, cal.norm, True)) if len(line_img) else np.array([])
    loo = []
    n_free = 9 if use_k1 else 8
    for k in range(len(pts_img)):
        keep = np.arange(len(pts_img)) != k
        if 2 * keep.sum() + len(line_img) <= n_free or keep.sum() < 4:
            loo = []
            break
        ck = fit_model(pts_img[keep], pts_court[keep], line_img, line_def, size, use_k1)
        loo.append(float(np.linalg.norm(ck.image_to_court(pts_img[k:k + 1])[0] - pts_court[k]) * 100))
    # Leave-one-LINE-out: refit without a whole line, then measure how far (cm, on the court)
    # its clicked points land from where that line really is. Clicks along a line are more
    # precise than at intersections, so this mostly measures the model, not the clicking.
    loo_lines = {}
    defs = [tuple(d) for d in line_def]
    for key in sorted(set(defs)):
        sel = np.array([d == key for d in defs])
        if sel.sum() < 2:
            continue
        rest = [d for d, s_ in zip(defs, sel) if not s_]
        if 2 * len(pts_img) + len(rest) <= n_free:
            continue
        ck = fit_model(pts_img, pts_court, line_img[~sel], rest, size, use_k1)
        xy = ck.image_to_court(line_img[sel])
        axis, v = key
        d = np.abs(xy[:, 0 if axis == "x" else 1] - v) * 100
        loo_lines[key] = d.tolist()
    return cal, err_px, err_cm, line_px, loo, loo_lines


# ---------------------------------------------------------------- video helpers
def ffmpeg_exe():
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        raise SystemExit("ffmpeg not found. Run:  pip install imageio-ffmpeg")


def read_frame(path, t):
    """Exact frame at time t (s), via ffmpeg (OpenCV seeking is unreliable on phone video)."""
    if Path(path).suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"):
        return cv2.imread(str(path))
    cmd = [ffmpeg_exe(), "-v", "error", "-ss", f"{t:.3f}", "-i", str(path), "-frames:v", "1",
           "-f", "image2pipe", "-vcodec", "png", "-"]
    data = subprocess.run(cmd, capture_output=True).stdout
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR) if data else None


def duration(path):
    cap = cv2.VideoCapture(str(path))
    n, fps = cap.get(cv2.CAP_PROP_FRAME_COUNT), cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return n / fps if fps else 0.0


# ---------------------------------------------------------------- clicking UI
def click_all(img, min_points=4):
    """Stage 1: named intersections. Stage 2: points along named lines. Full-res pixels."""
    scale = min(1.0, MAX_DISPLAY / img.shape[1])
    disp = cv2.resize(img, None, fx=scale, fy=scale)
    H_img, W_img = img.shape[:2]
    clicked, line_pts = [], []          # [(name, (u,v))], [(line_index, (u,v))]
    i, li, stage, pending, mouse = 0, 0, 1, [], [0, 0]

    def on_mouse(event, x, y, flags, param):
        mouse[0], mouse[1] = x, y
        if event == cv2.EVENT_LBUTTONDOWN:
            pending.append((x / scale, y / scale))

    win = "Calibrate court (see magnifier, top-right)"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)
    while True:
        if pending:
            if stage == 1 and i < len(POINTS):
                clicked.append((POINTS[i][0], pending[-1]))
                i += 1
            elif stage == 2:
                line_pts.append((li, pending[-1]))
        pending.clear()

        view = disp.copy()
        for name, (u, v) in clicked:
            p = (int(round(u * scale)), int(round(v * scale)))
            cv2.circle(view, p, 4, (0, 0, 255), 1)
            idx = [n for n, _ in POINTS].index(name) + 1
            cv2.putText(view, str(idx), (p[0] + 6, p[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        for k, (u, v) in line_pts:
            p = (int(round(u * scale)), int(round(v * scale)))
            cv2.circle(view, p, 3, (255, 255, 0) if k == li else (180, 120, 0), -1)

        # Magnifier from the full-resolution image
        x0 = int(round(mouse[0] / scale)) - ZOOM_HALF
        y0 = int(round(mouse[1] / scale)) - ZOOM_HALF
        patch = np.zeros((2 * ZOOM_HALF + 1, 2 * ZOOM_HALF + 1, 3), np.uint8)
        xa, ya = max(0, x0), max(0, y0)
        xb, yb = min(W_img, x0 + 2 * ZOOM_HALF + 1), min(H_img, y0 + 2 * ZOOM_HALF + 1)
        if xb > xa and yb > ya:
            patch[ya - y0:yb - y0, xa - x0:xb - x0] = img[ya:yb, xa:xb]
        for _, (u, v) in clicked + line_pts:
            pu, pv = int(round(u)) - x0, int(round(v)) - y0
            if 0 <= pu < patch.shape[1] and 0 <= pv < patch.shape[0]:
                patch[pv, pu] = (0, 0, 255)
        mag = cv2.resize(patch, None, fx=ZOOM, fy=ZOOM, interpolation=cv2.INTER_NEAREST)
        c = mag.shape[0] // 2
        cv2.line(mag, (c, 0), (c, mag.shape[0]), (0, 255, 255), 1)
        cv2.line(mag, (0, c), (mag.shape[1], c), (0, 255, 255), 1)
        cv2.rectangle(mag, (0, 0), (mag.shape[1] - 1, mag.shape[0] - 1), (255, 255, 255), 2)
        mh, mw = mag.shape[:2]
        if view.shape[0] > mh + 40 and view.shape[1] > mw:
            view[40:40 + mh, view.shape[1] - mw:] = mag

        if stage == 1:
            msg = (f"POINT {i + 1}/{len(POINTS)}: {POINTS[i][0]}" if i < len(POINTS) else "All points done")
            msg += f"  [{len(clicked)}]   s skip  u undo  ijkl nudge  Enter -> line points"
        else:
            n_here = sum(k == li for k, _ in line_pts)
            msg = (f"LINE {li + 1}/{len(LINES)}: click points along the {LINES[li][0]}  [{n_here}]"
                   "   n next  b back  u undo  ijkl nudge  Enter finish")
        cv2.rectangle(view, (0, 0), (view.shape[1], 34), (0, 0, 0), -1)
        cv2.putText(view, msg, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)
        cv2.imshow(win, view)

        key = cv2.waitKey(20) & 0xFF
        nudge = {ord("i"): (0, -1), ord("k"): (0, 1), ord("j"): (-1, 0), ord("l"): (1, 0)}
        if key in nudge:
            du, dv = nudge[key]
            if stage == 2 and line_pts:
                k, (u, v) = line_pts[-1]
                line_pts[-1] = (k, (u + du, v + dv))
            elif stage == 1 and clicked:
                name, (u, v) = clicked[-1]
                clicked[-1] = (name, (u + du, v + dv))
        elif key == ord("q"):
            cv2.destroyAllWindows()
            raise SystemExit("Quit without saving")
        elif stage == 1:
            if key == ord("s") and i < len(POINTS):
                i += 1
            elif key == ord("u") and i > 0:
                i -= 1
                if clicked and clicked[-1][0] == POINTS[i][0]:
                    clicked.pop()
            elif key == 13 and len(clicked) >= min_points:
                stage = 2
        else:
            if key == ord("n"):
                li = min(li + 1, len(LINES) - 1)
            elif key == ord("b"):
                li = max(li - 1, 0)
            elif key == ord("u") and line_pts:
                line_pts.pop()
            elif key == 13:
                break
    cv2.destroyAllWindows()
    return clicked, line_pts


# ---------------------------------------------------------------- drawing / checks
def draw_court(img, cal, points=(), line_pts=(), colour=(255, 0, 255)):
    out = img.copy()
    t = max(2, img.shape[1] // 800)
    for a, b in COURT_LINES:
        s = np.linspace(0, 1, 60)[:, None]
        pts = cal.court_to_image(np.asarray(a) * (1 - s) + np.asarray(b) * s)
        cv2.polylines(out, [np.int32(np.round(pts))], False, colour, t, cv2.LINE_AA)
    for _, (u, v) in points:
        cv2.circle(out, (int(u), int(v)), 4 * t, (0, 255, 0), t)
    for _, (u, v) in line_pts:
        cv2.circle(out, (int(u), int(v)), 2 * t, (255, 255, 0), -1)
    return out


def static_shift(ref, img, top_frac=0.35):
    """Shift (px) of the static top region (walls/ceiling) between two frames + match strength."""
    h = int(ref.shape[0] * top_frac)
    prep = lambda f: np.float32(cv2.equalizeHist(cv2.cvtColor(f[:h], cv2.COLOR_BGR2GRAY)))
    a, b = prep(ref), prep(img)
    win = cv2.createHanningWindow(a.shape[::-1], cv2.CV_32F)
    (dx, dy), response = cv2.phaseCorrelate(a, b, win)
    return float(np.hypot(dx, dy)), (float(dx), float(dy)), float(response)


# ---------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser()
    p.add_argument("source", help="video (or a still image)")
    p.add_argument("--camera", required=True, help="camera name, e.g. A or B")
    p.add_argument("--end", choices=["near", "far"], required=True)
    p.add_argument("--time", type=float, default=30.0, help="time (s) of the frame to click on")
    p.add_argument("--distortion", choices=["auto", "on", "off"], default="auto")
    p.add_argument("--check", type=int, default=6, help="frames in the check sheet (0 = no check)")
    p.add_argument("--timeline", type=int, default=40, help="frames sampled for the tripod timeline")
    p.add_argument("--valid", nargs=2, type=float, metavar=("FROM_S", "TO_S"),
                   help="record that this calibration is valid only for this time range")
    p.add_argument("--points-json", help="reuse clicks from a saved calibration instead of clicking")
    p.add_argument("--drop", nargs="*", default=[],
                   help="leave out these clicked points (exact names) when refitting, e.g. a misclick")
    p.add_argument("--auto-lines", help="JSON from detect_court_lines.py for the same frame size: its "
                   "line samples and near corners are added to your clicks")
    args = p.parse_args()

    frame = read_frame(args.source, args.time)
    if frame is None:
        raise SystemExit(f"Could not read a frame at {args.time} s from {args.source}")
    size = (frame.shape[1], frame.shape[0])

    if args.points_json:
        saved = json.loads(Path(args.points_json).read_text())
        clicked = [(q["name"], tuple(q["image"])) for q in saved["points"]]
        names = [n for n, _, _ in LINES]
        line_pts = [(names.index(q["line"]), tuple(q["image"])) for q in saved.get("line_points", [])]
    else:
        auto_n = 2 if args.auto_lines else 0
        clicked, line_pts = click_all(frame, min_points=max(1, 4 - auto_n))
    if args.auto_lines:
        auto = json.loads(Path(args.auto_lines).read_text())
        if list(auto["image_size"]) != [frame.shape[1], frame.shape[0]]:
            raise SystemExit(f"--auto-lines was made on a {auto['image_size']} image, but this frame is "
                             f"{frame.shape[1]}x{frame.shape[0]}: rerun detect_court_lines.py on the same video")
        names = [n for n, _, _ in LINES]
        have = {n for n, _ in clicked}
        added_l = added_c = 0
        for lname, L in auto["lines"].items():
            if lname in names:
                for q in L["samples"]:
                    line_pts.append((names.index(lname), tuple(q)))
                    added_l += 1
        for cname, c in auto["corners"].items():
            if cname not in have:                  # your own click wins if you made one
                clicked.append((cname, tuple(c)))
                added_c += 1
        print(f"Added from {args.auto_lines}: {added_l} line points, {added_c} corners")

    if args.drop:
        unknown = [d for d in args.drop if d not in {n for n, _ in clicked}]
        if unknown:
            raise SystemExit(f"--drop: no clicked point named {unknown}. Names: {[n for n, _ in clicked]}")
        clicked = [(n, uv) for n, uv in clicked if n not in args.drop]
        print(f"Dropped: {', '.join(args.drop)}")
    if len(clicked) < 4:
        raise SystemExit(f"Only {len(clicked)} intersection points: at least 4 are needed for a homography.")
    lookup = dict(POINTS)
    pts_img = np.array([uv for _, uv in clicked], float)
    pts_court = np.array([to_shared(lookup[n], args.end) for n, _ in clicked], float)
    line_img = np.array([uv for _, uv in line_pts], float).reshape(-1, 2)
    line_def = [line_to_shared(LINES[k][1], LINES[k][2], args.end) for k, _ in line_pts]

    # Fit without and (if possible) with lens distortion; keep the better by leave-one-out
    options = [False] if args.distortion == "off" or len(line_pts) < 6 else (
        [True] if args.distortion == "on" else [False, True])
    fits = {k1: evaluate(pts_img, pts_court, line_img, line_def, size, k1) for k1 in options}

    def score(f):          # held-out-line error if available (fairer), else intersection LOO
        held = [d for v in f[5].values() for d in v]
        return np.median(held) if held else (np.median(f[4]) if f[4] else np.inf)

    if len(fits) == 2:
        sc = {k1: score(f) for k1, f in fits.items()}
        sc = {k: (v if np.isfinite(v) and v < UNCONSTRAINED_CM else np.inf) for k, v in sc.items()}
        use_k1 = sc[True] < 0.9 * sc[False]
        print(f"\nHeld-out error: homography only {sc[False]:.1f} cm, with lens distortion {sc[True]:.1f} cm"
              f" -> using {'lens distortion' if use_k1 else 'homography only'}")
    else:
        use_k1 = options[0]
    cal, err_px, err_cm, line_px, loo, loo_lines = fits[use_k1]

    print(f"\nCamera {args.camera} ({args.end} end): {len(clicked)} intersections, "
          f"{len(line_pts)} line points" + (f", k1 = {cal.k1:+.4f}" if use_k1 else ""))
    for k, (name, _) in enumerate(clicked):
        extra = ""
        if loo:
            extra = (f"   leave-one-out {loo[k]:5.1f} cm" if loo[k] < UNCONSTRAINED_CM else
                     "   leave-one-out: unconstrained without this point (add line points near it)")
        print(f"  {name:42s} fit error {err_px[k]:4.1f} px {err_cm[k]:5.1f} cm{extra}")
    if len(line_px):
        print(f"  line points: median distance from their line {np.median(line_px):.1f} px, "
              f"max {line_px.max():.1f} px")
        for k in sorted({k for k, _ in line_pts}):
            sel = [j for j, (kk, _) in enumerate(line_pts) if kk == k]
            worst = line_px[sel].max()
            flag = "   <- check these clicks" if worst > 3 * ROBUST_PX else ""
            print(f"    {LINES[k][0]:28s} {len(sel)} points, worst {worst:4.1f} px{flag}")
    print(f"  RMS intersection fit error: {np.sqrt(np.mean(err_px ** 2)):.1f} px")
    if loo_lines:
        held = [d for v in loo_lines.values() for d in v]
        print("  held-out lines (each line left out of the fit, then checked):")
        names = {line_to_shared(a, v, args.end): n for n, a, v in LINES}
        for key, d in loo_lines.items():
            print(f"    {names[key]:28s} median {np.median(d):5.1f} cm, max {max(d):5.1f} cm")
        print(f"  ACCURACY ESTIMATE: median {np.median(held):.1f} cm over {len(held)} held-out line points")
    ok_loo = [x for x in loo if x < UNCONSTRAINED_CM]
    if ok_loo:
        print(f"  intersection leave-one-out: median {np.median(ok_loo):.1f} cm, max {max(ok_loo):.1f} cm"
              "  (includes the click error of the left-out point itself, so pessimistic)")
    if not loo and not loo_lines:
        print("  not enough clicks for an accuracy estimate: add line points")

    out_json = Path("data/annotations") / f"calib_{args.camera}.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_dir = Path("results/calib")
    out_dir.mkdir(parents=True, exist_ok=True)
    result = dict(camera=args.camera, end=args.end, source=Path(args.source).name,
                  frame_time_s=args.time, image_size=list(size),
                  court_frame="x 0-9 m across, y 0-18 m along; y=18 end = camera with --end near",
                  **cal.to_json(),
                  points=[dict(name=n, image=[round(u, 2), round(v, 2)], court=list(map(float, c)))
                          for (n, (u, v)), c in zip(clicked, pts_court)],
                  line_points=[dict(line=LINES[k][0], image=[round(u, 2), round(v, 2)])
                               for k, (u, v) in line_pts],
                  rms_px=float(np.sqrt(np.mean(err_px ** 2))), leave_one_out_cm=loo,
                  held_out_line_cm={f"{a}={v}": d for (a, v), d in loo_lines.items()})

    cv2.imwrite(str(out_dir / f"calib_{args.camera}_overlay.jpg"), draw_court(frame, cal, clicked, line_pts))

    # ---- Tripod timeline + check sheet ----
    if args.check and Path(args.source).suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp"):
        dur = duration(args.source)
        times = np.linspace(0.5, max(0.5, dur - 0.5), max(args.timeline, args.check))
        sheet_idx = set(np.round(np.linspace(0, len(times) - 1, args.check)).astype(int).tolist())
        tiles, shifts = [], []
        print(f"\nTripod timeline ({len(times)} frames):")
        for k, t in enumerate(times):
            f = read_frame(args.source, float(t))
            if f is None:
                continue
            s_px, (dx, dy), resp = static_shift(frame, f)
            shifts.append(dict(time_s=round(float(t), 1), shift_px=round(s_px, 2),
                               dx=round(dx, 2), dy=round(dy, 2), response=round(resp, 3)))
            if k in sheet_idx:
                tile = draw_court(f, cal)
                label = f"t={t:6.1f}s  shift {s_px:4.1f}px"
                colour = (0, 255, 0) if s_px <= SHIFT_OK_PX else (0, 0, 255)
                kk = tile.shape[1] / 1280
                cv2.putText(tile, label, (20, int(50 * kk)), cv2.FONT_HERSHEY_SIMPLEX, 1.2 * kk, (0, 0, 0), int(6 * kk))
                cv2.putText(tile, label, (20, int(50 * kk)), cv2.FONT_HERSHEY_SIMPLEX, 1.2 * kk, colour, int(3 * kk))
                tiles.append(cv2.resize(tile, (640, int(640 * tile.shape[0] / tile.shape[1]))))
        line = "".join("." if x["shift_px"] <= SHIFT_OK_PX else ("?" if x["response"] < 0.05 else "X")
                       for x in shifts)
        print(f"  {line}")
        print(f"  each character = {dur / max(1, len(shifts)):.0f} s;  . stable   X moved >{SHIFT_OK_PX:.0f} px"
              "   ? low-confidence match (lighting change / people in view)")
        stable = [x["time_s"] for x in shifts if x["shift_px"] <= SHIFT_OK_PX]
        if stable and len(stable) < len(shifts):
            print(f"  this calibration matches the video from about {min(stable)} s to {max(stable)} s")
        if tiles:
            while len(tiles) % 2:
                tiles.append(np.zeros_like(tiles[0]))
            sheet = np.vstack([np.hstack(tiles[i:i + 2]) for i in range(0, len(tiles), 2)])
            cv2.imwrite(str(out_dir / f"calib_{args.camera}_check.jpg"), sheet)
        result["tripod_check"] = shifts
    if args.valid:
        result["valid_from_s"], result["valid_to_s"] = args.valid

    out_json.write_text(json.dumps(result, indent=1))
    print(f"\nSaved {out_json}")
    print(f"Saved {out_dir}/calib_{args.camera}_overlay.jpg" + (" and _check.jpg" if args.check else ""))


if __name__ == "__main__":
    main()
