"""Ball candidates and 2D tracklets in one fixed camera (classical: motion + shape + colour).

Used by ball_two_cameras.py; can also be run alone to see what it finds in one camera:
    python scripts/ball_detect.py data/game2_A.mp4 --calib data/annotations/calib_A.json --start 30 --duration 5
Output: results/ball/<video>_candidates_from<start>.mp4
    grey boxes  = moving blobs rejected (wrong size or shape)
    thin boxes  = ball candidates (colour = how yellow/blue they are, white = not)
    lines       = tracklets (candidates linked over time), thicker = more ball-like

Per frame:
  1. Three-frame difference: a pixel is moving at frame k if it differs from both frame k-1
     and frame k+1 (largest change over the B, G, R channels). The tripods are fixed, so the
     background cancels. Using both neighbours keeps only the object's position at frame k
     (a two-frame difference also lights up where it was before). The threshold adapts to
     each frame (median + 4 x robust spread), which absorbs light flicker and video noise.
  2. Morphological opening (removes noise specks), closing (joins a motion-blurred ball), and
     connected components.
  3. Shape and size: the ball's image radius is f x 0.105 m / distance, so from the camera
     model (camera_3d.py) we know the smallest (far end of the court) and largest (2 m from the
     lens) possible ball size. Blobs outside that range, too elongated (blur streaks up to 5:1
     allowed) or too hollow are rejected.
  4. Colour score: fraction of the blob's pixels that are ball yellow or ball blue (HSV). Where
     the blob is clearly the ball's colour, a precise centre is also taken from the yellow pixels
     inside it (blur and the ball's shadow bias the blob centre); used for bounce contact points.
  5. Optional person boxes (YOLO, every few frames): a candidate inside a player is flagged.
     Most moving blobs are arms, heads and legs; the ball is usually away from bodies in flight.
  6. Isolation: share of the area around the blob (a box 3x its size) that is NOT moving.
     A ball in flight is alone; arms, heads and legs move next to the rest of the body.
Tracklets: candidates are linked frame to frame by predicted position (constant velocity),
Hungarian assignment, and a size-ratio check. The gate is measured in ball radii: a ball moves
at most ~25 m/s (8 radii per frame), and its image radius and image motion both scale with
1 / distance, so the same gate works near and far from the camera.
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).parent))

BALL_R = 0.105
MIN_DIFF = 14            # minimum colour change (0-255) counted as motion
MAX_ELONG = 5.0          # longest / shortest side of a blob (motion blur)
MIN_FILL = 0.25          # blob area / bounding-box area
GATE_R, GATE_V = 2.5, 0.35  # linking gate: GATE_R ball radii + GATE_V x speed (+ 8 px) per frame
                            # (wider gates mislinked the ball to players on real footage)
YELLOW = ((18, 42), 60, 70)     # hue range, min saturation, min value
BLUE = ((95, 130), 70, 50)


def radius_range(cam, size):
    """Smallest and largest image radius (px) of the ball anywhere it can be in play."""
    w, h = size
    f = cam.K[0, 0]
    grid = np.array([[x, y, z] for x in (-3, 4.5, 12) for y in (-6, 9, 24) for z in (0, 3, 8)], float)
    depth = (grid @ cam.R.T + cam.t)[:, 2]
    dmax = depth.max()
    return f * BALL_R / dmax, f * BALL_R / 2.0


class Candidates:
    def __init__(self, cam, size, persons_every=0, person_model="yolo11n.pt"):
        self.w, self.h = size
        self.s = self.w / 1920
        self.rmin, self.rmax = radius_range(cam, size)
        self.amin = max(4.0, 0.5 * np.pi * self.rmin ** 2)
        self.amax = np.pi * (1.3 * self.rmax) ** 2 * 3
        self.k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self.persons_every = persons_every
        self.person_boxes = np.zeros((0, 4))
        self.yolo = None
        if persons_every:
            from ultralytics import YOLO
            self.yolo = YOLO(person_model)

    @staticmethod
    def _threshold(d):
        sub = d[::8, ::8].astype(np.float32)
        med = float(np.median(sub))
        mad = float(np.median(np.abs(sub - med))) * 1.4826
        return med + max(MIN_DIFF, 4 * mad)

    def motion_mask(self, prev, cur, nxt):
        d1 = cv2.absdiff(cur, prev).max(axis=2)
        d2 = cv2.absdiff(nxt, cur).max(axis=2)
        m = ((d1 > self._threshold(d1)) & (d2 > self._threshold(d2))).astype(np.uint8) * 255
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, self.k_open)
        return cv2.morphologyEx(m, cv2.MORPH_CLOSE, self.k_close)

    def __call__(self, prev, cur, nxt, k):
        """Ball candidates at frame k (cur). Returns (candidates, rejected boxes)."""
        if self.yolo is not None and k % self.persons_every == 0:
            r = self.yolo.predict(cur, classes=[0], conf=0.3, imgsz=1280, verbose=False)[0]
            self.person_boxes = r.boxes.xyxy.cpu().numpy()
        m = self.motion_mask(prev, cur, nxt)
        n, lab, stats, cent = cv2.connectedComponentsWithStats(m, connectivity=8)
        integ = cv2.integral((m > 0).astype(np.uint8))
        out, rejected = [], []
        hsv = None
        for i in range(1, n):
            x, y, bw, bh, area = stats[i]
            if area < self.amin:
                continue                                       # noise: not even drawn
            elong = max(bw, bh) / max(1, min(bw, bh))
            fill = area / float(bw * bh)
            r = np.sqrt(area / np.pi) if elong < 1.8 else min(bw, bh) / 2.0
            if (area > self.amax or elong > MAX_ELONG or fill < MIN_FILL
                    or not 0.5 * self.rmin <= r <= 1.3 * self.rmax):
                rejected.append((x, y, bw, bh))
                continue
            if hsv is None:
                hsv = cv2.cvtColor(cur, cv2.COLOR_BGR2HSV)
            patch = hsv[y:y + bh, x:x + bw]
            sel = lab[y:y + bh, x:x + bw] == i
            hh, ss, vv = patch[..., 0][sel], patch[..., 1][sel], patch[..., 2][sel]
            yel = (hh >= YELLOW[0][0]) & (hh <= YELLOW[0][1]) & (ss >= YELLOW[1]) & (vv >= YELLOW[2])
            blu = (hh >= BLUE[0][0]) & (hh <= BLUE[0][1]) & (ss >= BLUE[1]) & (vv >= BLUE[2])
            u, v = cent[i]
            # A second, more precise centre (cu, cv) from the ball's own colour: the motion blob of
            # a fast or bouncing ball also contains blur and its shadow, which pulls its centroid
            # down (8 px close to a camera = ~16 cm on the floor). Yellow pixels inside the
            # (slightly grown) moving blob are the ball at this frame; still court lines are
            # excluded because they do not move. Tracking uses the motion centre (more robust on
            # real footage); bounce contact points use the colour centre.
            ph = patch[..., 0]
            yel_full = ((ph >= YELLOW[0][0]) & (ph <= YELLOW[0][1]) & (patch[..., 1] >= YELLOW[1])
                        & (patch[..., 2] >= YELLOW[2]))
            grown = cv2.dilate(sel.astype(np.uint8), self.k_open) > 0
            ys, xs = np.nonzero(yel_full & grown)
            cu, cv_ = (x + xs.mean(), y + ys.mean()) if len(xs) >= max(15, 0.3 * area) else (u, v)
            inside = bool(len(self.person_boxes) and np.any(
                (self.person_boxes[:, 0] <= u) & (u <= self.person_boxes[:, 2]) &
                (self.person_boxes[:, 1] <= v) & (v <= self.person_boxes[:, 3])))
            e = max(bw, bh)
            xa, ya = max(0, x - e), max(0, y - e)
            xb, yb = min(self.w, x + bw + e), min(self.h, y + bh + e)
            moving = integ[yb, xb] - integ[ya, xb] - integ[yb, xa] + integ[ya, xa] - area
            iso = 1.0 - moving / max(1.0, (xb - xa) * (yb - ya) - area)
            out.append(dict(k=k, u=float(u), v=float(v), cu=float(cu), cv=float(cv_), r=float(r), area=int(area),
                            elong=float(elong),
                            yellow=float(np.mean(yel)), colour=float(np.mean(yel | blu)), person=inside,
                            iso=float(iso),
                            box=(int(x), int(y), int(bw), int(bh))))
        return out, rejected


def link_tracklets(frames, scale, max_gap=2, step=1):
    """frames: list (one per frame) of candidate lists. Returns tracklets: lists of candidates."""
    done, active = [], []
    for cands in frames:
        if not cands:
            k = None
        else:
            k = cands[0]["k"]
        if active and cands:
            cost = np.full((len(active), len(cands)), 1e9)
            for i, tr in enumerate(active):
                last = tr["pts"][-1]
                gap = k - last["k"]
                pu, pv = last["u"] + tr["vu"] * gap, last["v"] + tr["vv"] * gap
                speed = np.hypot(tr["vu"], tr["vv"])
                # A ball moves at most ~25 m/s = 0.83 m per frame = 8 ball radii, and its image
                # radius scales with distance exactly like its image motion, so gates are in radii.
                if len(tr["pts"]) == 1:
                    gate = (8 * last["r"] + 10 * scale) * gap / step
                else:
                    gate = (GATE_R * last["r"] + GATE_V * speed * step + 8 * scale) * gap / step
                for j, c in enumerate(cands):
                    d = np.hypot(c["u"] - pu, c["v"] - pv)
                    ratio = c["r"] / max(last["r"], 1e-3)
                    if d <= gate and 0.5 <= ratio <= 2.0:
                        cost[i, j] = d
            rows, cols = linear_sum_assignment(cost)
            used = set()
            for i, j in zip(rows, cols):
                if cost[i, j] >= 1e9:
                    continue
                tr, c = active[i], cands[j]
                last = tr["pts"][-1]
                gap = c["k"] - last["k"]
                vu, vv = (c["u"] - last["u"]) / gap, (c["v"] - last["v"]) / gap
                if len(tr["pts"]) == 1:
                    tr["vu"], tr["vv"] = vu, vv
                else:
                    tr["vu"], tr["vv"] = 0.3 * tr["vu"] + 0.7 * vu, 0.3 * tr["vv"] + 0.7 * vv
                tr["pts"].append(c)
                used.add(j)
            new = [c for j, c in enumerate(cands) if j not in used]
        else:
            new = list(cands)
        for c in new:
            active.append(dict(pts=[c], vu=0.0, vv=0.0))
        if k is not None:
            still = []
            for tr in active:
                (still if k - tr["pts"][-1]["k"] <= max_gap * step else done).append(tr)
            active = still
    done += active
    return [tr["pts"] for tr in done]


def tracklet_features(pts, scale, step=1):
    n = len(pts)
    uv = np.array([[p["u"], p["v"]] for p in pts])
    ks = np.array([p["k"] for p in pts])
    # image speed in px per 1/30 s (at 1920 px width), so 30 and 60 fps tracklets compare
    speed = (np.linalg.norm(np.diff(uv, axis=0), axis=1) / np.diff(ks)).mean() * step / scale if n > 1 else 0.0
    return dict(n=n, speed=float(speed), colour=float(np.mean([p["colour"] for p in pts])),
                iso=float(np.mean([p["iso"] for p in pts])),
                person=float(np.mean([p["person"] for p in pts])), k0=int(ks[0]), k1=int(ks[-1]))


def score_2d(f):
    """How ball-like a tracklet is from one camera alone (used for display and pre-filtering)."""
    return f["n"] * (0.3 + f["colour"]) * (1 - 0.7 * f["person"]) * min(1.0, f["speed"] / 4.0)


def detect_video(video, cam, t0, t1, persons_every=3, keep_debug=False, label=""):
    """Candidates between t0 and t1 (the video's own clock).

    The frame difference always spans about 1/30 s, whatever the frame rate: at 60 fps, every
    second frame is processed and compared with the frames 2 before and 2 after. (Comparing
    neighbouring 60 fps frames, a slow ball overlaps itself and its blob comes out too small.)
    Returns fps, first frame index, list of candidate lists (one per processed frame), image
    size, debug data, and the frame step."""
    from collections import deque
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    step = max(1, int(round(fps / 30.0)))
    i0, i1 = max(step, int(np.floor(t0 * fps))), int(np.ceil(t1 * fps))
    for _ in range(i0 - step):
        if not cap.grab():
            raise SystemExit(f"{video} ends before {t0:.1f} s")
    buf = deque(maxlen=2 * step + 1)                 # frames i0-step ... i0+step
    for _ in range(2 * step):
        ok, f = cap.read()
        if not ok:
            raise SystemExit(f"Could not read {video} at {t0:.1f} s")
        buf.append(f)
    size = (buf[0].shape[1], buf[0].shape[0])
    det = Candidates(cam, size, persons_every * step if persons_every else 0)
    print(f"Camera {label}: {fps:.2f} fps (difference over {step} frame{'s' if step > 1 else ''}), "
          f"ball radius {det.rmin:.1f}-{det.rmax:.0f} px; frames {i0}-{i1} ...")
    frames, debug = [], []
    for k in range(i0, i1):
        ok, f = cap.read()
        if not ok:
            break
        buf.append(f)                                # buf[step] is frame k
        if (k - i0) % step:
            continue
        cands, rej = det(buf[0], buf[step], buf[2 * step], k)
        frames.append(cands)
        if keep_debug:
            debug.append((k, rej))
        if (k - i0 + 1) % int(5 * fps) < step:
            print(f"  {(k - i0 + 1) / fps:.0f} s")
    cap.release()
    print(f"  {np.mean([len(c) for c in frames]):.1f} ball candidates per processed frame")
    return fps, i0, frames, size, debug, step


def main():
    from camera_3d import load_camera
    from calibrate_camera import ffmpeg_exe
    import subprocess
    p = argparse.ArgumentParser()
    p.add_argument("video")
    p.add_argument("--calib", required=True)
    p.add_argument("--start", type=float, default=0.0)
    p.add_argument("--duration", type=float, default=5.0)
    p.add_argument("--persons-every", type=int, default=3, help="YOLO person boxes every N frames (0 = off)")
    p.add_argument("--width", type=int, default=1280)
    args = p.parse_args()
    cam, _, _ = load_camera(args.calib)
    fps, i0, frames, size, debug, step = detect_video(args.video, cam, args.start, args.start + args.duration,
                                                      args.persons_every, keep_debug=True)
    scale = size[0] / 1920
    tracks = [t for t in link_tracklets(frames, scale, step=step) if len(t) >= 3]
    feats = [tracklet_features(t, scale, step) for t in tracks]
    rej_by_k = dict(debug)
    cands_by_k = {c[0]["k"]: c for c in frames if c}
    best = max((score_2d(f) for f in feats), default=1.0)
    out = Path("results/ball")
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{Path(args.video).stem}_candidates_from{args.start:g}.mp4"
    cap = cv2.VideoCapture(args.video)
    for _ in range(i0):
        cap.grab()
    ff = None
    for j in range(len(frames) * step):
        ok, img = cap.read()
        if not ok:
            break
        k = i0 + j
        cands = cands_by_k.get(k - (k - i0) % step, [])
        for x, y, bw, bh in rej_by_k.get(k - (k - i0) % step, []):
            cv2.rectangle(img, (x, y), (x + bw, y + bh), (128, 128, 128), 1)
        for c in cands:
            x, y, bw, bh = c["box"]
            col = (0, int(255 * c["colour"]), 255) if c["colour"] > 0.1 else (255, 255, 255)
            cv2.rectangle(img, (x - 3, y - 3), (x + bw + 3, y + bh + 3), col, 1)
        for t, f in zip(tracks, feats):
            if f["k0"] <= k <= f["k1"] + 15:
                pts = np.int32([[p["u"], p["v"]] for p in t if p["k"] <= k])
                if len(pts) > 1:
                    th = 1 + int(4 * score_2d(f) / best)
                    cv2.polylines(img, [pts], False, (255, 255, 0) if th > 2 else (200, 120, 0), th, cv2.LINE_AA)
        cv2.putText(img, f"t={k / fps:.2f}s  candidates {len(cands)}", (12, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (255, 255, 255), 2)
        img = cv2.resize(img, (args.width, int(args.width * size[1] / size[0])))
        if ff is None:
            ff = subprocess.Popen([ffmpeg_exe(), "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
                                   "-s", f"{img.shape[1]}x{img.shape[0]}", "-r", f"{fps:.5f}", "-i", "-",
                                   "-c:v", "libx264", "-crf", "21", "-pix_fmt", "yuv420p", str(path)],
                                  stdin=subprocess.PIPE)
        ff.stdin.write(img.tobytes())
    ff.stdin.close()
    ff.wait()
    print(f"{len(tracks)} tracklets of 3+ frames. Saved {path}")


if __name__ == "__main__":
    main()
