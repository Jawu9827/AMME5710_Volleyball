"""Detect players in both synced cameras and plot their court positions on a top-down map.

Usage:
    python scripts/players_on_court.py data/game2_A.mp4 data/game2_B_sdr.mp4 \
        --calib-a data/annotations/calib_A.json --calib-b data/annotations/calib_B.json \
        --sync data/annotations/sync_game2_A_game2_B_sdr.json --start 30 --duration 30
    --detector hog     classical HOG + linear SVM people detector (OpenCV), for comparison
    --every 3          run detection on every 3rd frame (10 Hz); positions are held in between
    --show             watch while rendering (q quit, space pause)

Pipeline, per detection step:
  1. Person detection in each camera: YOLO11 (COCO 'person' class, pretrained, not fine-tuned)
     or OpenCV's HOG + SVM people detector.
  2. Foot point = bottom centre of each box. Only the feet touch the floor, so only they map
     correctly through the floor homography (a head would land metres behind the player).
     Boxes cut off by the image border are skipped (the feet are not visible).
  3. Map feet to court metres with each camera's calibration (lens distortion included).
     Keep points on or near the court (spectators and benches are outside it).
  4. Camera fusion: each camera is trusted for the half nearest to it, where it sees players
     at the highest resolution (A: y > 9, B: y < 9). Within 1 m of the net, where both see the
     player, matching detections (closer than MERGE_M) are averaged.
  5. Tracking in court coordinates: detections are matched to existing tracks by minimum total
     distance from their predicted position (Hungarian algorithm, gate GATE_M); new tracks need
     CONFIRM_HITS matches before they get an ID; duplicates within DEDUP_M are removed. Team = side of the net (volleyball teams stay on their side).

Outputs (results/players/):
    <A>_<B>_players_from<start>.mp4   camera views with boxes + top-down map with trails
    <A>_<B>_players_from<start>.csv   time_s, track_id, x_m, y_m, side, camera, confidence

Limitations: a jumping player's feet are in the air, so they map further away than they are;
heavily overlapping players can merge into one box; track IDs can swap when players cross.
"""
import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).parent))
from calibrate_camera import COURT_LINES, draw_court, ffmpeg_exe, load_calibration, read_frame  # noqa: E402

COURT_MARGIN = (2.0, 3.5)      # keep feet within this many metres outside the court (x, y)
NET_Y = 9.0
OVERLAP_M = 1.0                # both cameras accepted within this distance of the net
MERGE_M = 1.5                  # A and B detections closer than this near the net = same player
DEDUP_M = 0.4                  # two detections / tracks closer than this are the same player
GATE_M = 1.5                   # max distance from a track's predicted position to its next detection
MAX_SPEED = 8.0                # m/s, cap on a track's velocity estimate
CONFIRM_HITS = 3               # matches in a row before a track gets an ID and is drawn
LOST_S = 2.0                   # keep a confirmed track this long without a detection (occlusion)
SHOW_S = 0.4                   # only draw tracks seen within this time
SMOOTH = 0.5                   # weight of each new position (1 = no smoothing)
TRAIL_S = 2.0
FOOT_PATH = Path("data/annotations/foot_offset.json")   # written by eval_players.py
SIDE_COL = {"A": (0, 165, 255), "B": (80, 220, 80)}   # BGR: orange = camera A's side, green = B's


# ---------------------------------------------------------------- detectors
class YoloPeople:
    def __init__(self, model, conf, imgsz, device):
        from ultralytics import YOLO
        self.m, self.conf, self.imgsz, self.device = YOLO(model), conf, imgsz, device

    def __call__(self, img):
        r = self.m.predict(img, classes=[0], conf=self.conf, imgsz=self.imgsz,
                           device=self.device, verbose=False)[0]
        return r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()


class HogPeople:
    """OpenCV's HOG + linear SVM pedestrian detector (Dalal & Triggs), run at 960 px width."""
    def __init__(self, conf):
        if not hasattr(cv2, "HOGDescriptor"):
            raise RuntimeError("this OpenCV has no HOG detector (OpenCV 5 moved it to the contrib package). "
                               "Fix: pip uninstall -y opencv-python  then  pip install opencv-contrib-python")
        self.h = cv2.HOGDescriptor()
        self.h.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        self.conf = conf

    def __call__(self, img):
        s = min(1.0, 960 / img.shape[1])
        small = cv2.resize(img, None, fx=s, fy=s)
        rects, w = self.h.detectMultiScale(small, winStride=(8, 8), padding=(8, 8), scale=1.05)
        if len(rects) == 0:
            return np.zeros((0, 4)), np.zeros(0)
        w = np.asarray(w).ravel()
        keep = cv2.dnn.NMSBoxes([list(map(int, r)) for r in rects], w.tolist(), 0.0, 0.4)
        keep = np.asarray(keep).ravel() if len(keep) else np.zeros(0, int)
        boxes = np.array([[x, y, x + bw, y + bh] for x, y, bw, bh in rects[keep]], float) / s
        # HOG boxes have padding around the person: pull the bottom up ~5% to the feet
        boxes[:, 3] -= 0.05 * (boxes[:, 3] - boxes[:, 1])
        score = 1 / (1 + np.exp(-w[keep]))           # SVM margin -> 0..1 for display
        sel = score >= self.conf
        return boxes[sel], score[sel]


# ---------------------------------------------------------------- geometry / fusion
def load_foot_frac(model_name):
    """Foot offset (fraction of box height) fitted by eval_players.py for this detector, else 0."""
    try:
        return float(json.loads(FOOT_PATH.read_text()).get(Path(model_name).stem, 0.0))
    except (OSError, ValueError):
        return 0.0


def feet_to_court(boxes, conf, cal, img_shape, cam, end, foot_frac=0.0):
    """Foot points -> court metres; keep those on/near the court and in this camera's half.

    foot_frac raises the foot point above the box bottom by that fraction of the box height
    (the box bottom is slightly below where the feet touch the floor; see eval_players.py)."""
    h, w = img_shape[:2]
    out = []
    for (x1, y1, x2, y2), c in zip(boxes, conf):
        if y2 >= h - 3 or x1 <= 1 or x2 >= w - 2:        # cut off by the frame: feet unseen
            continue
        u, v = (x1 + x2) / 2, y2 - foot_frac * (y2 - y1)
        X, Y = cal.image_to_court([[u, v]])[0]
        mx, my = COURT_MARGIN
        if not (-mx <= X <= 9 + mx and -my <= Y <= 18 + my):
            continue
        near_half = Y >= NET_Y - OVERLAP_M if end == "near" else Y <= NET_Y + OVERLAP_M
        if near_half:
            out.append(dict(x=float(X), y=float(Y), conf=float(c), cam=cam, box=(x1, y1, x2, y2),
                            foot=(float(u), float(v))))
    return out


def fuse(dets_a, dets_b):
    """Merge A and B detections of the same player near the net, then remove duplicates.

    A-B pairs are matched by minimum total distance (Hungarian) within MERGE_M and averaged,
    weighted by confidence. Any two remaining detections closer than DEDUP_M are one person
    (e.g. a double box from one camera), so only the more confident one is kept.
    """
    fused, used_a, used_b = [], set(), set()
    if dets_a and dets_b:
        cost = np.array([[np.hypot(a["x"] - b["x"], a["y"] - b["y"]) for b in dets_b] for a in dets_a])
        for i, j in zip(*linear_sum_assignment(cost)):
            if cost[i, j] <= MERGE_M:
                a, b = dets_a[i], dets_b[j]
                wa, wb = a["conf"], b["conf"]
                fused.append(dict(x=(wa * a["x"] + wb * b["x"]) / (wa + wb),
                                  y=(wa * a["y"] + wb * b["y"]) / (wa + wb),
                                  conf=max(wa, wb), cam="A+B"))
                used_a.add(i)
                used_b.add(j)
    fused += [dict(d) for i, d in enumerate(dets_a) if i not in used_a]
    fused += [dict(d) for j, d in enumerate(dets_b) if j not in used_b]
    fused.sort(key=lambda d: -d["conf"])
    out = []
    for d in fused:
        if all(np.hypot(d["x"] - o["x"], d["y"] - o["y"]) > DEDUP_M for o in out):
            out.append(d)
    return out


class Tracker:
    """Tracks players in court coordinates.

    - Matching: Hungarian assignment on the distance from each track's predicted position
      (constant velocity) to each detection, gated at GATE_M.
    - A new track is tentative (no ID, not drawn) until it has been matched CONFIRM_HITS times
      in a row; a tentative track that misses once is deleted. This removes one-off false
      detections and stops the ID counter running up.
    - A confirmed track survives up to LOST_S without a match (occlusion), but is only drawn
      while it was seen in the last SHOW_S, so a lost player does not leave a stale dot.
    - Two tracks closer than DEDUP_M are the same player: the newer one is removed.
    """
    def __init__(self):
        self.tracks, self.next_id = [], 1

    def update(self, dets, t):
        for tr in self.tracks:                          # predict
            dt = t - tr["last"]
            tr["px"], tr["py"] = tr["x"] + tr["vx"] * dt, tr["y"] + tr["vy"] * dt
        matched_t, matched_d = set(), set()
        if self.tracks and dets:
            cost = np.array([[np.hypot(tr["px"] - d["x"], tr["py"] - d["y"]) for d in dets]
                             for tr in self.tracks])
            for r, c in zip(*linear_sum_assignment(cost)):
                if cost[r, c] > GATE_M:
                    continue
                tr, d = self.tracks[r], dets[c]
                dt = max(t - tr["last"], 1e-3)
                nx = (1 - SMOOTH) * tr["px"] + SMOOTH * d["x"]
                ny = (1 - SMOOTH) * tr["py"] + SMOOTH * d["y"]
                vx, vy = 0.5 * tr["vx"] + 0.5 * (nx - tr["x"]) / dt, 0.5 * tr["vy"] + 0.5 * (ny - tr["y"]) / dt
                sp = np.hypot(vx, vy)
                if sp > MAX_SPEED:
                    vx, vy = vx * MAX_SPEED / sp, vy * MAX_SPEED / sp
                tr.update(x=nx, y=ny, vx=vx, vy=vy, last=t, cam=d["cam"], conf=d["conf"], hits=tr["hits"] + 1)
                if tr["id"] is None and tr["hits"] >= CONFIRM_HITS:
                    tr["id"], self.next_id = self.next_id, self.next_id + 1
                tr["trail"].append((t, nx, ny))
                matched_t.add(r)
                matched_d.add(c)
        # tentative tracks die on their first miss; confirmed ones after LOST_S
        self.tracks = [tr for i, tr in enumerate(self.tracks)
                       if i in matched_t or (tr["id"] is not None and t - tr["last"] <= LOST_S)]
        for i, d in enumerate(dets):
            if i not in matched_d:
                self.tracks.append(dict(id=None, x=d["x"], y=d["y"], vx=0.0, vy=0.0, last=t, cam=d["cam"],
                                        conf=d["conf"], hits=1, born=t, trail=[(t, d["x"], d["y"])]))
        # duplicate tracks on one player: keep the older (more established) one
        self.tracks.sort(key=lambda tr: (tr["id"] is None, tr["born"]))
        kept = []
        for tr in self.tracks:
            if all(np.hypot(tr["x"] - k["x"], tr["y"] - k["y"]) > DEDUP_M for k in kept):
                kept.append(tr)
        self.tracks = kept
        for tr in self.tracks:
            tr["trail"] = [p for p in tr["trail"] if t - p[0] <= TRAIL_S]
        return self.visible(t)

    def visible(self, t):
        return [tr for tr in self.tracks if tr["id"] is not None and t - tr["last"] <= SHOW_S]


# ---------------------------------------------------------------- drawing
def side_of(y, ends):
    """Which camera's side of the net a court position is on."""
    near_cam = "A" if ends["A"] == "near" else "B"
    far_cam = "B" if near_cam == "A" else "A"
    return near_cam if y > NET_Y else far_cam


def court_map(height, tracks, ends):
    mx, my = 1.5, 3.5
    ppm = height / (18 + 2 * my)
    w = int(round((9 + 2 * mx) * ppm)) // 2 * 2
    img = np.full((height, w, 3), (90, 60, 40), np.uint8)
    P = lambda x, y: (int(round((x + mx) * ppm)), int(round((y + my) * ppm)))
    cv2.rectangle(img, P(0, 0), P(9, 18), (170, 140, 110), -1)
    for a, b in COURT_LINES:
        cv2.line(img, P(*a), P(*b), (0, 230, 255), 2, cv2.LINE_AA)
    cv2.line(img, P(-1, NET_Y), P(10, NET_Y), (255, 255, 255), 3, cv2.LINE_AA)
    for cam, end in ends.items():
        y = 18 + my - 0.8 if end == "near" else -my + 0.5
        cv2.putText(img, f"camera {cam}", (P(-1.2, y)), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 255, 255), 1, cv2.LINE_AA)
    for tr in tracks:
        col = SIDE_COL[side_of(tr["y"], ends)]
        pts = [P(x, y) for _, x, y in tr["trail"]]
        for p0, p1 in zip(pts, pts[1:]):
            cv2.line(img, p0, p1, col, 1, cv2.LINE_AA)
        cv2.circle(img, P(tr["x"], tr["y"]), max(5, int(0.28 * ppm)), col, -1, cv2.LINE_AA)
        cv2.circle(img, P(tr["x"], tr["y"]), max(5, int(0.28 * ppm)), (0, 0, 0), 1, cv2.LINE_AA)
    return img


def draw_view(img, cal, boxes_all, kept, cam, t, ends):
    v = draw_court(img, cal, colour=(255, 0, 255))
    kept_by_box = {tuple(d["box"]): d for d in kept}
    for (x1, y1, x2, y2) in boxes_all:
        d = kept_by_box.get((x1, y1, x2, y2))
        col = (200, 200, 200) if d is None else SIDE_COL[side_of(d["y"], ends)]
        cv2.rectangle(v, (int(x1), int(y1)), (int(x2), int(y2)), col, 2 if d else 1)
        if d:
            cv2.circle(v, tuple(int(round(q)) for q in d["foot"]), 6, (0, 0, 255), -1)
    k = img.shape[1] / 1280
    txt = f"camera {cam}  t={t:6.2f}s"
    cv2.putText(v, txt, (12, int(40 * k)), cv2.FONT_HERSHEY_SIMPLEX, 1.0 * k, (0, 0, 0), int(6 * k))
    cv2.putText(v, txt, (12, int(40 * k)), cv2.FONT_HERSHEY_SIMPLEX, 1.0 * k, (255, 255, 255), int(2 * k))
    return v


# ---------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser()
    p.add_argument("video_a")
    p.add_argument("video_b")
    p.add_argument("--calib-a", required=True)
    p.add_argument("--calib-b", required=True)
    p.add_argument("--sync", required=True)
    p.add_argument("--start", type=float, default=0.0, help="start time in camera A (s)")
    p.add_argument("--duration", type=float, default=30.0)
    p.add_argument("--every", type=int, default=3, help="detect every Nth frame")
    p.add_argument("--detector", choices=["yolo", "hog"], default="yolo")
    p.add_argument("--model", default="yolo11n.pt", help="YOLO weights (downloaded automatically)")
    p.add_argument("--imgsz", type=int, default=1280, help="YOLO input size (larger finds far players)")
    p.add_argument("--conf", type=float, default=0.3)
    p.add_argument("--foot-frac", type=float, default=None,
                   help="raise foot point by this fraction of box height (default: fitted value from "
                        "data/annotations/foot_offset.json, else 0)")
    p.add_argument("--device", default=None, help="e.g. 0 for GPU, cpu")
    p.add_argument("--width", type=int, default=960)
    p.add_argument("--show", action="store_true")
    args = p.parse_args()

    offset = float(json.loads(Path(args.sync).read_text())["offset_s"])
    cals, ends = {}, {}
    for cam, path in (("A", args.calib_a), ("B", args.calib_b)):
        cals[cam] = load_calibration(path)
        ends[cam] = json.loads(Path(path).read_text())["end"]
    detect = (YoloPeople(args.model, args.conf, args.imgsz, args.device) if args.detector == "yolo"
              else HogPeople(args.conf))
    foot_frac = args.foot_frac if args.foot_frac is not None else load_foot_frac(
        args.model if args.detector == "yolo" else "hog")
    print(f"Foot point: {100 * foot_frac:.1f}% of box height above the box bottom"
          + ("" if foot_frac else " (no fitted offset found; run eval_players.py evaluate)"))

    ca, cb = cv2.VideoCapture(args.video_a), cv2.VideoCapture(args.video_b)
    fa, fb = ca.get(cv2.CAP_PROP_FPS), cb.get(cv2.CAP_PROP_FPS)
    if args.start + offset < 0:
        raise SystemExit(f"Camera B has not started at A = {args.start} s; use --start {-offset:.1f} or later")
    a0 = int(round(args.start * fa))
    print(f"Detector: {args.detector}. Stepping to frame {a0} of A ...")
    for _ in range(a0):
        if not ca.grab():
            raise SystemExit("--start is beyond the end of video A")
    ia, ib, last_b = a0, 0, None

    stem = f"{Path(args.video_a).stem}_{Path(args.video_b).stem}_players_from{args.start:g}"
    out_dir = Path("results/players")
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, ff, tracker = [], None, Tracker()
    boxes = {"A": np.zeros((0, 4)), "B": np.zeros((0, 4))}
    kept = {"A": [], "B": []}
    tracks, det_ms = [], []
    paused = False
    import time
    for k in range(int(args.duration * fa)):
        ok, a = ca.read()
        if not ok:
            break
        t_a = ia / fa
        ia += 1
        target = int(round((t_a + offset) * fb))
        while ib < target and cb.grab():
            ib += 1
        if ib == target:
            okb, fr = cb.read()
            if okb:
                last_b, ib = fr, ib + 1
        b = last_b
        if b is None:
            break

        if k % args.every == 0:
            t0 = time.perf_counter()
            for cam, img in (("A", a), ("B", b)):
                bx, cf = detect(img)
                boxes[cam] = bx
                kept[cam] = feet_to_court(bx, cf, cals[cam], img.shape, cam, ends[cam], foot_frac)
            det_ms.append((time.perf_counter() - t0) * 1000)
            tracks = tracker.update(fuse(kept["A"], kept["B"]), t_a)
            for tr in tracks:
                if tr["last"] == t_a:
                    rows.append(dict(time_s=round(t_a, 3), track_id=tr["id"], x_m=round(tr["x"], 3),
                                     y_m=round(tr["y"], 3), side=side_of(tr["y"], ends),
                                     camera=tr["cam"], confidence=round(tr["conf"], 3)))

        views = []
        for cam, img in (("A", a), ("B", b)):
            v = draw_view(img, cals[cam], boxes[cam], kept[cam], cam, t_a + (offset if cam == "B" else 0), ends)
            vh = int(round(args.width * img.shape[0] / img.shape[1]))
            views.append(cv2.resize(v, (args.width, vh)))
        h = max(v.shape[0] for v in views)
        h += h % 2
        views = [cv2.copyMakeBorder(v, 0, h - v.shape[0], 0, 0, cv2.BORDER_CONSTANT) for v in views]
        frame = np.hstack(views + [court_map(h, tracks, ends)])

        if ff is None:
            ff = subprocess.Popen([ffmpeg_exe(), "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
                                   "-s", f"{frame.shape[1]}x{frame.shape[0]}", "-r", f"{fa:.5f}", "-i", "-",
                                   "-c:v", "libx264", "-preset", "veryfast", "-crf", "21",
                                   "-pix_fmt", "yuv420p", str(out_dir / f"{stem}.mp4")], stdin=subprocess.PIPE)
        ff.stdin.write(frame.tobytes())
        if args.show:
            cv2.imshow("players on court (q quit, space pause)", frame)
            key = cv2.waitKey(0 if paused else 1) & 0xFF
            if key == ord("q"):
                break
            if key == ord(" "):
                paused = not paused
        if k and k % int(10 * fa) == 0:
            print(f"  {k / fa:.0f} s done  (detection {np.mean(det_ms):.0f} ms per step, both cameras)")

    if ff:
        ff.stdin.close()
        ff.wait()
    ca.release(); cb.release()
    cv2.destroyAllWindows()
    with open(out_dir / f"{stem}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["time_s", "track_id", "x_m", "y_m", "side", "camera", "confidence"])
        w.writeheader()
        w.writerows(rows)
    if det_ms:
        n_steps = len(det_ms)
        print(f"Detection: {np.mean(det_ms):.0f} ms per step for both cameras ({n_steps} steps)")
        print(f"Players shown per step: {len(rows) / n_steps:.1f} on average (12 on court); "
              f"{tracker.next_id - 1} track IDs used")
    print(f"Saved {out_dir / (stem + '.mp4')} and .csv")


if __name__ == "__main__":
    main()