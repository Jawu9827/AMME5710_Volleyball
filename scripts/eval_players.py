"""Measure player detection accuracy against hand-clicked ground truth.

Step 1 - annotate (about 10 minutes): click the feet of every player in each camera's half
    python scripts/eval_players.py annotate data/game2_A.mp4 data/game2_B_sdr.mp4 \
        --calib-a data/annotations/calib_A.json --calib-b data/annotations/calib_B.json \
        --sync data/annotations/sync_game2_A_game2_B_sdr.json --n 10
  Frames are spread evenly over the overlap of the two videos (or use --start/--end).
  For each moment you see camera A, then camera B, with that camera's half of the court shaded.
  Click the floor point between the feet of EVERY player standing in the shaded half,
  including a server behind the end line and players hidden behind others (best guess).
    left click  add a player          right click  delete the nearest click
    u           undo last click
    Enter/space next image            s  skip this image (unusable, e.g. motion blur)
    q           save and stop (run again later to continue where you stopped)
  Saved to data/annotations/players_gt.json after every image.
  Redo:  --redo        start again from scratch (the old file is kept as players_gt_old.json)
         --only 3 7    redo just moments 3 and 7 (numbers as shown in the window title)

Step 2 - evaluate (no clicking): runs each detector on the same frames and scores it
    python scripts/eval_players.py evaluate --detectors yolo11n yolo11s hog
  For each camera, detections and clicks are both mapped to the court and kept only if they
  are in that camera's half (+ the serving zone and side margins used by players_on_court.py).
  A detection matches a clicked player if they are within --gate metres (one-to-one,
  Hungarian assignment). Reported per detector and camera:
    precision = TP / (TP + FP)   recall = TP / (TP + FN)   F1
    count error = mean |detected - true| players per image
    position error = median / mean distance (m) between matched detection and click
    time = ms per image
  "system" rows score the full two-camera pipeline (both cameras fused, as in
  players_on_court.py) against all clicked players at that moment, without tracking.

  Foot correction: the bottom of a detection box sits a little below where the feet touch the
  floor (box padding, the lowest shoe), which maps to a point too close to the camera. Each
  detector is scored twice:
    raw        foot point = bottom centre of the box
    corrected  foot point raised by a fraction of the box height, fitted as the median
               box-bottom-to-click offset. 2-fold cross-validated: the fraction used on the
               even moments is fitted on the odd ones and vice versa.
  The fraction fitted on all moments is saved to data/annotations/foot_offset.json and used by
  players_on_court.py.

Outputs (results/eval/):
    players_eval_summary.csv   one row per detector x raw/corrected x camera (+ printed)
    players_eval_frames.csv    per image counts
    vis/<detector>_<raw|corrected>_<cam>_t<time>.jpg   white ring = clicked player, green = matched detection,
                                       red = false detection, yellow X = missed player
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).parent))
from calibrate_camera import duration, load_calibration, read_frame  # noqa: E402
from players_on_court import (COURT_MARGIN, FOOT_PATH, NET_Y, HogPeople, YoloPeople,  # noqa: E402
                              feet_to_court, fuse)

HALF = {"near": ((0, 9), (9, 9), (9, 18), (0, 18)),     # camera behind y=18 judges y 9..18
        "far": ((0, 0), (9, 0), (9, 9), (0, 9))}        # camera behind y=0 judges y 0..9
COLOURS = {"A": (0, 200, 255), "B": (80, 230, 80)}     # BGR: orange for A, green for B


def shade_half(img, cal, end, colour, alpha=0.18):
    """Shade the half of the court a camera judges."""
    pts = []
    for (x0, y0), (x1, y1) in zip(HALF[end], HALF[end][1:] + HALF[end][:1]):
        s = np.linspace(0, 1, 30)[:, None]
        pts.append(cal.court_to_image(np.array([x0, y0]) * (1 - s) + np.array([x1, y1]) * s))
    poly = np.int32(np.round(np.vstack(pts)))
    over = img.copy()
    cv2.fillPoly(over, [poly], colour)
    return cv2.addWeighted(over, alpha, img, 1 - alpha, 0)

GT_PATH = Path("data/annotations/players_gt.json")
FRAME_DIR = Path("results/eval/frames")
OUT_DIR = Path("results/eval")
MAX_W = 1600
DIAG_GATE = 2.5     # m, loose matching used only to measure the detection-click offset


# ---------------------------------------------------------------- shared helpers
def in_half(x, y, end):
    mx, my = COURT_MARGIN
    if not (-mx <= x <= 9 + mx and -my <= y <= 18 + my):
        return False
    return y > NET_Y if end == "near" else y < NET_Y


def load_setup(gt):
    cals = {c: load_calibration(gt[f"calib_{c.lower()}"]) for c in "AB"}
    ends = {c: json.loads(Path(gt[f"calib_{c.lower()}"]).read_text())["end"] for c in "AB"}
    return cals, ends


def frame_path(cam, t):
    return FRAME_DIR / f"{cam}_t{t:08.3f}.png"


def get_frame(gt, cam, t_a):
    """Exact frame of camera cam at A-time t_a (cached as PNG so every run sees the same pixels)."""
    t = t_a + (gt["offset_s"] if cam == "B" else 0.0)
    p = frame_path(cam, t)
    if p.exists():
        return cv2.imread(str(p))
    img = read_frame(gt[f"video_{cam.lower()}"], t)
    if img is None:
        raise SystemExit(f"Could not read camera {cam} at {t:.2f} s")
    FRAME_DIR.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(p), img)
    return img


# ---------------------------------------------------------------- step 1: annotate
def click_players(img, title):
    """Returns (list of [u, v] full-res pixels, action) with action in next/skip/quit."""
    s = min(1.0, MAX_W / img.shape[1])
    base = cv2.resize(img, None, fx=s, fy=s)
    pts, mouse = [], [0, 0]

    def on_mouse(ev, x, y, flags, param):
        mouse[:] = [x, y]
        if ev == cv2.EVENT_LBUTTONDOWN:
            pts.append([x / s, y / s])
        elif ev == cv2.EVENT_RBUTTONDOWN and pts:       # delete the nearest click
            d = [np.hypot(u * s - x, v * s - y) for u, v in pts]
            if min(d) < 25:
                pts.pop(int(np.argmin(d)))

    win = "click players' feet  (u undo, Enter next, s skip, q quit)"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(win, on_mouse)
    while True:
        v = base.copy()
        for i, (u, w) in enumerate(pts):
            c = (int(u * s), int(w * s))
            cv2.circle(v, c, 6, (0, 0, 255), -1)
            cv2.circle(v, c, 7, (255, 255, 255), 1)
            cv2.putText(v, str(i + 1), (c[0] + 8, c[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        # 4x magnifier around the cursor, from the full-resolution image
        mx, my = int(mouse[0] / s), int(mouse[1] / s)
        r = 40
        crop = cv2.copyMakeBorder(img, r, r, r, r, cv2.BORDER_CONSTANT)[my:my + 2 * r, mx:mx + 2 * r]
        if crop.shape[:2] == (2 * r, 2 * r):
            mag = cv2.resize(crop, (8 * r, 8 * r), interpolation=cv2.INTER_NEAREST)
            cv2.drawMarker(mag, (4 * r, 4 * r), (0, 0, 255), cv2.MARKER_CROSS, 30, 1)
            v[-8 * r - 10:-10, -8 * r - 10:-10] = mag
        txt = f"{title}   players clicked: {len(pts)}"
        cv2.putText(v, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 5)
        cv2.putText(v, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.imshow(win, v)
        k = cv2.waitKey(20) & 0xFF
        if k == ord("u") and pts:
            pts.pop()
        elif k in (13, 10, 32):
            return pts, "next"
        elif k == ord("s"):
            return [], "skip"
        elif k == ord("q"):
            return pts, "quit"


def annotate(args):
    if args.redo and GT_PATH.exists():
        old = GT_PATH.with_name("players_gt_old.json")
        GT_PATH.replace(old)
        print(f"Old ground truth moved to {old}; starting again")
    if args.only:
        if not GT_PATH.exists():
            raise SystemExit("--only needs an existing ground truth file")
        gt = json.loads(GT_PATH.read_text())
        for m in args.only:
            if not 1 <= m <= len(gt["frames"]):
                raise SystemExit(f"Moment {m} does not exist (1-{len(gt['frames'])})")
            gt["frames"][m - 1] = dict(t_a=gt["frames"][m - 1]["t_a"])
        GT_PATH.write_text(json.dumps(gt, indent=1))
        print(f"Cleared moments {args.only}; click them again")
    if GT_PATH.exists():
        gt = json.loads(GT_PATH.read_text())
        print(f"Continuing {GT_PATH} ({sum(1 for f in gt['frames'] if 'A' in f and 'B' in f)} of "
              f"{len(gt['frames'])} moments done)")
    else:
        offset = float(json.loads(Path(args.sync).read_text())["offset_s"])
        d_a, d_b = duration(args.video_a), duration(args.video_b)
        lo = max(args.start, -offset, 0) + 2
        hi = min(args.end or 1e9, d_a, d_b - offset) - 2
        if hi <= lo:
            raise SystemExit("No overlap between the videos in that time range")
        times = np.linspace(lo, hi, args.n)
        gt = dict(video_a=args.video_a, video_b=args.video_b, calib_a=args.calib_a, calib_b=args.calib_b,
                  offset_s=offset, frames=[dict(t_a=round(float(t), 3)) for t in times])
        GT_PATH.parent.mkdir(parents=True, exist_ok=True)
        GT_PATH.write_text(json.dumps(gt, indent=1))
        print(f"{args.n} moments from A = {lo:.1f} s to {hi:.1f} s (B = A {offset:+.2f} s)")
    cals, ends = load_setup(gt)
    for i, fr in enumerate(gt["frames"]):
        for cam in "AB":
            if cam in fr or fr.get(f"skip_{cam}"):
                continue
            img = get_frame(gt, cam, fr["t_a"])
            shown = shade_half(img, cals[cam], ends[cam], COLOURS[cam], alpha=0.25)
            pts, action = click_players(shown, f"moment {i + 1}/{len(gt['frames'])}  camera {cam}  "
                                               f"t_A={fr['t_a']:.2f}s  - click feet in the shaded half")
            if action == "quit":
                GT_PATH.write_text(json.dumps(gt, indent=1))
                cv2.destroyAllWindows()
                print(f"Saved progress to {GT_PATH}; run the same command to continue")
                return
            if action == "skip":
                fr[f"skip_{cam}"] = True
            else:
                fr[cam] = [[round(u, 1), round(v, 1)] for u, v in pts]
            GT_PATH.write_text(json.dumps(gt, indent=1))
    cv2.destroyAllWindows()
    n = sum(len(fr.get(c, [])) for fr in gt["frames"] for c in "AB")
    print(f"Done: {n} players clicked. Saved {GT_PATH}. Next: python scripts/eval_players.py evaluate")


# ---------------------------------------------------------------- step 2: evaluate
def match(gt_xy, det_xy, gate):
    """One-to-one matching within gate metres. Returns (pairs, unmatched_gt, unmatched_det)."""
    if len(gt_xy) == 0 or len(det_xy) == 0:
        return [], list(range(len(gt_xy))), list(range(len(det_xy)))
    cost = np.linalg.norm(np.asarray(gt_xy)[:, None] - np.asarray(det_xy)[None], axis=2)
    pairs = [(r, c) for r, c in zip(*linear_sum_assignment(cost)) if cost[r, c] <= gate]
    mg, md = {r for r, _ in pairs}, {c for _, c in pairs}
    return (pairs, [i for i in range(len(gt_xy)) if i not in mg], [j for j in range(len(det_xy)) if j not in md])


def make_detector(name, conf):
    if name == "hog":
        return HogPeople(conf)
    return YoloPeople(name if name.endswith(".pt") else name + ".pt", conf, 1280, None)


def draw_vis(img, gt_uv, det_uv, pairs, miss, false, path):
    v = img.copy()
    k = max(1, img.shape[1] // 1280)
    for u, w in gt_uv:
        cv2.circle(v, (int(u), int(w)), 14 * k, (255, 255, 255), 2 * k)
    for _, c in pairs:
        cv2.circle(v, tuple(map(int, det_uv[c])), 6 * k, (0, 220, 0), -1)
    for c in false:
        cv2.circle(v, tuple(map(int, det_uv[c])), 6 * k, (0, 0, 255), -1)
    for r in miss:
        cv2.drawMarker(v, tuple(map(int, gt_uv[r])), (0, 255, 255), cv2.MARKER_TILTED_CROSS, 24 * k, 3 * k)
    path.parent.mkdir(parents=True, exist_ok=True)
    s = min(1.0, 1600 / v.shape[1])
    cv2.imwrite(str(path), cv2.resize(v, None, fx=s, fy=s))


def fit_foot_frac(cache, moments, gt, cals, ends, which):
    """Median fraction of box height between the box bottom and the clicked foot point,
    from loosely matched (raw, uncorrected) detections in the moments listed in `which`."""
    ratios = []
    for i in which:
        fr = moments[i]
        for cam in "AB":
            if cam not in fr:
                continue
            shape, boxes, conf, _ = cache[(i, cam)]
            dets = feet_to_court(boxes, conf, cals[cam], shape, cam, ends[cam], foot_frac=0.0)
            gt_uv = np.asarray(fr[cam], float).reshape(-1, 2)
            if not len(gt_uv) or not dets:
                continue
            g_xy = cals[cam].image_to_court(gt_uv)
            d_xy = np.array([[d["x"], d["y"]] for d in dets])
            for r, c in match(g_xy, d_xy, DIAG_GATE)[0]:
                x1, y1, x2, y2 = dets[c]["box"]
                ratios.append((y2 - gt_uv[r][1]) / max(y2 - y1, 1.0))
    return float(np.median(ratios)) if ratios else 0.0


def score(name, label, cache, moments, gt, cals, ends, fracs, args, per_frame):
    """Score one detector with foot fraction fracs[i] for moment i. Returns stats per camera."""
    stats = {c: dict(tp=0, fp=0, fn=0, err=[], cnt=[], ms=[], bias=[]) for c in ("A", "B", "system")}
    for i, fr in enumerate(moments):
        court_gt, court_det = [], {}
        for cam in "AB":
            shape, boxes, conf, ms = cache[(i, cam)]
            stats[cam]["ms"].append(ms)
            dets = feet_to_court(boxes, conf, cals[cam], shape, cam, ends[cam], foot_frac=fracs[i])
            court_det[cam] = dets
            if cam not in fr:
                continue
            gt_uv = np.asarray(fr[cam], float).reshape(-1, 2)
            gt_xy = cals[cam].image_to_court(gt_uv) if len(gt_uv) else np.zeros((0, 2))
            keep_g = [k for k, (x, y) in enumerate(gt_xy) if in_half(x, y, ends[cam])]
            keep_d = [j for j, d in enumerate(dets) if in_half(d["x"], d["y"], ends[cam])]
            g_xy, g_uv = gt_xy[keep_g], gt_uv[keep_g]
            d_xy = np.array([[dets[j]["x"], dets[j]["y"]] for j in keep_d]).reshape(-1, 2)
            d_uv = np.array([dets[j]["foot"] for j in keep_d]).reshape(-1, 2)
            court_gt += [tuple(q) for q in g_xy]
            pairs, miss, false = match(g_xy, d_xy, args.gate)
            st = stats[cam]
            st["tp"] += len(pairs); st["fn"] += len(miss); st["fp"] += len(false)
            st["err"] += [float(np.linalg.norm(g_xy[r] - d_xy[c])) for r, c in pairs]
            st["cnt"].append(abs(len(d_xy) - len(g_xy)))
            per_frame.append(dict(detector=name, foot=label, camera=cam, t_a=fr["t_a"], true=len(g_xy),
                                  detected=len(d_xy), tp=len(pairs), fp=len(false), fn=len(miss)))
            away = -1.0 if ends[cam] == "near" else 1.0     # +y is away from a far-end camera
            for r, c in match(g_xy, d_xy, DIAG_GATE)[0]:
                du, dv = d_uv[c] - g_uv[r]
                dx, dy = d_xy[c] - g_xy[r]
                st["bias"].append((du, dv, away * dy, dx))
            if args.vis:
                draw_vis(get_frame(gt, cam, fr["t_a"]), g_uv, d_uv, pairs, miss, false,
                         OUT_DIR / "vis" / f"{Path(name).stem}_{label}_{cam}_t{fr['t_a']:07.2f}.jpg")
        if "A" in fr and "B" in fr:                # full system: both cameras fused
            sys_det = fuse(court_det["A"], court_det["B"])
            d_xy = np.array([[d["x"], d["y"]] for d in sys_det
                             if in_half(d["x"], d["y"], "near") or in_half(d["x"], d["y"], "far")]).reshape(-1, 2)
            g_xy = np.asarray(court_gt).reshape(-1, 2)
            pairs, miss, false = match(g_xy, d_xy, args.gate)
            st = stats["system"]
            st["tp"] += len(pairs); st["fn"] += len(miss); st["fp"] += len(false)
            st["err"] += [float(np.linalg.norm(g_xy[r] - d_xy[c])) for r, c in pairs]
            st["cnt"].append(abs(len(d_xy) - len(g_xy)))
            st["ms"].append(stats["A"]["ms"][-1] + stats["B"]["ms"][-1])
            per_frame.append(dict(detector=name, foot=label, camera="system", t_a=fr["t_a"], true=len(g_xy),
                                  detected=len(d_xy), tp=len(pairs), fp=len(false), fn=len(miss)))
    return stats


def summarise(name, label, stats):
    rows = []
    for cam, st in stats.items():
        if not st["cnt"]:
            continue
        tp, fp, fn = st["tp"], st["fp"], st["fn"]
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        rows.append(dict(detector=Path(name).stem, foot=label, camera=cam, images=len(st["cnt"]),
                         players=tp + fn, tp=tp, fp=fp, fn=fn,
                         precision=round(p, 3), recall=round(r, 3),
                         f1=round(2 * p * r / (p + r), 3) if p + r else 0.0,
                         count_err=round(float(np.mean(st["cnt"])), 2),
                         pos_err_median_m=round(float(np.median(st["err"])), 3) if st["err"] else None,
                         pos_err_mean_m=round(float(np.mean(st["err"])), 3) if st["err"] else None,
                         ms_per_image=round(float(np.mean(st["ms"][1:] or st["ms"])), 0)))
    return rows


def print_bias(title, rows):
    print(f"\n{title}")
    for name, cam, b in rows:
        du, dv, depth, side = np.median(np.asarray(b), axis=0)
        print(f"  {name:>8} camera {cam}: {abs(dv):4.1f} px {'below' if dv > 0 else 'above'} the click, "
              f"{abs(du):4.1f} px {'right' if du > 0 else 'left'};  {abs(depth):.2f} m "
              f"{'further from' if depth > 0 else 'closer to'} the camera, {abs(side):.2f} m sideways  "
              f"({len(b)} pairs)")


def evaluate_all(args):
    if not GT_PATH.exists():
        raise SystemExit(f"{GT_PATH} not found - run the annotate step first")
    gt = json.loads(GT_PATH.read_text())
    cals, ends = load_setup(gt)
    moments = [f for f in gt["frames"] if "A" in f or "B" in f]
    print(f"{len(moments)} annotated moments, "
          f"{sum(len(f.get(c, [])) for f in moments for c in 'AB')} clicked players")
    folds = [list(range(0, len(moments), 2)), list(range(1, len(moments), 2))]
    summary, per_frame, bias_raw, bias_cor, fitted = [], [], [], [], {}
    for name in args.detectors:
        try:
            det = make_detector(name, args.conf)
        except Exception as e:                       # e.g. weights could not be downloaded
            print(f"Skipping {name}: {e}")
            continue
        stem = Path(name).stem
        cache = {}                                   # run each detector once per image
        for i, fr in enumerate(moments):
            for cam in "AB":
                img = get_frame(gt, cam, fr["t_a"])
                t0 = time.perf_counter()
                boxes, conf = det(img)
                cache[(i, cam)] = (img.shape, boxes, conf, (time.perf_counter() - t0) * 1000)

        # 1) raw: foot point = bottom of the box
        st = score(name, "raw", cache, moments, gt, cals, ends, [0.0] * len(moments), args, per_frame)
        summary += summarise(name, "raw", st)
        bias_raw += [(stem, c, s["bias"]) for c, s in st.items() if s["bias"]]

        # 2) corrected, 2-fold cross-validated: the foot offset used on even moments is fitted
        #    on odd moments and vice versa, so no moment is scored with an offset fitted on itself
        f_fold = [fit_foot_frac(cache, moments, gt, cals, ends, folds[1 - k]) for k in range(2)]
        fracs = [f_fold[0] if i % 2 == 0 else f_fold[1] for i in range(len(moments))]
        st = score(name, "corrected", cache, moments, gt, cals, ends, fracs, args, per_frame)
        summary += summarise(name, "corrected", st)
        bias_cor += [(stem, c, s["bias"]) for c, s in st.items() if s["bias"]]
        fitted[stem] = round(fit_foot_frac(cache, moments, gt, cals, ends, range(len(moments))), 4)
        print(f"  {stem}: foot offset = {100 * fitted[stem]:.1f}% of box height above the box bottom "
              f"(fold fits {100 * f_fold[0]:.1f}% / {100 * f_fold[1]:.1f}%)")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for path, rows in ((OUT_DIR / "players_eval_summary.csv", summary), (OUT_DIR / "players_eval_frames.csv", per_frame)):
        if rows:
            with open(path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
    if fitted:
        FOOT_PATH.write_text(json.dumps(fitted, indent=1))
    print_bias(f"Offset of detections from your clicks, RAW box bottom (median, loose matching within {DIAG_GATE} m):",
               bias_raw)
    print_bias("Offset AFTER the cross-validated foot correction:", bias_cor)
    cols = ["detector", "foot", "camera", "players", "tp", "fp", "fn", "precision", "recall", "f1",
            "count_err", "pos_err_median_m", "ms_per_image"]
    print("\n" + "  ".join(f"{c:>10}" for c in cols))
    for row in summary:
        print("  ".join(f"{str(row[c]):>10}" for c in cols))
    print(f"\nMatch gate {args.gate} m. Saved {OUT_DIR / 'players_eval_summary.csv'}"
          + (f" and images in {OUT_DIR / 'vis'}" if args.vis else "")
          + f".\nFitted foot offsets saved to {FOOT_PATH} (players_on_court.py uses them automatically).")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("annotate")
    a.add_argument("video_a")
    a.add_argument("video_b")
    a.add_argument("--calib-a", required=True)
    a.add_argument("--calib-b", required=True)
    a.add_argument("--sync", required=True)
    a.add_argument("--n", type=int, default=10, help="number of moments (each = one A and one B image)")
    a.add_argument("--start", type=float, default=0.0, help="earliest A time (s)")
    a.add_argument("--end", type=float, default=None, help="latest A time (s)")
    a.add_argument("--redo", action="store_true", help="start the ground truth again from scratch")
    a.add_argument("--only", type=int, nargs="+", help="redo only these moment numbers")
    e = sub.add_parser("evaluate")
    e.add_argument("--detectors", nargs="+", default=["yolo11n", "yolo11s", "hog"])
    e.add_argument("--conf", type=float, default=0.3)
    e.add_argument("--gate", type=float, default=1.0, help="max detection-click distance for a match (m)")
    e.add_argument("--no-vis", dest="vis", action="store_false")
    args = p.parse_args()
    annotate(args) if args.cmd == "annotate" else evaluate_all(args)


if __name__ == "__main__":
    main()