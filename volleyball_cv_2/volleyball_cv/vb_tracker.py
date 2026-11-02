"""
Volleyball player detection + court mapping prototype.

Pipeline (per frame):
  1. Person detection + 17-kpt pose (YOLO11-pose) on a court-ROI crop
  2. Ground contact point  = ankle midpoint (fallback: bbox bottom-centre)
  3. Image -> court plane via homography H (calibrated once per camera setup)
  4. Gate to the playing area (court + free-zone margin) -> drops adjacent
     court, spectators, subs
  5. Team = which side of the net (court y < 9 m or > 9 m)
  1b. Far court re-detected on a magnified tile, merged by NMS (small-object fix)
  6. Multi-object tracking (ByteTrack) + per-ID smoothing in court metres
  7. Render: annotated frame + top-down minimap with trails, CSV of positions

Court frame: x across (0..9 m, left->right from camera), y along
(0 = near baseline, 9 = net, 18 = far baseline).
"""
import argparse, csv, json, collections
import cv2, numpy as np
from ultralytics import YOLO

from common import run_args, open_video, prep, pkg_file
a = run_args()
a.video, a.calib, a.out, a.csv = a.cfg["video"], "calib.json", "tracking.mp4", "positions.csv"
a.start, a.dur, a.fps = a.cfg["start"], a.cfg["dur"], a.cfg["det_fps"]
a.model, a.imgsz, a.conf = pkg_file("yolo11m-pose.pt"), 1280, 0.25

C = json.load(open(a.calib))
H = cv2.findHomography(np.float32(C["world_m"]), np.float32(C["image_px"]))[0]
Hinv = np.linalg.inv(H)
x0, y0, x1, y1 = C["roi"]                      # detection crop (px)
GX, GY = C["gate_x"], C["gate_y"]              # playing-area gate (m)

def img2court(u, v):
    p = Hinv @ np.array([u, v, 1.0]); return p[:2] / p[2]
def court2img(x, y):
    p = H @ np.array([x, y, 1.0]); return (int(p[0] / p[2]), int(p[1] / p[2]))

# ---------------- minimap ----------------
S, PAD = 26, 40                                 # px per metre, padding
MW, MH = int(9 * S + 2 * PAD + 2 * S), int(18 * S + 2 * PAD + 2 * S)
def m2map(x, y):                                # far side at top
    return (int(PAD + S + x * S), int(PAD + S + (18 - y) * S))
def blank_map():
    m = np.full((MH, MW, 3), (60, 45, 30), np.uint8)
    cv2.rectangle(m, m2map(-1, -1), m2map(10, 19), (95, 75, 50), -1)
    cv2.rectangle(m, m2map(0, 0), m2map(9, 18), (200, 150, 90), -1)
    cv2.rectangle(m, m2map(0, 0), m2map(9, 18), (255, 255, 255), 2)
    for y in (6, 12): cv2.line(m, m2map(0, y), m2map(9, y), (255, 255, 255), 1)
    cv2.line(m, m2map(-0.5, 9), m2map(9.5, 9), (40, 40, 40), 4)   # net
    return m
TEAM_COL = {0: (60, 90, 255), 1: (255, 200, 40)}  # near = red, far = gold (BGR)

# ---------------- court overlay for the camera view ----------------
def draw_court(img):
    for y in (0, 6, 12, 18):
        cv2.line(img, court2img(0, y), court2img(9, y), (255, 255, 255), 1, cv2.LINE_AA)
    cv2.line(img, court2img(0, 0), court2img(0, 18), (255, 255, 255), 1, cv2.LINE_AA)
    cv2.line(img, court2img(9, 0), court2img(9, 18), (255, 255, 255), 1, cv2.LINE_AA)
    cv2.line(img, court2img(-0.5, 9), court2img(9.5, 9), (255, 0, 255), 2, cv2.LINE_AA)

import supervision as sv
model = YOLO(a.model)
TILES = [tuple(C["roi"])] + [tuple(t) for t in C.get("far_tiles", [])]
cap, src_fps = open_video(a.video)
step = max(1, round(src_fps / a.fps))
cap.set(cv2.CAP_PROP_POS_FRAMES, int(a.start * src_fps))
n_out = int(a.dur * src_fps / step)
tracker = sv.ByteTrack(track_activation_threshold=0.3, lost_track_buffer=int(2 * a.fps),
                       minimum_matching_threshold=0.8, frame_rate=int(a.fps))

OW = 1280; OH = 720
writer = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"), src_fps / step,
                         (OW + int(MW * OH / MH), OH))
smooth = {}                                   # id -> EMA court position
trails = collections.defaultdict(lambda: collections.deque(maxlen=int(3 * a.fps)))
last_seen = {}
fcsv = open(a.csv, "w", newline=""); W = csv.writer(fcsv)
W.writerow(["frame", "t_s", "track_id", "team", "x_m", "y_m", "u_px", "v_px", "conf", "foot_src"])

fi = int(a.start * src_fps)
frame_log = open("tracking_frames.txt", "w")   # source frame index of every output frame (used by render_overlay)
for k in range(n_out):
    for _ in range(step - 1): cap.grab(); fi += 1
    ok, frame = cap.read(); cur = fi; fi += 1
    if not ok: break
    frame = prep(frame); frame_log.write(f"{cur}\n")
    # --- two-scale tiled detection: whole court + magnified far court ---
    boxes, confs, feet, srcs = [], [], [], []
    for (tx0, ty0, tx1, ty1) in TILES:
        r = model.predict(frame[ty0:ty1, tx0:tx1], imgsz=a.imgsz, conf=a.conf,
                          classes=[0], verbose=False)[0]
        if r.boxes is None or not len(r.boxes): continue
        B = r.boxes.xyxy.cpu().numpy(); cf = r.boxes.conf.cpu().numpy()
        KP = r.keypoints.data.cpu().numpy()
        for i, (bx0, by0, bx1, by1) in enumerate(B):
            # drop boxes cut by an internal tile edge (the other tile sees them whole)
            if (tx0 > 0 and bx0 < 3) or (ty0 > 0 and by0 < 3) or (ty1 < frame.shape[0] and by1 > ty1 - ty0 - 3): continue
            bb = [bx0 + tx0, by0 + ty0, bx1 + tx0, by1 + ty0]
            # ground contact point: ankle midpoint, else bbox bottom-centre
            u, v, s = (bb[0] + bb[2]) / 2, bb[3], "bbox"
            good = [p for p in (KP[i, 15], KP[i, 16]) if p[2] > 0.5]
            if good:
                u = np.mean([p[0] for p in good]) + tx0
                v = max(p[1] for p in good) + ty0 + 0.012 * (bb[3] - bb[1]); s = "ankle"
            boxes.append(bb); confs.append(cf[i]); feet.append((u, v)); srcs.append(s)
    vis = frame.copy(); draw_court(vis)
    mp = blank_map()
    counts = {0: 0, 1: 0}
    if boxes:
        det = sv.Detections(xyxy=np.array(boxes, float), confidence=np.array(confs),
                            class_id=np.zeros(len(boxes), int),
                            data={"foot": np.array(feet), "src": np.array(srcs)})
        det = det.with_nms(threshold=0.5)
        det = tracker.update_with_detections(det)
    else:
        det = tracker.update_with_detections(sv.Detections.empty())
    if len(det):
        ids = det.tracker_id.astype(int); conf = det.confidence
        for i, (bx0, by0, bx1, by1) in enumerate(det.xyxy):
            u, v = det.data["foot"][i]; src = det.data["src"][i]
            cx, cy = img2court(u, v)
            if not (GX[0] <= cx <= GX[1] and GY[0] <= cy <= GY[1]):
                cv2.rectangle(vis, (int(bx0), int(by0)), (int(bx1), int(by1)), (120, 120, 120), 1)
                continue                               # not on our court
            team = 0 if cy < 9 else 1
            tid = ids[i]
            if tid >= 0:
                p = np.array([cx, cy])
                smooth[tid] = p if tid not in smooth else 0.6 * p + 0.4 * smooth[tid]
                cx, cy = smooth[tid]; trails[tid].append((cx, cy)); last_seen[tid] = k
            counts[team] += 1
            col = TEAM_COL[team]
            cv2.rectangle(vis, (int(bx0), int(by0)), (int(bx1), int(by1)), col, 2)
            cv2.circle(vis, (int(u), int(v)), 4, col, -1)
            cv2.putText(vis, f"#{tid} ({cx:.1f},{cy:.1f})", (int(bx0), int(by0) - 6),
                        0, 0.55, col, 2, cv2.LINE_AA)
            W.writerow([cur, round(cur / src_fps, 3), tid, "near" if team == 0 else "far",
                        round(cx, 3), round(cy, 3), round(u, 1), round(v, 1), round(float(conf[i]), 3), src])
    # minimap: trails of recently seen tracks, then dots
    for tid, tr in trails.items():
        if k - last_seen.get(tid, -99) > 5 or len(tr) < 2: continue
        team = 0 if tr[-1][1] < 9 else 1
        pts = np.array([m2map(*p) for p in tr], np.int32)
        cv2.polylines(mp, [pts], False, tuple(int(c * 0.6) for c in TEAM_COL[team]), 2, cv2.LINE_AA)
    for tid, tr in trails.items():
        if last_seen.get(tid) != k: continue
        team = 0 if tr[-1][1] < 9 else 1
        c = m2map(*tr[-1])
        cv2.circle(mp, c, 9, TEAM_COL[team], -1); cv2.circle(mp, c, 9, (255, 255, 255), 1)
        cv2.putText(mp, str(tid), (c[0] + 10, c[1] + 4), 0, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(mp, f"near {counts[0]}  far {counts[1]}", (10, 22), 0, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    left = cv2.resize(vis, (OW, OH))
    right = cv2.resize(mp, (int(MW * OH / MH), OH))
    writer.write(np.hstack([left, right]))
    if k % 25 == 0: print(f"{k}/{n_out}  near={counts[0]} far={counts[1]}", flush=True); fcsv.flush()
writer.release(); fcsv.close(); frame_log.close()
