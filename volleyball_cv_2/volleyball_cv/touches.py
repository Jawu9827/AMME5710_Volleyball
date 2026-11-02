"""
Touch attribution + volleyball rule checks.

For each ball impulse event:
  - detect players (two-scale YOLO tiles) and run RTMPose-m (top-down) on each box
  - distance from ball centre to the player's contact parts: hand (wrist + 0.35*forearm
    extension), forearm segment (elbow->wrist, digs/passes), head (nose)
  - normalised by the player's box height; min over frames event-2..event+2
  - toucher = closest player if dist < TOUCH_R, else bounce (ball low) / unattributed
  - team & track id from the court position (homography) matched to positions.csv tracks
Rules over each rally:
  - >3 consecutive touches by one team (block touches excluded)  -> FOUR_TOUCHES
  - same player touches twice in a row (not after a block)         -> DOUBLE_CONTACT
"""
import json, cv2, numpy as np, pandas as pd
from ultralytics import YOLO
from rtmlib import RTMPose
from camera_model import camera_from_calib, ball_3d
from common import run_args, open_video, prep, pkg_file, rtm_model
import sys
a = run_args()

VIDEO = a.cfg["video"]
TOUCH_R = 0.22           # x player height
C = json.load(open("calib.json"))
H = cv2.findHomography(np.float32(C["world_m"]), np.float32(C["image_px"]))[0]; Hinv = np.linalg.inv(H)
cam = camera_from_calib()
det = YOLO(pkg_file("yolo11m-pose.pt")); pose = RTMPose(rtm_model(), model_input_size=(192, 256))
TILES = [tuple(C["roi"])] + [tuple(t) for t in C["far_tiles"]]
import os
if not os.path.exists("positions.csv") or os.path.getsize("positions.csv") == 0:
    sys.exit("positions.csv is empty: the vb_tracker step did not finish. Re-run with --redo vb_tracker")
pos = pd.read_csv("positions.csv")
events = pd.read_csv("ball_events.csv")
if len(events) == 0:
    pd.DataFrame(columns=["frame", "t_s", "ball_u", "ball_v", "ball_x", "ball_y", "ball_z", "dist_norm", "kind", "team", "track_id", "player_x", "player_y"]).to_csv("touches.csv", index=False)
    print("no ball events -> empty touches.csv"); sys.exit(0)
track = pd.read_csv("ball_track.csv").set_index("frame")

def img2court(u, v):
    p = Hinv @ [u, v, 1.]; return p[:2] / p[2]

def people(frame):
    B = []
    for (x0, y0, x1, y1) in TILES:
        r = det.predict(frame[y0:y1, x0:x1], imgsz=1280, conf=0.25, classes=[0], verbose=False)[0]
        for b in r.boxes.xyxy.cpu().numpy():
            if (x0 > 0 and b[0] < 3) or (y1 < frame.shape[0] and b[3] > y1 - y0 - 3): continue
            B.append(b + [x0, y0, x0, y0])
    if not B: return [], [], []
    keep = np.array(cv2.dnn.NMSBoxes([[*b[:2], *(b[2:] - b[:2])] for b in B], [1.] * len(B), 0, .5)).flatten()
    B = [B[i] for i in keep]
    K, S = pose(frame, bboxes=[b.tolist() for b in B])
    return B, K, S

def seg_dist(p, a, b):
    ab = b - a; t = np.clip((p - a) @ ab / (ab @ ab + 1e-9), 0, 1); return np.linalg.norm(p - (a + t * ab))

def contact_dist(ball, k, s):
    d = []
    for e, w in ((7, 9), (8, 10)):                 # left/right elbow->wrist
        if s[w] > 0.3:
            hand = k[w] + 0.35 * (k[w] - k[e]) if s[e] > 0.3 else k[w]
            d.append(np.linalg.norm(ball - hand))
            if s[e] > 0.3: d.append(seg_dist(ball, k[e], hand))
    if s[0] > 0.3: d.append(np.linalg.norm(ball - k[0]))
    return min(d) if d else np.inf

cap, fps = open_video(VIDEO)
rows, sheet = [], []
for _, e in events.iterrows():
    best = (np.inf, None)
    for df in (-2, 0, 2):
        fr = int(e.frame) + df
        if fr not in track.index: continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, fr); ok, f = cap.read()
        if not ok: continue
        f = prep(f)
        ball = track.loc[fr, ["u", "v"]].values.astype(float)
        B, K, S = people(f)
        for b, k, s in zip(B, K, S):
            h = b[3] - b[1]; dn = contact_dist(ball, k, s) / h
            if dn < best[0]: best = (dn, dict(frame=fr, box=b, kps=k, sc=s, img=f, ball=ball))
    bx, by, bz = ball_3d(cam, e.u, e.v, e.size_px)
    row = dict(frame=int(e.frame), t_s=e.t_s, ball_u=round(e.u), ball_v=round(e.v),
               ball_x=round(bx, 2), ball_y=round(by, 2), ball_z=round(bz, 2), dist_norm=round(best[0], 3))
    if best[0] < TOUCH_R:
        b, k, s = best[1]["box"], best[1]["kps"], best[1]["sc"]
        ank = [k[j] for j in (15, 16) if s[j] > 0.3]
        fu, fv = (np.mean([a[0] for a in ank]), max(a[1] for a in ank)) if ank else ((b[0] + b[2]) / 2, b[3])
        cx, cy = img2court(fu, fv)
        near = pos[(pos.frame - best[1]["frame"]).abs() <= max(3, round(0.1 * fps))]
        tid = None
        if len(near):
            dd = np.hypot(near.x_m - cx, near.y_m - cy)
            if dd.min() < 1.2: tid = int(near.iloc[int(np.argmin(dd.values))].track_id)
        team = "near" if cy < 9 else "far"
        # block: at the net (<1.2 m) with both wrists above the head
        wr_up = all(s[j] > .3 and k[j][1] < k[0][1] for j in (9, 10)) if s[0] > .3 else False
        kind = "block" if abs(cy - 9) < 1.2 and wr_up else "touch"
        row.update(kind=kind, team=team, track_id=tid, player_x=round(cx, 2), player_y=round(cy, 2))
    else:
        row.update(kind="bounce" if bz < 0.6 else "unattributed", team=None, track_id=None)
    rows.append(row)
    # evidence crop
    if best[1] is not None:
        im = best[1]["img"].copy(); u, v = map(int, best[1]["ball"])
        if best[0] < TOUCH_R:
            k, s = best[1]["kps"], best[1]["sc"]
            for a_, b_ in [(5,7),(7,9),(6,8),(8,10),(5,6),(5,11),(6,12),(11,13),(13,15),(12,14),(14,16)]:
                if s[a_] > .3 and s[b_] > .3: cv2.line(im, tuple(map(int, k[a_])), tuple(map(int, k[b_])), (255, 0, 255), 2)
        cv2.circle(im, (u, v), 14, (0, 255, 0), 2)
        x0, y0 = max(0, u - 160), max(0, v - 160)
        c = cv2.resize(im[y0:y0 + 320, x0:x0 + 320], (240, 240))
        cv2.putText(c, f"{e.t_s:.1f}s {row['kind']}", (5, 20), 0, 0.6, (0, 0, 255), 2)
        cv2.putText(c, f"#{row.get('track_id')} {row.get('team') or ''} d={best[0]:.2f}", (5, 42), 0, 0.5, (0, 0, 255), 2)
        sheet.append(c)
    print(row, flush=True)

T = pd.DataFrame(rows)
# ---------------- rally logic ----------------
fouls, rally, cnt, last_team, last_pid, prev_block, prev_t = [], 0, 0, None, None, False, -99
T["rally"] = 0; T["team_touch_no"] = None; T["flag"] = ""
for i, r in T.iterrows():
    if r.t_s - prev_t > 4.0 or r.kind == "bounce":          # dead ball / new rally
        if r.kind == "bounce": T.at[i, "rally"] = rally; prev_t = r.t_s; rally += 1; cnt, last_team, last_pid = 0, None, None; continue
        rally += 1; cnt, last_team, last_pid, prev_block = 0, None, None, False
    T.at[i, "rally"] = rally; prev_t = r.t_s
    if r.kind not in ("touch", "block"): continue
    if r.team != last_team: cnt, last_pid, prev_block = 0, None, False
    if r.kind == "block": prev_block = True; last_team = r.team; last_pid = r.track_id; T.at[i, "team_touch_no"] = "block"; continue
    cnt += 1; T.at[i, "team_touch_no"] = cnt
    if cnt > 3: T.at[i, "flag"] = "FOUR_TOUCHES"
    if r.track_id is not None and r.track_id == last_pid and not prev_block: T.at[i, "flag"] += " DOUBLE_CONTACT"
    last_team, last_pid, prev_block = r.team, r.track_id, False
T.to_csv("touches.csv", index=False)
print(T[["t_s", "rally", "kind", "team", "track_id", "team_touch_no", "flag", "dist_norm", "ball_z"]].to_string())
if sheet:
    while len(sheet) % 5: sheet.append(np.zeros_like(sheet[0]))
    cv2.imwrite("touch_evidence.jpg", np.vstack([np.hstack(sheet[i:i + 5]) for i in range(0, len(sheet), 5)]))
