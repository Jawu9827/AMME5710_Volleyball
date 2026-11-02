"""Overlay ball track + contacts + rally/touch counter onto the player-tracking demo video."""
import cv2, json, numpy as np, pandas as pd
from common import run_args
a = run_args()
SRC, OUT = "tracking.mp4", "touches_overlay.mp4"
FPS = a.cfg["fps"]
SRC_FRAMES = [int(l) for l in open("tracking_frames.txt")]   # source frame index of each tracking.mp4 frame
IMG_H = json.load(open("calib.json")).get("image_size", [1920, 1080])[1]
import os, pickle
PRED = pickle.load(open("landing_pred.pkl", "rb")) if os.path.exists("landing_pred.pkl") else {}
LAND = pd.read_csv("landings.csv") if os.path.exists("landings.csv") and os.path.getsize("landings.csv") > 1 else pd.DataFrame()
PK = np.array(sorted(PRED))
trk = pd.read_csv("ball_track.csv").set_index("frame")
R = pd.read_csv("rallies.csv"); pos = pd.read_csv("positions.csv")
live = R[R.kind.isin(["touch", "block"])].copy()
def n(x):
    try: return str(int(float(x)))
    except (TypeError, ValueError): return str(x)
S, PAD, MH = 26, 40, 600; MS = 720 / MH
def m2px(x, y): return (int(1280 + (PAD + S + x * S) * MS), int((PAD + S + (18 - y) * S) * MS))
sc = 1280 / 1920; sy = 720 / IMG_H
cap = cv2.VideoCapture(SRC); W = int(cap.get(3)); Hh = int(cap.get(4))
wr = cv2.VideoWriter(OUT, cv2.VideoWriter_fourcc(*"mp4v"), cap.get(cv2.CAP_PROP_FPS) or 10, (W, Hh))
k = 0
while True:
    ok, f = cap.read()
    if not ok or k >= len(SRC_FRAMES): break
    fr = SRC_FRAMES[k]
    # ball trail (0.5 s at full rate)
    tr = trk.loc[(trk.index > fr - round(0.5 * FPS)) & (trk.index <= fr)]
    pts = (tr[["u", "v"]].values * [sc, sy]).astype(np.int32)
    for i in range(1, len(pts)):
        if tr.index[i] - tr.index[i - 1] <= max(1, round(3 * FPS / 59.94)):
            cv2.line(f, tuple(pts[i - 1]), tuple(pts[i]), (0, 255, 255), 2, cv2.LINE_AA)
    if fr in trk.index or (fr - 1) in trk.index:
        r = trk.loc[fr if fr in trk.index else fr - 1]
        cv2.circle(f, (int(r.u * sc), int(r.v * sy)), 10, (0, 255, 255), 2, cv2.LINE_AA)
    # current rally state
    past = R[R.frame <= fr]
    lastc = live[live.frame <= fr]
    hud = "no rally"
    if len(lastc):
        c = lastc.iloc[-1]
        ended = len(past) and past.iloc[-1].kind in ("bounce", "dead_ball_handling") and past.iloc[-1].frame > c.frame
        hud = f"RALLY {int(c.rally)}  |  {c.team.upper()} touch {n(c.team_touch)}/3" + ("  |  ball dead" if ended else "")
        if c.flag and isinstance(c.flag, str) and c.flag.strip(): hud += f"  |  FOUL: {c.flag.strip()}"
    cv2.rectangle(f, (0, 0), (1280, 44), (25, 25, 25), -1)
    cv2.putText(f, hud, (14, 31), 0, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    # flash recent contact / bounce (1 s)
    rec = R[(R.frame <= fr) & (R.frame > fr - round(FPS))]
    for _, c in rec.iterrows():
        u, v = trk.loc[c.frame, ["u", "v"]] if c.frame in trk.index else (None, None)
        if u is None: continue
        p = (int(u * sc), int(v * sy))
        if c.kind == "touch":
            lab, col = f"TOUCH {n(c.team_touch)}  #{'' if pd.isna(c.track_id) else int(c.track_id)}", (0, 255, 0)
        elif c.kind == "block": lab, col = "BLOCK", (255, 0, 255)
        elif c.kind == "bounce": lab, col = "floor", (0, 0, 255)
        elif c.kind == "dead_ball_handling": lab, col = "dead ball", (128, 128, 255)
        else: continue
        cv2.circle(f, p, 22, col, 3, cv2.LINE_AA)
        cv2.putText(f, lab, (p[0] + 24, p[1] - 10), 0, 0.75, col, 2, cv2.LINE_AA)
    # minimap: contact markers of the current rally
    if len(lastc):
        cur = lastc[lastc.rally == lastc.iloc[-1].rally]
        for _, c in cur.iterrows():
            near = pos[(pos.frame - c.frame).abs() <= round(0.1 * FPS)]
            if pd.isna(c.track_id): continue
            pr = near[near.track_id == c.track_id]
            if not len(pr): continue
            q = m2px(pr.iloc[0].x_m, pr.iloc[0].y_m)
            cv2.circle(f, q, 15, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(f, n(c.team_touch), (q[0] - 6, q[1] - 18), 0, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
    # ---- predicted landing while the ball is in the air ----
    if len(PK):
        j = PK[PK <= fr]
        if len(j) and PRED[int(j[-1])]["until"] >= fr and fr - j[-1] <= round(0.2 * FPS):
            P = PRED[int(j[-1])]; col = (60, 200, 60) if P["call"].startswith("IN") else (60, 60, 230)
            pts = (P["path"] * [sc, sy]).astype(np.int32)
            for i in range(0, len(pts) - 1, 2): cv2.line(f, tuple(pts[i]), tuple(pts[i + 1]), (255, 0, 255), 2, cv2.LINE_AA)
            lp = (int(P["land_px"][0] * sc), int(P["land_px"][1] * sy))
            cv2.drawMarker(f, lp, (255, 0, 255), cv2.MARKER_TILTED_CROSS, 18, 3)
            cv2.putText(f, f"predicted: {P['call']}", (lp[0] + 14, lp[1] + 26), 0, 0.7, (255, 0, 255), 2, cv2.LINE_AA)
            q = m2px(*P["land"]); cv2.drawMarker(f, q, (255, 0, 255), cv2.MARKER_TILTED_CROSS, 16, 2)
    # ---- measured line call for 1.5 s after the bounce ----
    for _, l in LAND.iterrows():
        if 0 <= fr - l.frame <= round(1.5 * FPS):
            col = (60, 200, 60) if str(l.call).startswith("IN") else (60, 60, 230)
            q = m2px(l.x_m, l.y_m); cv2.circle(f, q, 10, col, -1); cv2.circle(f, q, 10, (255, 255, 255), 2)
            cv2.rectangle(f, (930, 52), (1270, 112), (25, 25, 25), -1)
            cv2.putText(f, f"{l.call}  ({l.margin_m:+.2f} m)", (944, 94), 0, 1.0, col, 3, cv2.LINE_AA)
    wr.write(f); k += 1
wr.release(); print(f"touches_overlay.mp4: {k} frames")
