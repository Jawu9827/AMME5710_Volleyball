"""
Rally segmentation + rule checks (runs on cached outputs, no detection needed).

Inputs : ball_track.csv, touches.csv (attribution from touches.py), flights.csv
Outputs: rallies.csv (one row per contact with rally / count / flags) and a printed summary

Ball state per frame (size-corrected monocular 3-D):
  dead  = ball low (z < 0.35 m), slow, floor-consistent size, for >= 0.4 s -> on floor / rolling
  bounce= impulse event, nobody in reach, floor-consistent        -> ball hit the floor
  lost  = untracked for > 1.5 s                                       -> rally over (or out of view)
A rally runs from its first live touch to the first dead/lost state.
Counting is only CERTAIN when the ball was continuously tracked between consecutive touches
(gap <= 0.35 s); otherwise the count is marked uncertain and no foul is raised.
"""
import numpy as np, pandas as pd
from scipy.signal import savgol_filter
from camera_model import camera_from_calib, ball_3d
from common import run_args
a = run_args()
FPS = a.cfg["fps"]; FR = 59.94 / FPS
cam = camera_from_calib()
trk = pd.read_csv("ball_track.csv"); T = pd.read_csv("touches.csv"); F = pd.read_csv("flights.csv")
if len(trk) < 10 or len(T) == 0:
    pd.DataFrame(columns=["t_s", "frame", "rally", "kind", "team", "track_id", "ball_z", "team_touch", "certain", "crossed_by_flight", "flag"]).to_csv("rallies.csv", index=False)
    print("not enough ball data for rally logic -> empty rallies.csv"); raise SystemExit(0)

# ---- per-frame ball state ----
z = np.array([ball_3d(cam, r.u, r.v, r.size_px)[2] for r in trk.itertuples()])
trk["z"] = savgol_filter(z, 9, 1) if len(z) > 9 else z
sp = np.r_[0, np.hypot(np.diff(trk.u), np.diff(trk.v))]; sp[np.r_[True, np.diff(trk.frame) != 1]] = 0
trk["speed"] = savgol_filter(sp, 9, 1) if len(sp) > 9 else sp
# floor-consistency: treat the ball's lowest pixel as a floor contact, map it through the court
# homography, and check the apparent size matches a 21 cm ball at THAT floor point's depth
import json, cv2
from camera_model import SIZE_CORR, BALL_D
Cj = json.load(open("calib.json"))
Hm = cv2.findHomography(np.float32(Cj["world_m"]), np.float32(Cj["image_px"]))[0]; Hinv = np.linalg.inv(Hm)
def floor_ok(u, v, d):
    p = Hinv @ [u, v + d / 2, 1.]; x, y = p[:2] / p[2]
    if not (-3 <= x <= 12 and -4 <= y <= 22): return False
    Zc = (cam["R"] @ np.array([x, y, BALL_D / 2]) + cam["t"])[2]          # depth of a ball resting there
    ratio = d * SIZE_CORR / (cam["f"] * BALL_D / Zc)
    return 0.7 <= ratio <= 1.4
trk["floor_ok"] = [floor_ok(r.u, r.v, r.size_px) for r in trk.itertuples()]
low = (trk.z < 0.35) & (trk.speed < 6 * FR) & trk.floor_ok
run = (low != low.shift()).cumsum()
trk["dead"] = low & low.groupby(run).transform("size").ge(int(0.4 * FPS))   # rolling/resting >= 0.4 s
dead_frames = trk.loc[trk.dead, "frame"].values
frames = trk.frame.values

fl_ok = trk.set_index("frame").floor_ok
bounce_frames = np.array([r.frame for r in T.itertuples() if r.kind == "bounce" and r.ball_z < 0.6
                          and fl_ok.get(r.frame, False)])               # floor-consistent impulse, nobody in reach
def dead_between(f0, f1):
    return np.any((dead_frames > f0) & (dead_frames < f1)) or np.any((bounce_frames > f0) & (bounce_frames < f1))
def max_gap(f0, f1):
    fs = frames[(frames >= f0) & (frames <= f1)]
    if len(fs) < 2: return (f1 - f0) / FPS
    return max(np.diff(np.r_[f0, fs, f1])) / FPS
def is_dead_at(f): return np.any(np.abs(dead_frames - f) <= max(1, round(3 / FR)))

# re-evaluate contact type with the corrected height + dead-ball state
for i, r in T.iterrows():
    T.at[i, "ball_z"] = round(float(ball_3d(cam, r.ball_u, r.ball_v, trk.set_index("frame").size_px.get(r.frame, 12))[2]), 2)
    # dead-ball handling = the ball was ALREADY on the floor just before this contact
    # (height alone is unreliable when the ball overlaps a player, so test the history instead)
    if r.kind in ("touch", "block") and np.any((dead_frames > r.frame - round(0.75 * FPS)) & (dead_frames < r.frame - max(1, round(2 / FR)))):
        T.at[i, "kind"] = "dead_ball_handling"

rows, rally, prev = [], 0, None
cnt, team, pid, after_block, certain = 0, None, None, False, True
for _, r in T.sort_values("frame").iterrows():
    live = r.kind in ("touch", "block")
    new_rally = rally == 0 or prev is None or dead_between(prev.frame, r.frame) or (r.frame - prev.frame) / FPS > 4.0
    if new_rally and live:
        rally += 1; cnt, team, pid, after_block, certain = 0, None, None, False, True
    out = dict(t_s=r.t_s, frame=r.frame, rally=rally if live else None, kind=r.kind, team=r.team,
               track_id=None if pd.isna(r.track_id) else int(r.track_id), ball_z=r.ball_z,
               team_touch=None, certain=None, crossed_by_flight=None, flag="")
    if live:
        gap_ok = prev is None or new_rally or max_gap(prev.frame, r.frame) <= 0.35
        fl = F[(F.f0 >= (prev.frame if prev is not None else 0)) & (F.f1 <= r.frame) & F.valid]
        crossed = bool(fl.crosses_net.any()) if len(fl) else None
        out["crossed_by_flight"] = crossed
        if r.team != team:                                   # possession changed -> ball crossed the net
            cnt, pid, after_block, certain = 0, None, False, True
        elif crossed:                                        # same team but ball went over and back unseen
            cnt, pid, after_block, certain = 0, None, False, False
        if not gap_ok: certain = False
        if r.kind == "block":
            out["team_touch"] = "block (not counted)"; after_block = True
        else:
            cnt += 1; out["team_touch"] = cnt
            if certain and cnt > 3: out["flag"] += "FOUR_TOUCHES "
            if certain and pid is not None and out["track_id"] == pid and not after_block: out["flag"] += "DOUBLE_CONTACT "
            after_block = False
        out["certain"] = certain
        team, pid = r.team, out["track_id"]
        prev = r
    else:
        if r.kind != "dead_ball_handling": prev = r if prev is None else prev
    rows.append(out)
R = pd.DataFrame(rows); R.to_csv("rallies.csv", index=False)
print(R.to_string())
print(f"\ndead-ball frames: {trk.dead.sum()}  |  rallies: {rally}  |  live contacts: {(R.kind.isin(['touch','block'])).sum()}"
      f"  |  fouls flagged: {(R.flag != '').sum()}")
