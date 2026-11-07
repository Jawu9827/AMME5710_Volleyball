"""One video with everything: both synced cameras and a court map showing players, the 3D ball,
contacts and who made them, the rally / touch count and fouls, the live landing prediction, and
the line call.

Usage (after the other steps for the same window; run_pipeline.py runs them all):
    python scripts/render_full.py results/ball/game2_A_game2_B_sdr_ball_from30.json
    --players <csv>    players_on_court.py output (default: same videos and start time)
    --show             watch while rendering (q quit, space pause)

Adapted from render_overlay.py in volleyball_cv_inout/ (rally HUD, contact flashes, predicted
landing). Inputs, all looked up from the ball run's file names:
    results/ball/<stem>.csv                        ball 2D in each camera and 3D
    results/players/<A>_<B>_players_from<s>.csv    player court positions and track IDs
    results/touches/<stem>_touches.csv             contacts, teams, hit numbers, flags, bounces
    results/landing/<stem>_landing_pred.csv        predicted landing during each flight
Any missing input is skipped. Players are drawn from their court position (a box 1.8 m tall,
projected into each camera), so the detector doesn't run again.

Output: results/full/<A>_<B>_full_from<start>.mp4
  camera views: court lines (magenta), players (orange = camera A's half, green = B's, with track
  ID), ball (2D detection = cyan circle, 3D = magenta cross with its floor shadow), contacts
  (ring + "A2 #5"), predicted path and landing spot (dashed, "pred IN"), the measured bounce.
  map: the same from above, the ball trail coloured by height, the rally's contacts numbered.
  top bar: rally number, team and hit number (e.g. "A hit 2/3"), fouls, point after a bounce.
"""
import argparse
import csv
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from calibrate_camera import COURT_LINES, ffmpeg_exe  # noqa: E402
from camera_3d import load_camera  # noqa: E402
from line_calls import G  # noqa: E402

NET_Y = 9.0
SIDE_COL = {"A": (0, 165, 255), "B": (80, 220, 80), None: (200, 200, 200)}   # BGR
TRAIL_S = 1.0
CONTACT_SHOW_S = 1.0
CALL_SHOW_S = 2.0
PLAYER_HOLD_S = 0.2     # players are detected every 3rd frame; hold positions this long


def height_colour(z):
    v = np.uint8([[[int(np.clip(z / 6.0, 0, 1) * 255)]]])
    return tuple(int(c) for c in cv2.applyColorMap(v, cv2.COLORMAP_TURBO)[0, 0])


def fnum(v):
    return None if v in ("", None) else float(v)


# ---------------------------------------------------------------- inputs
def load_ball(path):
    rows = []
    for r in csv.DictReader(open(path)):
        a = (float(r["a_u"]), float(r["a_v"])) if r["a_u"] else None
        b = (float(r["b_u"]), float(r["b_v"])) if r["b_u"] else None
        X = np.array([float(r["x_m"]), float(r["y_m"]), float(r["z_m"])]) if r["x_m"] else None
        rows.append(dict(t=float(r["time_a_s"]), a=a, a_meas=r["a_state"] == "measured", b=b,
                         b_meas=r["b_measured"] == "1", X=X, status=r["status"]))
    return rows


def load_players(path):
    by_t = defaultdict(list)
    if path and Path(path).exists():
        for r in csv.DictReader(open(path)):
            by_t[float(r["time_s"])].append(dict(id=int(r["track_id"]), x=float(r["x_m"]), y=float(r["y_m"]),
                                                 side=r["side"]))
    return np.array(sorted(by_t)), by_t


def load_events(path):
    if not path.exists():
        return []
    out = []
    for r in csv.DictReader(open(path)):
        out.append(dict(t=float(r["time_a_s"]), kind=r["kind"], detail=r["detail"], x=float(r["x_m"]),
                        y=float(r["y_m"]), z=float(r["z_m"]), team=r["team"] or None,
                        track=r["track_id"] or None, rally=int(r["rally"]) if r["rally"] else None,
                        hit=r["team_hit"], certain=r["certain"], flags=r["flags"], call=r["call"],
                        point_to=r["point_to"]))
    return out


def load_preds(path):
    if not path.exists():
        return []
    out = []
    for r in csv.DictReader(open(path)):
        out.append(dict(t=float(r["time_a_s"]), t_land=float(r["t_land_s"]), x=float(r["x_m"]), y=float(r["y_m"]),
                        call=r["call"], close=r["close"] == "1", t_ref=float(r["t_ref"]),
                        p=np.array([[float(r["x0"]), float(r["vx"])], [float(r["y0"]), float(r["vy"])],
                                    [float(r["z0"]), float(r["vz"])]]), bounce=int(r["bounce"])))
    return out


def arc_points(pred, t0, n=24):
    s = np.linspace(t0, pred["t_land"], n) - pred["t_ref"]
    p = pred["p"]
    return np.c_[p[0, 0] + p[0, 1] * s, p[1, 0] + p[1, 1] * s, p[2, 0] + p[2, 1] * s - 0.5 * G * s ** 2]


def contact_label(e):
    if e["detail"] == "serve":
        lab = f"{e['team']} serve"
    elif e["detail"] == "block":
        lab = f"{e['team']} BLOCK"
    elif e["team"]:
        lab = f"{e['team']}{e['hit']}"
    else:
        lab = "?"
    if e["track"]:
        lab += f" #{e['track']}"
    return lab


def flagged(e):
    return any(f in e["flags"] for f in ("FOUR_TOUCHES", "DOUBLE_CONTACT"))


# ---------------------------------------------------------------- state at time t
class State:
    def __init__(self, ball, players, events, preds):
        self.ball, (self.pt, self.pby), self.events, self.preds = ball, players, events, preds
        self.floors = [e for e in events if e["kind"] == "floor"]

    def players(self, t):
        if not len(self.pt):
            return []
        i = np.searchsorted(self.pt, t + 1e-6) - 1
        return self.pby[self.pt[i]] if i >= 0 and t - self.pt[i] <= PLAYER_HOLD_S else []

    def recent_contacts(self, t):
        return [e for e in self.events if e["kind"] == "contact" and 0 <= t - e["t"] <= CONTACT_SHOW_S]

    def recent_calls(self, t):
        return [e for e in self.floors if 0 <= t - e["t"] <= CALL_SHOW_S]

    def prediction(self, t):
        best = None
        for p in self.preds:
            if p["t"] <= t < p["t_land"] and t - p["t"] <= 0.2:
                best = p
        return best

    def rally_contacts(self, t):
        past = [e for e in self.events if e["t"] <= t]
        cs = [e for e in past if e["kind"] == "contact" and e["rally"] is not None]
        if not cs:
            return []
        r = cs[-1]["rally"]
        return [e for e in cs if e["rally"] == r]

    def hud(self, t):
        past = [e for e in self.events if e["t"] <= t]
        cs = [e for e in past if e["kind"] == "contact"]
        if not cs:
            return "waiting for the first contact", None
        last = cs[-1]
        after = [e for e in past if e["kind"] == "floor" and e["t"] > last["t"]]
        rally = f"RALLY {last['rally']}" if last["rally"] is not None else "between rallies"
        if after:
            f = after[-1]
            txt = f"{rally}  |  ball down {f['call']}" + (f"  |  point to {f['point_to']}" if f["point_to"] else "")
            return txt, (60, 200, 60) if f["call"] == "IN" else (60, 60, 230)
        if last["detail"] == "serve":
            txt = f"{rally}  |  {last['team']} serve"
        elif last["detail"] == "block":
            txt = f"{rally}  |  {last['team']} block (not counted)"
        elif last["team"]:
            txt = f"{rally}  |  {last['team']} hit {last['hit']}/3"
        else:
            txt = f"{rally}  |  contact, toucher not found"
        if last["certain"] == "False":
            txt += " (count uncertain)"
        fouls = [e["flags"] for e in cs if e["rally"] == last["rally"] and flagged(e)]
        if fouls:
            txt += "  |  FOUL: " + fouls[-1].replace("TEAM_GUESSED", "").replace("NO_TOUCHER", "").strip()
            return txt, (60, 60, 230)
        return txt, None


# ---------------------------------------------------------------- drawing
def put(img, txt, org, scale, col, thick=1):
    """Text on a dark box (readable on any background)."""
    (w, h), base = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    x, y = int(org[0]), int(org[1])
    x0, y0, x1, y1 = max(0, x - 2), max(0, y - h - 3), min(img.shape[1], x + w + 2), min(img.shape[0], y + base)
    if x1 > x0 and y1 > y0:
        img[y0:y1, x0:x1] = (img[y0:y1, x0:x1] * 0.35).astype(np.uint8)
    cv2.putText(img, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, col, thick, cv2.LINE_AA)


def draw_camera(img, cam, name, st, k, t, s, label):
    """img is already resized by s; cam projects to full-resolution pixels."""
    P = lambda X: cam.project(np.asarray(X, float).reshape(-1, 3)) * s
    ip = lambda q: tuple(int(round(v)) for v in q)
    for a, b in COURT_LINES:
        u = np.linspace(0, 1, 40)[:, None]
        pts = P(np.c_[np.asarray(a) * (1 - u) + np.asarray(b) * u, np.zeros(40)])
        cv2.polylines(img, [np.int32(np.round(pts))], False, (255, 0, 255), 1, cv2.LINE_AA)
    for pl in st.players(t):
        foot, head = P([pl["x"], pl["y"], 0.0])[0], P([pl["x"], pl["y"], 1.8])[0]
        hh = foot[1] - head[1]
        if hh < 6 or not (0 <= foot[0] < img.shape[1]):
            continue
        col = SIDE_COL.get(pl["side"], SIDE_COL[None])
        cv2.rectangle(img, ip((foot[0] - 0.22 * hh, head[1])), ip((foot[0] + 0.22 * hh, foot[1])), col, 1, cv2.LINE_AA)
        put(img, f"#{pl['id']}", ip((foot[0] - 0.22 * hh, head[1] - 3)), 0.4, col)
    key, meas = ("a", "a_meas") if name == "A" else ("b", "b_meas")
    trail = [r[key] for r in st.ball[max(0, k - 60):k + 1] if r[key] is not None and t - r["t"] <= TRAIL_S]
    for p0, p1 in zip(trail, trail[1:]):
        cv2.line(img, ip(np.multiply(p0, s)), ip(np.multiply(p1, s)), (255, 255, 0), 1, cv2.LINE_AA)
    r = st.ball[k]
    if r[key] is not None:
        cv2.circle(img, ip(np.multiply(r[key], s)), 9, (255, 220, 0) if r[meas] else (0, 170, 255), 2, cv2.LINE_AA)
    if r["X"] is not None:
        q, g = P(r["X"])[0], P([r["X"][0], r["X"][1], 0.0])[0]
        cv2.drawMarker(img, ip(q), (255, 0, 255), cv2.MARKER_CROSS, 16, 2)
        cv2.line(img, ip(q), ip(g), (255, 0, 255), 1, cv2.LINE_AA)
        cv2.circle(img, ip(g), 3, (255, 0, 255), -1)
    pred = st.prediction(t)
    if pred is not None:
        pts = P(arc_points(pred, t))
        for i in range(0, len(pts) - 1, 2):
            cv2.line(img, ip(pts[i]), ip(pts[i + 1]), (255, 0, 255), 2, cv2.LINE_AA)
        lp = P([pred["x"], pred["y"], 0.0])[0]
        col = (60, 200, 60) if pred["call"] == "IN" else (60, 60, 230)
        cv2.drawMarker(img, ip(lp), col, cv2.MARKER_TILTED_CROSS, 16, 3)
        put(img, f"pred {pred['call']}" + (" (close)" if pred["close"] else ""), ip(lp + [10, 18]), 0.5, col)
    for e in st.recent_contacts(t):
        c = P([e["x"], e["y"], e["z"]])[0]
        col = (60, 60, 230) if flagged(e) else SIDE_COL.get(e["team"], SIDE_COL[None])
        cv2.circle(img, ip(c), 16, col, 2, cv2.LINE_AA)
        put(img, contact_label(e), ip(c + [18, -8]), 0.5, col)
    for e in st.recent_calls(t):
        c = P([e["x"], e["y"], 0.0])[0]
        col = (60, 200, 60) if e["call"] == "IN" else (60, 60, 230)
        cv2.circle(img, ip(c), 10, col, -1, cv2.LINE_AA)
        cv2.circle(img, ip(c), 10, (255, 255, 255), 1, cv2.LINE_AA)
        put(img, e["call"], ip(c + [14, 6]), 0.7, col, 2)
    put(img, label, (8, img.shape[0] - 10), 0.55, (255, 255, 255))
    return img


def draw_map(height, st, k, t):
    mx, my = 1.5, 3.0
    ppm = height / (18 + 2 * my)
    w = int(round((9 + 2 * mx) * ppm)) // 2 * 2
    img = np.full((height, w, 3), (90, 60, 40), np.uint8)
    P = lambda x, y: (int(round((x + mx) * ppm)), int(round((y + my) * ppm)))
    cv2.rectangle(img, P(0, 0), P(9, 18), (170, 140, 110), -1)
    for a, b in COURT_LINES:
        cv2.line(img, P(*a), P(*b), (0, 230, 255), 2, cv2.LINE_AA)
    cv2.line(img, P(-1, NET_Y), P(10, NET_Y), (255, 255, 255), 3, cv2.LINE_AA)
    put(img, "B", P(-1.3, 1.0), 0.5, SIDE_COL["B"])
    put(img, "A", P(-1.3, 17.5), 0.5, SIDE_COL["A"])
    for pl in st.players(t):
        col = SIDE_COL.get(pl["side"], SIDE_COL[None])
        cv2.circle(img, P(pl["x"], pl["y"]), max(4, int(0.25 * ppm)), col, -1, cv2.LINE_AA)
        put(img, str(pl["id"]), (P(pl["x"], pl["y"])[0] + 6, P(pl["x"], pl["y"])[1] - 4), 0.35, (255, 255, 255))
    trail = [r for r in st.ball[max(0, k - 60):k + 1] if r["X"] is not None and t - r["t"] <= 1.5]
    for r0, r1 in zip(trail, trail[1:]):
        if r1["t"] - r0["t"] <= 0.2:
            cv2.line(img, P(*r0["X"][:2]), P(*r1["X"][:2]), height_colour(r1["X"][2]), 2, cv2.LINE_AA)
    r = st.ball[k]
    if r["X"] is not None:
        cv2.circle(img, P(*r["X"][:2]), 6, height_colour(r["X"][2]), -1, cv2.LINE_AA)
        cv2.circle(img, P(*r["X"][:2]), 7, (255, 255, 255), 1, cv2.LINE_AA)
        put(img, f"ball {r['X'][2]:.1f} m up", (4, height - 8), 0.4, (255, 255, 255))
    for e in st.rally_contacts(t):
        col = (60, 60, 230) if flagged(e) else SIDE_COL.get(e["team"], SIDE_COL[None])
        cv2.circle(img, P(e["x"], e["y"]), 5, col, 2, cv2.LINE_AA)
        lab = "S" if e["detail"] == "serve" else ("BL" if e["detail"] == "block" else str(e["hit"] or "?"))
        put(img, lab, (P(e["x"], e["y"])[0] - 4, P(e["x"], e["y"])[1] - 8), 0.4, col)
    pred = st.prediction(t)
    if pred is not None:
        col = (60, 200, 60) if pred["call"] == "IN" else (60, 60, 230)
        cv2.drawMarker(img, P(pred["x"], pred["y"]), col, cv2.MARKER_TILTED_CROSS, 14, 2)
    for e in st.recent_calls(t):
        col = (60, 200, 60) if e["call"] == "IN" else (60, 60, 230)
        cv2.circle(img, P(e["x"], e["y"]), 6, col, -1, cv2.LINE_AA)
        cv2.circle(img, P(e["x"], e["y"]), 7, (255, 255, 255), 1, cv2.LINE_AA)
    return img


# ---------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser()
    p.add_argument("run_json", help="the .json written by ball_two_cameras.py")
    p.add_argument("--players", help="players_on_court.py .csv (default: matching file name)")
    p.add_argument("--width", type=int, default=960, help="width of each camera view")
    p.add_argument("--show", action="store_true")
    args = p.parse_args()
    run = json.loads(Path(args.run_json).read_text())
    stem = Path(run["csv"]).stem
    va, vb = Path(run["video_a"]).stem, Path(run["video_b"]).stem
    players_csv = args.players or f"results/players/{va}_{vb}_players_from{run['start']:g}.csv"
    ball = load_ball(run["csv"])
    st = State(ball, load_players(players_csv), load_events(Path("results/touches") / f"{stem}_touches.csv"),
               load_preds(Path("results/landing") / f"{stem}_landing_pred.csv"))
    print(f"Ball: {len(ball)} frames; players: {len(st.pt)} detection steps"
          + ("" if Path(players_csv).exists() else f" ({players_csv} not found)")
          + f"; events: {len(st.events)}; landing predictions: {len(st.preds)}")
    cams = [load_camera(run["calib_a"])[0], load_camera(run["calib_b"])[0]]
    offset = run["offset_s"]

    ca, cb = cv2.VideoCapture(run["video_a"]), cv2.VideoCapture(run["video_b"])
    fa, fb = ca.get(cv2.CAP_PROP_FPS), cb.get(cv2.CAP_PROP_FPS)
    ia0 = int(round(ball[0]["t"] * fa))
    for _ in range(ia0):
        ca.grab()
    out = Path("results/full")
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{va}_{vb}_full_from{run['start']:g}.mp4"
    ib, last_b, ff, paused = 0, None, None, False
    for k, r in enumerate(ball):
        ok, a = ca.read()
        if not ok:
            break
        target = int(round((r["t"] + offset) * fb))
        while ib < target and cb.grab():
            ib += 1
        if ib == target:
            okb, fr = cb.read()
            if okb:
                last_b, ib = fr, ib + 1
        if last_b is None:
            continue
        views = []
        for img, cam, name, tt in ((a, cams[0], "A", r["t"]), (last_b, cams[1], "B", r["t"] + offset)):
            s = args.width / img.shape[1]
            v = cv2.resize(img, (args.width, int(round(img.shape[0] * s))), interpolation=cv2.INTER_AREA)
            views.append(draw_camera(v, cam, name, st, k, r["t"], s, f"camera {name}  t={tt:6.2f}s"))
        h = max(v.shape[0] for v in views)
        h += h % 2
        views = [cv2.copyMakeBorder(v, 0, h - v.shape[0], 0, 0, cv2.BORDER_CONSTANT) for v in views]
        body = np.hstack(views + [draw_map(h, st, k, r["t"])])
        bar = np.full((40, body.shape[1], 3), 25, np.uint8)
        txt, col = st.hud(r["t"])
        if col is not None:
            bar[:] = tuple(int(0.45 * c) for c in col)
        put(bar, txt, (12, 28), 0.8, (255, 255, 255), 2)
        frame = np.vstack([bar, body])
        if frame.shape[0] % 2:
            frame = frame[:-1]
        if ff is None:
            ff = subprocess.Popen([ffmpeg_exe(), "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
                                   "-s", f"{frame.shape[1]}x{frame.shape[0]}", "-r", f"{fa:.5f}", "-i", "-",
                                   "-c:v", "libx264", "-preset", "veryfast", "-crf", "21", "-pix_fmt", "yuv420p",
                                   str(path)], stdin=subprocess.PIPE)
        ff.stdin.write(frame.tobytes())
        if args.show:
            cv2.imshow("full pipeline (q quit, space pause)", frame)
            key = cv2.waitKey(0 if paused else 1) & 0xFF
            if key == ord("q"):
                break
            if key == ord(" "):
                paused = not paused
    if ff:
        ff.stdin.close()
        ff.wait()
    ca.release()
    cb.release()
    cv2.destroyAllWindows()
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
