"""Ball contacts in the 3D track, who touched the ball, and the 3-touch / double-contact rules.

Usage (after ball_two_cameras.py; players_on_court.py for the same window is optional but
needed for DOUBLE_CONTACT, since that needs player identities):
    python scripts/touches_rules.py results/ball/game2_A_game2_B_sdr_ball_from30.json \
        --players results/players/game2_A_game2_B_sdr_players_from30.csv
    --pose-model yolo11m-pose.pt   pose model (downloaded automatically on first use)
    --no-evidence                  skip the evidence image (faster: no frames are read)

Adapted from the single-camera pipeline in volleyball_cv_inout/ (ball_events.py, touches.py,
rules.py), but working on the two-camera 3D track instead of a 2D pixel track:

1. Contacts. Between contacts the ball is in free flight, so its 3D path is a fixed-gravity arc.
   At every 3D sample, ballistic arcs are fitted to the 0.3 s before and the 0.3 s after
   (line_calls.fit_arc, weighted by ray angle) and compared at that moment: the change in
   velocity |v_after - v_before| is ~0 in free flight and several m/s at a contact. Peaks above
   DV_MIN are contacts; the contact time and point are where the two arcs meet. Across a gap in
   the track (ball hidden or seen by one camera only), the arcs on either side are extrapolated:
   if they join up smoothly there was no contact, if they meet with a kink the contact is placed
   there, and if they don't meet at all the gap is marked as a possible unseen contact.
   Floor bounces come from line_calls.find_bounces (with their IN / OUT call).
2. Who touched it. The 3D contact point is projected into both cameras at the frames just before,
   at and after the contact (the ball position at each frame comes from the arcs, so it is known
   even where the 2D detector missed the ball). YOLO11-pose runs on a crop around it, scaled to
   about 4 x 5 m at the ball's distance, so distant players are seen at a usable size. For each
   person: gap from the ball's surface to the hands (wrist + 0.35 x forearm, as in touches.py),
   the forearms (passes and digs) and the head, converted to metres at the ball's distance. The
   person's feet (ankles, else box bottom) are mapped to the court; a person more than REACH_M
   from the ball horizontally can't have touched it (this rejects players behind the ball along
   the camera ray, e.g. on the other side of the net). For contacts above JUMP_BALL_Z the player
   may be jumping, and a jumping player's feet map metres too far from the camera, so the feet
   may be anywhere on the camera ray from the floor up to JUMP_M. The closest person within
   TOUCH_GAP_M touched it. Team = side of the net where the toucher stands (A = camera A's half,
   y > 9). If that side is ambiguous (a jumping player whose possible feet positions straddle
   the net), the team is guessed as the side the ball came from, flagged TEAM_GUESSED, and the
   count is not trusted.
3. Rules (FIVB), per rally:
   - A rally starts at a serve (contact with the ball behind an end line) or the first contact
     after a dead ball, and ends at a floor bounce, or when the ball is unseen for LOST_S.
   - Possession changes when a different team touches it, or the 3D track crosses the net.
   - A block (toucher within 1 m of the net, both wrists above the head, ball above
     NET_H - 0.3 m, ball arriving from the other side) is not counted as one of the team's hits,
     and the blocker may touch the ball again.
   - FOUR_TOUCHES: a 4th hit by the same team. DOUBLE_CONTACT: the same player twice in a row
     (not after a block). Identity = track id from players_on_court.py (nearest track within
     1 m at the contact time).
   - A count is only CERTAIN if the ball was observed (3D or either camera) with no gap longer
     than MAX_UNSEEN_S since the previous contact; otherwise a contact could have been missed and
     no foul is flagged.
   - Point: floor bounce IN -> point to the team on the other side; OUT -> point to the opponents
     of the last team to touch it.

Outputs (results/touches/):
    <stem>_touches.csv     one row per event (contact or floor): time, 3D point, velocity change,
                           kind, team, toucher position, track id, gap, rally, team hit number,
                           certain, flags
    <stem>_touches.png     ball height over time with every event labelled, and contact points
                           on the court
    <stem>_evidence.jpg    for each contact, the crop used, with the skeleton and the ball
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from calibrate_camera import read_frame  # noqa: E402
from camera_3d import load_camera  # noqa: E402
from line_calls import BALL_R, G, fit_arc, find_bounces, load_track, make_call  # noqa: E402

NET_Y, NET_H = 9.0, 2.43
WIN_S = 0.3            # s of track on each side of a sample for the before/after arcs
MIN_ARC = 4            # samples needed for an arc
MAX_ARC_RMS = 0.3      # m; worse arcs are not free flight (3D noise in flight is 0.05-0.13 m)
DV_MIN = 4.0           # m/s; velocity change that counts as a contact
PEAK_S = 0.2           # one contact per this many seconds
MAX_GAP = 0.2          # s; longer gaps split the 3D track (as in line_calls.py)
MAX_LINK_S = 1.2       # s; across longer gaps the arcs are not extrapolated
JOIN_M = 0.3           # arcs meeting closer than this with no velocity change: same flight
KINK_M = 0.8           # arcs meeting within this: contact in the gap
FLOOR_S = 0.15         # a contact this close to a floor bounce is the bounce
MIN_CONTACT_Z = 0.2    # m; lower "contacts" are the ball on the floor (radius 0.105 m)
REACH_M = 1.8          # max horizontal distance from the toucher's feet to the ball
JUMP_BALL_Z = 2.0      # m; for contacts at least this high the toucher may be jumping ...
JUMP_M = 0.6           # ... with the feet up to this high
TOUCH_GAP_M = 0.3      # max gap from ball surface to hand / forearm / head
BLOCK_NET_M = 1.0
MAX_UNSEEN_S = 0.5     # longer unseen stretch between contacts: count uncertain
LOST_S = 2.0           # unseen this long: rally over
SERVE_MARGIN = 0.5     # ball within this of / behind an end line at a rally's first contact: serve
SKELETON = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15),
            (12, 14), (14, 16), (0, 5), (0, 6)]


# ---------------------------------------------------------------- arcs and contacts
def arc_at(params, t_ref, t):
    """Position and velocity of a fit_arc arc at time t."""
    s = np.asarray(t, float) - t_ref
    pos = np.stack([params[0, 0] + params[0, 1] * s, params[1, 0] + params[1, 1] * s,
                    params[2, 0] + params[2, 1] * s - 0.5 * G * s ** 2], -1)
    vel = np.stack([np.full_like(s, params[0, 1]), np.full_like(s, params[1, 1]), params[2, 1] - G * s], -1)
    return pos, vel


def side_arc(t, X, ang, mask, t_ref):
    if mask.sum() < MIN_ARC:
        return None
    p, _, rms = fit_arc(t[mask], X[mask], t_ref, ang[mask])
    return (p, t_ref) if rms[0] <= MAX_ARC_RMS else None


def meet(arc_b, arc_a, t0, t1, n=61):
    """Time in [t0, t1] where the two arcs are closest; distance and velocity change there."""
    ts = np.linspace(t0, t1, n)
    pb, vb = arc_at(*arc_b, ts)
    pa, va = arc_at(*arc_a, ts)
    d = np.linalg.norm(pb - pa, axis=1)
    k = int(np.argmin(d))
    return ts[k], d[k], float(np.linalg.norm(va[k] - vb[k])), 0.5 * (pb[k] + pa[k])


def find_contacts(t, X, ang):
    """Contacts inside continuous stretches, plus contacts / unknowns across gaps."""
    contacts, unknown = [], []
    if len(t) < 2 * MIN_ARC:
        return contacts, unknown
    cuts = np.where(np.diff(t) > MAX_GAP)[0] + 1
    segs = np.split(np.arange(len(t)), cuts)
    for seg in segs:
        ts, Xs, angs = t[seg], X[seg], ang[seg]
        dv = np.zeros(len(seg))
        for i in range(len(seg)):
            b = side_arc(ts, Xs, angs, (ts < ts[i]) & (ts >= ts[i] - WIN_S), ts[i])
            a = side_arc(ts, Xs, angs, (ts > ts[i]) & (ts <= ts[i] + WIN_S), ts[i])
            if b and a:
                dv[i] = np.linalg.norm(arc_at(*a, ts[i])[1] - arc_at(*b, ts[i])[1])
        for i in np.where(dv >= DV_MIN)[0]:
            near = np.abs(ts - ts[i]) <= PEAK_S
            if dv[i] < dv[near].max():
                continue
            tc = ts[i]
            for _ in range(2):            # refit with the samples split at the current estimate
                dt = 1.5 * np.median(np.diff(ts)) if len(ts) > 1 else 0.05
                b = side_arc(ts, Xs, angs, (ts < tc) & (ts >= tc - WIN_S), tc)
                a = side_arc(ts, Xs, angs, (ts > tc) & (ts <= tc + WIN_S), tc)
                if not (b and a):
                    break
                tc, d, dvc, P = meet(b, a, tc - dt, tc + dt)
            if b and a:
                contacts.append(dict(t=float(tc), P=P, dv=dvc, miss_m=float(d), source="track",
                                     arc_before=b, arc_after=a))
    # across gaps
    for s0, s1 in zip(segs, segs[1:]):
        te, tb = t[s0[-1]], t[s1[0]]
        if tb - te > MAX_LINK_S:
            continue
        b = side_arc(t, X, ang, np.isin(np.arange(len(t)), s0) & (t >= te - 0.4), te)
        a = side_arc(t, X, ang, np.isin(np.arange(len(t)), s1) & (t <= tb + 0.4), tb)
        if not (b and a):
            unknown.append((float(te), float(tb)))
            continue
        tc, d, dvc, P = meet(b, a, te, tb)
        if d < JOIN_M and dvc < DV_MIN:
            continue                                   # same flight, just not seen
        if d < KINK_M:
            contacts.append(dict(t=float(tc), P=P, dv=dvc, miss_m=float(d), source="gap",
                                 arc_before=b, arc_after=a))
        else:
            unknown.append((float(te), float(tb)))
    contacts.sort(key=lambda c: c["t"])
    return contacts, unknown


def ball_at(c, t):
    arc = c["arc_before"] if t < c["t"] else c["arc_after"]
    return arc_at(*arc, t)[0]


# ---------------------------------------------------------------- who touched it
def project_with_depth(cam, X):
    Xc = cam.R @ np.asarray(X, float) + cam.t
    return cam.project(X)[0], float(Xc[2])


def hand_points(k, s, thr=0.3):
    """Contact parts in pixels: hands (wrist + 0.35 forearm), forearm segments, head."""
    pts, segs = [], []
    for e, w in ((7, 9), (8, 10)):
        if s[w] > thr:
            hand = k[w] + 0.35 * (k[w] - k[e]) if s[e] > thr else k[w]
            pts.append(hand)
            if s[e] > thr:
                segs.append((k[e], hand))
    if s[0] > thr:
        pts.append(k[0])
    return pts, segs


def point_seg_dist(p, a, b):
    ab = b - a
    u = np.clip((p - a) @ ab / (ab @ ab + 1e-9), 0, 1)
    return float(np.linalg.norm(p - (a + u * ab)))


def feet_pixel(k, s, box, crop_box, img_shape):
    """Pixel of a person's feet: ankle keypoints, else the box bottom if not cut off."""
    ank = [k[j] for j in (15, 16) if s[j] > 0.5]
    if ank:
        return np.array([np.mean([p[0] for p in ank]), max(p[1] for p in ank)])
    if box[3] < min(crop_box[3], img_shape[0]) - 3:
        return np.array([(box[0] + box[2]) / 2, box[3]])
    return None


def foot_line(cam, uv, h_max):
    """Court (x, y) of the feet if they are on the floor (p0) and if they are h_max up (p1): the
    point on the camera ray through the feet pixel at that height. A jumping player's feet map
    too far from the camera through the floor homography (README section 7)."""
    d = cam.ray(uv)
    if d[2] >= -1e-6:
        return None, None                       # ray doesn't reach the floor
    pts = [cam.C + (h - cam.C[2]) / d[2] * d for h in (0.0, min(h_max, cam.C[2] - 0.05))]
    return pts[0][:2], pts[1][:2]


def seg_param(p, a, b):
    ab = b - a
    return float(np.clip((p - a) @ ab / (ab @ ab + 1e-9), 0, 1))


class Attributor:
    def __init__(self, model, cams, videos, offset, fps):
        from ultralytics import YOLO
        self.m = YOLO(model)
        self.cams, self.videos, self.offset, self.dt = cams, videos, offset, 1.0 / fps

    def people(self, img, crop):
        x0, y0, x1, y1 = crop
        r = self.m.predict(img[y0:y1, x0:x1], imgsz=640, conf=0.25, verbose=False)[0]
        if r.boxes is None or not len(r.boxes):
            return []
        boxes = r.boxes.xyxy.cpu().numpy() + [x0, y0, x0, y0]
        kp = r.keypoints.data.cpu().numpy()
        kp[:, :, 0] += x0
        kp[:, :, 1] += y0
        return [(b, k[:, :2], k[:, 2]) for b, k in zip(boxes, kp)]

    def crop_for(self, cam, X, shape):
        (u, v), depth = project_with_depth(cam, X)
        ppm = cam.K[0, 0] / max(depth, 0.5)                    # pixels per metre at the ball
        floor_v = cam.project([X[0], X[1], 0.0])[0][1]
        half = max(96, 2.0 * ppm)
        y0, y1 = v - max(1.2 * ppm, 64), max(floor_v, v) + 0.4 * ppm
        h, w = shape[:2]
        c = [int(max(0, u - half)), int(max(0, y0)), int(min(w, u + half)), int(min(h, y1))]
        return c, (u, v), ppm

    def __call__(self, c):
        """Best toucher candidate for contact c (dict) or None, plus an evidence crop."""
        best, fallback = None, None
        for ci, (cam, video) in enumerate(zip(self.cams, self.videos)):
            name = "AB"[ci]
            for k_off in (-1, 0, 1):
                t_a = c["t"] + k_off * self.dt
                img = read_frame(video, t_a + (self.offset if ci == 1 else 0.0))
                if img is None:
                    continue
                Xb = ball_at(c, t_a)
                crop, uv, ppm = self.crop_for(cam, Xb, img.shape)
                if crop[2] - crop[0] < 16 or crop[3] - crop[1] < 16:
                    continue
                r_px = BALL_R * ppm
                own = (name == "A") == (c["P"][1] > NET_Y)
                if fallback is None or (own and k_off == 0):
                    fallback = dict(img=img, crop=crop, uv=uv, r_px=r_px, cam=name)
                for box, k, s in self.people(img, crop):
                    pts, segs = hand_points(k, s)
                    if not pts:
                        continue
                    d_px = min([float(np.linalg.norm(np.asarray(uv) - p)) for p in pts] +
                               [point_seg_dist(np.asarray(uv), a, b) for a, b in segs])
                    gap = max(0.0, d_px - r_px) / ppm
                    # feet: on the floor for low balls; for high balls the player may be jumping, so
                    # the feet are somewhere on the camera ray between the floor and JUMP_M up
                    feet_uv = feet_pixel(k, s, box, crop, img.shape)
                    feet, side, h_need = None, None, 0.0
                    if feet_uv is not None:
                        h_max = JUMP_M if Xb[2] >= JUMP_BALL_Z else 0.0
                        p0, p1 = foot_line(cam, feet_uv, h_max)
                        if p0 is None:
                            continue
                        frac = seg_param(Xb[:2], p0, p1)
                        feet = p0 + frac * (p1 - p0)
                        if np.hypot(*(feet - Xb[:2])) > REACH_M:
                            continue
                        h_need = frac * h_max
                        if (p0[1] - NET_Y) * (p1[1] - NET_Y) > 0:   # whole line on one side of the net
                            side = "A" if p0[1] > NET_Y else "B"
                    # prefer the camera on the ball's side (closer, sharper) on near ties, and
                    # standing players over ones that would have to be jumping
                    score = gap + (0.0 if own else 0.05) + 0.2 * h_need
                    if best is None or score < best["score"]:
                        wr_up = s[0] > 0.3 and all(s[j] > 0.3 and k[j][1] < k[0][1] for j in (9, 10))
                        best = dict(score=score, gap=gap, feet=feet, side=side, jump_m=h_need, cam=name,
                                    k_off=k_off, box=box, k=k, s=s, wrists_up=bool(wr_up), img=img,
                                    crop=crop, uv=uv, r_px=r_px)
        if best is not None and best["gap"] <= TOUCH_GAP_M:
            return best, best
        return None, (best or fallback)


def evidence_tile(ev, label, ok):
    if ev is None:
        return np.zeros((240, 240, 3), np.uint8)
    img = ev["img"].copy()
    if ok and "k" in ev:
        k, s = ev["k"], ev["s"]
        for a, b in SKELETON:
            if s[a] > 0.3 and s[b] > 0.3:
                cv2.line(img, tuple(map(int, k[a])), tuple(map(int, k[b])), (255, 0, 255), 2, cv2.LINE_AA)
    cv2.circle(img, tuple(map(int, ev["uv"])), max(4, int(ev["r_px"])), (0, 255, 0) if ok else (0, 0, 255), 2)
    x0, y0, x1, y1 = ev["crop"]
    tile = cv2.resize(img[y0:y1, x0:x1], (240, 240))
    for i, txt in enumerate(label):
        cv2.putText(tile, txt, (5, 18 + 20 * i), 0, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(tile, txt, (5, 18 + 20 * i), 0, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return tile


# ---------------------------------------------------------------- identities and rules
def load_players(path):
    if not path:
        return None
    rows = list(csv.DictReader(open(path)))
    return np.array([[float(r["time_s"]), int(r["track_id"]), float(r["x_m"]), float(r["y_m"])] for r in rows])


def track_id_at(players, t, xy):
    if players is None or xy is None or not len(players):
        return None
    near = players[np.abs(players[:, 0] - t) <= 0.2]
    if not len(near):
        return None
    d = np.hypot(near[:, 2] - xy[0], near[:, 3] - xy[1])
    return int(near[np.argmin(d), 1]) if d.min() <= 1.0 else None


def observed_times(csv_path):
    """Camera A frame times at which the ball was seen by either camera (2D or 3D)."""
    out = []
    for r in csv.DictReader(open(csv_path)):
        if r["status"] in ("3d", "3d weak", "one camera") or r["a_state"] == "measured" or r["b_measured"] == "1":
            out.append(float(r["time_a_s"]))
    return np.array(sorted(out))


def max_unseen(obs, t0, t1):
    inside = obs[(obs > t0) & (obs < t1)]
    return float(np.max(np.diff(np.r_[t0, inside, t1])))


def crossed_net(t, X, t0, t1):
    m = (t > t0) & (t < t1)
    y = X[m, 1]
    return bool(len(y) >= 2 and np.any(np.sign(y[1:] - NET_Y) != np.sign(y[:-1] - NET_Y)))


def apply_rules(events, t3, X3, obs, unknown):
    rally, team, count, last_pid, after_block, certain = 0, None, 0, None, False, True
    prev = None
    for e in events:
        e.update(rally=None, team_hit=None, certain=None, flags=[])
        if e["kind"] == "floor":
            if prev is not None and rally:
                e["rally"] = rally
                loser = ("A" if e["y"] > NET_Y else "B") if e["call"] == "IN" else team
                if loser:
                    e["point_to"] = "B" if loser == "A" else "A"
            prev, team = None, None                  # dead ball
            continue
        if e["kind"] != "contact":
            continue
        unseen = max_unseen(obs, prev["t"], e["t"]) if prev is not None else np.inf
        new_rally = (prev is None or unseen > LOST_S or e["t"] - prev["t"] > 4.0)
        serve = new_rally and (e["y"] < SERVE_MARGIN or e["y"] > 18 - SERVE_MARGIN)
        if new_rally:
            rally += 1
            team, count, last_pid, after_block, certain = None, 0, None, False, True
        e["rally"] = rally
        e["serve"] = serve
        if e["team"] is None:
            e["flags"].append("NO_TOUCHER")
            certain = False
            prev = e
            continue
        crossed = prev is not None and not new_rally and crossed_net(t3, X3, prev["t"], e["t"])
        gap_unknown = prev is not None and any(prev["t"] < u1 and u0 < e["t"] for u0, u1 in unknown)
        if e["team"] != team:
            count, last_pid, after_block, certain = 0, None, False, True
        elif crossed:                                # over the net and back without a seen touch
            count, last_pid, after_block, certain = 0, None, False, False
        if prev is not None and not new_rally and (unseen > MAX_UNSEEN_S or gap_unknown):
            certain = False
        if not e.get("team_sure", True):
            certain = False
            e["flags"].append("TEAM_GUESSED")
        arrived_from_other_side = prev is not None and (prev.get("team") not in (None, e["team"]) or crossed)
        block = (abs(e["toucher_y"] - NET_Y) <= BLOCK_NET_M if e["toucher_y"] is not None else
                 abs(e["y"] - NET_Y) <= BLOCK_NET_M) and e["wrists_up"] and e["z"] >= NET_H - 0.3 \
            and arrived_from_other_side
        if block:
            e["kind_detail"] = "block"
            e["team_hit"] = "block"
            after_block = True
        else:
            e["kind_detail"] = "serve" if serve else "hit"
            count += 1
            e["team_hit"] = count
            if certain and count > 3:
                e["flags"].append("FOUR_TOUCHES")
            if certain and e["track_id"] is not None and e["track_id"] == last_pid and not after_block:
                e["flags"].append("DOUBLE_CONTACT")
            after_block = False
        e["certain"] = certain
        team, last_pid = e["team"], e["track_id"]
        prev = e
    return events


# ---------------------------------------------------------------- figure
def plot(events, t, X, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    fig = plt.figure(figsize=(13, 6.5))
    ax = fig.add_axes([0.05, 0.1, 0.62, 0.8])
    ax.plot(t, X[:, 2], ".", ms=3, color="#9aa8bd")
    ax.axhline(NET_H, color="k", lw=0.8, ls="--")
    ax.text(t.min() if len(t) else 0, NET_H + 0.05, "net top", fontsize=8)
    col = {"A": "#e08214", "B": "#1a9850", None: "#888888"}
    for e in events:
        if e["kind"] == "floor":
            c = "#1a9850" if e["call"] == "IN" else "#d73027"
            ax.axvline(e["t"], color=c, lw=1.2)
            ax.text(e["t"], 0.1, f" floor {e['call']}" + (f"\n point {e['point_to']}" if e.get("point_to") else ""),
                    fontsize=8, color=c)
            continue
        lab = {"block": "BLK", "serve": "SRV"}.get(e.get("kind_detail"), str(e["team_hit"] or "?"))
        lab = f"{e['team'] or '?'}{lab if lab != 'SRV' else ' serve'}"
        if e.get("track_id") is not None:
            lab += f" #{e['track_id']}"
        if e["flags"]:
            lab += "\n" + " ".join(e["flags"])
        ax.plot(e["t"], e["z"], "o", ms=8, mfc="none", mec=col[e["team"]], mew=2)
        ax.annotate(lab, (e["t"], e["z"]), xytext=(0, 9), textcoords="offset points", ha="center", fontsize=8,
                    color="#d73027" if e["flags"] and e["flags"] != ["NO_TOUCHER"] else col[e["team"]])
        if e["certain"] is False:
            ax.plot(e["t"], e["z"], "x", ms=4, color="#555555")
    ax.set_xlabel("time in camera A (s)")
    ax.set_ylabel("ball height (m)")
    ax.set_title("contacts: team + hit number (A = camera A's half, y > 9); x = count uncertain", fontsize=9)
    ax2 = fig.add_axes([0.72, 0.1, 0.25, 0.8])
    ax2.add_patch(Rectangle((0, 0), 9, 18, fc="#c9a77c", ec="#e8c400", lw=2))
    for y in (6, 12):
        ax2.plot([0, 9], [y, y], color="#e8c400", lw=1)
    ax2.plot([-1, 10], [NET_Y, NET_Y], color="k", lw=3)
    ax2.plot(X[:, 0], X[:, 1], ".", ms=1.5, color="#888888")
    for e in events:
        if e["kind"] == "contact":
            ax2.plot(e["x"], e["y"], "o", color=col[e["team"]], ms=6)
        else:
            ax2.plot(e["x"], e["y"], "s", color="#1a9850" if e["call"] == "IN" else "#d73027", ms=6)
    ax2.set_xlim(-3, 12)
    ax2.set_ylim(21, -3)
    ax2.set_aspect("equal")
    ax2.set_title("contact (o) and bounce (square) points", fontsize=9)
    fig.suptitle(title, fontsize=11)
    fig.savefig(path, dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser()
    p.add_argument("run_json", help="the .json written by ball_two_cameras.py")
    p.add_argument("--players", help="players_on_court.py .csv for the same window (track ids)")
    p.add_argument("--pose-model", default="yolo11m-pose.pt")
    p.add_argument("--no-evidence", action="store_true")
    args = p.parse_args()
    run = json.loads(Path(args.run_json).read_text())
    t, X, weak, ang = load_track(run["csv"])
    if len(t) == 0:
        raise SystemExit("No 3D ball samples in " + run["csv"])
    obs = observed_times(run["csv"])
    contacts, unknown = find_contacts(t, X, ang)
    bounces = [make_call(b, 0.03, 0.05) for b in find_bounces(t, X, weak, ang)]
    # a "contact" at a bounce, or with the ball centre within 10 cm of resting on the floor
    # (rolling, bouncing), is the floor, not a player
    contacts = [c for c in contacts if all(abs(c["t"] - b["t"]) > FLOOR_S for b in bounces)
                and c["P"][2] >= MIN_CONTACT_Z]
    print(f"{len(contacts)} contact(s), {len(bounces)} floor bounce(s), {len(unknown)} gap(s) that may hide a "
          f"contact, from {len(t)} 3D samples")

    cams = [load_camera(run["calib_a"])[0], load_camera(run["calib_b"])[0]]
    cap = cv2.VideoCapture(run["video_a"])
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    attrib = Attributor(args.pose_model, cams, (run["video_a"], run["video_b"]), run["offset_s"], fps)
    players = load_players(args.players)
    events, tiles = [], []
    for n, c in enumerate(contacts, 1):
        who, ev = attrib(c)
        e = dict(kind="contact", t=c["t"], x=float(c["P"][0]), y=float(c["P"][1]), z=float(c["P"][2]),
                 dv=c["dv"], source=c["source"], team=None, toucher_x=None, toucher_y=None, track_id=None,
                 gap_m=None, camera=None, wrists_up=False)
        if ev is not None and "gap" in ev:
            e["gap_m"] = ev["gap"]
        if who is not None:
            feet = who["feet"]
            team, team_sure = who["side"], who["side"] is not None
            if team is None:
                # can't tell which side of the net the toucher is on (jumping at the net, or feet
                # not visible): guess the side the ball came from, and don't trust the count
                y_before = ball_at(c, c["t"] - 0.15)[1]
                team = "A" if y_before > NET_Y else "B"
            e.update(team=team, team_sure=team_sure, jump_m=who["jump_m"], camera=who["cam"],
                     wrists_up=who["wrists_up"],
                     toucher_x=None if feet is None else float(feet[0]),
                     toucher_y=None if feet is None else float(feet[1]),
                     track_id=track_id_at(players, c["t"], feet if feet is not None else c["P"][:2]))
        events.append(e)
        tiles.append((e, ev, who is not None))
        print(f"  contact {n}/{len(contacts)} t = {c['t']:.2f} s, z = {e['z']:.2f} m, dv = {c['dv']:.1f} m/s -> "
              + (f"team {e['team']} (camera {e['camera']}, gap {e['gap_m']:.2f} m"
                 + (f", track {e['track_id']}" if e["track_id"] is not None else "") + ")"
                 if who is not None else "no toucher found"), flush=True)
    for b in bounces:
        events.append(dict(kind="floor", t=b["t"], x=b["x"], y=b["y"], z=BALL_R, dv=None, source="bounce",
                           call=b["call"], team=None, track_id=None))
    events.sort(key=lambda e: e["t"])
    apply_rules(events, t, X, obs, unknown)

    stem = Path(run["csv"]).stem
    out = Path("results/touches")
    out.mkdir(parents=True, exist_ok=True)
    cols = ["time_a_s", "kind", "detail", "x_m", "y_m", "z_m", "dv_ms", "source", "team", "toucher_x_m",
            "toucher_y_m", "track_id", "gap_m", "camera", "rally", "team_hit", "certain", "flags", "call", "point_to"]
    r2 = lambda v: None if v is None else round(float(v), 2)
    with open(out / f"{stem}_touches.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for e in events:
            w.writerow([round(e["t"], 3), e["kind"], e.get("kind_detail", ""), r2(e["x"]), r2(e["y"]), r2(e["z"]),
                        r2(e["dv"]), e["source"], e["team"], r2(e.get("toucher_x")), r2(e.get("toucher_y")),
                        e["track_id"], r2(e.get("gap_m")), e.get("camera"), e.get("rally"), e.get("team_hit"),
                        e.get("certain"), " ".join(e.get("flags", [])), e.get("call", ""), e.get("point_to", "")])
    print("\n  time    event        team  hit   track  rally  certain  flags")
    for e in events:
        if e["kind"] == "floor":
            print(f"  {e['t']:6.2f}  floor {e['call']:<4}                     {e.get('rally') or '':>5}"
                  + (f"           point to {e['point_to']}" if e.get("point_to") else ""))
        else:
            print(f"  {e['t']:6.2f}  {e.get('kind_detail', 'contact'):<12} {e['team'] or '?':<4} "
                  f"{str(e['team_hit'] or ''):<5} {str(e['track_id'] if e['track_id'] is not None else ''):<6} "
                  f"{e['rally']:>5}  {str(e['certain']):<7}  {' '.join(e['flags'])}")
    plot(events, t, X, out / f"{stem}_touches.png", f"{stem}: contacts and rules")
    if not args.no_evidence and tiles:
        imgs = []
        for e, ev, ok in tiles:
            lab = [f"{e['t']:.2f}s z={e['z']:.1f}m dv={e['dv']:.0f}",
                   (f"team {e['team']} hit {e['team_hit']} cam {e['camera']}" if ok else "no toucher")
                   + (f" #{e['track_id']}" if e["track_id"] is not None else ""),
                   " ".join(e["flags"])]
            imgs.append(evidence_tile(ev, lab, ok))
        while len(imgs) % 6:
            imgs.append(np.zeros_like(imgs[0]))
        cv2.imwrite(str(out / f"{stem}_evidence.jpg"),
                    np.vstack([np.hstack(imgs[i:i + 6]) for i in range(0, len(imgs), 6)]))
    print(f"\nSaved {out / (stem + '_touches.csv')}, _touches.png" + ("" if args.no_evidence else ", _evidence.jpg"))


if __name__ == "__main__":
    main()
