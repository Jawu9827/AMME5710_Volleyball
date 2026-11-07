"""Render a synthetic two-camera volleyball clip with exactly known ground truth.

Usage (from the repo root):
    python scripts/synthetic_render.py                 # writes data/synthetic/
    python scripts/synthetic_render.py --out data/synthetic

Real games have no ground truth for where the ball landed, so this draws one: two virtual
pinhole cameras placed like our phones (behind each end line, ~1.5 m high, facing each other),
a court with lines, posts and net, twelve simple moving players, and a ball that follows pure
physics through a toss and three serves. Everything is projected with OpenCV from known 3D
geometry (no 3D engine).

The three serves test the call logic:
    1  lands at (2.50, 2.00) m      2 m inside the end line          -> IN
    2  lands at (9.30, 15.00) m     30 cm outside the sideline       -> OUT
    3  lands at (9.02, 4.00) m      centre 2 cm outside the sideline -> IN (the ball touches the line)

Camera B's timeline is shifted by OFF_TRUE = 2.017 s; sync.json gives the pipeline a starting
guess of 2.000 s (17 ms off), so ball_two_cameras.py's sync refinement is tested too.

Outputs in --out:
    A.mp4, B.mp4                 the two camera videos (1920x1080, 30 fps)
    calib_A.json, calib_B.json   each camera's exact calibration, in calibrate_camera.py's format
    sync.json                    starting sync guess
    truth.npy                    true ball position per camera-A frame: t, x, y, z (NaN = no ball)
    A_frame.jpg                  one example frame

Then run the normal pipeline on it and score it (see synthetic_score.py).
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from calibrate_camera import COURT_LINES, CourtCalibration, ffmpeg_exe  # noqa: E402
from camera_3d import Camera3D  # noqa: E402

W, H = 1920, 1080
FPS = 30.0
OFF_TRUE = 2.017          # s, true time offset of camera B
G = 9.81
RB = 0.105                # m, ball radius

# camera: centre (m), aim point (m), focal length (px), lens k1, which end it is behind
CAMERAS = {"A": ((4.2, 24.0, 1.45), (4.5, 9, 0.0), 1450, -0.02, "near"),
           "B": ((4.9, -5.5, 1.6), (4.5, 9, 0.0), 1550, -0.012, "far")}

# true landing points of the three serves (x, y in m) and the correct call
TRUE_LANDINGS = [((2.50, 2.00), "IN"), ((9.30, 15.00), "OUT"), ((9.02, 4.00), "IN")]


# ----------------------------------------------------------------------------- cameras
def make_cam(C, target, f, size=(W, H), k1=0.0):
    """Pinhole camera at C looking at target -> (K, R, t, CourtCalibration of its floor)."""
    C = np.array(C, float)
    fwd = np.array(target, float) - C
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0, 0, 1.0])
    right /= np.linalg.norm(right)
    if right[0] * (1 if C[1] > 9 else -1) < 0:      # A (behind y=18): +x to the right; B: -x
        right = -right
    down = np.cross(fwd, right)
    if down[2] > 0:
        down = -down
    R = np.vstack([right, down, fwd])
    t = -R @ C
    K = np.array([[f, 0, size[0] / 2], [0, f, size[1] / 2], [0, 0, 1]])
    Hc = K @ np.c_[R[:, 0], R[:, 1], t]
    Hc /= Hc[2, 2]
    cal = CourtCalibration(Hc, k1, (size[0] / 2, size[1] / 2), size[0] / 2)
    return K, R, t, cal


def save_calib(name, end, cal, path):
    d = dict(camera=name, end=end, frame_time_s=0.0, image_size=[W, H], **cal.to_json())
    Path(path).write_text(json.dumps(d))


# ----------------------------------------------------------------------------- ball path
def _shot(t0, P0, P1, T, e=0.6, after=0.7):
    """Ballistic flight from P0 to P1 (landing, z = ball radius) taking T s, then a bounce."""
    P0, P1 = np.array(P0, float), np.array(P1, float)
    vh = (P1[:2] - P0[:2]) / T
    vz = (P1[2] - P0[2] + 0.5 * G * T * T) / T

    def f(t):
        s = t - t0
        if s < 0 or s > T + after:
            return None
        if s <= T:
            return np.r_[P0[:2] + vh * s, P0[2] + vz * s - 0.5 * G * s * s]
        u = s - T
        vz2 = -e * (vz - G * T)
        return np.r_[P1[:2] + 0.7 * vh * u, RB + vz2 * u - 0.5 * G * u * u]
    return f


_SERVES = [(1.0, (6.0, 19.5, 2.9), 1.15),     # from A's side, lands IN on B's side
           (3.0, (3.0, -0.5, 2.6), 1.0),      # from B's side, lands just OUT
           (5.3, (4.0, 19.0, 2.8), 1.1)]      # from A's side, centre 2 cm outside x = 9: IN (close)
_SHOTS = [_shot(t0, P0, (*land, RB), T) for (t0, P0, T), (land, _) in zip(_SERVES, TRUE_LANDINGS)]


def ball(t):
    """True 3D ball centre at time t (camera A's clock), or None when there is no ball."""
    for s in _SHOTS:
        X = s(t)
        if X is not None:
            return X
    for t0, P, _ in _SERVES:                 # toss before each serve
        if t0 - 0.5 <= t < t0:
            return np.r_[P[0], P[1], P[2] - 3 * (t0 - t) ** 2]
    return None


# ----------------------------------------------------------------------------- drawing
def background(cam):
    img = np.zeros((H, W, 3), np.uint8)
    img[:] = (150, 160, 170)
    rng = np.random.default_rng(0)
    img = (img + rng.normal(0, 4, img.shape)).clip(0, 255).astype(np.uint8)
    # floor = image points whose back-projection lands in front of the camera
    yy, xx = np.mgrid[0:H:4, 0:W:4]
    uv = np.c_[xx.ravel(), yy.ravel()].astype(float)
    pu = cam.cal.undistort(uv)
    h = np.c_[pu, np.ones(len(pu))] @ cam.cal.H_i2c.T
    ru = cam.cal.undistort(cam.cal.court_to_image([[4.5, 9]]))[0]
    sref = np.sign((cam.cal.H_i2c @ np.r_[ru, 1])[2])
    m = (np.sign(h[:, 2]) == sref).reshape(yy.shape).astype(np.uint8)
    m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
    img[m] = (185, 135, 60)
    s = np.linspace(0, 1, 50)[:, None]
    for a, b in COURT_LINES:
        pts = cam.project(np.c_[np.asarray(a) * (1 - s) + np.asarray(b) * s, np.zeros(50)])
        cv2.polylines(img, [np.int32(pts)], False, (40, 215, 235), 4, cv2.LINE_AA)
    for x in (-0.6, 9.6):                    # posts
        post = cam.project(np.c_[np.full(50, x), np.full(50, 9.0), 2.5 * s[:, 0]])
        cv2.polylines(img, [np.int32(post)], False, (40, 40, 200), 8)
    net = cam.project(np.c_[9 * s[:, 0], np.full(50, 9.0), np.full(50, 2.43)])
    cv2.polylines(img, [np.int32(net)], False, (250, 250, 250), 5)
    return img


def ball_patch(path):
    p = cv2.imread(str(path)) if path and Path(path).exists() else None
    if p is None:                            # fallback: a drawn yellow/blue ball
        p = np.full((64, 64, 3), (40, 200, 230), np.uint8)
        cv2.ellipse(p, (32, 32), (30, 12), 30, 0, 360, (150, 80, 30), -1)
    return p


def draw_ball(cam, X, img, patch):
    d = (cam.R @ X + cam.t)[2]
    r = max(2, cam.K[0, 0] * RB / d)
    u, v = cam.project(X)[0]
    sz = int(round(2 * r))
    if sz < 3 or not (0 <= u < W and 0 <= v < H):
        return img
    p = cv2.resize(patch, (sz, sz), interpolation=cv2.INTER_AREA)
    m = np.zeros((sz, sz), np.float32)
    cv2.circle(m, (sz // 2, sz // 2), sz // 2, 1, -1)
    m = cv2.GaussianBlur(m, (3, 3), 0)[..., None]
    x0, y0 = int(round(u - sz / 2)), int(round(v - sz / 2))
    if x0 < 0 or y0 < 0 or x0 + sz > W or y0 + sz > H:
        return img
    roi = img[y0:y0 + sz, x0:x0 + sz].astype(np.float32)
    img[y0:y0 + sz, x0:x0 + sz] = (m * p + (1 - m) * roi).astype(np.uint8)
    return img


def make_people():
    rng = np.random.default_rng(5)
    shirts = [(40, 40, 40), (200, 200, 200), (60, 60, 160), (30, 200, 230), (120, 80, 40), (90, 90, 200)]
    people = []
    for i in range(12):
        side = 1 if i < 6 else -1
        base = np.array([rng.uniform(1, 8), 9 + side * rng.uniform(1.5, 7.5)])
        people.append(dict(base=base, amp=rng.uniform(0.3, 1.2, 2), w=rng.uniform(0.5, 1.5, 2),
                           ph=rng.uniform(0, 6, 4), col=shirts[i % 6], arm=rng.uniform(4, 9)))
    return people


def draw_people(img, cam, t, people):
    order = sorted(people, key=lambda p: -(cam.R @ np.r_[p["base"], 1.0] + cam.t)[2])   # far first
    for p in order:
        xy = p["base"] + p["amp"] * np.sin(p["w"] * t + p["ph"][:2])
        jump = max(0, np.sin(1.3 * t + p["ph"][2])) ** 8 * 0.5

        def seg(a, b, wid, col):
            A = np.r_[xy, 0] + a
            B = np.r_[xy, 0] + b
            A[2] += jump
            B[2] += jump
            d = (cam.R @ ((A + B) / 2) + cam.t)[2]
            px = max(1, int(cam.K[0, 0] * wid / d))
            ua, ub = cam.project(np.vstack([A, B]))
            cv2.line(img, tuple(np.int32(ua)), tuple(np.int32(ub)), col, px, cv2.LINE_AA)

        seg(np.r_[-0.1, 0, 0], np.r_[-0.1, 0, 0.9], 0.13, (60, 50, 50))         # legs
        seg(np.r_[0.1, 0, 0], np.r_[0.1, 0, 0.9], 0.13, (60, 50, 50))
        seg(np.r_[0, 0, 0.9], np.r_[0, 0, 1.5], 0.38, p["col"])                 # torso
        sw = np.sin(p["arm"] * t + p["ph"][3])
        for sx in (-0.25, 0.25):                                                # arms
            seg(np.r_[sx, 0, 1.45], np.r_[sx, 0.45 * sw, 1.45 - 0.5 * abs(np.cos(p["arm"] * t))],
                0.09, (150, 180, 220))
        seg(np.r_[0, 0, 1.62], np.r_[0, 0, 1.78], 0.2, (120, 160, 210))         # head
    return img


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synthetic", help="output folder")
    ap.add_argument("--patch", default="data/annotations/ball_patch.png",
                    help="image of the ball to paste (a crop of the real ball; if missing, a drawn ball)")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    patch = ball_patch(args.patch)

    cams = {}
    for n, (C, tg, f, k1, end) in CAMERAS.items():
        K, R, t, cal = make_cam(C, tg, f, k1=k1)
        save_calib(n, end, cal, out / f"calib_{n}.json")
        cams[n] = Camera3D(K, R, t, cal)
    (out / "sync.json").write_text(json.dumps({"offset_s": 2.0}))    # a guess 17 ms off the truth

    people = make_people()
    spare = np.array([-2.0, 12.0, RB])                                 # stationary decoy ball
    for n, dur, shift in (("A", 7.5, 0.0), ("B", 10.0, OFF_TRUE)):
        bg = background(cams[n])
        ff = subprocess.Popen([ffmpeg_exe(), "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
                               "-s", f"{W}x{H}", "-r", "30", "-i", "-", "-c:v", "libx264", "-crf", "18",
                               "-pix_fmt", "yuv420p", str(out / f"{n}.mp4")], stdin=subprocess.PIPE)
        for k in range(int(dur * FPS)):
            t = k / FPS - shift
            img = draw_people(bg.copy(), cams[n], t, people)
            img = draw_ball(cams[n], spare, img, patch)
            X = ball(t)
            if X is not None:
                img = draw_ball(cams[n], X, img, patch)
            img = np.clip(img.astype(np.float32) * (1 + 0.02 * np.sin(2 * np.pi * 7 * t)), 0, 255)  # 7 Hz flicker
            ff.stdin.write(img.astype(np.uint8).tobytes())
        ff.stdin.close()
        ff.wait()
        print(f"wrote {out / (n + '.mp4')}")
    cv2.imwrite(str(out / "A_frame.jpg"),
                draw_ball(cams["A"], ball(1.6), draw_people(background(cams["A"]), cams["A"], 1.6, people), patch))
    truth = [[k / FPS, *(ball(k / FPS) if ball(k / FPS) is not None else [np.nan] * 3)] for k in range(int(7.5 * FPS))]
    np.save(out / "truth.npy", np.array(truth))
    print(f"wrote calibrations, sync.json and truth.npy in {out}  (true offset {OFF_TRUE} s)")


if __name__ == "__main__":
    main()
