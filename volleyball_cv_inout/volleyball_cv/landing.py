"""
IN / OUT line calls: measured at the bounce, and predicted during the flight.

MEASURED  At each floor bounce that ends a flight (after the rally's last live touch), the ball's
          lowest pixel is ON the floor plane, where the court homography is exact:
          (u, v + r_px) --H^-1--> court (x, y) in metres. Volleyball rule: a ball touching the line
          is IN (lines belong to the 9 x 18 m court), so the call uses the signed distance to the
          nearest boundary. A call closer than the local measurement uncertainty is marked CLOSE.
PREDICTED While the ball is in the air, on the frames seen SO FAR:
          WHEN  the gravity-constrained 3-D fit X(t) = X0 + V0 t + 1/2 g t^2 reaches z = ball radius;
          WHERE the pixel track (quadratic in t, as in the Medium/cvzone article) extrapolated to that
                time, its lowest pixel mapped through the exact floor homography.
          Pixel-only extrapolation cannot know when the ball meets the floor (the floor is not a fixed
          pixel row under perspective); the 3-D fit supplies that time.
FOCAL     The homography's focal length is weakly constrained (~+/-15 %). It is refined from physics:
          the f for which ballistic fits land exactly when the ball is seen to land. Each landing is
          evaluated with f refined on the OTHER landings (leave-one-out).
Outputs: landings.csv (one row per landing: measured spot, call, margin, prediction error vs the
         fraction of flight seen) and landing_pred.pkl (per-frame predicted path for the overlay).
"""
import json, os, pickle, cv2, numpy as np, pandas as pd
from scipy.optimize import least_squares
from camera_model import camera_from_calib, ball_3d, SIZE_CORR, BALL_D
from common import run_args

a = run_args()
FPS = a.cfg["fps"]; G = np.array([0, 0, -9.81]); R_BALL = BALL_D / 2
cam = camera_from_calib()
C = json.load(open("calib.json"))
H = cv2.findHomography(np.float32(C["world_m"]), np.float32(C["image_px"]))[0]; Hinv = np.linalg.inv(H)
trk = pd.read_csv("ball_track.csv").set_index("frame")
R = pd.read_csv("rallies.csv").sort_values("frame") if os.path.getsize("rallies.csv") > 1 else pd.DataFrame(columns=["frame", "kind"])

def camera_with_f(f, W, Hh):
    """K, R, t from the court homography for a GIVEN focal length (plane-based calibration)"""
    K = np.array([[f, 0, W / 2], [0, f, Hh / 2], [0, 0, 1.]])
    M = np.linalg.inv(K) @ H; lam = 2 / (np.linalg.norm(M[:, 0]) + np.linalg.norm(M[:, 1]))
    r1, r2, t = lam * M[:, 0], lam * M[:, 1], lam * M[:, 2]
    if t[2] < 0: r1, r2, t = -r1, -r2, -t
    Rm = np.column_stack([r1, r2, np.cross(r1, r2)]); U, _, Vt = np.linalg.svd(Rm); Rm = U @ Vt
    if np.linalg.det(Rm) < 0: Rm[:, 2] *= -1
    Cw = -Rm.T @ t
    if Cw[2] < 0: Rm[:, 2] *= -1; Cw = -Rm.T @ t
    return dict(K=K, R=Rm, t=t, C=Cw, f=f)
IMG_W, IMG_H = C.get("image_size", [1920, 1080])

def img2court(u, v):
    p = Hinv @ [u, v, 1.]; return p[:2] / p[2]
def project(X):
    P = (cam["K"] @ (cam["R"] @ np.atleast_2d(X).T + cam["t"][:, None])).T
    return P[:, :2] / P[:, 2:3]
def margin_in(x, y):
    """signed distance (m) to the nearest court boundary: + inside, - outside"""
    return min(x, 9 - x, y, 18 - y)
def m_per_px(x, y):
    """local floor resolution: metres covered by one pixel at (x, y), worst direction"""
    p0 = (H @ [x, y, 1.]); p0 = p0[:2] / p0[2]
    px = (H @ [x + 0.1, y, 1.]); px = px[:2] / px[2]
    py = (H @ [x, y + 0.1, 1.]); py = py[:2] / py[2]
    return 0.1 / max(1e-6, min(np.linalg.norm(px - p0), np.linalg.norm(py - p0)))

def fit(fr, u, v, size):
    """gravity-constrained 3-D flight fit (bounded, robust); returns params, rms px"""
    t = (fr - fr[0]) / FPS
    X0 = ball_3d(cam, u[0], v[0], np.median(size[:5])); X1 = ball_3d(cam, u[-1], v[-1], np.median(size[-5:]))
    V0 = (X1 - X0) / max(t[-1], 1e-3) - 0.5 * G * t[-1]
    obs = np.c_[u, v]
    def res(p):
        X = p[:3] + np.outer(t, p[3:]) + 0.5 * np.outer(t ** 2, G)
        return np.r_[(project(X) - obs).ravel(), 0.15 * np.linalg.norm(p[:3] - X0)]
    lo = np.r_[-6, -8, 0.0, -35, -35, -35]; hi = np.r_[15, 26, 12, 35, 35, 35]
    s = least_squares(res, np.clip(np.r_[X0, V0], lo + 1e-3, hi - 1e-3), bounds=(lo, hi), loss="soft_l1", f_scale=3.0)
    return s.x, float(np.sqrt(np.mean(res(s.x)[:-1] ** 2)))

def land(p):
    """time and (x, y) where the fitted flight reaches z = ball radius (descending root)"""
    z0, vz = p[2], p[5]
    disc = vz ** 2 + 2 * 9.81 * (z0 - R_BALL)
    if disc < 0: return None
    tl = (vz + np.sqrt(disc)) / 9.81
    if tl <= 0: return None
    return tl, p[0] + p[3] * tl, p[1] + p[4] * tl

def call_of(x, y, unc):
    m = margin_in(x, y)
    if abs(m) < unc: return ("IN" if m >= 0 else "OUT") + " (close)", m
    return ("IN" if m >= 0 else "OUT"), m

# ---- 1. landings = first floor bounce after the last live touch of each flight (measured, f-independent) ----
L = []
last_touch = None
for _, r in R.iterrows():
    if r.kind in ("touch", "block"): last_touch = r; continue
    if r.kind != "bounce" or last_touch is None: continue
    fb = int(r.frame)
    win = trk.loc[(trk.index >= fb - 4) & (trk.index <= fb + 4)]
    if not len(win): last_touch = None; continue
    k = int(win.v.idxmax()); u, v, d = win.loc[k, ["u", "v", "size_px"]]
    x, y = img2court(u, v + d * SIZE_CORR / 2)                    # lowest pixel of the ball, on the floor plane
    unc = 3 * m_per_px(x, y)                                      # +/- 3 px of contact-point error
    fl = trk.loc[(trk.index > last_touch.frame + 2) & (trk.index < k - 2)]
    fl = fl[fl.src == "det"] if "src" in fl else fl
    ok_flight = len(fl) >= 12 and np.diff(fl.index.values).max() <= 8
    L.append(dict(touch=last_touch, k=k, x=x, y=y, unc=unc, fl=fl if ok_flight else None))
    last_touch = None

# ---- 2. focal-length self-calibration from physics: a ballistic fit must land when the ball lands ----
def time_err(cm, lands):
    global cam
    saved = cam; cam = cm; e = []
    for l in lands:
        fl = l["fl"]; p, rms = fit(fl.index.values.astype(float), fl.u.values, fl.v.values, fl.size_px.values)
        Ld = land(p); e.append(abs(Ld[0] - (l["k"] - fl.index[0]) / FPS) if Ld else 1.0)
    cam = saved; return float(np.mean(e))
def best_f(lands):
    if not lands: return cam["f"]
    grid = cam["f"] * np.linspace(0.7, 1.4, 29)
    return float(min(grid, key=lambda f: time_err(camera_with_f(f, IMG_W, IMG_H), lands)))
calib_lands = [l for l in L if l["fl"] is not None]
f_all = best_f(calib_lands)
print(f"focal length: homography estimate {cam['f']:.0f} px -> physics-refined {f_all:.0f} px "
      f"(from {len(calib_lands)} tracked landing flight(s))")

# ---- 3. measured call + prediction (leave-one-out: each landing judged with f refined on the OTHERS) ----
rows, preds = [], {}
for i, l in enumerate(L):
    call, m = call_of(l["x"], l["y"], l["unc"])
    row = dict(rally=l["touch"].rally, frame=l["k"], t_s=round(l["k"] / FPS, 2), x_m=round(l["x"], 2), y_m=round(l["y"], 2),
               call=call, margin_m=round(m, 2), uncertainty_m=round(l["unc"], 2), last_touch_t=l["touch"].t_s)
    fl = l["fl"]
    if fl is not None:
        others = [o for o in calib_lands if o is not l]
        f_eval = best_f(others) if others else cam["f"]            # never calibrate on the landing being judged
        cam_eval = camera_with_f(f_eval, IMG_W, IMG_H)
        row["f_used"] = round(f_eval)
        saved = cam; cam = cam_eval
        n = len(fl); f0 = fl.index[0]; d_end = np.median(fl.size_px.values[-5:])
        def predict(sub):
            """3-D fit gives WHEN it lands; pixel-quadratic extrapolation + exact floor homography gives WHERE"""
            p, rms = fit(sub.index.values.astype(float), sub.u.values, sub.v.values, sub.size_px.values)
            Ld = land(p)
            if Ld is None or rms > 6: return None
            tt = (sub.index.values - sub.index[0]) / FPS
            pu, pv = np.polyfit(tt, sub.u.values, 2), np.polyfit(tt, sub.v.values, 2)
            ul, vl = np.polyval(pu, Ld[0]), np.polyval(pv, Ld[0])
            x, y = img2court(ul, vl + d_end * SIZE_CORR / 2)
            ts = np.linspace(tt[-1], Ld[0], 20)
            return x, y, np.c_[np.polyval(pu, ts), np.polyval(pv, ts)], (ul, vl)
        for frac in (0.25, 0.5, 0.75):
            P = predict(fl.iloc[:max(8, int(round(frac * n)))])
            row[f"err_{int(frac * 100)}"] = None if P is None else round(float(np.hypot(P[0] - l["x"], P[1] - l["y"])), 2)
            row[f"call_{int(frac * 100)}"] = None if P is None else call_of(P[0], P[1], l["unc"])[0]
        cam = camera_with_f(f_all, IMG_W, IMG_H)                    # live overlay uses all landings
        for j in range(max(8, int(0.2 * FPS)), n + 1, max(1, int(FPS / 30))):
            P = predict(fl.iloc[:j])
            if P is None: continue
            preds[int(fl.index[j - 1])] = dict(path=P[2], land=(P[0], P[1]), call=call_of(P[0], P[1], l["unc"])[0],
                                               land_px=P[3], until=l["k"])
        cam = saved
    rows.append(row)

out = pd.DataFrame(rows)
if len(out) == 0: out = pd.DataFrame(columns=["rally", "frame", "t_s", "x_m", "y_m", "call", "margin_m", "uncertainty_m"])
out.to_csv("landings.csv", index=False); pickle.dump(preds, open("landing_pred.pkl", "wb"))
print(out.to_string() if len(out) else "no landings found")
