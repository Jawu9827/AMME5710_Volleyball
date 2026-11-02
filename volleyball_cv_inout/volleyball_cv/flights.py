"""Monocular 3-D ballistic flight reconstruction: between contacts the ball obeys
X(t) = X0 + V0 t + 0.5 g t^2, g = (0,0,-9.81). Known metric gravity fixes the scale, so the
6 unknowns (X0, V0) are recoverable from one camera's 2-D track by least squares on
reprojection error (initialised from apparent ball size)."""
import numpy as np, pandas as pd
from scipy.optimize import least_squares
from camera_model import camera_from_calib, ball_3d, project
from common import run_args
a = run_args()
cam = camera_from_calib(); G = np.array([0, 0, -9.81]); FPS = a.cfg["fps"]
def fit_flight(fr, u, v, size):
    t = (fr - fr[0]) / FPS
    X0 = ball_3d(cam, u[0], v[0], np.median(size[:5])); X1 = ball_3d(cam, u[-1], v[-1], np.median(size[-5:]))
    V0 = (X1 - X0) / max(t[-1], 1e-3) - 0.5 * G * t[-1]
    obs = np.c_[u, v]
    def res(p):
        X = p[:3] + np.outer(t, p[3:]) + 0.5 * np.outer(t ** 2, G)
        P = (cam["K"] @ (cam["R"] @ X.T + cam["t"][:, None])).T
        r = (P[:, :2] / P[:, 2:3] - obs).ravel()
        prior = 0.15 * (np.linalg.norm(p[:3] - X0))          # weak size-depth prior
        return np.r_[r, prior]
    lo = np.r_[-6, -8, 0.0, -35, -35, -35]; hi = np.r_[15, 26, 12, 35, 35, 35]
    x0 = np.clip(np.r_[X0, V0], lo + 1e-3, hi - 1e-3)
    s = least_squares(res, x0, bounds=(lo, hi), loss="soft_l1", f_scale=3.0)
    X = s.x[:3] + np.outer(t, s.x[3:]) + 0.5 * np.outer(t ** 2, G)
    rms = np.sqrt(np.mean(res(s.x)[:-1] ** 2))
    return X, rms
def flights(track_csv="ball_track.csv", events_csv="ball_events.csv"):
    d = pd.read_csv(track_csv); ev = sorted(pd.read_csv(events_csv).frame.tolist())
    seg = np.r_[0, (np.diff(d.frame) > 3).cumsum()]
    out = []
    for g, s in d.groupby(seg):
        cuts = [s.frame.iloc[0]] + [e for e in ev if s.frame.iloc[0] < e < s.frame.iloc[-1]] + [s.frame.iloc[-1] + 1]
        for a, b in zip(cuts, cuts[1:]):
            f = s[(s.frame >= a + 2) & (s.frame < b - 2)]
            if len(f) < max(6, round(10 * FPS / 59.94)): continue
            X, rms = fit_flight(f.frame.values, f.u.values, f.v.values, f.size_px.values)
            ys = X[:, 1]
            out.append(dict(f0=int(f.frame.iloc[0]), f1=int(f.frame.iloc[-1]), t0=round(f.frame.iloc[0] / FPS, 2),
                            t1=round(f.frame.iloc[-1] / FPS, 2), y_start=round(ys[0], 1), y_end=round(ys[-1], 1),
                            x_start=round(X[0, 0], 1), z_start=round(X[0, 2], 1), z_max=round(X[:, 2].max(), 1),
                            z_end=round(X[-1, 2], 1), crosses_net=bool((ys[0] - 9) * (ys[-1] - 9) < 0),
                            rms_px=round(rms, 1), valid=bool(rms < 4 and X[:, 2].min() > -0.3 and -6 < ys.min() and ys.max() < 26)))
    return pd.DataFrame(out)
if __name__ == "__main__":
    F = flights()
    if len(F) == 0: F = pd.DataFrame(columns=["f0", "f1", "t0", "t1", "y_start", "y_end", "x_start", "z_start", "z_max", "z_end", "crosses_net", "rms_px", "valid"])
    F.to_csv("flights.csv", index=False); print(f"{len(F)} flights fitted, {int(F.valid.sum()) if len(F) else 0} valid")
