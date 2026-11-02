"""Camera intrinsics + pose from the court homography (plane-based calibration, Zhang-style,
principal point at image centre, square pixels). Gives a full projection P = K[R|t]."""
import json, cv2, numpy as np
def camera_from_calib(calib_path="calib.json", W=None, Hh=None):
    C = json.load(open(calib_path))
    W, Hh = W or C.get("image_size", [1920, 1080])[0], Hh or C.get("image_size", [1920, 1080])[1]
    H = cv2.findHomography(np.float32(C["world_m"]), np.float32(C["image_px"]))[0]
    T = np.array([[1, 0, -W / 2], [0, 1, -Hh / 2], [0, 0, 1.]])
    Hc = T @ H                                    # principal point -> origin
    a, b = Hc[:, 0], Hc[:, 1]
    f2_orth = -(a[0] * b[0] + a[1] * b[1]) / (a[2] * b[2])                     # r1 . r2 = 0
    f2_norm = -((a[0]**2 + a[1]**2) - (b[0]**2 + b[1]**2)) / (a[2]**2 - b[2]**2)  # |r1| = |r2|
    f = float(np.sqrt(np.mean([v for v in (f2_orth, f2_norm) if v > 0])))
    K = np.array([[f, 0, W / 2], [0, f, Hh / 2], [0, 0, 1.]])
    M = np.linalg.inv(K) @ H
    lam = 2 / (np.linalg.norm(M[:, 0]) + np.linalg.norm(M[:, 1]))
    r1, r2, t = lam * M[:, 0], lam * M[:, 1], lam * M[:, 2]
    if t[2] < 0: r1, r2, t = -r1, -r2, -t
    R = np.column_stack([r1, r2, np.cross(r1, r2)])
    U, _, Vt = np.linalg.svd(R); R = U @ Vt       # nearest rotation
    if np.linalg.det(R) < 0: R[:, 2] *= -1
    # world z must point UP: camera centre should have z > 0
    Cw = -R.T @ t
    if Cw[2] < 0:
        R[:, 2] *= -1; Cw = -R.T @ t
    return dict(K=K, R=R, t=t, C=Cw, f=f, sqrt_f2=(np.sqrt(abs(f2_orth)), np.sqrt(abs(f2_norm))))

BALL_D = 0.21  # m
SIZE_CORR = 1.22  # yellow-only mask under-measures a yellow/blue ball; calibrated on rolling-ball frames (z -> 0.1 m)
def ball_3d(cam, u, v, d_px):
    """Back-project pixel along its ray to the depth implied by apparent diameter."""
    Z = cam["f"] * BALL_D / max(d_px * SIZE_CORR, 1e-3)       # camera-frame depth
    xc = np.linalg.inv(cam["K"]) @ np.array([u, v, 1.]) * Z
    return cam["R"].T @ (xc - cam["t"])           # world (x, y, z)

def project(cam, X):
    p = cam["K"] @ (cam["R"] @ np.asarray(X, float) + cam["t"]); return p[:2] / p[2]

if __name__ == "__main__":
    cam = camera_from_calib()
    print("f px %.0f (orth %.0f, norm %.0f)" % (cam["f"], *cam["sqrt_f2"]))
    print("camera centre (m):", cam["C"].round(2))
    for X in ([0, 9, 2.43], [9, 9, 2.43], [-0.92, 9, 0], [4.5, 9, 2.43]):   # net top (men's height) and post base
        print(X, "->", project(cam, X).round(0))
