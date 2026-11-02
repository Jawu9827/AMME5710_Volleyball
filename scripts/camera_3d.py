"""Recover each camera's 3D pose from its floor calibration, and check it against the net.

Usage:
    python scripts/camera_3d.py data/annotations/calib_A.json data/annotations/calib_B.json \
        --video-a data/game2_A.mp4 --video-b data/game2_B_sdr.mp4 --net-height 2.43

A court calibration is a homography H from the floor (z = 0) to the image. For a pinhole
camera, H = K [r1 r2 t] (up to scale), where K holds the focal length f and principal point,
and r1, r2 are the first two columns of the rotation. Assuming square pixels and the principal
point at the image centre (true for uncropped phone video), f is the only unknown in K:
it is the value that makes r1 and r2 orthogonal and of equal length (Zhang's constraints,
solved here as a 1-D minimisation). Then r3 = r1 x r2, and the full projection matrix
P = K [r1 r2 r3 t] maps any 3D point (x, y, z) in metres to the image.

This gives:
  - each camera's position (x, y) behind the court and its height above the floor
  - the focal length and horizontal field of view (compare with the phone's lens)
  - P for both cameras, so a ball seen by both can be triangulated in 3D

Check: the net is NOT used anywhere in the calibration, but it is at a known height
(men 2.43 m, women 2.24 m; antennas on the sidelines, 0.8 m above the net). Its top is drawn
on each camera's calibration frame; if it lines up with the real net tape, the 3D model works.

Outputs: results/calib/camera3d_<camera>_net.jpg, and the summary printed.
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import minimize_scalar

sys.path.insert(0, str(Path(__file__).parent))
from calibrate_camera import draw_court, load_calibration, read_frame  # noqa: E402


class Camera3D:
    """Pinhole camera (K, R, t) plus the calibration's radial distortion."""

    def __init__(self, K, R, t, cal):
        self.K, self.R, self.t, self.cal = K, R, t.reshape(3), cal
        self.Kinv = np.linalg.inv(K)
        self.P = K @ np.c_[R, self.t]
        self.C = -R.T @ self.t                       # camera centre in court coordinates

    def project(self, X, distort=True):
        """3D court points (N x 3, metres) -> image pixels (with lens distortion)."""
        X = np.asarray(X, float).reshape(-1, 3)
        h = np.c_[X, np.ones(len(X))] @ self.P.T
        uv = h[:, :2] / h[:, 2:3]
        return self.cal.distort(uv) if distort else uv

    def normalised(self, uv):
        """Distorted pixels -> normalised camera coordinates (K^-1 of the undistorted pixel)."""
        pu = self.cal.undistort(np.asarray(uv, float).reshape(-1, 2))
        h = np.c_[pu, np.ones(len(pu))] @ self.Kinv.T
        return h[:, :2] / h[:, 2:3]

    def ray(self, uv):
        """Direction (court frame) of the ray through a pixel."""
        xn = self.normalised(uv)[0]
        d = self.R.T @ np.array([xn[0], xn[1], 1.0])
        return d / np.linalg.norm(d)


def _orthonormality_error(f, A):
    Kinv = np.diag([1 / f, 1 / f, 1.0])
    r1, r2 = Kinv @ A[:, 0], Kinv @ A[:, 1]
    n1, n2 = np.linalg.norm(r1), np.linalg.norm(r2)
    return (r1 @ r2 / (n1 * n2)) ** 2 + ((n1 - n2) / (n1 + n2)) ** 2


def camera_from_calibration(cal, size):
    """Camera3D + info dict from a CourtCalibration and the image size (w, h)."""
    w, h = size
    cx, cy = w / 2.0, h / 2.0
    T = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1.0]])
    A = T @ cal.H_c2i
    A = A / np.linalg.norm(A[:, 2])
    # 1-D search for f: coarse log grid, then bounded refinement
    grid = np.geomspace(0.25 * w, 5 * w, 400)
    errs = [_orthonormality_error(f, A) for f in grid]
    k = int(np.argmin(errs))
    lo, hi = grid[max(k - 1, 0)], grid[min(k + 1, len(grid) - 1)]
    f = float(minimize_scalar(_orthonormality_error, bounds=(lo, hi), args=(A,), method="bounded").x)
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]])
    M = np.linalg.inv(K) @ cal.H_c2i
    lam = 2.0 / (np.linalg.norm(M[:, 0]) + np.linalg.norm(M[:, 1]))
    # sign: the court centre must be in front of the camera (positive depth)
    if (M @ np.array([4.5, 9.0, 1.0]))[2] * lam < 0:
        lam = -lam
    r1, r2, t = lam * M[:, 0], lam * M[:, 1], lam * M[:, 2]
    U, _, Vt = np.linalg.svd(np.c_[r1, r2, np.cross(r1, r2)])
    R = U @ Vt                                       # nearest true rotation
    C = -R.T @ t
    # The court frame (x across, y along) is fixed by the calibration; choose z so that the
    # camera is ABOVE the floor (z up). If needed, flip the z axis (this makes R a reflection,
    # which is fine for projection and triangulation).
    if C[2] < 0:
        R = R @ np.diag([1, 1, -1.0])
    cam = Camera3D(K, R, t, cal)
    # how well P (orthonormalised) still reproduces the floor homography, over the court
    gx, gy = np.meshgrid(np.linspace(0, 9, 10), np.linspace(0, 18, 19))
    floor = np.c_[gx.ravel(), gy.ravel()]
    diff = np.linalg.norm(cam.project(np.c_[floor, np.zeros(len(floor))]) - cal.court_to_image(floor), axis=1)
    info = dict(f_px=f, hfov_deg=float(np.degrees(2 * np.arctan(w / (2 * f)))),
                vfov_deg=float(np.degrees(2 * np.arctan(h / (2 * f)))),
                position_m=[float(v) for v in cam.C], floor_consistency_px=float(np.median(diff)),
                orthonormality_error=float(_orthonormality_error(f, A)))
    return cam, info


def load_camera(calib_path):
    d = json.loads(Path(calib_path).read_text())
    cal = load_calibration(calib_path)
    cam, info = camera_from_calibration(cal, d["image_size"])
    return cam, info, d


def triangulate(cams, uvs):
    """Linear (DLT) triangulation from 2+ cameras in normalised coordinates.

    Returns (X, reprojection errors in px per camera, angle between the rays in degrees).
    A small angle (rays nearly parallel) means the depth along the rays is poorly determined.
    """
    rows = []
    for cam, uv in zip(cams, uvs):
        xn = cam.normalised(uv)[0]
        Pn = np.c_[cam.R, cam.t]
        rows += [xn[0] * Pn[2] - Pn[0], xn[1] * Pn[2] - Pn[1]]
    _, _, Vt = np.linalg.svd(np.asarray(rows))
    Xh = Vt[-1]
    X = Xh[:3] / Xh[3]
    err = [float(np.linalg.norm(cam.project(X)[0] - np.asarray(uv, float).reshape(2))) for cam, uv in zip(cams, uvs)]
    d0, d1 = cams[0].ray(uvs[0]), cams[1].ray(uvs[1])
    angle = float(np.degrees(np.arcsin(min(1.0, np.linalg.norm(np.cross(d0, d1))))))
    return X, err, angle


def draw_net(img, cam, net_h):
    v = draw_court(img, cam.cal, colour=(255, 0, 255))
    k = max(2, img.shape[1] // 800)
    s = np.linspace(0, 1, 40)[:, None]
    for z, col in ((net_h, (0, 0, 255)), (net_h - 1.0, (0, 140, 255))):    # net top and bottom
        line = cam.project(np.c_[9 * s, np.full_like(s, 9.0), np.full_like(s, z)])
        cv2.polylines(v, [np.int32(np.round(line))], False, col, k, cv2.LINE_AA)
    for x in (0.0, 9.0):                                                     # antennas
        ant = cam.project(np.c_[np.full_like(s, x), np.full_like(s, 9.0), net_h - 1.0 + 1.8 * s])
        cv2.polylines(v, [np.int32(np.round(ant))], False, (255, 255, 255), k, cv2.LINE_AA)
    for x in (0.0, 9.0):                                                     # 2 m height poles on far... corners
        for y in (0.0, 18.0):
            pole = cam.project(np.c_[np.full_like(s, x), np.full_like(s, y), 2.0 * s])
            cv2.polylines(v, [np.int32(np.round(pole))], False, (0, 255, 0), k, cv2.LINE_AA)
    return v


def main():
    p = argparse.ArgumentParser()
    p.add_argument("calibs", nargs="+", help="calib_<camera>.json files")
    p.add_argument("--video-a", help="video for the first calibration (draws the net check)")
    p.add_argument("--video-b", help="video for the second calibration")
    p.add_argument("--net-height", type=float, default=2.43, help="men 2.43, women 2.24, mixed often 2.35")
    args = p.parse_args()
    videos = [args.video_a, args.video_b]
    out = Path("results/calib")
    out.mkdir(parents=True, exist_ok=True)
    cams = []
    for i, path in enumerate(args.calibs):
        cam, info, d = load_camera(path)
        cams.append(cam)
        x, y, z = info["position_m"]
        print(f"Camera {d['camera']} ({path}):")
        print(f"  focal length {info['f_px']:.0f} px, field of view {info['hfov_deg']:.1f} x {info['vfov_deg']:.1f} deg")
        print(f"  position x = {x:.2f} m, y = {y:.2f} m, height = {z:.2f} m  "
              f"(court is x 0-9, y 0-18; so {abs(y - 18) if y > 9 else abs(y):.1f} m behind its end line)")
        print(f"  check: 3D model reproduces the floor calibration to {info['floor_consistency_px']:.2f} px "
              f"(orthonormality error {info['orthonormality_error']:.1e})")
        video = videos[i] if i < len(videos) else None
        img = read_frame(video, d["frame_time_s"]) if video else None
        if img is not None:
            if [img.shape[1], img.shape[0]] != list(d["image_size"]):
                print(f"  (skipping net image: {video} is not the calibrated size)")
            else:
                f = out / f"camera3d_{d['camera']}_net.jpg"
                cv2.imwrite(str(f), draw_net(img, cam, args.net_height))
                print(f"  net check image: {f}  (red = net top at {args.net_height} m, orange = net bottom, "
                      f"white = antennas, green = 2 m poles at the court corners)")
    if len(cams) == 2:
        d = np.linalg.norm(cams[0].C - cams[1].C)
        print(f"Distance between the cameras: {d:.1f} m")


if __name__ == "__main__":
    main()
