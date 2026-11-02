"""
One-time court calibration per camera placement (camera must not move).

  python calibrate.py --run runs/IMG_4833          (run_all.py calls this for you)

Click the landmarks in the order printed (press 's' to skip one you can't see,
'u' to undo, Enter when done). >=4 non-collinear points needed; 6-8 is better
(RANSAC + reprojection error reported). Coordinates: x across 0..9 m (left->right
as seen from THIS camera), y along 0..18 m (0 = baseline nearest this camera).
"""
import json, cv2, numpy as np
from common import run_args, open_video, prep

LANDMARKS = [  # name, (x, y) metres
    ("near-left corner (baseline x sideline)", (0, 0)),
    ("near-right corner", (9, 0)),
    ("near attack line x left sideline", (0, 6)),
    ("near attack line x right sideline", (9, 6)),
    ("centre line x left sideline (under net)", (0, 9)),
    ("centre line x right sideline", (9, 9)),
    ("far attack line x left sideline", (0, 12)),
    ("far attack line x right sideline", (9, 12)),
    ("far-left corner", (0, 18)),
    ("far-right corner", (9, 18)),
]

a = run_args(); a.out = "calib.json"
import os
img = cv2.imread("bg.png") if os.path.exists("bg.png") else None   # empty court = easiest to click
if img is None:
    cap, fps = open_video(a.cfg["video"]); cap.set(cv2.CAP_PROP_POS_MSEC, a.cfg["start"] * 1000)
    ok, img = cap.read(); assert ok, "could not read frame"; img = prep(img)
h, w = img.shape[:2]; sc = min(1.0, 1280 / w)
clicks, idx = [], [0]

def redraw():
    d = cv2.resize(img, None, fx=sc, fy=sc)
    for (u, v), (_, (x, y)) in clicks:
        cv2.circle(d, (int(u * sc), int(v * sc)), 5, (0, 0, 255), -1)
        cv2.putText(d, f"({x},{y})", (int(u * sc) + 6, int(v * sc) - 6), 0, 0.5, (0, 0, 255), 1)
    if idx[0] < len(LANDMARKS):
        cv2.putText(d, "Click: " + LANDMARKS[idx[0]][0], (10, 30), 0, 0.8, (0, 255, 255), 2)
    cv2.imshow("calib", d)

def on_mouse(ev, x, y, *_):
    if ev == cv2.EVENT_LBUTTONDOWN and idx[0] < len(LANDMARKS):
        clicks.append(((x / sc, y / sc), LANDMARKS[idx[0]])); idx[0] += 1; redraw()

cv2.namedWindow("calib"); cv2.setMouseCallback("calib", on_mouse); redraw()
while True:
    k = cv2.waitKey(0) & 0xFF
    if k in (13, 10): break
    if k == ord("s") and idx[0] < len(LANDMARKS): idx[0] += 1
    if k == ord("u") and clicks: clicks.pop(); idx[0] = LANDMARKS.index(clicks[-1][1]) + 1 if clicks else 0
    redraw()
cv2.destroyAllWindows()

img_px = np.float32([c[0] for c in clicks]); wld = np.float32([c[1][1] for c in clicks])
H, inl = cv2.findHomography(wld, img_px, cv2.RANSAC if len(clicks) > 4 else 0, 4.0)
proj = cv2.perspectiveTransform(wld[None], H)[0]
err = np.linalg.norm(proj - img_px, axis=1)
print("reprojection error px:", np.round(err, 1), "mean", err.mean().round(2))

# detection ROI = court + 2 m margin, padded upward by a player height (~2.2 m)
ring = np.float32([[-2, -2], [11, -2], [11, 20], [-2, 20]])
rp = cv2.perspectiveTransform(ring[None], H)[0]
x0, x1 = int(max(0, rp[:, 0].min())), int(min(w, rp[:, 0].max()))
far_y = int(rp[2:, 1].min()); y0 = max(0, far_y - 120); y1 = h
far_net = cv2.perspectiveTransform(np.float32([[[-1, 9], [10, 9], [-1, 20], [10, 20]]]), H)[0]
fx0, fx1 = int(max(0, far_net[:, 0].min() - 60)), int(min(w, far_net[:, 0].max() + 60))
fy1 = int(min(h, far_net[:2, 1].max() + 90))
json.dump({"image_px": img_px.tolist(), "world_m": wld.tolist(), "image_size": [w, h],
           "roi": [x0, y0, x1, y1], "far_tiles": [[fx0, y0, fx1, fy1]],
           "gate_x": [-1.5, 10.5], "gate_y": [-2.5, 20.5]}, open(a.out, "w"), indent=1)
chk = img.copy(); P = lambda x, y: tuple(int(v) for v in cv2.perspectiveTransform(np.float32([[[x, y]]]), H)[0, 0])
for p, q in [((0, 0), (9, 0)), ((9, 0), (9, 18)), ((9, 18), (0, 18)), ((0, 18), (0, 0)), ((0, 6), (9, 6)), ((0, 12), (9, 12))]:
    cv2.line(chk, P(*p), P(*q), (0, 0, 255), 2)
cv2.line(chk, P(-0.5, 9), P(9.5, 9), (255, 0, 255), 3)
cv2.imwrite("calib_check.jpg", chk)
print("saved", a.out, "+ calib_check.jpg (red court model, magenta net): it should sit on the painted lines")
