"""Median background of the static camera (players removed) -> bg.png in the run folder."""
import cv2, numpy as np
from common import run_args, open_video, prep
a = run_args(lambda ap: ap.add_argument("--n", type=int, default=60))
cap, fps = open_video(a.cfg["video"])
n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
f0 = int(a.cfg["start"] * fps); f1 = min(n_total - 1, int((a.cfg["start"] + max(a.cfg["dur"], 60)) * fps))
frames = []
for i in np.linspace(f0, f1, a.n).astype(int):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(i)); ok, f = cap.read()
    if ok: frames.append(prep(f))
bg = np.median(np.stack(frames), 0).astype(np.uint8)
cv2.imwrite("bg.png", bg); print(f"bg.png from {len(frames)} frames, size {bg.shape[1]}x{bg.shape[0]}")
