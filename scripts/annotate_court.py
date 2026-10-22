"""Click ground-truth court points on a random sample of frames.

Usage:
    python scripts/annotate_court.py frames/volleyballmatch --n 15

For each frame, click six points in this order (the next one is shown on screen):
    1. far-left corner        2. far-right corner
    3. near-right corner      4. near-left corner
    5. near attack line meets LEFT sideline
    6. near attack line meets RIGHT sideline
Points 5 and 6 are never used by court_detect.py, so they test the detector
fairly as well as the corners.

Keys: r = redo this frame, s = skip frame (replay, point hidden), Enter = accept,
      q = save and quit.
Progress is saved after every frame, so you can stop and run it again later;
frames already done are not shown again.

Output: data/annotations/<folder name>_gt.json
"""
import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np

POINT_NAMES = [
    "far-left corner", "far-right corner", "near-right corner", "near-left corner",
    "attack line / LEFT sideline", "attack line / RIGHT sideline",
]
COURT_POINTS = [[0, 0], [9, 0], [9, 18], [0, 18], [0, 12], [9, 12]]  # metres
MAX_DISPLAY = 1400


def annotate(img, title):
    """Return list of six (x, y) clicks, 'skip', or 'quit'."""
    scale = min(1.0, MAX_DISPLAY / img.shape[1])
    display = cv2.resize(img, None, fx=scale, fy=scale)
    clicks = []

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(clicks) < len(POINT_NAMES):
            clicks.append((x / scale, y / scale))

    cv2.namedWindow(title)
    cv2.setMouseCallback(title, on_mouse)
    while True:
        view = display.copy()
        for i, (x, y) in enumerate(clicks):
            p = (int(x * scale), int(y * scale))
            cv2.circle(view, p, 5, (0, 0, 255), -1)
            cv2.putText(view, str(i + 1), (p[0] + 7, p[1] - 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        msg = (f"Click {len(clicks) + 1}: {POINT_NAMES[len(clicks)]}"
               if len(clicks) < len(POINT_NAMES) else "Enter = accept, r = redo")
        msg += "   |  s = skip, q = quit"
        cv2.rectangle(view, (0, 0), (view.shape[1], 36), (0, 0, 0), -1)
        cv2.putText(view, msg, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
        cv2.imshow(title, view)
        key = cv2.waitKey(20) & 0xFF
        if key == ord("r"):
            clicks.clear()
        elif key == ord("s"):
            cv2.destroyWindow(title)
            return "skip"
        elif key == ord("q"):
            cv2.destroyWindow(title)
            return "quit"
        elif key == 13 and len(clicks) == len(POINT_NAMES):
            cv2.destroyWindow(title)
            return clicks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder")
    parser.add_argument("--n", type=int, default=15, help="frames to annotate in total")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", help="default: data/annotations/<folder>_gt.json")
    args = parser.parse_args()

    folder = Path(args.folder)
    out = Path(args.out or f"data/annotations/{folder.name}_gt.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    data = (json.loads(out.read_text()) if out.exists()
            else {"court_points": COURT_POINTS, "point_names": POINT_NAMES,
                  "frames": {}, "skipped": []})

    frames = sorted(p.name for p in folder.glob("frame_*.jpg") if p.stem.count("_") == 1)
    random.Random(args.seed).shuffle(frames)   # same order every run, so re-runs resume
    todo = [f for f in frames if f not in data["frames"] and f not in data["skipped"]]

    while len(data["frames"]) < args.n and todo:
        name = todo.pop(0)
        img = cv2.imread(str(folder / name))
        done = len(data["frames"])
        result = annotate(img, f"{name}  ({done + 1}/{args.n})")
        if result == "quit":
            break
        if result == "skip":
            data["skipped"].append(name)
        else:
            data["frames"][name] = [[round(x, 1), round(y, 1)] for x, y in result]
        out.write_text(json.dumps(data, indent=1))

    cv2.destroyAllWindows()
    print(f"{len(data['frames'])} frames annotated, {len(data['skipped'])} skipped -> {out}")


if __name__ == "__main__":
    main()
