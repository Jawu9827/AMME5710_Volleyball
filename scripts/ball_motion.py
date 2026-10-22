"""First pass at ball candidates using background subtraction.

Usage:
    python scripts/ball_motion.py data/raw/myclip.mp4 --start 30 --duration 15

Pipeline per frame: MOG2 background subtraction -> drop shadows ->
morphology -> contours filtered by area and circularity.
Writes <name>_candidates.mp4 with candidates circled, and
<name>_candidates.csv with frame, time_s, x_px, y_px, radius_px, circularity.
"""
import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

# Tuning parameters (depend on resolution and camera distance)
MIN_AREA = 20           # pixels^2
MAX_AREA = 1500         # pixels^2
MIN_CIRCULARITY = 0.55  # 1.0 is a perfect circle; motion blur lowers this


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video")
    parser.add_argument("--start", type=float, default=0.0, help="start time (s)")
    parser.add_argument("--duration", type=float, default=15.0, help="seconds to process")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Could not open {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    start_frame = int(args.start * fps)
    n = int(args.duration * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    stem = Path(args.video).stem
    writer = cv2.VideoWriter(f"{stem}_candidates.mp4",
                             cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    bg = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=25, detectShadows=True)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    with open(f"{stem}_candidates.csv", "w", newline="") as f:
        out = csv.writer(f)
        out.writerow(["frame", "time_s", "x_px", "y_px", "radius_px", "circularity"])

        for i in range(n):
            ok, frame = cap.read()
            if not ok:
                break
            idx = start_frame + i

            fg = bg.apply(frame)
            fg = np.where(fg == 255, 255, 0).astype(np.uint8)  # shadows are 127
            fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel)
            fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel, iterations=2)

            contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                area = cv2.contourArea(c)
                if not MIN_AREA < area < MAX_AREA:
                    continue
                perimeter = cv2.arcLength(c, True)
                if perimeter == 0:
                    continue
                circularity = 4 * np.pi * area / perimeter ** 2
                if circularity < MIN_CIRCULARITY:
                    continue
                (x, y), r = cv2.minEnclosingCircle(c)
                out.writerow([idx, f"{idx / fps:.3f}", f"{x:.1f}", f"{y:.1f}",
                              f"{r:.1f}", f"{circularity:.2f}"])
                cv2.circle(frame, (int(x), int(y)), int(r) + 4, (0, 255, 255), 2)

            writer.write(frame)

    cap.release()
    writer.release()
    print(f"Saved {stem}_candidates.mp4 and {stem}_candidates.csv")


if __name__ == "__main__":
    main()
