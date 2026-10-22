"""First pass at automatic court-line detection on a single frame.

Usage:
    python scripts/court_lines.py frames/myclip/frame_000000.jpg

Pipeline: bright, low-saturation pixels (HSV) (white lines) -> morphology ->
Canny edges -> probabilistic Hough lines.
Saves <name>_lines_debug.jpg with four panels: mask, edges, raw lines, original.
Tune the thresholds below for your footage; coloured court lines need a
different colour mask.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

# Tuning parameters
MIN_VALUE = 170          # HSV value: line pixels are bright
MAX_SATURATION = 60      # HSV saturation: white lines are unsaturated
CANNY_LOW, CANNY_HIGH = 50, 150
HOUGH_THRESHOLD = 80
MIN_LINE_LENGTH = 80     # pixels
MAX_LINE_GAP = 20        # pixels


def detect_lines(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    _, sat, val = cv2.split(hsv)
    mask = ((val > MIN_VALUE) & (sat < MAX_SATURATION)).astype(np.uint8) * 255

    # Remove speckle, then join small gaps along the lines.
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    edges = cv2.Canny(mask, CANNY_LOW, CANNY_HIGH)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, HOUGH_THRESHOLD,
                            minLineLength=MIN_LINE_LENGTH, maxLineGap=MAX_LINE_GAP)
    lines = [] if lines is None else lines.reshape(-1, 4)
    return mask, edges, lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    args = parser.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        raise SystemExit(f"Could not read {args.image}")

    mask, edges, lines = detect_lines(img)
    print(f"Found {len(lines)} line segments")

    drawn = img.copy()
    for x1, y1, x2, y2 in lines:
        cv2.line(drawn, (x1, y1), (x2, y2), (0, 0, 255), 2)

    to_bgr = lambda g: cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    top = np.hstack([to_bgr(mask), to_bgr(edges)])
    bottom = np.hstack([drawn, img])
    panel = np.vstack([top, bottom])
    panel = cv2.resize(panel, None, fx=0.5, fy=0.5)

    out = Path(args.image).with_suffix("")
    cv2.imwrite(f"{out}_lines_debug.jpg", panel)
    print(f"Saved {out}_lines_debug.jpg")


if __name__ == "__main__":
    main()
