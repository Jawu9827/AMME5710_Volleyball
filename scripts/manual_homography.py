"""Click four court reference points to get a ground-truth homography.

Usage:
    python scripts/manual_homography.py frames/myclip/frame_000000.jpg
    python scripts/manual_homography.py frames/myclip/frame_000000.jpg --points full

Point sets (--points), clicked in the order shown, as seen from the camera:
    near-half (default): use when the far court is hidden behind the net.
        1. attack line meets LEFT sideline   2. attack line meets RIGHT sideline
        3. near-RIGHT corner                 4. near-LEFT corner
    full: use when the whole court is visible.
        1. far-left   2. far-right   3. near-right   4. near-left
Click the centre of each line intersection. Press 'r' to reset, Enter or 'q' to finish.

Outputs (next to the image):
    <name>_H.npy        3x3 homography, image pixels -> court metres
    <name>_overlay.jpg  court model projected back onto the image (check alignment)
    <name>_topdown.jpg  bird's-eye view of the court
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

# Court frame in metres: x across the 9 m width, y along the 18 m length.
# Far end line is y = 0, near end line is y = 18, centre line is y = 9.
# Lines are 5 cm wide and part of the court; these points are line centres.
POINT_SETS = {
    "near-half": np.array([[0, 12], [9, 12], [9, 18], [0, 18]], dtype=np.float32),
    "full": np.array([[0, 0], [9, 0], [9, 18], [0, 18]], dtype=np.float32),
}
COURT_LINES = [
    ((0, 0), (9, 0)),     # far end line
    ((9, 0), (9, 18)),    # right side line
    ((9, 18), (0, 18)),   # near end line
    ((0, 18), (0, 0)),    # left side line
    ((0, 9), (9, 9)),     # centre line
    ((0, 6), (9, 6)),     # far attack line
    ((0, 12), (9, 12)),   # near attack line
]
MAX_DISPLAY = 1280  # shrink large frames so they fit on screen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    parser.add_argument("--points", choices=POINT_SETS, default="near-half")
    args = parser.parse_args()
    court_pts = POINT_SETS[args.points]

    img = cv2.imread(args.image)
    if img is None:
        raise SystemExit(f"Could not read {args.image}")
    scale = min(1.0, MAX_DISPLAY / max(img.shape[:2]))
    display = cv2.resize(img, None, fx=scale, fy=scale)
    clicks = []

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(clicks) < 4:
            clicks.append((x / scale, y / scale))

    win = f"Click 4 points ({args.points}) - see script header for order"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)
    while True:
        view = display.copy()
        for i, (x, y) in enumerate(clicks):
            p = (int(x * scale), int(y * scale))
            cv2.circle(view, p, 6, (0, 0, 255), -1)
            cv2.putText(view, str(i + 1), (p[0] + 8, p[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.imshow(win, view)
        key = cv2.waitKey(20) & 0xFF
        if key == ord("r"):
            clicks.clear()
        if key in (ord("q"), 13) and len(clicks) == 4:
            break
    cv2.destroyAllWindows()

    img_pts = np.array(clicks, dtype=np.float32)
    H = cv2.getPerspectiveTransform(img_pts, court_pts)  # image -> court
    H_inv = np.linalg.inv(H)                                  # court -> image

    stem = Path(args.image).with_suffix("")
    np.save(f"{stem}_H.npy", H)
    np.save(f"{stem}_clicks.npy", img_pts)  # keep the raw clicks as ground truth

    # Project the full court model back onto the image to check the fit.
    overlay = img.copy()
    for a, b in COURT_LINES:
        pts = np.array([[a, b]], dtype=np.float32).reshape(-1, 1, 2)
        pa, pb = cv2.perspectiveTransform(pts, H_inv).reshape(-1, 2)
        cv2.line(overlay, tuple(map(int, pa)), tuple(map(int, pb)), (0, 255, 0), 2)
    cv2.imwrite(f"{stem}_overlay.jpg", overlay)

    # Bird's-eye view at 40 px per metre, with a 2 m margin.
    ppm, margin = 40, 2
    to_pixels = np.array([[ppm, 0, margin * ppm], [0, ppm, margin * ppm], [0, 0, 1]])
    size = (int((9 + 2 * margin) * ppm), int((18 + 2 * margin) * ppm))
    topdown = cv2.warpPerspective(img, to_pixels @ H, size)
    cv2.imwrite(f"{stem}_topdown.jpg", topdown)

    print("Homography (image -> court metres):\n", H)
    print(f"Saved {stem}_H.npy, {stem}_overlay.jpg, {stem}_topdown.jpg")


if __name__ == "__main__":
    main()
