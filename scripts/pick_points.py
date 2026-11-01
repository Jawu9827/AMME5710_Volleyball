"""Click points on a frame, save their pixel coordinates, and plot saved points.

Usage:
    python scripts/pick_points.py frames/game2_A/t0060.00.png
    python scripts/pick_points.py data/game2_A.mp4 --time 60
    python scripts/pick_points.py data/game2_A.mp4 --time 60 --names "near-left" "near-right" "net-left"
    python scripts/pick_points.py frames/game2_A/t0120.00.png --show results/points/game2_A_t60.json

Clicking: a 5x magnifier (top-right) follows the mouse; put its crosshair on the spot and
click. With --names it asks for each name in turn; otherwise points are numbered.
Keys: i/j/k/l nudge the last point 1 px (up/left/down/right), u undo, Enter save, q quit.

Saves results/points/<name>.json  ({"points": [{"name", "x", "y"}], image size, source})
and   results/points/<name>.png   (the frame with the points drawn and labelled).
--show draws a saved JSON onto any image or video frame (e.g. to check a later frame).
"""
import argparse
import json
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np

MAX_DISPLAY = 1400
ZOOM, HALF = 5, 30


def ffmpeg_exe():
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        raise SystemExit("ffmpeg not found. Run:  pip install imageio-ffmpeg")


def load_image(source, t):
    if Path(source).suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp"):
        img = cv2.imread(str(source))
    else:
        cmd = [ffmpeg_exe(), "-v", "error", "-ss", f"{t:.3f}", "-i", str(source), "-frames:v", "1",
               "-f", "image2pipe", "-vcodec", "png", "-"]
        data = subprocess.run(cmd, capture_output=True).stdout
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR) if data else None
    if img is None:
        raise SystemExit(f"Could not read an image from {source}")
    return img


def draw_points(img, points, colour=(0, 0, 255)):
    out = img.copy()
    k = max(1.0, img.shape[1] / 1280)
    for pnt in points:
        x, y = int(round(pnt["x"])), int(round(pnt["y"]))
        cv2.drawMarker(out, (x, y), colour, cv2.MARKER_CROSS, int(18 * k), max(1, int(2 * k)))
        cv2.putText(out, pnt["name"], (x + int(8 * k), y - int(8 * k)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6 * k, (0, 0, 0), int(4 * k))
        cv2.putText(out, pnt["name"], (x + int(8 * k), y - int(8 * k)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6 * k, (255, 255, 255), max(1, int(1.5 * k)))
    return out


def click(img, names):
    scale = min(1.0, MAX_DISPLAY / img.shape[1])
    disp = cv2.resize(img, None, fx=scale, fy=scale)
    H, W = img.shape[:2]
    pts, pending, mouse = [], [], [0, 0]

    def cb(event, x, y, flags, param):
        mouse[0], mouse[1] = x, y
        if event == cv2.EVENT_LBUTTONDOWN:
            pending.append((x / scale, y / scale))

    win = "Pick points (magnifier top-right) | ijkl nudge  u undo  Enter save  q quit"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, cb)
    while True:
        if pending and (not names or len(pts) < len(names)):
            x, y = pending[-1]
            pts.append(dict(name=names[len(pts)] if names else f"P{len(pts) + 1}", x=x, y=y))
        pending.clear()
        view = cv2.resize(draw_points(img, pts), (disp.shape[1], disp.shape[0]))
        # magnifier
        x0, y0 = int(round(mouse[0] / scale)) - HALF, int(round(mouse[1] / scale)) - HALF
        patch = np.zeros((2 * HALF + 1, 2 * HALF + 1, 3), np.uint8)
        xa, ya, xb, yb = max(0, x0), max(0, y0), min(W, x0 + 2 * HALF + 1), min(H, y0 + 2 * HALF + 1)
        if xb > xa and yb > ya:
            patch[ya - y0:yb - y0, xa - x0:xb - x0] = img[ya:yb, xa:xb]
        for q in pts:
            u, v = int(round(q["x"])) - x0, int(round(q["y"])) - y0
            if 0 <= u <= 2 * HALF and 0 <= v <= 2 * HALF:
                patch[v, u] = (0, 0, 255)
        mag = cv2.resize(patch, None, fx=ZOOM, fy=ZOOM, interpolation=cv2.INTER_NEAREST)
        c = mag.shape[0] // 2
        cv2.line(mag, (c, 0), (c, mag.shape[0]), (0, 255, 255), 1)
        cv2.line(mag, (0, c), (mag.shape[1], c), (0, 255, 255), 1)
        if view.shape[0] > mag.shape[0] + 40:
            view[40:40 + mag.shape[0], view.shape[1] - mag.shape[1]:] = mag
        u, v = mouse[0] / scale, mouse[1] / scale
        msg = (f"Click: {names[len(pts)]}" if names and len(pts) < len(names) else
               "All named points done - Enter to save" if names else f"Click point P{len(pts) + 1}")
        msg += f"   cursor ({u:.0f}, {v:.0f})"
        cv2.rectangle(view, (0, 0), (view.shape[1], 34), (0, 0, 0), -1)
        cv2.putText(view, msg, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.imshow(win, view)
        key = cv2.waitKey(20) & 0xFF
        nudge = {ord("i"): (0, -1), ord("k"): (0, 1), ord("j"): (-1, 0), ord("l"): (1, 0)}
        if key in nudge and pts:
            pts[-1]["x"] += nudge[key][0]
            pts[-1]["y"] += nudge[key][1]
        elif key == ord("u") and pts:
            pts.pop()
        elif key == 13 and pts:
            break
        elif key == ord("q"):
            cv2.destroyAllWindows()
            raise SystemExit("Quit without saving")
    cv2.destroyAllWindows()
    return pts


def main():
    p = argparse.ArgumentParser()
    p.add_argument("source", help="image, or video with --time")
    p.add_argument("--time", type=float, default=0.0)
    p.add_argument("--names", nargs="*", help="names to click, in order")
    p.add_argument("--show", help="plot points from this JSON instead of clicking")
    p.add_argument("--out", help="output name (default: from the source and time)")
    args = p.parse_args()

    img = load_image(args.source, args.time)
    is_video = Path(args.source).suffix.lower() not in (".png", ".jpg", ".jpeg", ".bmp")
    stem = args.out or (Path(args.source).stem + (f"_t{args.time:g}" if is_video else ""))
    out_dir = Path("results/points")
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.show:
        pts = json.loads(Path(args.show).read_text())["points"]
        out = out_dir / f"{stem}_with_{Path(args.show).stem}.png"
        cv2.imwrite(str(out), draw_points(img, pts))
        print(f"Saved {out}")
        return

    pts = click(img, args.names)
    for q in pts:
        print(f"  {q['name']:>20s}: ({q['x']:.1f}, {q['y']:.1f})")
    data = dict(source=str(args.source), time_s=args.time if is_video else None,
                image_size=[img.shape[1], img.shape[0]],
                points=[dict(name=q["name"], x=round(q["x"], 2), y=round(q["y"], 2)) for q in pts])
    (out_dir / f"{stem}.json").write_text(json.dumps(data, indent=1))
    cv2.imwrite(str(out_dir / f"{stem}.png"), draw_points(img, pts))
    print(f"Saved {out_dir / (stem + '.json')} and .png")


if __name__ == "__main__":
    main()
