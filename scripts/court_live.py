"""Live court detection on a video file or a camera, with the court drawn on top.

Usage:
    python scripts/court_live.py data/volleyballmatch.mp4
    python scripts/court_live.py data/volleyballmatch.mp4 --start 300 --save results/court_live.mp4
    python scripts/court_live.py 0                      # webcam / phone camera index 0

Keys: q = quit, space = pause/resume, s = save a screenshot

Each frame:
    1. Detect the court on a downscaled copy (fast; corners are scaled back up).
    2. Quality checks from court_batch_eval.py decide whether to trust it.
    3. Temporal filtering: the camera is still during a rally, so corners are
       smoothed with an exponential moving average to remove jitter. A large
       jump (zoom or camera cut) snaps straight to the new position instead.
       If detection fails, the last good court is held for a short time.
    4. Draw the court lines, a top-down mini-map, and the status / timing.
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from court_batch_eval import quality_check   # noqa: E402
from court_detect import COURT_CORNERS, COURT_LINES, detect_court, project  # noqa: E402

SNAP_FRAC = 0.03     # corner jump (x image width) treated as a zoom/cut -> snap, don't blend
HOLD_FRAMES = 15     # keep the last good court this many frames after detection fails
MAP_HEIGHT_FRAC = 0.32  # mini-map height as a fraction of the frame height


def draw_court(frame, H_inv, colour_outline=(0, 255, 0), colour_inner=(0, 200, 255)):
    for name, (a, b) in COURT_LINES.items():
        pa, pb = project(H_inv, [a, b])
        inner = "attack" in name or name == "centre"
        cv2.line(frame, tuple(map(int, pa)), tuple(map(int, pb)),
                 colour_inner if inner else colour_outline, 2, cv2.LINE_AA)


def draw_minimap(frame, state):
    """Top-down court in the top-right corner (over the crowd). Ball positions will go here later."""
    m = 1.0  # margin in metres
    ppm = MAP_HEIGHT_FRAC * frame.shape[0] / (18 + 2 * m)
    w, h = int((9 + 2 * m) * ppm), int((18 + 2 * m) * ppm)
    mini = np.full((h, w, 3), (120, 70, 30), np.uint8)       # free zone
    to_px = lambda x, y: (int((x + m) * ppm), int((y + m) * ppm))
    cv2.rectangle(mini, to_px(0, 0), to_px(9, 18), (170, 210, 230), -1)
    for _, (a, b) in COURT_LINES.items():
        cv2.line(mini, to_px(*a), to_px(*b), (255, 255, 255), max(1, int(ppm / 8)))
    if state != "TRACKING":
        mini = (mini * 0.4).astype(np.uint8)
    H, W = frame.shape[:2]
    top = int(0.06 * H)                       # below the broadcast scoreboard
    frame[top:top + h, W - w - 10:W - 10] = mini


def put_text(frame, text, org, colour=(255, 255, 255), scale=0.8):
    k = frame.shape[1] / 1280                       # keep text readable at any resolution
    org = (int(org[0] * k), int(org[1] * k))
    t = max(1, int(2 * k))
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale * k, (0, 0, 0), t + 3, cv2.LINE_AA)
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale * k, colour, t, cv2.LINE_AA)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", help="video file, or a camera index such as 0")
    parser.add_argument("--start", type=float, default=0.0, help="start time in seconds (files only)")
    parser.add_argument("--detect-width", type=int, default=960, help="width used for detection")
    parser.add_argument("--smooth", type=float, default=0.3, help="EMA weight of each new detection (1 = none)")
    parser.add_argument("--display-width", type=int, default=1280)
    parser.add_argument("--save", help="write the annotated video to this .mp4")
    parser.add_argument("--no-display", action="store_true", help="process without a window")
    parser.add_argument("--max-frames", type=int, default=0, help="stop after this many frames (0 = all)")
    args = parser.parse_args()

    src = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise SystemExit(f"Could not open {args.source}")
    is_file = not isinstance(src, int)
    fps_src = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if is_file and args.start > 0:
        cap.set(cv2.CAP_PROP_POS_MSEC, args.start * 1000)

    writer = None
    corners_s = None          # smoothed corners (full resolution)
    missed = HOLD_FRAMES + 1
    paused = False
    frame_i = 0
    t_prev = time.perf_counter()
    fps_disp = 0.0
    shot_dir = Path("results/screenshots")

    while True:
        if not paused:
            ok, frame = cap.read()
            if not ok:
                break
            frame_i += 1
            H_img, W_img = frame.shape[:2]

            # 1-2. Detect on a downscaled copy, then quality-check it
            t0 = time.perf_counter()
            scale = min(1.0, args.detect_width / W_img)
            small = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            good = False
            try:
                _, corners, _, scans, lines, _ = detect_court(small)
                good, reason, _ = quality_check(small, corners, scans, lines)
            except (ValueError, np.linalg.LinAlgError, SystemExit) as e:
                reason = str(e).splitlines()[0][:50]
            detect_ms = (time.perf_counter() - t0) * 1000

            # 3. Temporal filtering
            if good:
                corners = corners / scale
                jump = (np.abs(corners - corners_s).max() if corners_s is not None else np.inf)
                if jump > SNAP_FRAC * W_img:
                    corners_s = corners.copy()                        # zoom / cut: snap
                else:
                    corners_s = (1 - args.smooth) * corners_s + args.smooth * corners
                missed = 0
            else:
                missed += 1

            if corners_s is not None and missed == 0:
                state, colour = "TRACKING", (0, 255, 0)
            elif corners_s is not None and missed <= HOLD_FRAMES:
                state, colour = "HOLD", (0, 200, 255)
            else:
                state, colour = "NO COURT", (0, 0, 255)

            # 4. Draw
            out = frame.copy()
            if state != "NO COURT":
                H = cv2.getPerspectiveTransform(np.float32(corners_s), COURT_CORNERS)
                draw_court(out, np.linalg.inv(H),
                           colour_outline=(0, 255, 0) if state == "TRACKING" else (0, 200, 255))
            draw_minimap(out, state)

            t_now = time.perf_counter()
            fps_disp = 0.9 * fps_disp + 0.1 / max(t_now - t_prev, 1e-6) if fps_disp else 1 / max(t_now - t_prev, 1e-6)
            t_prev = t_now
            put_text(out, f"{state}" + ("" if good else f"  ({reason})"), (20, 75), colour)
            put_text(out, f"detect {detect_ms:4.1f} ms   display {fps_disp:4.1f} fps   "
                          f"t = {frame_i / fps_src + (args.start if is_file else 0):6.1f} s",
                     (20, 105), (255, 255, 255), 0.6)

            if args.save:
                if writer is None:
                    Path(args.save).parent.mkdir(parents=True, exist_ok=True)
                    writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"),
                                             fps_src, (W_img, H_img))
                writer.write(out)

            if args.max_frames and frame_i >= args.max_frames:
                break

        if args.no_display:
            continue
        d = min(1.0, args.display_width / out.shape[1])
        cv2.imshow("Court detection (q quit, space pause, s screenshot)",
                   cv2.resize(out, None, fx=d, fy=d) if d < 1 else out)

        # Play files at their real speed; cameras as fast as they arrive
        wait = 1
        if is_file and not paused:
            spare = 1000 / fps_src - (time.perf_counter() - t0) * 1000
            wait = max(1, int(spare))
        key = cv2.waitKey(wait) & 0xFF
        if key == ord("q"):
            break
        if key == ord(" "):
            paused = not paused
        if key == ord("s"):
            shot_dir.mkdir(parents=True, exist_ok=True)
            p = shot_dir / f"court_{frame_i:06d}.jpg"
            cv2.imwrite(str(p), out)
            print(f"Saved {p}")

    cap.release()
    if writer is not None:
        writer.release()
        print(f"Saved {args.save}")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
