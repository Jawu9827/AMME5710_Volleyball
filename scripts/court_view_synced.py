"""Both cameras, synced, with each camera's calibrated court drawn on it, plus a top-down map.

Usage:
    python scripts/court_view_synced.py data/game2_A.mp4 data/game2_B_sdr.mp4 \
        --calib-a data/annotations/calib_A.json --calib-b data/annotations/calib_B.json \
        --sync data/annotations/sync_game2_A_game2_B_sdr.json --start 30 --duration 60
    add --show to watch it live as well (q quit, space pause)

Each camera judges the half of the court nearest to it (it sees that half at the best
resolution), so that half is shaded on its view and on the map:
    camera A (--end near): y = 9..18 m      camera B (--end far): y = 0..9 m
The map is the shared court frame (x 0-9 m across, y 0-18 m along, net at y = 9).
Ball bounces will be plotted on it in the next step.

Also checks, every second, that each camera has not moved since its calibration frame
(phase correlation of the walls/ceiling, as in calibrate_camera.py); a moved camera is
flagged in red, because its court lines would then be wrong.

Output: results/synced/<A>_<B>_court_from<start>.mp4 (H.264, with camera A's sound)
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from calibrate_camera import (COURT_LINES, draw_court, ffmpeg_exe, load_calibration,  # noqa: E402
                              read_frame, static_shift)

HALF = {"near": ((0, 9), (9, 9), (9, 18), (0, 18)),     # camera behind y=18 judges y 9..18
        "far": ((0, 0), (9, 0), (9, 9), (0, 9))}        # camera behind y=0 judges y 0..9
COLOURS = {"A": (0, 200, 255), "B": (80, 230, 80)}     # BGR: orange for A, green for B (blue vanishes on the floor)


def shade_half(img, cal, end, colour, alpha=0.18):
    pts = []
    for (x0, y0), (x1, y1) in zip(HALF[end], HALF[end][1:] + HALF[end][:1]):
        s = np.linspace(0, 1, 30)[:, None]
        pts.append(cal.court_to_image(np.array([x0, y0]) * (1 - s) + np.array([x1, y1]) * s))
    poly = np.int32(np.round(np.vstack(pts)))
    over = img.copy()
    cv2.fillPoly(over, [poly], colour)
    return cv2.addWeighted(over, alpha, img, 1 - alpha, 0)


def court_map(height, ends, moved):
    """Top-down court: 9 x 18 m with 1 m margin, net at y = 9."""
    m = 1.0
    ppm = height / (18 + 2 * m)
    w = int(round((9 + 2 * m) * ppm)) // 2 * 2
    img = np.full((height, w, 3), (90, 60, 40), np.uint8)
    P = lambda x, y: (int(round((x + m) * ppm)), int(round((y + m) * ppm)))
    cv2.rectangle(img, P(0, 0), P(9, 18), (170, 140, 110), -1)
    for cam, end in ends.items():
        (x0, y0), _, (x1, y1), _ = HALF[end]
        over = img.copy()
        cv2.rectangle(over, P(x0, y0), P(x1, y1), COLOURS[cam], -1)
        img = cv2.addWeighted(over, 0.25 if not moved[cam] else 0.08, img, 0.75 if not moved[cam] else 0.92, 0)
        cy = 18.5 if end == "near" else -0.5
        cv2.putText(img, f"camera {cam}", (P(0, cy)[0], P(0, cy)[1] + 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (255, 255, 255), 1, cv2.LINE_AA)
    for a, b in COURT_LINES:
        cv2.line(img, P(*a), P(*b), (0, 230, 255), 2, cv2.LINE_AA)
    cv2.line(img, P(-0.5, 9), P(9.5, 9), (255, 255, 255), 3, cv2.LINE_AA)   # net
    return img


def label(img, text, colour, y=34):
    cv2.putText(img, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 5, cv2.LINE_AA)
    cv2.putText(img, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, colour, 2, cv2.LINE_AA)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("video_a")
    p.add_argument("video_b")
    p.add_argument("--calib-a", required=True)
    p.add_argument("--calib-b", required=True)
    p.add_argument("--sync", required=True, help="sync JSON (offset_s = t_B - t_A)")
    p.add_argument("--start", type=float, default=0.0, help="start time in camera A (s)")
    p.add_argument("--duration", type=float, default=60.0)
    p.add_argument("--width", type=int, default=960, help="width of each camera view")
    p.add_argument("--show", action="store_true", help="also display while rendering")
    args = p.parse_args()

    offset = float(json.loads(Path(args.sync).read_text())["offset_s"])
    cals, ends, refs = {}, {}, {}
    for cam, path, video in (("A", args.calib_a, args.video_a), ("B", args.calib_b, args.video_b)):
        d = json.loads(Path(path).read_text())
        cals[cam], ends[cam] = load_calibration(path), d["end"]
        refs[cam] = read_frame(video, d["frame_time_s"])      # calibration frame, for the tripod check
        if refs[cam] is None:
            raise SystemExit(f"Could not read {video}")
        if list(d["image_size"]) != [refs[cam].shape[1], refs[cam].shape[0]]:
            raise SystemExit(f"{path} was made at {d['image_size']} but {video} is "
                             f"{refs[cam].shape[1]}x{refs[cam].shape[0]}")
    if ends["A"] == ends["B"]:
        print("Warning: both calibrations use the same --end; cameras should be at opposite ends")

    ca, cb = cv2.VideoCapture(args.video_a), cv2.VideoCapture(args.video_b)
    fa, fb = ca.get(cv2.CAP_PROP_FPS), cb.get(cv2.CAP_PROP_FPS)
    a0 = int(round(args.start * fa))
    if args.start + offset < 0:
        raise SystemExit(f"Camera B has not started at A = {args.start} s; use --start {-offset:.1f} or later")
    print(f"Offset B - A = {offset:+.3f} s. Stepping to frame {a0} of A (decoding, not seeking) ...")
    for _ in range(a0):
        if not ca.grab():
            raise SystemExit("--start is beyond the end of video A")
    ia, ib = a0, 0
    last_b = None

    # Output: frames piped to ffmpeg (H.264) with camera A's audio for the same time span
    # Each view keeps its own aspect ratio; the shorter one is padded to the common height
    vh = {cam: int(round(args.width * refs[cam].shape[0] / refs[cam].shape[1])) for cam in refs}
    h = max(vh.values()) // 2 * 2 + (2 if max(vh.values()) % 2 else 0)
    map_w = court_map(h, ends, {"A": False, "B": False}).shape[1]
    W = 2 * args.width + map_w
    out = Path("results/synced") / f"{Path(args.video_a).stem}_{Path(args.video_b).stem}_court_from{args.start:g}.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    ff = subprocess.Popen([ffmpeg_exe(), "-v", "error", "-y",
                           "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{h}", "-r", f"{fa:.5f}", "-i", "-",
                           "-ss", f"{args.start:.3f}", "-t", f"{args.duration:.3f}", "-i", args.video_a,
                           "-map", "0:v", "-map", "1:a?", "-c:v", "libx264", "-preset", "veryfast",
                           "-crf", "20", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(out)],
                          stdin=subprocess.PIPE)

    moved = {"A": False, "B": False}
    shift = {"A": 0.0, "B": 0.0}
    n_frames = int(args.duration * fa)
    paused = False
    for k in range(n_frames):
        ok, a = ca.read()
        if not ok:
            break
        t_a = ia / fa
        ia += 1
        target = int(round((t_a + offset) * fb))
        while ib < target and cb.grab():
            ib += 1
        if ib == target:
            okb, fr = cb.read()
            if okb:
                last_b, ib = fr, ib + 1
        b = last_b
        if b is None:
            break

        if k % max(1, int(round(fa))) == 0:                 # tripod check once a second
            for cam, img in (("A", a), ("B", b)):
                shift[cam] = static_shift(refs[cam], img)[0]
                moved[cam] = shift[cam] > 2.0

        views = []
        for cam, img in (("A", a), ("B", b)):
            v = shade_half(img, cals[cam], ends[cam], COLOURS[cam])
            v = draw_court(v, cals[cam], colour=(255, 0, 255) if not moved[cam] else (0, 0, 255))
            v = cv2.resize(v, (args.width, vh[cam]))
            v = cv2.copyMakeBorder(v, 0, h - vh[cam], 0, 0, cv2.BORDER_CONSTANT)
            status = f"camera {cam}  t={t_a + (offset if cam == 'B' else 0):6.2f}s"
            label(v, status, (255, 255, 255))
            label(v, "camera moved - court lines invalid" if moved[cam] else f"tripod OK ({shift[cam]:.1f}px)",
                  (0, 0, 255) if moved[cam] else (0, 255, 0), y=64)
            views.append(v)
        frame = np.hstack(views + [court_map(h, ends, moved)])
        ff.stdin.write(frame.tobytes())

        if args.show:
            cv2.imshow("synced court view (q quit, space pause)", frame)
            key = cv2.waitKey(0 if paused else 1) & 0xFF
            if key == ord("q"):
                break
            if key == ord(" "):
                paused = not paused
        if k and k % int(10 * fa) == 0:
            print(f"  {k / fa:.0f} s done")

    ff.stdin.close()
    ff.wait()
    ca.release(); cb.release()
    cv2.destroyAllWindows()
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
