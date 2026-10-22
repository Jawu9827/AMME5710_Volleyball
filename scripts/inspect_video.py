"""Print basic video info and save still frames for court-line work.

Usage:
    python scripts/inspect_video.py data/raw/myclip.mp4 --every 10
Saves one frame every N seconds to frames/<video name>/.
"""
import argparse
from pathlib import Path

import cv2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video")
    parser.add_argument("--every", type=float, default=10.0, help="seconds between saved frames")
    parser.add_argument("--out", default="frames")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Could not open {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Resolution: {w}x{h}")
    print(f"FPS:        {fps:.2f}")
    print(f"Frames:     {n_frames}")
    print(f"Duration:   {n_frames / fps:.1f} s")

    out_dir = Path(args.out) / Path(args.video).stem
    out_dir.mkdir(parents=True, exist_ok=True)

    step = max(1, int(round(fps * args.every)))
    saved = 0
    for idx in range(0, n_frames, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            break
        path = out_dir / f"frame_{idx:06d}.jpg"
        cv2.imwrite(str(path), frame)
        saved += 1

    cap.release()
    print(f"Saved {saved} frames to {out_dir}")


if __name__ == "__main__":
    main()
