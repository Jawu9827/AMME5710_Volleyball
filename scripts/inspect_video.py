"""Inspect a video: format, frame-rate consistency, and frames saved for later steps.

Usage:
    python scripts/inspect_video.py data/game2_A.mp4
    python scripts/inspect_video.py data/game2_A.mp4 --every 30          # a frame every 30 s
    python scripts/inspect_video.py data/game2_A.mp4 --times 12.5 60 95  # specific times (s)

Prints: resolution, frame rate, duration, bitrate, and whether the frame rate is constant
(HandBrake and phones can produce variable frame rate, which upsets frame-based timing).
Saves: frames/<video>/t0060.00.png ... (lossless PNG, exact frame via ffmpeg)
       frames/<video>/contact_sheet.jpg  (all saved frames on one page)
"""
import argparse
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np


def ffmpeg_exe():
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        raise SystemExit("ffmpeg not found. Run:  pip install imageio-ffmpeg")


def read_frame(path, t):
    """Exact frame at time t (s) via ffmpeg (OpenCV seeking is unreliable on phone video)."""
    cmd = [ffmpeg_exe(), "-v", "error", "-ss", f"{t:.3f}", "-i", str(path), "-frames:v", "1",
           "-f", "image2pipe", "-vcodec", "png", "-"]
    data = subprocess.run(cmd, capture_output=True).stdout
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR) if data else None


def codec_info(path):
    """Codec / bitrate lines from ffmpeg's own description of the file."""
    r = subprocess.run([ffmpeg_exe(), "-hide_banner", "-i", str(path)], capture_output=True, text=True)
    return [l.strip() for l in r.stderr.splitlines() if "Stream #" in l or "Duration:" in l]


def frame_timing(path, n=600):
    """Gaps between the first n frame timestamps: constant frame rate -> all gaps equal."""
    cap = cv2.VideoCapture(str(path))
    ts = []
    for _ in range(n):
        if not cap.grab():
            break
        ts.append(cap.get(cv2.CAP_PROP_POS_MSEC))
    cap.release()
    gaps = np.diff(ts)
    return gaps[gaps > 0]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("video")
    p.add_argument("--every", type=float, default=60.0, help="save a frame every N seconds")
    p.add_argument("--times", type=float, nargs="*", help="save frames at these times (s) instead")
    p.add_argument("--out", default="frames")
    args = p.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Could not open {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    dur = n / fps if fps else 0
    size_mb = Path(args.video).stat().st_size / 1e6

    print(f"File:        {args.video}  ({size_mb:.0f} MB)")
    print(f"Resolution:  {w} x {h}")
    print(f"Frame rate:  {fps:.3f} fps   ({n} frames, {dur:.1f} s = {int(dur // 60)}:{dur % 60:04.1f})")
    print(f"Bitrate:     {size_mb * 8 / max(dur, 1e-9):.1f} Mbit/s overall")
    info = codec_info(args.video)
    for line in info:
        print(f"  {line}")
    if any("arib-std-b67" in l or "smpte2084" in l for l in info):
        print("  -> HDR video (iPhone HLG): colours decode washed out. The scripts adapt, but for new "
              "recordings turn HDR off (iPhone Settings > Camera > Record Video > HDR Video).")

    gaps = frame_timing(args.video)
    if len(gaps):
        med = np.median(gaps)
        print(f"Frame gaps (first {len(gaps)}): median {med:.2f} ms, "
              f"min {gaps.min():.2f}, max {gaps.max():.2f}")
        if gaps.max() > 1.5 * med or gaps.min() < 0.5 * med:
            print("  -> VARIABLE frame rate (frames dropped or doubled). Re-export with constant frame "
                  "rate (HandBrake: Video tab -> 'Constant Framerate'), or timing between cameras drifts.")
        elif gaps.max() - gaps.min() > 0.5:
            print("  -> nearly constant (small timing jitter, under a tenth of a frame): fine")
        else:
            print("  -> constant frame rate: good")
    if w < 1920:
        print("  -> below 1080p: the ball will be small; export at the original resolution if you can")

    times = args.times if args.times else list(np.arange(0, max(dur - 0.5, 0) + 1e-6, args.every))
    out_dir = Path(args.out) / Path(args.video).stem
    out_dir.mkdir(parents=True, exist_ok=True)
    thumbs = []
    for t in times:
        f = read_frame(args.video, float(t))
        if f is None:
            print(f"  could not read t={t:.2f} s")
            continue
        path = out_dir / f"t{t:07.2f}.png"
        cv2.imwrite(str(path), f)
        th = cv2.resize(f, (480, int(480 * f.shape[0] / f.shape[1])))
        cv2.putText(th, f"t={t:.1f}s", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
        cv2.putText(th, f"t={t:.1f}s", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        thumbs.append(th)
    if thumbs:
        cols = 4
        while len(thumbs) % cols:
            thumbs.append(np.zeros_like(thumbs[0]))
        sheet = np.vstack([np.hstack(thumbs[i:i + cols]) for i in range(0, len(thumbs), cols)])
        cv2.imwrite(str(out_dir / "contact_sheet.jpg"), sheet)
    print(f"\nSaved {len(times)} frames + contact_sheet.jpg to {out_dir}")


if __name__ == "__main__":
    main()