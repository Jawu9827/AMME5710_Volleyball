"""Put two synced videos side by side (or stacked) in one file, from a chosen start time.

Usage (with the offset from sync_videos.py):
    python scripts/combine_videos.py data/game2_A.mp4 data/game2_B_sdr.mp4 \
        --sync data/annotations/sync_game2_A_game2_B_sdr.json --start 0 --duration 3:00
Other examples:
    python scripts/combine_videos.py data/Game1Trim.mp4 data/Game1Trim2.mp4 --start 180
    python scripts/combine_videos.py data/Game1Trim.mp4 data/Game1Trim2.mp4 --start 180 --duration 60
    python scripts/combine_videos.py A.mp4 B.mp4 --start 180 --offset -0.345      (if not already synced)
    python scripts/combine_videos.py A.mp4 B.mp4 --start 180 --layout stack       (one above the other)

--start is a time in video A (seconds, or mm:ss such as 3:00).
--offset is t_B - t_A for the same event; leave it at 0 if you trimmed the videos to line up.
--sync reads offset_s from a sync JSON (overrides --offset).
Audio is taken from video A. Both videos are converted to A's frame rate so frames pair up.

Output: results/combined/<A>_<B>_from<start>.mp4  (or --out PATH)
"""
import argparse
import shutil
import subprocess
from pathlib import Path


def ffmpeg_exe():
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        raise SystemExit("ffmpeg not found. Run:  pip install imageio-ffmpeg")


def parse_time(text):
    """'180', '3:00' or '1:02:03.5' -> seconds."""
    parts = [float(p) for p in str(text).split(":")]
    secs = 0.0
    for p in parts:
        secs = secs * 60 + p
    return secs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("video_a")
    p.add_argument("video_b")
    p.add_argument("--start", default="0", help="start time in A: seconds or mm:ss")
    p.add_argument("--duration", default=None, help="length to export (default: until a video ends)")
    p.add_argument("--offset", type=float, default=0.0, help="t_B - t_A in seconds (default 0)")
    p.add_argument("--sync", help="sync JSON from sync_videos.py (uses its offset_s)")
    p.add_argument("--layout", choices=["side", "stack"], default="side")
    p.add_argument("--width", type=int, default=960, help="width of each video in the output")
    p.add_argument("--fps", type=float, help="output frame rate (default: video A's)")
    p.add_argument("--out")
    args = p.parse_args()

    if args.sync:
        import json
        args.offset = float(json.loads(Path(args.sync).read_text())["offset_s"])
        print(f"Offset from {args.sync}: B - A = {args.offset:+.3f} s")
    import cv2
    def info(path):
        cap = cv2.VideoCapture(path)
        f, n = cap.get(cv2.CAP_PROP_FPS), cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cap.release()
        return f, (n / f if f else 0.0)
    fps_a, dur_a = info(args.video_a)
    _, dur_b = info(args.video_b)
    if args.fps is None:
        args.fps = round(fps_a, 3) if fps_a else 30.0
    start_a = parse_time(args.start)
    start_b = start_a + args.offset
    if start_b < 0:
        raise SystemExit(f"With offset {args.offset:+.3f} s, B has not started at A = {start_a} s. "
                         f"Use --start {-args.offset:.2f} or later.")

    # How much of A (from start_a) is also covered by B?
    avail = min(dur_a - start_a, dur_b - (start_a + args.offset))
    want = parse_time(args.duration) if args.duration else avail
    if want > avail + 0.05:
        print(f"Note: only {avail:.1f} s of overlap from this start; exporting {avail:.1f} s")
        want = avail
    args.duration = f"{want:.3f}"
    tag = args.start.replace(":", "m")
    out = Path(args.out or f"results/combined/{Path(args.video_a).stem}_{Path(args.video_b).stem}_from{tag}.mp4")
    out.parent.mkdir(parents=True, exist_ok=True)

    stack = "hstack" if args.layout == "side" else "vstack"
    # scale=W:-2 keeps the aspect ratio with an even height; fps= makes both 30 fps so frames
    # pair one-to-one; setpts restarts each clip's clock at 0 after the seek.
    scale = f"scale={args.width}:-2,fps={args.fps},setpts=PTS-STARTPTS"
    filt = f"[0:v]{scale}[a];[1:v]{scale}[b];[a][b]{stack}=inputs=2:shortest=1[v]"

    cmd = [ffmpeg_exe(), "-v", "error", "-stats", "-y",
           # -ss before -i seeks each input; ffmpeg decodes from the previous keyframe,
           # so the cut is frame-accurate (unlike OpenCV seeking)
           "-ss", f"{start_a:.3f}", "-i", args.video_a,
           "-ss", f"{start_b:.3f}", "-i", args.video_b,
           "-filter_complex", filt, "-map", "[v]", "-map", "0:a?",
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
           "-c:a", "aac", "-b:a", "128k", "-shortest"]
    if args.duration:
        cmd += ["-t", f"{parse_time(args.duration):.3f}"]
    cmd.append(str(out))

    print(f"A from {start_a:.2f} s, B from {start_b:.2f} s  ({args.layout}, {args.width} px each)")
    subprocess.run(cmd, check=True)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
