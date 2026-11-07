"""Convert iPhone HDR video (HLG or Dolby Vision/HDR10) to normal SDR colours, properly tone-mapped.

Usage:
    python scripts/hdr_to_sdr.py "C:/path/IMG_1234.MOV" data/game2_B.mp4
    python scripts/hdr_to_sdr.py data/game2_B.mp4 data/game2_B_sdr.mp4          (from a HandBrake copy)
    python scripts/hdr_to_sdr.py IMG_1234.MOV test.mp4 --start 60 --duration 10  (quick test first)
    python scripts/hdr_to_sdr.py IMG_1234.MOV data/game2_B.mp4 --width 1920       (4K original -> 1080p)

Why: HDR video stores brightness with a different curve (HLG) and wider colour primaries
(BT.2020). Read as if it were normal video, colours come out washed out (our floor:
saturation 22 instead of ~65). This converts it properly:
    HLG -> linear light (reference white 203 nits, ITU-R BT.2408)
        -> BT.2020 to BT.709 colour primaries
        -> tone map highlights into SDR range (Mobius curve: leaves normal tones unchanged)
        -> standard SDR video (BT.709), H.264, quality CRF 18, audio copied.
Checked by a round trip (SDR -> HLG -> back): floor colour (hue, sat, val) 100/51/165 ->
100/49/163, mean pixel difference 3 levels out of 255, versus 13 for reading HDR naively.

Best input: the ORIGINAL file from the iPhone (10-bit). A HandBrake 8-bit copy also works,
with slightly more banding. Uses the ffmpeg bundled with imageio-ffmpeg (has zscale).
Also saves <output>_compare.jpg: the same frame before (naive) and after conversion.
"""
import argparse
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np


def ffmpeg_exe():
    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        exe = shutil.which("ffmpeg")
    if not exe:
        raise SystemExit("ffmpeg not found. Run:  pip install imageio-ffmpeg")
    return exe


def describe(path):
    r = subprocess.run([ffmpeg_exe(), "-hide_banner", "-i", str(path)], capture_output=True, text=True)
    return "\n".join(l for l in r.stderr.splitlines() if "Video:" in l)


def frame(path, t):
    cmd = [ffmpeg_exe(), "-v", "error", "-ss", f"{t:.3f}", "-i", str(path), "-frames:v", "1",
           "-f", "image2pipe", "-vcodec", "png", "-"]
    data = subprocess.run(cmd, capture_output=True).stdout
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR) if data else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("input")
    p.add_argument("output")
    p.add_argument("--start", type=float, help="convert from this time (s), for a quick test")
    p.add_argument("--duration", type=float, help="convert only this many seconds")
    p.add_argument("--tonemap", default="mobius", choices=["mobius", "hable", "reinhard", "clip"])
    p.add_argument("--white", type=float, default=203, help="HDR reference white in nits (203 = BT.2408)")
    p.add_argument("--crf", type=int, default=18, help="H.264 quality: lower = better, bigger file")
    p.add_argument("--width", type=int, help="resize to this width (e.g. 1920 if the original is 4K, "
                   "so pixel coordinates match your other 1080p files)")
    args = p.parse_args()

    src = Path(args.input)
    if not src.exists():
        # Windows hides extensions: try the usual iPhone/video ones
        found = [src.with_name(src.name + e) for e in (".MOV", ".mov", ".mp4", ".MP4", ".m4v")
                 if src.with_name(src.name + e).exists()]
        if not found:
            raise SystemExit(f"File not found: {src}\n(Run 'dir {src.parent}' to see exact names; "
                             "Windows may be hiding the extension, e.g. IMG_4857.MOV)")
        src = found[0]
        print(f"Using {src}")
    args.input = str(src)
    info = describe(args.input)
    if not info:
        r = subprocess.run([ffmpeg_exe(), "-hide_banner", "-i", args.input], capture_output=True, text=True)
        raise SystemExit("ffmpeg could not read a video stream from this file:\n" + r.stderr.strip()[-600:])
    print(info.strip())
    if "arib-std-b67" in info:
        transfer = "arib-std-b67"           # HLG (iPhone default HDR)
    elif "smpte2084" in info:
        transfer = "smpte2084"              # PQ (HDR10 / Dolby Vision base layer)
    else:
        raise SystemExit("This video is not tagged as HDR (no HLG/PQ transfer): nothing to convert.")
    print(f"HDR transfer: {'HLG' if transfer == 'arib-std-b67' else 'PQ'} -> converting to SDR BT.709")

    vf = (f"zscale=tin={transfer}:min=bt2020nc:pin=bt2020:rin=tv:t=linear:npl={args.white},"
          "format=gbrpf32le,zscale=p=bt709,"
          f"tonemap=tonemap={args.tonemap}:desat=0,"
          "zscale=t=bt709:m=bt709:r=tv,format=yuv420p")
    if args.width:
        vf += f",scale={args.width}:-2:flags=lanczos"
    cmd = [ffmpeg_exe(), "-v", "error", "-stats", "-y"]
    if args.start is not None:
        cmd += ["-ss", f"{args.start:.3f}"]
    cmd += ["-i", args.input]
    if args.duration is not None:
        cmd += ["-t", f"{args.duration:.3f}"]
    cmd += ["-vf", vf, "-c:v", "libx264", "-preset", "medium", "-crf", str(args.crf),
            "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
            "-c:a", "copy", "-movflags", "+faststart", args.output]
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(cmd, check=True)
    print(f"Saved {args.output}")

    # Before/after comparison at the same moment
    t_rel = min(1.0, args.duration / 2) if args.duration else 1.0
    before, after = frame(args.input, (args.start or 0) + t_rel), frame(args.output, t_rel)
    if before is not None and after is not None:
        w = 960
        a = cv2.resize(before, (w, int(w * before.shape[0] / before.shape[1])))
        b = cv2.resize(after, (w, int(w * after.shape[0] / after.shape[1])))
        for img, label in ((a, "before (HDR read naively)"), (b, "after (tone-mapped SDR)")):
            cv2.putText(img, label, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 5)
            cv2.putText(img, label, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        out = Path(args.output).with_name(Path(args.output).stem + "_compare.jpg")
        cv2.imwrite(str(out), np.vstack([a, b]))
        for name, img in (("before", before), ("after", after)):
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            h, w0 = hsv.shape[:2]
            fl = np.median(hsv[int(.75 * h):int(.95 * h), int(.3 * w0):int(.7 * w0)].reshape(-1, 3), axis=0)
            print(f"  floor colour {name:6s}: hue {fl[0]:.0f}, saturation {fl[1]:.0f}, brightness {fl[2]:.0f}")
        print(f"Saved {out}")


if __name__ == "__main__":
    main()
