"""Find the time offset between two recordings of the same rally/match.

Usage:
    python scripts/sync_videos.py data/camA.mp4 data/camB.mp4
    python scripts/sync_videos.py data/camA.mp4 data/camB.mp4 --guess 12.4 --search 2
    python scripts/sync_videos.py data/camA.mp4 data/camB.mp4 --offset 12.37   (manual, preview only)

If automatic sync is unreliable, find ONE clear event in both videos (a serve contact,
a whistle, the ball landing), note its time in each, and pass --guess (time_B - time_A).
The script then refines it with audio within +-search seconds.

Result convention
    offset_s = (time of an event in B) - (time of the same event in A)
    So frame f of A matches frame round((f / fps_A + offset_s) * fps_B) of B.

Methods
    audio  (default, reference). Both phones hear the same whistles, ball contacts and
           shouts. Each soundtrack becomes a multi-band "onset" signal at 1 kHz (the audio
           is split into frequency bands; in each band, rises in log energy are kept),
           which is robust to the two phones' different microphones and to echo.
           Voting: every 30 s chunk of A is matched independently against all of B. If
           many chunks vote for the same offset, it is reliable; the answer is the median
           of the agreeing chunks. A trend across them would reveal clock drift.
           Correlation is normalised by the amount of overlap at each lag, so partially
           overlapping recordings (trimmed separately) are handled.
    vision (--vision, optional). Uses no sound: a motion-energy signal per frame (how much
           of the image changed since the last frame) is cross-correlated between the two
           cameras. Players start and stop moving at the same moments in both views.
           Resolution is one frame. Compare it against the audio result.

Caveat: sound travels 343 m/s. Cameras at opposite ends ~20 m apart hear a sound made
next to one camera up to ~60 ms (~2 frames) apart. Sounds from mid-court (the referee's
whistle at the net, most ball contacts) arrive almost together, and correlating over a
whole recording averages the rest, but expect residual errors of a frame or so.

Outputs
    data/annotations/sync_<A>_<B>.json   offset, per-chunk checks, confidence
    results/sync/<A>_<B>_preview.mp4     side-by-side synced clip to check by eye
"""
import argparse
import json
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
from scipy import signal

ENV_RATE = 1000          # onset signal sample rate (Hz) -> 1 ms resolution (fine stage)
COARSE_RATE = 100        # coarse stage rate (Hz); signals blurred so that sounds reaching
COARSE_BLUR_S = 0.04     #   the two phones up to ~+-60 ms apart still overlap
FINE_WINDOW_S = 0.15     # fine stage searches +-this around the coarse answer
AUDIO_RATE = 16000       # audio decoded at this rate
CHUNK_S = 30.0           # chunk length for voting
AGREE_S = 0.1            # chunks within this of each other vote for the same offset
MIN_OVERLAP_S = 20.0     # ignore lags where the recordings overlap less than this
BANDS = [(300, 600), (600, 1200), (1200, 2400), (2400, 4800), (4800, 7900)]  # Hz


# ---------------------------------------------------------------- utilities
def ffmpeg_exe():
    """System ffmpeg, else the copy bundled with the imageio-ffmpeg pip package."""
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        raise SystemExit("ffmpeg not found. Run:  pip install imageio-ffmpeg")


def video_info(path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise SystemExit(f"Could not open {path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return dict(fps=fps, frames=n, width=w, height=h, seconds=n / fps if fps else 0)


def load_audio(path):
    cmd = [ffmpeg_exe(), "-v", "error", "-i", str(path), "-vn", "-ac", "1",
           "-ar", str(AUDIO_RATE), "-f", "s16le", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    if not raw:
        raise SystemExit(f"No audio track found in {path}")
    return np.frombuffer(raw, np.int16).astype(np.float32) / 32768.0


def onset_signal(audio):
    """Multi-band onset strength at ENV_RATE: peaks where sharp sounds start (whistle, hits)."""
    hop = AUDIO_RATE // ENV_RATE
    n = len(audio) // hop
    total = np.zeros(n)
    for lo, hi in BANDS:
        sos = signal.butter(4, (lo, hi), "bandpass", fs=AUDIO_RATE, output="sos")
        x = signal.sosfilt(sos, audio)
        energy = (x[:n * hop] ** 2).reshape(n, hop).mean(axis=1)
        log_e = np.log(energy + 1e-10)
        log_e = signal.convolve(log_e, np.ones(5) / 5, mode="same")   # 5 ms smoothing
        rise = np.maximum(np.diff(log_e, prepend=log_e[0]), 0)
        # Normalise each band so no single band (e.g. hum, a shoe squeak) dominates
        total += rise / (np.percentile(rise, 99) + 1e-9)
    total = np.minimum(total, np.percentile(total, 99.9))            # cap very loud bangs
    return (total - total.mean()) / (total.std() + 1e-9)


def motion_signal(path, width=160):
    """Fraction of changed pixels between consecutive frames, one value per frame."""
    cap = cv2.VideoCapture(str(path))
    prev, out = None, []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        g = cv2.cvtColor(cv2.resize(frame, (width, int(width * h / w))), cv2.COLOR_BGR2GRAY)
        g = cv2.GaussianBlur(g, (3, 3), 0)
        out.append(0.0 if prev is None else float(np.mean(cv2.absdiff(g, prev) > 12)))
        prev = g
    cap.release()
    m = np.asarray(out)
    m = m - signal.medfilt(m, 31)                         # keep changes, not slow trends
    return (m - m.mean()) / (m.std() + 1e-9)


def coarse(x):
    """Blur the 1 kHz onset signal and downsample to COARSE_RATE."""
    sigma = COARSE_BLUR_S * ENV_RATE
    k = np.exp(-0.5 * (np.arange(-3 * sigma, 3 * sigma + 1) / sigma) ** 2)
    y = signal.fftconvolve(x, k / k.sum(), mode="same")
    step = ENV_RATE // COARSE_RATE
    y = y[: len(y) // step * step].reshape(-1, step).mean(axis=1)
    return (y - y.mean()) / (y.std() + 1e-9)


def correlation(a, b, rate):
    """Cross-correlation as a z-score per lag (fair across overlap lengths). b(t + lag) ~ a(t)."""
    c = signal.correlate(b, a, mode="full", method="fft")
    overlap = signal.correlate(np.ones(len(b)), np.ones(len(a)), mode="full", method="fft")
    overlap = np.maximum(np.rint(overlap), 1)
    lags = np.arange(-len(a) + 1, len(b)) / rate
    # Both signals have zero mean and unit variance, so with no real match the sum over
    # N overlapping samples has spread ~sqrt(N). Dividing by sqrt(N) gives a z-score that
    # treats short and long overlaps fairly (dividing by N would let tiny overlaps win by chance).
    c = c / np.sqrt(overlap)
    min_ov = min(MIN_OVERLAP_S * rate, 0.5 * min(len(a), len(b)))
    c[overlap < min_ov] = -np.inf
    return lags, c


def best_lag(a, b, rate, lo_s=None, hi_s=None):
    """Lag (s) maximising correlation, defined so that b(t + lag) ~ a(t). Returns lag, confidence."""
    lags, c = correlation(a, b, rate)
    keep = np.isfinite(c)
    if lo_s is not None:
        keep &= lags >= lo_s
    if hi_s is not None:
        keep &= lags <= hi_s
    c, lags = c[keep], lags[keep]
    i = int(np.argmax(c))
    # Sub-sample peak by parabolic interpolation
    if 0 < i < len(c) - 1:
        y0, y1, y2 = c[i - 1], c[i], c[i + 1]
        d = 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2 + 1e-12)
    else:
        d = 0.0
    lag = lags[i] + d / rate
    # Confidence: peak height relative to the best peak away from it
    # Confidence: how far the peak stands above the rest, in units of the typical spread,
    # relative to the best peak elsewhere (>1.5 means a clear winner)
    excl = np.abs(lags - lags[i]) > 0.1
    base = np.median(c)
    second = c[excl].max() if excl.any() else base
    conf = float((c[i] - base) / (second - base)) if second > base else float("inf")
    return float(lag), conf


# ---------------------------------------------------------------- methods
def sync_audio(path_a, path_b, lo_s=None, hi_s=None):
    oa, ob = onset_signal(load_audio(path_a)), onset_signal(load_audio(path_b))
    ca, cb = coarse(oa), coarse(ob)
    R = COARSE_RATE

    # Coarse stage on blurred signals (tolerant of sound-travel delays)
    lag, conf = best_lag(ca, cb, R, lo_s, hi_s)

    # Voting: each chunk of A is matched independently against the whole of B.
    chunks = []
    n = int(CHUNK_S * R)
    for s in range(0, len(ca) - n + 1, n):
        seg = ca[s:s + n]
        if seg.std() < 1e-6:
            continue
        lo = None if lo_s is None else lo_s + s / R
        hi = None if hi_s is None else hi_s + s / R
        try:
            local, c = best_lag(seg, cb, R, lo, hi)
        except ValueError:        # no valid lag for this chunk (outside the overlap)
            continue
        chunks.append(dict(start_s=s / R, offset_s=local - s / R, confidence=c))

    result = dict(global_offset_s=lag, global_confidence=conf, chunks=chunks,
                  n_chunks=len(chunks), n_agree=0, offset_s=lag, reliable=False)
    if chunks:
        offs = np.array([ch["offset_s"] for ch in chunks])
        support = np.array([(np.abs(offs - o) <= AGREE_S).sum() for o in offs])
        best = offs[np.argmax(support)]
        agree = np.abs(offs - best) <= AGREE_S
        result["n_agree"] = int(agree.sum())
        result["offset_s"] = float(np.median(offs[agree]))
        result["agree_spread_ms"] = float(np.ptp(offs[agree]) * 1000)
        # Reliable: a clear majority of chunks (and at least 3) point at the same offset
        result["reliable"] = bool(agree.sum() >= max(3, 0.5 * len(chunks)))
        if agree.sum() >= 3:
            t = np.array([ch["start_s"] for ch in chunks])[agree]
            result["drift_ms_per_min"] = float(np.polyfit(t, offs[agree], 1)[0] * 60 * 1000)

    # If the chunks don't reach a majority, they're individually too noisy (e.g. heavily
    # compressed audio). The whole-recording match pools all the evidence, so trust it
    # when its peak is clear, but flag it for a visual check.
    if result["reliable"]:
        result["basis"] = "chunk votes"
    elif conf >= 1.5:
        result.update(offset_s=lag, basis="whole recording")
    else:
        result["basis"] = "none clear"

    # Fine stage: 1 kHz signals with light blur, searched only near the coarse answer
    c0 = result["offset_s"]
    light = lambda x: signal.convolve(x, np.ones(9) / 9, mode="same")
    try:
        fine, fconf = best_lag(light(oa), light(ob), ENV_RATE, c0 - FINE_WINDOW_S, c0 + FINE_WINDOW_S)
        result.update(coarse_offset_s=c0, offset_s=fine, fine_confidence=fconf)
    except ValueError:
        pass
    result["_signals"] = (ca, cb)
    return result


def sync_vision(path_a, path_b, fps_a, fps_b, lo_s=None, hi_s=None):
    ma, mb = motion_signal(path_a), motion_signal(path_b)
    if abs(fps_a - fps_b) > 0.01:
        # Resample B onto A's frame rate before correlating
        tb = np.arange(len(mb)) / fps_b
        mb = np.interp(np.arange(int(tb[-1] * fps_a)) / fps_a, tb, mb)
    lag, conf = best_lag(ma, mb, fps_a, lo_s, hi_s)
    return dict(offset_s=lag, confidence=conf)


def save_diagnostics(audio, out_path, lo_s=None, hi_s=None):
    """Correlation curve and chunk votes, to see why sync did or didn't work."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    oa, ob = audio["_signals"]
    lags, c = correlation(oa, ob, COARSE_RATE)
    ok = np.isfinite(c)
    if lo_s is not None:
        ok &= (lags >= lo_s) & (lags <= hi_s)
    fig, ax = plt.subplots(3, 1, figsize=(10, 8))
    ax[0].plot(lags[ok], c[ok], lw=0.5)
    ax[0].axvline(audio["offset_s"], color="r", ls="--", lw=1, label="chosen offset")
    ax[0].set_xlabel("offset B - A (s)"); ax[0].set_ylabel("correlation")
    ax[0].set_title("Whole-recording audio correlation: one tall, narrow spike = clear match")
    ax[0].legend()
    z = ok & (np.abs(lags - audio["offset_s"]) < 1.0)
    ax[1].plot(lags[z], c[z])
    ax[1].axvline(audio["offset_s"], color="r", ls="--", lw=1)
    ax[1].set_xlabel("offset B - A (s), zoomed to +-1 s")
    ch = audio["chunks"]
    if ch:
        t = [x["start_s"] for x in ch]
        o = [x["offset_s"] for x in ch]
        agree = [abs(x - audio["offset_s"]) <= AGREE_S for x in o]
        ax[2].scatter(t, o, c=["tab:green" if a else "tab:grey" for a in agree], s=18)
        ax[2].axhline(audio["offset_s"], color="r", ls="--", lw=1)
        ax[2].set_xlabel("chunk start in A (s)")
        ax[2].set_ylabel("offset voted (s)")
        ax[2].set_title(f"Chunk votes: {audio['n_agree']}/{audio['n_chunks']} agree (green)")
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------- preview
def write_preview(path_a, path_b, offset_s, start_s, seconds, out_path, width=960):
    """Side-by-side synced clip. Frames are counted by decoding from the start rather than
    seeking: OpenCV seeking in phone videos is unreliable (it can silently land at 0)."""
    ca, cb = cv2.VideoCapture(str(path_a)), cv2.VideoCapture(str(path_b))
    fa, fb = ca.get(cv2.CAP_PROP_FPS), cb.get(cv2.CAP_PROP_FPS)
    a_start = int(round(start_s * fa))
    if a_start > 300:
        print(f"  stepping to frame {a_start} of A (a few seconds per minute of video) ...")
    for _ in range(a_start):
        if not ca.grab():
            raise SystemExit(f"--preview-start {start_s} s is beyond the end of {path_a}")
    ia, ib = a_start, 0          # index of the next frame each reader will return
    last_b, writer = None, None
    for _ in range(int(seconds * fa)):
        ok, a = ca.read()
        if not ok:
            break
        fa_idx, ia = ia, ia + 1
        t_a = fa_idx / fa
        target = int(round((t_a + offset_s) * fb))      # B frame showing the same instant
        b, b_label = None, "B  (not started yet)"
        if target >= 0:
            while ib < target and cb.grab():
                ib += 1
            if ib == target:
                okb, frame_b = cb.read()
                if okb:
                    last_b, ib = frame_b, ib + 1
            b = last_b
            if b is not None:
                b_label = f"B  frame {ib - 1}  t={(ib - 1) / fb:7.2f}s"
        h = int(width * a.shape[0] / a.shape[1])
        a = cv2.resize(a, (width, h))
        b = cv2.resize(b, (width, h)) if b is not None else np.zeros_like(a)
        for img, label in ((a, f"A  frame {fa_idx}  t={t_a:7.2f}s"), (b, b_label)):
            cv2.putText(img, label, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 5)
            cv2.putText(img, label, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        frame = np.hstack([a, b])
        if writer is None:
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fa,
                                     (frame.shape[1], frame.shape[0]))
        writer.write(frame)
    ca.release(); cb.release()
    if writer:
        writer.release()


# ---------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser()
    p.add_argument("video_a")
    p.add_argument("video_b")
    p.add_argument("--vision", action="store_true", help="also run the motion-based method (slower)")
    p.add_argument("--guess", type=float, help="rough offset t_B - t_A in seconds (from one event)")
    p.add_argument("--search", type=float, default=2.0, help="search +- this many seconds around --guess")
    p.add_argument("--offset", type=float, help="use this offset (no estimation); writes JSON + preview")
    p.add_argument("--preview-start", type=float, default=None,
                   help="A-time (s) to start the side-by-side preview (default: 30 s in)")
    p.add_argument("--preview-seconds", type=float, default=10.0)
    p.add_argument("--no-preview", action="store_true")
    args = p.parse_args()

    ia, ib = video_info(args.video_a), video_info(args.video_b)
    for name, i in (("A", ia), ("B", ib)):
        print(f"{name}: {i['width']}x{i['height']}  {i['fps']:.3f} fps  {i['seconds']:.1f} s")

    stem_a, stem_b = Path(args.video_a).stem, Path(args.video_b).stem
    lo = hi = None
    if args.guess is not None:
        lo, hi = args.guess - args.search, args.guess + args.search
        print(f"\nSearching offsets {lo:+.2f} .. {hi:+.2f} s")

    result = dict(video_a=Path(args.video_a).name, video_b=Path(args.video_b).name,
                  fps_a=ia["fps"], fps_b=ib["fps"],
                  convention="t_B = t_A + offset_s; frame_B = round((frame_A / fps_A + offset_s) * fps_B)")

    if args.offset is not None:
        off = args.offset
        result.update(offset_s=off, method="manual")
        print(f"\nUsing manual offset B - A = {off:+.3f} s")
    else:
        print("\nAudio sync ...")
        audio = sync_audio(args.video_a, args.video_b, lo, hi)
        off = audio["offset_s"]
        print(f"  offset B - A = {off:+.3f} s  ({off * ia['fps']:+.2f} frames of A)")
        print(f"  chunk votes: {audio['n_agree']}/{audio['n_chunks']} thirty-second chunks agree"
              + (f" (within {audio['agree_spread_ms']:.0f} ms)" if audio["n_agree"] > 1 else ""))
        if "fine_confidence" in audio:
            print(f"  fine refinement: {audio['coarse_offset_s']:+.3f} -> {off:+.3f} s")
        print(f"  whole-recording match: {audio['global_offset_s']:+.3f} s, "
              f"confidence {audio['global_confidence']:.2f} (>1.5 clear)")
        if "drift_ms_per_min" in audio:
            print(f"  drift between phone clocks: {audio['drift_ms_per_min']:+.2f} ms per minute")
        print(f"  answer based on: {audio['basis']}")
        print("  RESULT: " + ("reliable" if audio["reliable"] else
              "PROBABLE - check the preview at a ball contact before using it"
              if audio["basis"] == "whole recording" else
              "NOT reliable - find one event by eye and rerun with --guess (see header)"))
        diag = Path("results/sync") / f"{stem_a}_{stem_b}_audio_diag.png"
        save_diagnostics(audio, diag, lo, hi)
        print(f"  diagnostics: {diag}")
        audio.pop("_signals")
        result.update(offset_s=off, method="audio" + ("+guess" if lo is not None else ""),
                      reliable=audio["reliable"], audio=audio)

    if args.vision and args.offset is None:
        print("\nVision sync (decodes both videos; can take a few minutes) ...")
        vis = sync_vision(args.video_a, args.video_b, ia["fps"], ib["fps"], lo, hi)
        diff_frames = (vis["offset_s"] - off) * ia["fps"]
        print(f"  offset B - A = {vis['offset_s']:+.3f} s   confidence {vis['confidence']:.2f}")
        print(f"  difference from audio: {diff_frames:+.2f} frames")
        result["vision"] = vis
        result["vision_minus_audio_frames"] = diff_frames

    out = Path("data/annotations") / f"sync_{stem_a}_{stem_b}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1))
    print(f"\nSaved {out}")

    if not args.no_preview:
        start = args.preview_start if args.preview_start is not None else min(30.0, ia["seconds"] / 2)
        prev = Path("results/sync") / f"{stem_a}_{stem_b}_preview.mp4"
        write_preview(args.video_a, args.video_b, off, start, args.preview_seconds, prev)
        print(f"Saved {prev}  <- watch a ball contact or whistle: both halves should match")


if __name__ == "__main__":
    main()
