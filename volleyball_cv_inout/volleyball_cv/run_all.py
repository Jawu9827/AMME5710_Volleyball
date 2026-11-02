"""
One command for the whole pipeline.

  python run_all.py --video "C:/path/IMG_4833.mp4" --start 0 --dur 60

Creates runs/<video name>/ and runs, in order:
  1 make_background  -> bg.png              (median of 60 frames = empty court)
  2 calibrate        -> calib.json          (opens a window: click court landmarks; once per camera placement)
  3 vb_tracker       -> positions.csv, tracking.mp4        (players, pose, court minimap)
  4 ball_track       -> ball_track.csv                     (full frame rate)
  5 ball_events      -> ball_events.csv
  6 touches          -> touches.csv, touch_evidence.jpg    (RTMPose attribution)
  7 flights          -> flights.csv                        (ballistic 3-D fits)
  8 rules            -> rallies.csv                        (rallies, 3-touch, double contact)
  9 landing          -> landings.csv                       (IN/OUT: measured at bounce + predicted in flight)
 10 render_overlay   -> touches_overlay.mp4                (final video)
Re-running skips finished steps unless --redo names them, e.g. --redo rules (later steps re-run automatically)
"""
import argparse, json, os, shutil, subprocess, sys, time, cv2
HERE = os.path.dirname(os.path.abspath(__file__))
STEPS = [("make_background", "bg.png"), ("calibrate", "calib.json"), ("vb_tracker", "positions.csv"),
         ("ball_track", "ball_track.csv"), ("ball_events", "ball_events.csv"), ("touches", "touches.csv"),
         ("flights", "flights.csv"), ("rules", "rallies.csv"), ("landing", "landings.csv"), ("render_overlay", "touches_overlay.mp4")]
ap = argparse.ArgumentParser()
ap.add_argument("--video", required=True)
ap.add_argument("--start", type=float, default=0, help="seconds")
ap.add_argument("--dur", type=float, default=60, help="seconds to process")
ap.add_argument("--det-fps", type=float, default=10, help="player-detection rate (ball runs at full rate)")
ap.add_argument("--calib", default=None, help="reuse a calib.json from an earlier run (same camera placement)")
ap.add_argument("--run", default=None, help="run folder (default runs/<video name>)")
ap.add_argument("--redo", nargs="*", default=[], help="steps to force re-run (or 'all')")
a = ap.parse_args()

video = os.path.abspath(a.video)
cap = cv2.VideoCapture(video)
if not cap.isOpened(): sys.exit(f"cannot open {video}")
fps = cap.get(cv2.CAP_PROP_FPS); n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
w, h = int(cap.get(3)), int(cap.get(4)); cap.release()
length = n / fps if fps else 0
dur = min(a.dur, max(0, length - a.start))
print(f"video {os.path.basename(video)}: {w}x{h} @ {fps:.2f} fps, {length:.1f} s -> processing {a.start:.1f}-{a.start + dur:.1f} s")
run = os.path.abspath(a.run or os.path.join(HERE, "runs", os.path.splitext(os.path.basename(video))[0]))
os.makedirs(run, exist_ok=True)
cfg = dict(video=video, fps=fps, start=a.start, dur=dur, det_fps=a.det_fps, width=w, height=h)
old = json.load(open(os.path.join(run, "run.json"))) if os.path.exists(os.path.join(run, "run.json")) else None
json.dump(cfg, open(os.path.join(run, "run.json"), "w"), indent=1)
if old and (old["start"], old["dur"], old["det_fps"]) != (cfg["start"], cfg["dur"], cfg["det_fps"]):
    a.redo = list({*a.redo, "vb_tracker", "ball_track", "ball_events", "touches", "flights", "rules", "landing", "render_overlay"})
    print("time window changed -> re-running analysis steps")
if a.calib: shutil.copy(a.calib, os.path.join(run, "calib.json"))

# download the YOLO weights once into the package folder
if not os.path.exists(os.path.join(HERE, "yolo11m-pose.pt")):
    cwd = os.getcwd(); os.chdir(HERE)
    from ultralytics import YOLO; YOLO("yolo11m-pose.pt"); os.chdir(cwd)

env = dict(os.environ, KMP_DUPLICATE_LIB_OK="TRUE")   # Anaconda + PyTorch OpenMP clash ("OMP: Error #15")
cascade = False
for step, out in STEPS:
    marker = os.path.join(run, f".done_{step}")
    path = os.path.join(run, out)
    have = os.path.exists(path) and os.path.getsize(path) > 0
    # bg / calibration: an existing non-empty file is enough; analysis steps need a success marker
    done = have and (step in ("make_background", "calibrate") or os.path.exists(marker))
    if done and not cascade and step not in a.redo and "all" not in a.redo:
        print(f"[skip] {step} ({out} done)"); continue
    print(f"\n=== {step} ===", flush=True); t0 = time.time()
    if os.path.exists(marker): os.remove(marker)
    r = subprocess.run([sys.executable, os.path.join(HERE, step + ".py"), "--run", run], env=env)
    if r.returncode != 0:
        sys.exit(f"\nstep {step} FAILED (exit code {r.returncode}). Scroll up for the first error message, "
                 f"fix it, then re-run the same command: finished steps are skipped.")
    open(marker, "w").write(time.strftime("%Y-%m-%d %H:%M:%S"))
    if step not in ("make_background", "calibrate"): cascade = True   # later steps depend on this output
    print(f"[done] {step} in {time.time() - t0:.0f} s")
print(f"\nAll outputs in: {run}\nWatch: {os.path.join(run, 'touches_overlay.mp4')}")
