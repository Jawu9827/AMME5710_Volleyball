"""Run the whole two-camera pipeline on one time window, ending with one combined video.

Usage (Game 2 files are the defaults):
    python scripts/run_pipeline.py --start 30 --duration 30
    python scripts/run_pipeline.py --start 30 --duration 30 --redo touches      # re-run from a step
    python scripts/run_pipeline.py --video-a data/x_A.mp4 --video-b data/x_B.mp4 \
        --calib-a ... --calib-b ... --sync ... --start 0 --duration 20

Needs the one-off setup steps first (README steps 2-5): calibrations and the audio sync.
Steps, each skipped if its output already exists (like run_all.py in volleyball_cv_inout/):
  1 players   players_on_court.py  -> results/players/<A>_<B>_players_from<s>.csv
  2 ball      ball_two_cameras.py  -> results/ball/<A>_<B>_ball_from<s>.json (+ .csv)
  3 calls     line_calls.py        -> results/calls/<stem>_calls.csv
  4 touches   touches_rules.py     -> results/touches/<stem>_touches.csv
  5 landing   landing_predict.py   -> results/landing/<stem>_landing.csv
  6 render    render_full.py       -> results/full/<A>_<B>_full_from<s>.mp4
--redo <step> forces that step and every later step (they depend on it); --redo all re-runs everything.
A step that changed the time window is detected from the ball run's .json and re-run.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
STEPS = ["players", "ball", "calls", "touches", "landing", "render"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video-a", default="data/game2_A.mp4")
    p.add_argument("--video-b", default="data/game2_B_sdr.mp4")
    p.add_argument("--calib-a", default="data/annotations/calib_A.json")
    p.add_argument("--calib-b", default="data/annotations/calib_B.json")
    p.add_argument("--sync", default="data/annotations/sync_game2_A_game2_B_sdr.json")
    p.add_argument("--start", type=float, required=True, help="start time in camera A (s)")
    p.add_argument("--duration", type=float, default=30.0)
    p.add_argument("--redo", nargs="*", default=[], choices=STEPS + ["all"])
    p.add_argument("--show", action="store_true", help="watch the final video while it renders")
    args = p.parse_args()

    va, vb = Path(args.video_a).stem, Path(args.video_b).stem
    s = f"{args.start:g}"
    ball_stem = f"{va}_{vb}_ball_from{s}"
    run_json = Path("results/ball") / f"{ball_stem}.json"
    players_csv = Path("results/players") / f"{va}_{vb}_players_from{s}.csv"
    common = [args.video_a, args.video_b, "--calib-a", args.calib_a, "--calib-b", args.calib_b,
              "--sync", args.sync, "--start", s, "--duration", f"{args.duration:g}"]
    steps = {
        "players": (["players_on_court.py"] + common, players_csv),
        "ball": (["ball_two_cameras.py"] + common, run_json),
        "calls": (["line_calls.py", str(run_json)], Path("results/calls") / f"{ball_stem}_calls.csv"),
        "touches": (["touches_rules.py", str(run_json), "--players", str(players_csv)],
                    Path("results/touches") / f"{ball_stem}_touches.csv"),
        "landing": (["landing_predict.py", str(run_json)], Path("results/landing") / f"{ball_stem}_landing.csv"),
        "render": (["render_full.py", str(run_json), "--players", str(players_csv)] + (["--show"] if args.show else []),
                   Path("results/full") / f"{va}_{vb}_full_from{s}.mp4"),
    }
    for f in (args.video_a, args.video_b, args.calib_a, args.calib_b, args.sync):
        if not Path(f).exists():
            raise SystemExit(f"{f} not found (calibrate and sync first: README steps 2-5)")

    redo_from = len(STEPS)
    if "all" in args.redo:
        redo_from = 0
    elif args.redo:
        redo_from = min(STEPS.index(r) for r in args.redo)
    # an existing ball run with a different duration is stale (the start is in the file name)
    if run_json.exists() and json.loads(run_json.read_text()).get("duration") != args.duration:
        print("Ball run has a different duration -> re-running from the players step")
        redo_from = 0

    t_all = time.time()
    for i, name in enumerate(STEPS):
        cmd, out = steps[name]
        if out.exists() and i < redo_from:
            print(f"[skip] {name}: {out} exists")
            continue
        print(f"\n=== {i + 1}/{len(STEPS)} {name}: {cmd[0]} ===", flush=True)
        t0 = time.time()
        r = subprocess.run([sys.executable, str(HERE / cmd[0])] + cmd[1:])
        if r.returncode != 0:
            raise SystemExit(f"\nStep {name} failed (exit code {r.returncode}). Fix the error above and run the "
                             f"same command again: finished steps are skipped.")
        redo_from = min(redo_from, i + 1)          # later steps depend on this output
        print(f"[done] {name} in {time.time() - t0:.0f} s")
    print(f"\nPipeline finished in {time.time() - t_all:.0f} s. Watch: {steps['render'][1]}")


if __name__ == "__main__":
    main()
