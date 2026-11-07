"""Score the pipeline's output on the synthetic clip against the exact ground truth.

Usage (from the repo root, after synthetic_render.py, ball_two_cameras.py and line_calls.py):
    python scripts/synthetic_score.py results/ball/A_B_ball_from0.3.csv results/calls/A_B_ball_from0.3_calls.csv

Reports
    3D ball   frames tracked in 3D / frames with a ball in flight, false 3D points (more than 0.5 m
              from the true ball), median and 90th-percentile 3D error
    sync      refined offset (from the run's .json) against the true 2.017 s
    calls     each true landing matched to the nearest detected bounce within MATCH_M (else
              MISSED): estimated contact point,
              error across the judged line, and whether the call is correct
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from synthetic_render import OFF_TRUE, TRUE_LANDINGS, ball  # noqa: E402

MATCH_M = 0.5    # a detected bounce further than this from a true landing is not that landing


def score_3d(path):
    rows = list(csv.DictReader(open(path)))
    err, false3d = [], 0
    for r in rows:
        if r["status"] in ("3d", "3d weak"):
            T = ball(float(r["time_a_s"]))
            X = np.array([float(r["x_m"]), float(r["y_m"]), float(r["z_m"])])
            if T is None or np.linalg.norm(X - T) > 0.5:
                false3d += 1
            else:
                err.append(np.linalg.norm(X - T))
    inflight = sum(1 for r in rows if ball(float(r["time_a_s"])) is not None)
    print(f"3D ball: {len(err)} / {inflight} frames with a ball tracked in 3D; false 3D points: {false3d}")
    if err:
        print(f"         error median {100 * np.median(err):.1f} cm, 90th percentile {100 * np.percentile(err, 90):.1f} cm")
    run = Path(path).with_suffix(".json")
    if run.exists():
        off = json.loads(run.read_text()).get("offset_s")
        if off is not None:
            print(f"sync:    refined offset {off:.3f} s, true {OFF_TRUE:.3f} s, error {1000 * abs(off - OFF_TRUE):.0f} ms")


def score_calls(path):
    rows = list(csv.DictReader(open(path)))
    if not rows:
        print("calls:   no bounces found")
        return
    correct = 0
    print("calls:")
    for i, ((x, y), true_call) in enumerate(TRUE_LANDINGS, 1):
        r = min(rows, key=lambda q: np.hypot(float(q["x_m"]) - x, float(q["y_m"]) - y))
        ex, ey = float(r["x_m"]), float(r["y_m"])
        if np.hypot(ex - x, ey - y) > MATCH_M:          # nearest bounce is another serve's
            print(f"  serve {i}: true ({x:.2f}, {y:.2f}) {true_call:3s} | no bounce found within "
                  f"{100 * MATCH_M:.0f} cm | MISSED")
            continue
        across = abs(ex - x) if "sideline" in r["nearest_line"] else abs(ey - y)
        ok = r["call"] == true_call
        correct += ok
        close = " (close)" if r.get("close") == "1" else ""
        print(f"  serve {i}: true ({x:.2f}, {y:.2f}) {true_call:3s} | estimated ({ex:.3f}, {ey:.3f}) "
              f"{r['call']}{close} | error across the line {100 * across:.1f} cm | {'correct' if ok else 'WRONG'}")
    print(f"  {correct} / {len(TRUE_LANDINGS)} calls correct")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    score_3d(sys.argv[1])
    if len(sys.argv) > 2:
        score_calls(sys.argv[2])
