"""Predict where the ball will land while it is still in the air, and score the prediction.

Usage (after line_calls.py and touches_rules.py for the same run):
    python scripts/landing_predict.py results/ball/game2_A_game2_B_sdr_ball_from30.json

Adapted from landing.py in the team's single-camera prototype. There, one camera needs a 3D gravity fit for
WHEN the ball lands and a pixel extrapolation for WHERE, plus a physics-tuned focal length. With
two cameras the ball's 3D position is measured, so one fit gives both.

For every floor bounce (from line_calls.py, via the touches CSV):
1. Flight = the 3D samples since the last contact before the bounce (touches_rules.py), or since
   the track's last gap of more than MAX_LINK_S if no contact was seen.
2. At each sample during the flight, a fixed-gravity arc (line_calls.fit_arc, weighted by ray
   angle) is fitted to the samples seen SO FAR, and its descending crossing of z = ball radius
   gives the predicted landing time and spot. The spot's uncertainty comes from the fit
   covariance (across and along the court separately), plus the calibration error.
3. The predicted spot is called IN / OUT with the same rule as line_calls.py (lines are in; the
   3 cm contact patch must reach the line), and "close" under 2 sigma.
4. Score: distance from each prediction to the measured contact point, and whether the predicted
   call matches, after 25 / 50 / 75 % of the flight time (no air drag in the model, so long
   flights are predicted slightly long).

Outputs (results/landing/):
    <stem>_landing.csv        one row per bounce: measured spot and call, flight length, error and
                              call after 25 / 50 / 75 % of the flight
    <stem>_landing_pred.csv   one row per prediction (per sample during each flight): time,
                              predicted spot, time and call, sigma, and the arc parameters
                              (used by render_full.py to draw the predicted path)
    <stem>_landing.png        prediction error against the share of the flight seen
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from line_calls import BALL_R, G, boundary_distance, fit_arc, load_track  # noqa: E402

MIN_SAMPLES = 5        # samples needed before predicting
MAX_LINK_S = 0.5       # without a seen contact, the flight starts after the last gap this long
MAX_FLIGHT_S = 3.0
CONTACT_R = 0.03
CALIB_SIGMA = 0.05
FRACTIONS = (0.25, 0.5, 0.75)


def predict(t, X, ang):
    """Fit an arc to the samples so far; return the landing prediction or None."""
    t_ref = t[-1]
    p, cov, (rms, rms_xyz) = fit_arc(t, X, t_ref, ang)
    z0, vz = p[2]
    disc = vz ** 2 + 2 * G * (z0 - BALL_R)
    if disc < 0:
        return None
    s = (vz + np.sqrt(disc)) / G                 # descending root, relative to the last sample
    if s < -0.05:
        return None
    a = np.array([1.0, s])
    xy = np.array([p[0] @ a, p[1] @ a])
    sig = rms_xyz[:2] * float(np.sqrt(a @ cov @ a))
    d, line = boundary_distance(*xy)
    s_line = sig[0] if line.startswith("sideline") else sig[1]
    sigma = float(np.hypot(s_line, CALIB_SIGMA))
    margin = d + CONTACT_R
    return dict(t_land=float(t_ref + s), x=float(xy[0]), y=float(xy[1]), sigma=sigma, rms=rms,
                call="IN" if margin >= 0 else "OUT", close=bool(abs(margin) < 2 * sigma),
                params=p, t_ref=float(t_ref))


def read_events(path):
    rows = list(csv.DictReader(open(path)))
    contacts = [float(r["time_a_s"]) for r in rows if r["kind"] == "contact"]
    floors = [dict(t=float(r["time_a_s"]), x=float(r["x_m"]), y=float(r["y_m"]), call=r["call"]) for r in rows
              if r["kind"] == "floor"]
    return contacts, floors


def flight_samples(t, X, ang, contacts, tb):
    """3D samples of the flight that ends at the bounce at time tb."""
    before = [c for c in contacts if tb - MAX_FLIGHT_S < c < tb]
    if before:
        t0 = before[-1] + 0.03
    else:
        idx = np.where(t < tb)[0]
        if not len(idx):
            return None
        gaps = np.where(np.diff(t[idx]) > MAX_LINK_S)[0]
        t0 = t[idx][gaps[-1] + 1] if len(gaps) else t[idx][0]
        t0 = max(t0, tb - MAX_FLIGHT_S)
    m = (t >= t0) & (t < tb - 0.02)
    return (t[m], X[m], ang[m], t0) if m.sum() >= MIN_SAMPLES else None


def plot(summary, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for b in summary:
        fr = [p["seen"] for p in b["preds"]]
        err = [p["err"] for p in b["preds"]]
        ax.plot(fr, err, "-o", ms=3, label=f"bounce at {b['t']:.1f} s ({b['call']}, {b['dur']:.1f} s flight)")
    ax.set_xlabel("share of the flight seen")
    ax.set_ylabel("predicted landing spot error (m)")
    ax.set_xlim(0, 1)
    ax.set_yscale("log")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run_json", help="the .json written by ball_two_cameras.py")
    args = p.parse_args()
    run = json.loads(Path(args.run_json).read_text())
    stem = Path(run["csv"]).stem
    touches = Path("results/touches") / f"{stem}_touches.csv"
    if not touches.exists():
        raise SystemExit(f"{touches} not found: run touches_rules.py first")
    t, X, _, ang = load_track(run["csv"])
    contacts, floors = read_events(touches)
    out = Path("results/landing")
    out.mkdir(parents=True, exist_ok=True)
    summary, pred_rows = [], []
    for n, b in enumerate(floors, 1):
        fl = flight_samples(t, X, ang, contacts, b["t"])
        if fl is None:
            print(f"  bounce {n} at {b['t']:.2f} s: too few 3D samples of the flight, no prediction")
            continue
        tf, Xf, af, t0 = fl
        dur = b["t"] - t0
        preds = []
        for i in range(MIN_SAMPLES, len(tf) + 1):
            r = predict(tf[:i], Xf[:i], af[:i])
            if r is None:
                continue
            r["seen"] = (tf[i - 1] - t0) / dur
            r["err"] = float(np.hypot(r["x"] - b["x"], r["y"] - b["y"]))
            preds.append(r)
            pa = r["params"]
            pred_rows.append([n, round(tf[i - 1], 3), round(r["seen"], 3), round(r["t_land"], 3), round(r["x"], 3),
                              round(r["y"], 3), r["call"], int(r["close"]), round(r["sigma"], 3), round(r["err"], 3),
                              round(r["t_ref"], 3)] + [round(float(v), 4) for v in pa.ravel()])
        if not preds:
            continue
        row = dict(n=n, t=b["t"], x=b["x"], y=b["y"], call=b["call"], dur=dur, preds=preds)
        for f in FRACTIONS:
            seen = [q for q in preds if q["seen"] <= f + 1e-9]
            q = seen[-1] if seen else None
            row[f] = q
        summary.append(row)
    with open(out / f"{stem}_landing.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        hdr = ["bounce", "time_a_s", "x_m", "y_m", "call", "flight_s", "samples"]
        for f in FRACTIONS:
            k = int(100 * f)
            hdr += [f"err_{k}_m", f"call_{k}", f"t_ahead_{k}_s"]
        w.writerow(hdr)
        for r in summary:
            line = [r["n"], round(r["t"], 3), round(r["x"], 3), round(r["y"], 3), r["call"], round(r["dur"], 2),
                    len(r["preds"])]
            for f in FRACTIONS:
                q = r[f]
                line += ["", "", ""] if q is None else [round(q["err"], 3), q["call"] + (" (close)" if q["close"] else ""),
                                                        round(r["t"] - q["t_ref"], 2)]
            w.writerow(line)
    with open(out / f"{stem}_landing_pred.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["bounce", "time_a_s", "flight_seen", "t_land_s", "x_m", "y_m", "call", "close", "sigma_m",
                    "error_m", "t_ref", "x0", "vx", "y0", "vy", "z0", "vz"])
        w.writerows(pred_rows)
    print(f"{len(summary)} of {len(floors)} bounce(s) predicted in flight:")
    for r in summary:
        parts = []
        for f in FRACTIONS:
            q = r[f]
            parts.append(f"{int(100 * f)}%: " + ("-" if q is None else
                         f"{100 * q['err']:.0f} cm {q['call']}{'' if q['call'] == r['call'] else ' (WRONG)'}"))
        print(f"  bounce {r['n']} t = {r['t']:.2f} s, {r['call']}, {r['dur']:.2f} s flight | " + " | ".join(parts))
    if summary:
        plot(summary, out / f"{stem}_landing.png")
    print(f"Saved {out / (stem + '_landing.csv')}, _landing_pred.csv" + (", _landing.png" if summary else ""))


if __name__ == "__main__":
    main()
