"""Run automatic court detection on many frames and measure accuracy and robustness.

Usage (recommended): per-frame ground truth from annotate_court.py
    python scripts/court_batch_eval.py frames/volleyballmatch \
        --gt-json data/annotations/volleyballmatch_gt.json

Usage (single click set, only valid if the camera never zooms or moves):
    python scripts/court_batch_eval.py frames/volleyballmatch \
        --gt data/annotations/volleyballmatch_frame_000300_clicks.npy --gt-points near-half

Detection and quality checks run on every frame; error is measured only on
frames that have ground truth. With --gt-json the error is reported separately
for the corners and for the attack-line points, which the detector never uses.

Quality checks (a frame is "accepted" only if all pass):
    - every edge has at least MIN_POINTS scan points and MIN_INLIER_FRAC RANSAC inliers
    - the four corners form a convex quadrilateral
    - the near end line is wider in the image than the far end line (perspective)
    - all corners lie within the image, allowing a small margin

Outputs in results/court_batch/<folder name>/:
    per_frame.csv   one row per frame: status, reason, corners, inlier fractions, errors
    summary.txt     detection rate, error statistics, false accepts
    error_plot.png  error per frame through the match
    worst.jpg       overlays of the highest-error accepted frames, for the report
    rejected.jpg    a sample of rejected frames, to check the rejections are sensible
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from court_detect import (COURT_LINES, GT_POINT_SETS, compare_to_gt,  # noqa: E402
                          detect_court, project)

MIN_POINTS = 10
MIN_INLIER_FRAC = 0.4
CORNER_MARGIN_FRAC = 0.05
GOOD_CM = 15.0      # accepted frames with mean error above this count as false accepts


def quality_check(img, corners, scans, lines):
    h, w = img.shape[:2]
    fracs = {}
    for k in ("far", "near", "left", "right"):
        n = len(scans[k])
        fracs[k] = lines[k][2].sum() / n if n else 0.0
        if n < MIN_POINTS:
            return False, f"too few {k} points ({n})", fracs
        if fracs[k] < MIN_INLIER_FRAC:
            return False, f"{k} edge inliers {fracs[k]:.0%}", fracs
    if not cv2.isContourConvex(corners.reshape(-1, 1, 2)):
        return False, "corners not convex", fracs
    far_w = np.linalg.norm(corners[1] - corners[0])
    near_w = np.linalg.norm(corners[2] - corners[3])
    if near_w <= far_w:
        return False, "near edge not wider than far edge", fracs
    m = CORNER_MARGIN_FRAC
    if (corners[:, 0].min() < -m * w or corners[:, 0].max() > (1 + m) * w or
            corners[:, 1].min() < -m * h or corners[:, 1].max() > (1 + m) * h):
        return False, "corner outside image", fracs
    return True, "", fracs


def draw_overlay(img, H, label):
    out = img.copy()
    H_inv = np.linalg.inv(H)
    for name, (a, b) in COURT_LINES.items():
        pa, pb = project(H_inv, [a, b])
        colour = (0, 200, 255) if "attack" in name or name == "centre" else (0, 255, 0)
        cv2.line(out, tuple(map(int, pa)), tuple(map(int, pb)), colour, 3)
    cv2.putText(out, label, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 0), 8)
    cv2.putText(out, label, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (255, 255, 255), 3)
    return out


def montage(images, cols=3, width=640):
    if not images:
        return None
    tiles = [cv2.resize(im, (width, int(im.shape[0] * width / im.shape[1]))) for im in images]
    th = max(t.shape[0] for t in tiles)
    tiles = [cv2.copyMakeBorder(t, 0, th - t.shape[0], 0, 0, cv2.BORDER_CONSTANT) for t in tiles]
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
    return np.vstack(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", help="folder of frames from inspect_video.py")
    parser.add_argument("--gt-json", help="per-frame ground truth from annotate_court.py")
    parser.add_argument("--gt", help="single clicks .npy from manual_homography.py")
    parser.add_argument("--gt-points", choices=GT_POINT_SETS, default="near-half")
    parser.add_argument("--out", default="results/court_batch")
    args = parser.parse_args()

    # Only original frames, not the debug images other scripts write next to them
    frames = sorted(p for p in Path(args.folder).glob("frame_*.jpg") if p.stem.count("_") == 1)
    if not frames:
        raise SystemExit(f"No frame_*.jpg files in {args.folder}")
    out_dir = Path(args.out) / Path(args.folder).name
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.gt_json:
        gtj = json.loads(Path(args.gt_json).read_text())
        gt_court = np.float32(gtj["court_points"])
        gt_frames = {k: np.float32(v) for k, v in gtj["frames"].items()}
        gt_for = lambda name: gt_frames.get(name)
        independent = np.array([p[1] not in (0, 18) for p in gtj["court_points"]])
    elif args.gt:
        gt_court = GT_POINT_SETS[args.gt_points]
        single = np.load(args.gt)
        gt_for = lambda name: single
        independent = None
    else:
        gt_for = lambda name: None
        independent = None
    has_gt = bool(args.gt_json or args.gt)
    corner_err, indep_err = [], []

    rows, accepted_err, rejected_imgs, accepted = [], [], [], []
    for i, path in enumerate(frames, 1):
        img = cv2.imread(str(path))
        row = {"frame": path.name, "status": "", "reason": ""}
        try:
            H, corners, _, scans, lines, _ = detect_court(img)
        except (ValueError, np.linalg.LinAlgError, SystemExit) as e:
            row.update(status="failed", reason=str(e).splitlines()[0][:60])
            rows.append(row)
            rejected_imgs.append((path, img, "failed"))
            print(f"[{i}/{len(frames)}] {path.name}: failed ({row['reason']})")
            continue

        ok, reason, fracs = quality_check(img, corners, scans, lines)
        row.update({f"inliers_{k}": f"{v:.2f}" for k, v in fracs.items()})
        row.update({f"c{j}_{ax}": f"{corners[j][a]:.1f}" for j in range(4) for a, ax in enumerate("xy")})
        gt = gt_for(path.name)
        if gt is not None:
            err_px, err_m = compare_to_gt(H, gt, gt_court)
            row.update(mean_err_px=f"{err_px.mean():.2f}", mean_err_cm=f"{err_m.mean() * 100:.2f}",
                       max_err_cm=f"{err_m.max() * 100:.2f}")
            if independent is not None and ok:
                corner_err.extend(err_m[~independent] * 100)
                indep_err.extend(err_m[independent] * 100)
        row["status"] = "accepted" if ok else "rejected"
        row["reason"] = reason
        rows.append(row)

        if ok:
            err_cm = err_m.mean() * 100 if gt is not None else -1.0
            if gt is not None:
                accepted_err.append(err_cm)
            accepted.append((err_cm, path, img, H))
        else:
            rejected_imgs.append((path, img, reason))
        msg = f"{row.get('mean_err_cm', '-')} cm" if ok else f"rejected: {reason}"
        print(f"[{i}/{len(frames)}] {path.name}: {msg}")

    # ---- CSV ----
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(out_dir / "per_frame.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=keys)
        wr.writeheader()
        wr.writerows(rows)

    # ---- Summary ----
    n = len(rows)
    n_acc = sum(r["status"] == "accepted" for r in rows)
    n_rej = sum(r["status"] == "rejected" for r in rows)
    n_fail = sum(r["status"] == "failed" for r in rows)
    lines_out = [
        f"Frames:   {n}",
        f"Accepted: {n_acc} ({n_acc / n:.0%})",
        f"Rejected: {n_rej} ({n_rej / n:.0%})  (quality checks failed)",
        f"Failed:   {n_fail} ({n_fail / n:.0%})  (detector could not run)",
    ]
    if has_gt and accepted_err:
        e = np.array(accepted_err)
        good = e <= GOOD_CM
        lines_out += [
            "",
            f"Error against hand-clicked ground truth, {len(e)} accepted frames with GT (cm on court):",
            f"  median {np.median(e):.1f}   mean {e.mean():.1f}   "
            f"90th pct {np.percentile(e, 90):.1f}   max {e.max():.1f}",
            f"  within {GOOD_CM:.0f} cm: {good.sum()}/{len(e)}",
            f"  false accepts (> {GOOD_CM:.0f} cm): {(~good).sum()}  "
            "<- check worst.jpg: camera moved, or a check is too loose",
            "",
            f"Mean error on good frames only: {e[good].mean():.1f} cm" if good.any() else "",
        ]
        if indep_err:
            ce, ie = np.array(corner_err), np.array(indep_err)
            lines_out += [
                "",
                "Per point type (accepted frames):",
                f"  corners (used by the detector):        median {np.median(ce):.1f}  mean {ce.mean():.1f} cm",
                f"  attack-line points (never used):       median {np.median(ie):.1f}  mean {ie.mean():.1f} cm",
            ]
    reasons = {}
    for r in rows:
        if r["status"] != "accepted":
            reasons[r["reason"]] = reasons.get(r["reason"], 0) + 1
    if reasons:
        lines_out += ["", "Rejection / failure reasons:"]
        lines_out += [f"  {c:4d}  {k}" for k, c in sorted(reasons.items(), key=lambda x: -x[1])]
    summary = "\n".join(lines_out)
    (out_dir / "summary.txt").write_text(summary + "\n")
    print("\n" + summary)

    # ---- Error plot ----
    if has_gt:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        idx = np.arange(n)
        err = np.array([float(r.get("mean_err_cm", "nan") or "nan") for r in rows])
        acc = np.array([r["status"] == "accepted" for r in rows]) & ~np.isnan(err)
        fig, ax = plt.subplots(figsize=(10, 3.5))
        ax.scatter(idx[acc], err[acc], s=14, label="accepted")
        rej = np.array([r["status"] != "accepted" for r in rows])
        ax.scatter(idx[rej], np.full(rej.sum(), 0), s=14, marker="x", color="grey",
                   label="rejected / failed")
        ax.axhline(GOOD_CM, ls="--", lw=1, color="red", label=f"{GOOD_CM:.0f} cm")
        ax.set_yscale("symlog", linthresh=10)
        ax.set_xlabel("frame index (sampled through the match)")
        ax.set_ylabel("mean error vs ground truth (cm)")
        ax.legend(loc="upper right", fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / "error_plot.png", dpi=150)

    # ---- Image sheets ----
    accepted.sort(key=lambda t: -t[0])
    worst = [draw_overlay(img, H, f"{p.stem}  {e:.1f} cm" if e >= 0 else p.stem)
             for e, p, img, H in accepted[:6]]
    sheet = montage(worst)
    if sheet is not None:
        cv2.imwrite(str(out_dir / "worst.jpg"), sheet)
    rej = []
    for p, img, reason in rejected_imgs[:9]:
        im = img.copy()
        cv2.putText(im, f"{p.stem}: {reason}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 0), 8)
        cv2.putText(im, f"{p.stem}: {reason}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 3)
        rej.append(im)
    sheet = montage(rej)
    if sheet is not None:
        cv2.imwrite(str(out_dir / "rejected.jpg"), sheet)

    print(f"\nSaved results to {out_dir}")


if __name__ == "__main__":
    main()
