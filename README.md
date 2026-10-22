# AMME5710 Volleyball Line Judge

A low-cost computer vision line judge for amateur indoor volleyball: phones on tripods
courtside, no calibration target, automatic in/out calls.

AMME5710 Computer Vision and Image Processing, Major Project 2026, University of Sydney.

## Status

| Component | Status |
|---|---|
| Court detection and homography (single camera) | ✅ Done: 5.8 cm mean error, runs live |
| Ball candidate detection | 🟡 Starter script only (`ball_motion.py`) |
| Ball tracking and bounce detection | ⬜ Not started |
| Two-camera sync and 3D triangulation | ⬜ Not started |
| In/out calls | ⬜ Not started |

---

## Setup

Python 3.11 recommended.

**Windows (PowerShell)**
```
py -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```
If activation is blocked: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, then activate again.

**Mac / Linux**
```
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Check it works:
```
python -c "import cv2, numpy, matplotlib; print('OpenCV', cv2.__version__)"
```

Run all commands from the repo root with the environment activated.

## Data

Videos are **not** in Git: they're too large, and GitHub rejects files over 100 MB.
Put video files in `data/`, for example `data/volleyballmatch.mp4`.

| Folder | In Git? | Contents |
|---|---|---|
| `data/` | No | Raw videos |
| `data/annotations/` | **Yes** | Hand-clicked ground truth (small JSON / `.npy` files) |
| `frames/` | No | Frames extracted by `inspect_video.py`, plus debug images |
| `results/` | No | Evaluation outputs, saved videos, screenshots |

Development footage so far: a broadcast match (USA vs Brazil) filmed from a raised position
behind one end line. The camera holds still during rallies but **zooms and reframes between rallies**.

---

## Court detection

### How it works (`scripts/court_detect.py`)

White court lines on a light floor are almost invisible (roughly 245 vs 241 brightness),
so detecting the lines directly doesn't work (see `court_lines.py`, kept as the failed
baseline). Instead the detector uses the strong colour boundary between the court and
the free zone around it. Lines count as part of the court, so that boundary is the court edge.

1. **Free-zone colour model.** Sample the floor colour in a strip at the bottom of the
   frame, just outside the near end line, and mark every pixel with that hue (HSV hue
   distance plus saturation/value thresholds), then clean the mask with a morphological
   opening. The colour isn't hard-coded, so other free-zone colours should work.
2. **Boundary scanning.** Walk along columns and rows from inside the court until a run of
   free-zone pixels is reached, collecting candidate points on the near end line, far end
   line and both sidelines.
3. **RANSAC line fitting** for each edge (vectorised), with a total-least-squares refit on
   the inliers. Players standing on or over lines become outliers.
4. **Homography.** Intersect the four lines to get four corners, then map them to the
   9 m × 18 m court model with `cv2.getPerspectiveTransform`.
5. **Independent check.** The centre line and attack lines are never used in the fit;
   they're projected back into the image to check the homography.

Court frame (metres): `x` across the 9 m width, `y` along the 18 m length.
Far end line `y = 0`, far attack line `y = 6`, centre line `y = 9`,
near attack line `y = 12`, near end line `y = 18`.

### Results

Evaluated on 213 frames sampled every 10 s through a full match, against hand-clicked
ground truth on 15 randomly chosen frames covering different zoom levels:

| Points | Median error | Mean error |
|---|---|---|
| All 6 points | 5.8 cm | 5.8 cm |
| Corners (used by the detector) | 4.7 cm | 6.0 cm |
| Attack-line points (**never used** by the detector) | 4.9 cm | 5.4 cm |

- All 15 frames are within 15 cm; the worst is 8.0 cm.
- Speed: about 11–15 ms per frame when detecting at 960 px width, so it keeps up with 30 fps video.

**Known limitations**
- Tested on one video at one venue so far.
- Relies on the free zone being a different colour from the court.
- The quality checks have not yet rejected any frames, because this video has no replays or
  close-ups, so rejection is untested.

### Scripts and how to run them

**1. Inspect a video and extract frames**
```
python scripts/inspect_video.py data/volleyballmatch.mp4 --every 10
```
Prints resolution, fps and duration, and saves one frame every 10 s to `frames/volleyballmatch/`.

**2. Detect the court in one frame**
```
python scripts/court_detect.py frames/volleyballmatch/frame_000300.jpg
```
Saves `_auto_H.npy` (image → court metres), `_auto_corners.npy`, `_auto_overlay.jpg` and
`_auto_debug.jpg`. The debug image has four panels: free-zone mask, scanned points
(coloured = kept, red = rejected by RANSAC), fitted court, and the original frame.

Add `--gt <clicks.npy>` to compare with hand-clicked points from `manual_homography.py`.

**3. Run live on a video or camera**
```
python scripts/court_live.py data/volleyballmatch.mp4
python scripts/court_live.py data/volleyballmatch.mp4 --start 300 --save results/court_live.mp4
python scripts/court_live.py 0          # camera index 0
```
Keys: `space` pause, `s` screenshot (saved to `results/screenshots/`), `q` quit.

- Status shows **TRACKING**, **HOLD** (detection failed; the last good court is kept for
  0.5 s) or **NO COURT**.
- Corners are smoothed over time while the camera is still. A large jump, such as a zoom
  or a cut, snaps straight to the new position instead.
- Options: `--detect-width` (default 960), `--smooth` (default 0.3; 1 = no smoothing),
  `--display-width`, `--no-display`, `--max-frames`.

**4. Hand-click ground truth on random frames**
```
python scripts/annotate_court.py frames/volleyballmatch --n 15
```
Click six points per frame, in the order shown on screen:
1. far-left corner
2. far-right corner
3. near-right corner
4. near-left corner
5. near attack line meets the left sideline
6. near attack line meets the right sideline

Keys: `Enter` accept, `r` redo, `s` skip, `q` save and quit. It saves after every frame and
resumes where you left off. Output: `data/annotations/volleyballmatch_gt.json`.

**5. Evaluate on many frames**
```
python scripts/court_batch_eval.py frames/volleyballmatch --gt-json data/annotations/volleyballmatch_gt.json
```
Outputs go to `results/court_batch/volleyballmatch/`:
- `summary.txt`: accepted / rejected counts, error statistics, per-point-type errors
- `per_frame.csv`: status, corners, inlier fractions and error for each frame
- `error_plot.png`: error through the match
- `worst.jpg`: overlays of the highest-error frames
- `rejected.jpg`: sample of rejected frames

A frame is accepted only if every edge has enough RANSAC inliers, the corners form a
convex shape, the near edge is wider than the far edge, and all corners are inside the image.

**6. Manual homography (single frame)**
```
python scripts/manual_homography.py frames/volleyballmatch/frame_000300.jpg
```
Click four points and get a homography plus an overlay to check it.
- `--points near-half` (default): near attack line left and right, then near-right and near-left corners.
- `--points full`: the four outer corners.

### Other scripts

| Script | Purpose |
|---|---|
| `court_lines.py` | First attempt: white-line mask + Canny + Hough. Fails on light courts; kept as a baseline for the report. |
| `ball_motion.py` | Starter ball candidates: MOG2 background subtraction + size / circularity filter. Writes an annotated video and a CSV of candidates. |

```
python scripts/ball_motion.py data/volleyballmatch.mp4 --start 30 --duration 15
```

---

## Working on the repo

- **Never commit directly to `main`.** Make a branch per task:
  ```
  git checkout main
  git pull
  git checkout -b ball-tracking
  ```
- Push your branch, open a pull request on GitHub, and merge when it works.
- Before committing, run `git status` and make sure no videos, frames or results are listed.

## Next steps

1. Ball detection and tracking: reduce false positives from `ball_motion.py` and link detections into trajectories.
2. Bounce detection: find the frame and pixel where the ball hits the floor.
3. Map the bounce through the court homography and call it in or out.
4. Record our own two-phone footage with taped ground-truth marks.
5. Two-camera sync and triangulation.
