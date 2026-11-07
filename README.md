# AMME5710 Volleyball Line Judge

A low-cost computer vision line judge for amateur indoor volleyball: two phones on tripods
behind the end lines, no calibration target, automatic in/out calls.

AMME5710 Computer Vision and Image Processing, Major Project 2026, University of Sydney.

## Status

| Component | Status |
|---|---|
| Court detection on broadcast footage (free-zone boundary) | Done: 5.8 cm mean error, runs live |
| Game 2 footage: HDR to SDR conversion | Done |
| Game 2: automatic yellow-line detection | Done (both cameras) |
| Game 2: camera calibration | Done: camera A 6.3 cm, camera B 2.8 cm (held-out line error) |
| Game 2: audio sync between cameras | Done: offset +9.71 s, 6/8 chunks agree |
| Game 2: synced two-camera court view | Done |
| Player detection and top-down court map | Done: YOLO11n, F1 0.93, median position error 0.25 m |
| Ball tracking in 3D (two cameras) | Done: 77% of frames on an 8 s real rally, no false balls |
| Bounce detection and in/out calls | Done: 3/3 correct on synthetic serves (error across the line 0.5-6.9 cm) |
| Synthetic test with exact ground truth | Done: 3D error median 1.6 cm, sync refined to 1 ms |
| Touches, who touched it, 3-touch / double-contact rules on the 3D track | Done: 7/9 contacts attributed on a real rally; not yet scored against ground truth |
| Single-camera pipeline: players, touches, 3-touch rule (`volleyball_cv_inout/`) | Done (see its own README) |

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

`requirements.txt` must include `opencv-contrib-python`, `numpy`, `scipy`, `matplotlib`,
`imageio-ffmpeg` and `ultralytics`.
- `imageio-ffmpeg` provides ffmpeg (with the `zscale` filter needed for HDR conversion), so
  ffmpeg does not need to be installed separately.
- `ultralytics` provides YOLO for player detection. It installs PyTorch (large; the CPU
  version is fine). Model weights (`yolo11n.pt`, `yolo11s.pt`, and `yolo11m-pose.pt` for
  `touches_rules.py`) download automatically on
  first use and are ignored by Git (`*.pt`).
- Use `opencv-contrib-python`, not `opencv-python`: OpenCV 5 moved the HOG people detector
  to the contrib package. Don't install both (they share files). If you already have
  `opencv-python`: `pip uninstall -y opencv-python`, then `pip install -r requirements.txt`.

Run all commands from the repo root with the environment activated.

## Data

Videos are **not** in Git: they're too large, and GitHub rejects files over 100 MB.

| Folder | In Git? | Contents |
|---|---|---|
| `data/` | No | Videos, e.g. `data/game2_A.mp4`, `data/game2_B_sdr.mp4` |
| `data/annotations/` | **Yes** | Calibrations, sync offsets, hand-clicked ground truth, fitted foot offset (small JSON files) |
| `frames/` | No | Frames saved by `inspect_video.py` |
| `results/` | No | Debug images, overlays, combined, synced and player videos, evaluation output |

### Game 2 footage (current)

Two iPhones on tripods, one behind each end line of an indoor court with **yellow
volleyball lines on a blue floor**, recorded at 1920x1080, about 29.97 fps.

| Camera | File | Position (`--end`) | Notes |
|---|---|---|---|
| A | `data/game2_A.mp4` | behind the y = 18 end line (`near`) | normal (SDR) video |
| B | `data/game2_B_sdr.mp4` | behind the y = 0 end line (`far`) | recorded in HDR; converted with `hdr_to_sdr.py` |

Court frame (metres), shared by both cameras: `x` 0–9 across the width, `y` 0–18 along the
length. Far end line `y = 0`, far attack line `y = 6`, centre line (net) `y = 9`,
near attack line `y = 12`, near end line `y = 18`. Each camera judges the half nearest to
it: camera A `y = 9..18`, camera B `y = 0..9`.

For future recordings: turn **HDR Video off** on both phones
(Settings > Camera > Record Video), and record a clap at the start of filming.

---

## Two-camera pipeline (Game 2)

### 1. Inspect each video
```
python scripts/inspect_video.py data/game2_A.mp4 --every 30
```
Prints resolution, frame rate, bitrate and codec, checks for dropped or doubled frames, and
flags HDR video. Saves exact frames (PNG, via ffmpeg) and a `contact_sheet.jpg` to
`frames/game2_A/`. Use `--times 12.5 60` for specific times.

### 2. Convert HDR video to normal colours
```
python scripts/hdr_to_sdr.py data/IMG_4857.MOV data/game2_B_sdr.mp4 --width 1920
```
iPhone HDR (HLG, BT.2020) looks washed out when read as normal video: our floor saturation
dropped from about 65 to 22. The script converts properly: HLG to linear light (reference
white 203 nits, ITU-R BT.2408), BT.2020 to BT.709 primaries, Mobius tone mapping, H.264
CRF 18, audio copied. A round-trip check (SDR to HLG and back) recovered the floor colour to
within 2 saturation levels, with a mean pixel difference of 3/255 (13/255 when read naively).

- Use the original `.MOV` from the phone (10-bit) if possible.
- `--width 1920` keeps a 4K original the same size as the other camera.
- `--white 350` gives a darker result (camera B came out about 30% brighter than A at 203).
- Test first with `--start 60 --duration 10`; it saves a before/after `_compare.jpg`.

### 3. Detect the yellow court lines automatically
```
python scripts/detect_court_lines.py data/game2_A.mp4 --time 60
```
1. **Colour model:** the floor colour is measured at the bottom centre of the frame, and
   saturation/brightness thresholds are set relative to it, so the same code works on normal
   and converted HDR footage. Yellow is hue 22–42, which excludes the orange lines of other
   sports (hue 9–20).
2. **Floor mask:** pixels near the floor's hue. Thin lines from other sports are bridged by a
   morphological closing first, then every large floor region is kept.
3. **Line mask:** yellow pixels on the floor; compact blobs (skin, shoes) are removed by shape.
4. **Hough transform** gives segments, grouped by angle into the near end line, left
   sideline and right sideline.
5. **RANSAC line fit** for each group (players on the line are outliers), least-squares
   refit on the inliers, and the near corners as line intersections.

Saves `results/court_lines/<video>_t60.json` (lines, corners, and sample points taken from
the actual line pixels) and `_debug.jpg` (mask and segments above, fitted lines below).
On our footage the detected line pixels lie within about 1–2 px of straight lines.

### 4. Calibrate each camera
```
python scripts/calibrate_camera.py data/game2_A.mp4 --camera A --end near --time 60 --auto-lines results/court_lines/game2_A_t60.json
python scripts/calibrate_camera.py data/game2_B_sdr.mp4 --camera B --end far --time 24 --auto-lines results/court_lines/game2_B_sdr_t24.json
```
The detected lines and corners are added automatically. You then click intersections the
detector can't find, such as the attack-line ends and the far court. Names are always from
the camera's own point of view ("near-left" = nearest corner on its left), and `--end` maps
them into the shared court frame.

- **Clicking:** a 5x magnifier follows the mouse. `i/j/k/l` nudge the last click 1 px,
  `s` skips a point, `u` undoes, `Enter` moves to the next stage. In the second stage you can
  click extra points along any line (`n` next line, `b` back, `Enter` finish).
- **Fit:** a homography (court to image) by robust least squares on the pixel distances of
  the intersections and of the line points from their lines, plus an optional radial lens
  distortion term `k1`. That term is kept only if it reduces the held-out error.
- **Accuracy estimate:** each line is left out of the fit in turn, and its points are checked
  against where the line really is, in centimetres on the court. This is conservative: in
  simulation it was about twice the true error.
- **Tripod check:** 40 frames across the video are compared with the calibration frame by
  phase correlation of the walls and ceiling. A shift over 2 px means the phone moved.
- **Refit without re-clicking:** `--points-json data/annotations/calib_B.json`. Leave a bad
  click out with `--drop "centre line / right sideline (under net)"`.

Outputs: `data/annotations/calib_<camera>.json` (commit this), and in `results/calib/` the
`_overlay.jpg` (court drawn on the calibration frame) and `_check.jpg` (court on frames
through the video). Other scripts load a calibration with:
```python
from calibrate_camera import load_calibration
cal = load_calibration("data/annotations/calib_A.json")
court_xy = cal.image_to_court([[u, v]])      # pixels -> metres, for points on the floor
```

**Results (Game 2):**

| | Accuracy estimate | Lens correction k1 | Tripod |
|---|---|---|---|
| Camera A | 6.3 cm | -0.021 | stable for the whole video |
| Camera B | 2.8 cm | -0.012 | stable from about 7 s onwards |

The centre line is too dark to click reliably near the right-hand net post in both views,
so that point is left out with `--drop`. Far-court points have large errors in centimetres
(one pixel covers 15–20 cm beyond the net), which doesn't matter because the other camera
judges that half.

### 5. Sync the two cameras
```
python scripts/sync_videos.py data/game2_A.mp4 data/game2_B_sdr.mp4
```
Both phones recorded the same whistles and ball contacts. Each soundtrack becomes a
multi-band onset signal (rises in log energy in five frequency bands), and the two are
cross-correlated, coarse to fine:
1. Blurred signals at 100 Hz, so sounds reaching the phones up to about 60 ms apart (20 m
   at 343 m/s) still line up.
2. Refined at 1 ms within ±0.15 s of that answer.

Each correlation is scored as a z-score per lag, which treats short and long overlaps fairly.
Every 30 s chunk of A votes independently for an offset. The result is "reliable" if most
chunks agree; otherwise it falls back to the whole-recording match ("PROBABLE") if that peak
is clear.

Outputs: `data/annotations/sync_<A>_<B>.json` (offset, `t_B = t_A + offset_s`),
`results/sync/..._audio_diag.png` (correlation curve and chunk votes), and a side-by-side
`_preview.mp4`.

If it's unreliable: `--guess <seconds> --search 2` narrows the search, and
`--offset <seconds>` sets the offset by hand.

**Result (Game 2):** offset +9.71 s (camera B started about 9.7 s before camera A), with 6
of 8 chunks agreeing and one clear correlation peak. Sound travel between the phones limits
the accuracy to about ±1–2 frames, so confirm it on a serve contact.

### 6. Synced views
Side by side, with no overlays:
```
python scripts/combine_videos.py data/game2_A.mp4 data/game2_B_sdr.mp4 --sync data/annotations/sync_game2_A_game2_B_sdr.json --start 0 --duration 3:00
```
Courts drawn on both cameras, each camera's half shaded, a top-down court map, and a
once-per-second tripod check:
```
python scripts/court_view_synced.py data/game2_A.mp4 data/game2_B_sdr.mp4 --calib-a data/annotations/calib_A.json --calib-b data/annotations/calib_B.json --sync data/annotations/sync_game2_A_game2_B_sdr.json --start 30 --duration 60
```
Add `--show` to watch while it renders. Outputs go to `results/combined/` and `results/synced/`.

### 7. Player detection and court positions
```
python scripts/players_on_court.py data/game2_A.mp4 data/game2_B_sdr.mp4 --calib-a data/annotations/calib_A.json --calib-b data/annotations/calib_B.json --sync data/annotations/sync_game2_A_game2_B_sdr.json --start 30 --duration 30
```
Both synced camera views with player boxes, plus a top-down court map with a dot and a short
trail for each player, coloured by side of the net (orange = camera A's side, green = B's).

1. **Detection:** YOLO11n (pretrained COCO `person` class) on each camera, every 3rd frame
   by default. `--detector hog` uses OpenCV's classical HOG + linear SVM people detector.
2. **Foot point:** bottom centre of each box, raised by a fitted fraction of the box height
   (see step 8). Only the feet touch the floor, so only they map correctly through the floor
   homography. Boxes cut off by the image edge are skipped.
3. **Court position:** foot point mapped to metres with that camera's calibration. People
   well outside the court (benches, spectators, walls) are dropped and shown with grey boxes.
4. **Two-camera fusion:** each camera covers its own half; within 1 m of the net both are
   used, and matching detections (Hungarian assignment, within 1.5 m) are averaged.
   Detections closer than 0.4 m are one person.
5. **Tracking in court coordinates:** detections are matched to each track's predicted
   position (constant velocity, Hungarian assignment, 1.5 m gate). A new track must be matched
   3 times in a row before it is drawn, and a track is only drawn while it was seen in the
   last 0.4 s, so false detections and lost players don't leave stray dots.

Options: `--show` (watch while rendering), `--every 1` (detect on every frame, about 3x
slower), `--model yolo11s.pt` (slightly more accurate, about 2.4x slower), `--device 0`
(NVIDIA GPU), `--foot-frac 0` (no foot correction).

Outputs in `results/players/`: the video, and a `.csv` of court positions per detection step
(`time_s, track_id, x_m, y_m, side, camera, confidence`).

**Limitations:** a jumping player's feet are off the floor, so they map further from the
camera than they are; a player hidden behind another in the camera covering their half can
drop out; track IDs can change after a long occlusion or when players cross.

### 8. Player detection accuracy
Ground truth: the floor point between the feet of every player, clicked by hand at 10 moments
spread over the game (each camera's half, both cameras). About 10 minutes of clicking:
```
python scripts/eval_players.py annotate data/game2_A.mp4 data/game2_B_sdr.mp4 --calib-a data/annotations/calib_A.json --calib-b data/annotations/calib_B.json --sync data/annotations/sync_game2_A_game2_B_sdr.json --n 10
```
Left click adds a player, right click deletes the nearest click, `u` undoes, `Enter` moves to
the next image, `s` skips, `q` saves and stops (run again to continue). `--redo` starts
again; `--only 3 7` redoes single moments. Saved to `data/annotations/players_gt.json`
(committed, so the evaluation can be rerun without clicking).

Score the detectors (no clicking):
```
python scripts/eval_players.py evaluate
```
- Detections and clicks are mapped to the court and kept if they are in that camera's half.
  A detection is correct if it is within 1.0 m of a click (one-to-one, Hungarian matching;
  change with `--gate`).
- Reports precision, recall, F1, count error, median position error and time per image, per
  camera and for the whole two-camera **system** (fused, single frames, no tracking).
- **Foot correction:** the bottom of a detection box lies about 10 px below where the feet
  touch the floor. At these shallow viewing angles that is 0.66-0.84 m too close to the
  camera, with almost no sideways error. Each detector is scored with the raw box bottom and
  with the foot point raised by the median offset (a fraction of the box height), fitted with
  2-fold cross-validation (the value used on odd moments is fitted on even moments and vice
  versa). The value fitted on all moments is saved to `data/annotations/foot_offset.json`
  and used by `players_on_court.py`.

Outputs in `results/eval/`: `players_eval_summary.csv`, `players_eval_frames.csv`, and
images in `vis/` (white ring = clicked player, green = correct detection, red = false
detection, yellow X = missed player).

**Results (Game 2):** 10 moments, 105 players, system rows (both cameras), 1.0 m match gate.

| Detector | Foot point | Precision | Recall | F1 | Median position error | Time, both cameras |
|---|---|---|---|---|---|---|
| HOG + SVM | raw | 0.03 | 0.02 | 0.02 | 0.62 m | 446 ms |
| HOG + SVM | corrected (13.6%) | 0.71 | 0.50 | 0.58 | 0.51 m | 446 ms |
| YOLO11n | raw | 0.67 | 0.69 | 0.68 | 0.66 m | 264 ms |
| **YOLO11n** | **corrected (5.3%)** | **0.94** | **0.91** | **0.93** | **0.25 m** | **264 ms** |
| YOLO11s | raw | 0.67 | 0.73 | 0.70 | 0.65 m | 631 ms |
| YOLO11s | corrected (5.6%) | 0.94 | 0.95 | 0.94 | 0.25 m | 631 ms |

- The foot correction is the largest single improvement (YOLO11n F1 0.68 to 0.93). The two
  cross-validation folds gave the same value (5.3% / 5.3%), so it is stable.
- YOLO11s is only marginally better at 2.4x the cost, so YOLO11n is used.
- HOG finds about half the players even after correction: its fixed window misses small,
  distant and crouching players, and its boxes are loose.
- Remaining misses (5-6 per camera) are mostly players hidden behind others. The position
  error includes the error of the hand clicks.
- Limitations: 10 moments from one game, one annotator, single frames (tracking not scored).
  Times were measured on our development PC.

### 9. Ball in 3D from both cameras
```
python scripts/ball_two_cameras.py data/game2_A.mp4 data/game2_B_sdr.mp4 --calib-a data/annotations/calib_A.json --calib-b data/annotations/calib_B.json --sync data/annotations/sync_game2_A_game2_B_sdr.json --start 30 --duration 10
```
No learned ball detector; the second camera does the filtering.
- **Each camera** (`ball_detect.py`): three-frame difference (frames compared a fixed *time*
  apart, so 30 and 60 fps behave the same), opening/closing, then size (the 21 cm ball's
  possible image size from the 3D camera), shape (up to 5:1 for motion blur, at least 25% fill),
  ball colour (yellow/blue share, plus a precise colour centre for bounces), person boxes from
  YOLO every 3rd frame, isolation. Candidates are linked into 2D tracklets (constant velocity,
  Hungarian, gate in ball radii).
- **Both cameras**: for every pair of tracklets that overlap in time, camera B is interpolated
  to each camera A frame time and the pair is triangulated (DLT). A pair is the ball only if
  the rays meet (< 12 px) on 10+ frames and 80% of the overlap, median speed >= 3 m/s, it fits a
  fixed-gravity parabola within 3 cm (weighted by ray angle), it is ball-coloured (>= 45%), and
  the blob size matches a 21 cm ball at the triangulated distance (0.6-2.5x). The script prints
  how many pairs each rule rejected.
- **Sub-frame sync**: the audio offset is refined on the accepted pairs (+-0.15 s in 2 ms steps,
  lowest median reprojection error), then matching is repeated.
- Rays closer than 3 deg to parallel are flagged `3d weak` (near camera height between the
  cameras); near the floor the geometry is good.

Outputs in `results/ball/`: `<A>_<B>_ball_from<start>.mp4` (both views, top and side view),
`.csv` (per camera A frame: 2D positions, x, y, z, reprojection error, ray angle, status),
`_trajectory.png`, `.json` (run details and refined offset) and `_tracks2d.csv` (per-camera
2D tracks with colour centres, used for single-camera bounces).

**Real footage:** 3D coverage on an 8 s rally went from 15% to 77% of frames with no false
balls after the time-based frame step, a 3 m/s speed floor, reusing tracklets in different
pairs, the angle-weighted gravity test and the colour threshold.

### 10. Bounces and in/out calls
```
python scripts/line_calls.py results/ball/game2_A_game2_B_sdr_ball_from30.json
```
1. Bounce candidates: low points (< 0.45 m) where the ball stops descending, per continuous
   stretch of 3D track (gaps > 0.2 s split it).
2. Contact point: ballistic arcs (g fixed) fitted to up to 0.4 s before and after, weighted by
   sin^2(ray angle); contact is where the ball centre is one radius (0.105 m) above the floor.
3. Uncertainty from the fit covariance, separately across and along the court, plus 5 cm of
   calibration error (`--calib-sigma`).
4. Call (FIVB: lines are in, court measured to their outer edges): IN if the 3 cm contact patch
   reaches the line (`--contact-radius`), else OUT; "close" when the margin is under 2 sigma.
5. Bounce seen by one camera only: quadratics fitted to its image path before and after the
   turn meet at the contact pixel; its ray at z = 0.105 m gives the floor point. If the curves
   miss by more than 6 px (e.g. uneven frame timing) the call is UNRELIABLE.

Outputs in `results/calls/`: `_calls.csv` (one row per bounce), `_calls.png` (court with
2-sigma circles and a height plot), `_bounce<N>.jpg` (zoomed replay from both cameras).

### 11. Synthetic test (exact ground truth)
Real games have no ground truth for landing spots, so `synthetic_render.py` draws a clip where
everything is known: two virtual cameras placed like ours (exact calibrations saved), a court,
twelve moving players, a decoy ball, and a ball flying a toss and three serves (IN 2 m inside,
OUT 30 cm outside, and IN with the centre 2 cm outside the sideline). Camera B is shifted by
2.017 s; the pipeline starts from a 2.000 s guess.
```
python scripts/synthetic_render.py
cd data/synthetic
python ../../scripts/ball_two_cameras.py A.mp4 B.mp4 --calib-a calib_A.json --calib-b calib_B.json --sync sync.json --start 0.3 --duration 7.2
python ../../scripts/line_calls.py results/ball/A_B_ball_from0.3.json
python ../../scripts/synthetic_score.py results/ball/A_B_ball_from0.3.csv results/calls/A_B_ball_from0.3_calls.csv
```
| Measure | Result |
|---|---|
| Calls | 3 / 3 correct (serve 3 called IN, close) |
| Contact error across the judged line | 1.2, 6.9, 0.5 cm |
| 3D ball error | median 1.6 cm, 90th percentile 6.6 cm, 99 of 196 frames, no false points |
| Sync refinement | 2.016 s vs true 2.017 s (1 ms) |

Not tested by it: calibration error (cameras are exact), audio sync, motion blur, 60 fps / HDR,
uneven frame timing. The ball image is `data/annotations/ball_patch.png` (a crop of the real
ball; a drawn ball is used if it is missing).

### 12. Touches, the 3-touch rule and double contacts
```
python scripts/touches_rules.py results/ball/game2_A_game2_B_sdr_ball_from30.json --players results/players/game2_A_game2_B_sdr_players_from30.csv
```
Adapted from the single-camera pipeline in `volleyball_cv_inout/` (its `ball_events.py`,
`touches.py` and `rules.py`), but on the two-camera 3D ball track. `--players` (the
`players_on_court.py` CSV for the same window) is optional; it gives track IDs, which are needed for
double contacts.
1. **Contacts:** a ball in free flight follows a fixed-gravity arc. At each 3D sample, arcs fitted to
   the 0.3 s before and the 0.3 s after are compared. Their velocity change is about 0 in flight and
   several m/s at a contact (threshold 4 m/s). The contact time and point are where the two arcs meet.
   Across a gap in the 3D track, the arcs either side are extrapolated: if they join smoothly there was
   no contact, if they meet with a kink the contact is placed there, and if they don't meet the gap is
   marked as a possible unseen contact. Floor bounces (and their IN/OUT call) come from `line_calls.py`.
   Contacts with the ball centre below 0.2 m are the ball on the floor.
2. **Who touched it:** the 3D contact point is projected into both cameras at the frames before, at
   and after the contact. YOLO11m-pose runs on a crop of about 4 x 5 m around it, so far players are
   seen at a usable size. For each person: the gap from the ball's surface to the hands (wrist + 0.35
   x forearm), forearms and head, in metres at the ball's distance. People whose feet (ankles, else
   box bottom) are more than 1.8 m from the ball horizontally are rejected; this removes players
   behind the ball along the camera ray. The closest person within 0.3 m touched it.
   Team = side of the net where they stand (A = camera A's half, y > 9).
3. **Rules (FIVB), per rally:** a rally starts at a serve (first contact with the ball behind an end
   line) or the first contact after a dead ball, and ends at a floor bounce or when the ball is unseen
   for 2 s. Possession changes when the other team touches the ball or the 3D track crosses the net.
   A block (toucher within 1 m of the net, both wrists above the head, ball within 0.3 m of the net
   top or higher, arriving from the other side) doesn't count as a hit and the blocker may touch
   again. `FOUR_TOUCHES` = a 4th team hit; `DOUBLE_CONTACT` = the same track ID twice in a row (not
   after a block). A count is only **certain** if the ball was seen (3D or either camera) with no gap
   over 0.5 s since the previous contact; otherwise no foul is flagged. A floor bounce IN gives the
   point to the team on the other side; OUT gives it to the opponents of the last team to touch it.

Outputs in `results/touches/`: `_touches.csv` (one row per contact or bounce), `_touches.png`
(ball height with every event labelled, plus contact points on the court) and `_evidence.jpg` (the
crop used for each contact, with the skeleton and the ball).

What changed from `volleyball_cv_inout/`: contacts are found from 3D velocity changes rather than
2D pixel kinks; the ball's position at each frame comes from the arcs (known where the 2D detector
missed it); the toucher has to be within reach in court metres, not just close in the image;
possession uses real net crossings; YOLO11-pose replaces RTMPose (no extra install).

For contacts at 2 m or higher the toucher may be jumping, and a jumping player's feet map too far
from the camera (step 7). So their feet are allowed anywhere on the camera ray from the floor up to
0.6 m. If that leaves the side of the net ambiguous, the team is guessed from where the ball came
from, flagged `TEAM_GUESSED`, and the count is not trusted.

**Result (Game 2, 30-60 s, 3D on 46% of frames):** 9 contacts and 1 floor bounce found, 7 contacts
attributed. The first rally reads B serve, A hit 1, A hit 2, an unattributed contact at the net,
floor IN on A's side, point to B. No fouls in this footage (the rule logic was checked on
constructed sequences: 4 hits, the same player twice, a block followed by 3 hits and a touch by the
blocker, and a hidden stretch that must not raise a foul). Not yet measured against hand-labelled
contacts; contacts during unseen stretches (e.g. the serve at about 48 s) are missed.

### Utility: click and plot points
```
python scripts/pick_points.py data/game2_A.mp4 --time 60 --names ball landing
python scripts/pick_points.py frames/game2_A/t0120.00.png --show results/points/game2_A_t60.json
```
Saves pixel coordinates to `results/points/<name>.json` with an annotated image, or redraws
saved points on another frame. Use it to hand-label ball landing spots as ground truth.

---

## Earlier work: court detection on broadcast footage

Development footage: a broadcast match (USA vs Brazil) filmed from behind one end line. The
camera holds still during rallies but zooms and reframes between rallies.

White lines on that light floor are nearly invisible (brightness 245 vs 241), so
`court_detect.py` uses the colour boundary between the court and the blue free zone:
1. A free-zone colour model, sampled at the bottom of the frame.
2. Boundary scanning along rows and columns.
3. Vectorised RANSAC line fits for the four edges.
4. Corners from line intersections, then a homography to the 9 x 18 m court.
5. The centre and attack lines, never used in the fit, as an independent check.

**Results:** 213 frames across a full match; 15 randomly chosen frames hand-labelled across
different zoom levels.

| Points | Median error | Mean error |
|---|---|---|
| All 6 points | 5.8 cm | 5.8 cm |
| Corners (used by the detector) | 4.7 cm | 6.0 cm |
| Attack-line points (never used by the detector) | 4.9 cm | 5.4 cm |

It runs at about 11–15 ms per frame at 960 px width.

**Limitations:** tested at one venue only, and it needs a free zone of a different colour
from the court, which is why Game 2 uses line detection instead.

| Script | Purpose |
|---|---|
| `court_detect.py` | Free-zone boundary detector, one image: `python scripts/court_detect.py <image> [--gt clicks.npy]` |
| `court_live.py` | Live court detection on a video or camera, with smoothing and HOLD / NO COURT states |
| `annotate_court.py` | Hand-click 6 ground-truth points on random frames |
| `court_batch_eval.py` | Run the detector on a folder of frames and measure the error against ground truth |
| `manual_homography.py` | Four-click homography with an overlay check |
| `court_lines.py` | First attempt (white-line mask, Canny, Hough); fails on light courts, kept as a baseline |

`annotate_court.py` and `court_batch_eval.py` expect frames named `frame_XXXXXX.jpg`, from
the earlier version of `inspect_video.py`. The current version names frames `tSSSS.SS.png`.

---

## Working on the repo

- **Never commit directly to `main`.** Make a branch per task:
  ```
  git checkout main
  git pull
  git checkout -b my-feature
  ```
- Push your branch, open a pull request on GitHub, and merge when it works.
- Before committing, run `git status` and make sure no videos, frames or results are listed.

## Next steps

1. Record both phones at 60 fps with Auto FPS off (camera A's uneven frame timing made a real
   corner bounce UNRELIABLE).
2. Mount the cameras higher, or add a third phone at the side, for high balls and depth near the net.
3. Hand-label real landing spots with `pick_points.py` to measure in/out accuracy on real rallies.
4. Refit both calibrations with the line-centre convention (`calibrate_camera.py --points-json`).
5. Compare the ball detector with TrackNetV3 on the same clips.

## Single-camera pipeline (`volleyball_cv_inout/volleyball_cv/`)
A separate, one-camera pipeline for players (YOLO11-pose, ByteTrack, court minimap), ball
touches (RTMPose), the 3-touch rule and single-camera in/out calls. Run with `run_all.py`;
setup and outputs are in its own README.
