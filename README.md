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
| Ball tracking | On the `Ball-Detection-+-Tracking` branch, not merged yet |
| Bounce detection and in/out calls | Not started |

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

`requirements.txt` must include `opencv-python`, `numpy`, `scipy`, `matplotlib` and
`imageio-ffmpeg`. The last one provides ffmpeg (with the `zscale` filter needed for HDR
conversion), so ffmpeg does not need to be installed separately.

Run all commands from the repo root with the environment activated.

## Data

Videos are **not** in Git: they're too large, and GitHub rejects files over 100 MB.

| Folder | In Git? | Contents |
|---|---|---|
| `data/` | No | Videos, e.g. `data/game2_A.mp4`, `data/game2_B_sdr.mp4` |
| `data/annotations/` | **Yes** | Calibrations, sync offsets, hand-clicked ground truth (small JSON files) |
| `frames/` | No | Frames saved by `inspect_video.py` |
| `results/` | No | Debug images, overlays, combined and synced videos |

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

1. Merge the ball tracker from `Ball-Detection-+-Tracking` and run it on both Game 2 videos.
2. Bounce detection: the frame where the ball's vertical image motion reverses at the floor.
3. Map the bottom of the ball at the bounce through that camera's calibration into court
   coordinates, and call it in or out (line contact counts as in).
4. Hand-label landing spots with `pick_points.py` to measure call accuracy.
