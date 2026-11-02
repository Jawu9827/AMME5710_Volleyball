# Volleyball CV: players, ball, touches and the 3-touch rule

Stage 1 (players) and stage 2 (ball, touches, rules) are ONE pipeline. `run_all.py` runs everything in order,
so the only command you need per video is:

```
python run_all.py --video "C:\path\to\IMG_4833.mp4" --start 0 --dur 30
```

## One-time setup (Windows, VS Code terminal / PowerShell)

```
cd C:\path\to\volleyball_cv
python -m venv .venv
.\.venv\Scripts\Activate.ps1          # if blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
pip install -r requirements.txt
```
Then unzip the RTMPose file you downloaded (`rtmpose-m_simcc-body7_pt-body7_420e-256x192-...zip`) into
`models\` so that a file `models\...\end2end.onnx` exists (any sub-folder depth is fine).
The YOLO weights (`yolo11m-pose.pt`, 40 MB) download automatically on the first run.

Optional, much faster if your laptop has an NVIDIA GPU (check with `nvidia-smi`):
```
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

## Running

```
python run_all.py --video "C:\path\to\IMG_4833.mp4" --start 0 --dur 30
```
- A window opens for **calibration** (once per camera placement): click the court landmarks in the order shown
  at the top (yellow text). Press **s** to skip one you can't see, **u** to undo, **Enter** when done (≥4 points;
  6-8 is better). It shows an empty court (median background), so lines are easy to see.
  Then open `runs\IMG_4833\calib_check.jpg`: the red court model must sit on the painted lines. If not,
  delete `calib.json` and re-run to click again.
- Everything lands in `runs\IMG_4833\`. Watch `touches_overlay.mp4` (VLC plays it; Windows' player may not).
- Re-running the same command **resumes** (finished steps are skipped). Force steps with `--redo rules render_overlay`, or `--redo all`.
- Same tripod position for another clip → reuse the calibration: `--calib runs\IMG_4833\calib.json`.

Speed on a laptop CPU: ~25-35 min per minute of video (player detection at 10 fps is the slow part).
Test with `--dur 30` first, then lengthen. `--det-fps 5` halves the player-detection time.

## Outputs (in the run folder)

| file | what |
|---|---|
| `tracking.mp4` | players, IDs, team colours, court minimap |
| `touches_overlay.mp4` | + ball trail, touch / floor markers, rally and touch counter |
| `positions.csv` | per player per frame: court x, y (m), team, track id |
| `ball_track.csv` | ball pixel track at full frame rate |
| `touches.csv`, `touch_evidence.jpg` | each ball event, who touched it, evidence crops |
| `flights.csv` | 3-D ballistic fits between contacts |
| `rallies.csv` | rally number, team touch count, FOUR_TOUCHES / DOUBLE_CONTACT flags |
| `landings.csv` | IN / OUT per landing: measured spot (m), margin to the line, uncertainty, predicted-landing error after 25/50/75 % of the flight |

## Steps (each can be run alone with `python <step>.py --run runs\IMG_4833`)
make_background → calibrate → vb_tracker → ball_track → ball_events → touches → flights → rules → landing → render_overlay

## IN / OUT calls (landing.py)
- **Measured:** at the bounce the ball's lowest pixel is on the floor, where the homography is exact → court (x, y).
  A ball touching the line is IN; calls within ±3 px of floor resolution are marked "(close)".
- **Predicted (live, during flight):** the 3-D gravity fit says *when* the ball meets the floor; the pixel track,
  extrapolated with a quadratic (as in the cvzone article), says *where* in the image; the homography maps that to metres.
- **Focal length is refined from physics:** the f at which ballistic fits land exactly when the ball is seen to land.
  Needs ≥1 landing with a tracked flight; each landing is evaluated with f refined on the others.

## Assumptions that matter for a new video
- Camera is **static** (tripod). If it moves, the homography and background are wrong.
- Ball is **yellow** (HSV hue 14-36 in `ball/ball_cands.py`). Change the range for another ball colour.
- Frames are resized to 1920 px wide internally; any fps works (thresholds scale from 60 fps).
