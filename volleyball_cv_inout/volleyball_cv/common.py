"""Shared helpers: run folder, video access, frame normalisation.

Every script works inside one RUN FOLDER (e.g. runs/IMG_4833) that holds run.json, calib.json,
bg.png and all outputs. Frames are always resized to 1920 px wide so that pixel thresholds
(ball size, gates, tiles) mean the same thing for 1080p, 4K or 720p input.
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")   # Anaconda + PyTorch OpenMP clash
import argparse, glob, json, sys, cv2
PKG = os.path.dirname(os.path.abspath(__file__))
W_STD = 1920

def run_args(extra=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run folder created by run_all.py")
    if extra: extra(ap)
    a = ap.parse_args()
    a.run = os.path.abspath(a.run)
    os.chdir(a.run)
    a.cfg = json.load(open("run.json"))
    return a

def prep(frame):
    h, w = frame.shape[:2]
    if w == W_STD: return frame
    return cv2.resize(frame, (W_STD, round(h * W_STD / w)), interpolation=cv2.INTER_AREA)

def open_video(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened(): sys.exit(f"cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    return cap, fps

def rtm_model():
    hits = glob.glob(os.path.join(PKG, "models", "**", "end2end.onnx"), recursive=True)
    if not hits: sys.exit("RTMPose model not found: unzip the rtmpose-m ONNX zip into models/ (see README)")
    return hits[0]

def pkg_file(name): return os.path.join(PKG, name)
