import os
import argparse
import json
import sys
import time
import types
from datetime import datetime
from importlib import metadata, util

os.environ.setdefault("MEDIAPIPE_DISABLE_TENSORFLOW", "1")

import cv2
import numpy as np

# Stub the top-level mediapipe package to avoid importing optional TensorFlow-heavy modules.
_mp_spec = util.find_spec("mediapipe")
if _mp_spec and _mp_spec.submodule_search_locations:
    _mp_stub = types.ModuleType("mediapipe")
    _mp_stub.__path__ = list(_mp_spec.submodule_search_locations)
    sys.modules.setdefault("mediapipe", _mp_stub)

from mediapipe.python.solutions import drawing_utils as mp_draw
from mediapipe.python.solutions import pose as mp_pose

import angles
import io_utils

# Capture the installed MediaPipe version without importing the top-level package (which pulls TensorFlow).
MEDIAPIPE_VERSION = metadata.version("mediapipe")

# --- Configuration and Setup ---
try:
    with open('config.json', 'r') as f:
        config = json.load(f)
except FileNotFoundError:
    print("E109: config.json not found. Please ensure the configuration file exists.")
    exit()

POSE = mp_pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# --- Argument Parsing ---
parser = argparse.ArgumentParser(description="Record canonical dances from video or webcam.")
parser.add_argument('--source', type=str, required=True, choices=['webcam', 'video'], help="Source of the video stream.")
parser.add_argument('--path', type=str, help="Path to the video file (required if source is 'video').")
parser.add_argument('--out', type=str, required=True, help="Prefix for the output CSV and JSON files (e.g., 'dances/output/MyDance').")
args = parser.parse_args()

# --- Main Recording Logic ---
def main():
    if args.source == 'video':
        record_from_video()
    elif args.source == 'webcam':
        record_from_webcam()

def record_from_video():
    """Records angles from a video file."""
    print(f"Step: Opening video file... (Why: Loading video for processing from {args.path})")
    if not args.path:
        print("E101: --path argument is required when --source is 'video'.")
        return

    cap = cv2.VideoCapture(args.path)
    if not cap.isOpened():
        print(f"E101: Input video not found at {args.path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Result: Video opened successfully. FPS: {fps}, Frames: {frame_count}")

    header = ['frame', 'timestamp', 'fps', 'pose_confidence'] + config['joints']
    rows = []
    
    for frame_num in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = POSE.process(rgb_frame)
        
        timestamp = frame_num / fps
        
        if results.pose_landmarks:
            angle_dict, pose_confidence = angles.outputToAngleDict(
                results.pose_landmarks.landmark, mp_pose, config['visibility_threshold']
            )
            
            row = [frame_num, timestamp, fps, pose_confidence] + [angle_dict.get(j, np.nan) for j in config['joints']]
            rows.append(row)
        else:
            # Append NaNs if no pose is detected
            rows.append([frame_num, timestamp, fps, 0.0] + [np.nan] * len(config['joints']))

        if frame_num % 100 == 0:
            print(f"Processing frame {frame_num}/{frame_count}...")

    cap.release()

    metadata_dict = {
        "source_file": args.path,
        "capture_type": "video",
        "fps": fps,
        "frame_count": frame_count,
        "angle_list": config['joints'],
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "mediapipe_version": MEDIAPIPE_VERSION,
        "notes": f"Recorded by recorder.py --source video"
    }
    
    io_utils.write_angles_csv(args.out, header, rows, metadata_dict)

def record_from_webcam():
    """Records angles from a live webcam feed."""
    print("Step: Opening webcam... (Why: Preparing for live recording)")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("E102: Webcam open failed. Check permissions or device index.")
        return

    is_recording = False
    recorded_data = []
    start_time = None
    
    print("Result: Webcam opened. Press 'r' to start/stop recording, 's' to save, 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        
        # --- Display and Recording Logic ---
        hud_text = ""
        if is_recording:
            hud_text = "REC"
            cv2.circle(frame, (30, 30), 15, (0, 0, 255), -1)
            
            # Process frame
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = POSE.process(rgb_frame)
            
            if start_time is None:
                start_time = time.time()

            timestamp = time.time() - start_time
            fps = 1 / (time.time() - (prev_time if 'prev_time' in locals() else time.time() - 0.03))
            
            if results.pose_landmarks:
                angle_dict, pose_confidence = angles.outputToAngleDict(
                    results.pose_landmarks.landmark, mp_pose, config['visibility_threshold']
                )
                row = [len(recorded_data), timestamp, fps, pose_confidence] + [angle_dict.get(j, np.nan) for j in config['joints']]
                recorded_data.append(row)
                
                mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        cv2.putText(frame, hud_text, (60, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow("Webcam Recorder", frame)
        
        prev_time = time.time()
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('r'):
            is_recording = not is_recording
            if is_recording:
                print("Recording started...")
                recorded_data = []
                start_time = time.time()
            else:
                print("Recording stopped.")
        elif key == ord('s'):
            if recorded_data:
                print("Saving data...")
                header = ['frame', 'timestamp', 'fps', 'pose_confidence'] + config['joints']
                metadata_dict = {
                    "source_file": "webcam",
                    "capture_type": "webcam",
                    "fps": np.mean([row[2] for row in recorded_data]),
                    "frame_count": len(recorded_data),
                    "angle_list": config['joints'],
                    "timestamp_utc": datetime.utcnow().isoformat() + "Z",
                    "mediapipe_version": MEDIAPIPE_VERSION,
                    "notes": "Recorded by recorder.py --source webcam"
                }
                io_utils.write_angles_csv(args.out, header, recorded_data, metadata_dict)
                recorded_data = [] # Clear after saving
            else:
                print("No data to save. Record first using 'r'.")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
