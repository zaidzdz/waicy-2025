import cv2
import mediapipe as mp
import json
from datetime import datetime
import numpy as np

import angles
import io_utils

# This script is now a simple example of how to use the new utilities to process a video.
# For more advanced recording, use recorder.py.

# --- Configuration ---
try:
    with open('config.json', 'r') as f:
        config = json.load(f)
except FileNotFoundError:
    print("E109: config.json not found.")
    exit()

video_path = 'dances/train/Dance1.mp4'
output_prefix = 'dances/output/Dance1_legacy'

# --- Processing Logic ---
print(f"Step: Processing video... (Why: Extracting angles from {video_path})")
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"E101: Could not open video file {video_path}")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
rows = []
header = ['frame', 'timestamp', 'fps', 'pose_confidence'] + config['joints']

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, model_complexity=1)

for frame_num in range(frame_count):
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(rgb_frame)
    
    timestamp = frame_num / fps

    if result.pose_landmarks:
        angle_dict, pose_confidence = angles.outputToAngleDict(
            result.pose_landmarks.landmark, mp_pose, config['visibility_threshold']
        )
        row = [frame_num, timestamp, fps, pose_confidence] + [angle_dict.get(j, np.nan) for j in config['joints']]
        rows.append(row)
    else:
        rows.append([frame_num, timestamp, fps, 0.0] + [np.nan] * len(config['joints']))

    if frame_num % 100 == 0:
        print(f"Processed frame {frame_num}/{frame_count}")

cap.release()
pose.close()
print("Result: Video processing complete.")

# --- Save Data ---
metadata = {
    "source_file": video_path,
    "capture_type": "video",
    "fps": fps,
    "frame_count": len(rows),
    "angle_list": config['joints'],
    "timestamp_utc": datetime.utcnow().isoformat() + "Z",
    "mediapipe_version": mp.__version__,
    "notes": f"Recorded by legacy video-train.py"
}

io_utils.write_angles_csv(output_prefix, header, rows, metadata)
