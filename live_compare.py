import os

os.environ.setdefault("MEDIAPIPE_DISABLE_TENSORFLOW", "1")

import argparse
import json
import sys
import time
import types
from importlib import util

import cv2
import matplotlib.pyplot as plt
import numpy as np

# Optional audio cue support on Windows.
try:
    import winsound  # type: ignore
except ImportError:  # pragma: no cover
    winsound = None

# Stub mediapipe to avoid heavy optional imports triggered by package __init__.
_mp_spec = util.find_spec("mediapipe")
if _mp_spec and _mp_spec.submodule_search_locations:
    _mp_stub = types.ModuleType("mediapipe")
    _mp_stub.__path__ = list(_mp_spec.submodule_search_locations)
    sys.modules.setdefault("mediapipe", _mp_stub)

from mediapipe.python.solutions import drawing_utils as mp_draw
from mediapipe.python.solutions import pose as mp_pose

import angles
import io_utils
import preprocess

# --- Configuration and Setup ---
try:
    with open('config.json', 'r') as f:
        config = json.load(f)
except FileNotFoundError:
    print("E109: config.json not found.")
    exit()

POSE = mp_pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# --- Argument Parsing ---
parser = argparse.ArgumentParser(description="Compare live webcam feed to a canonical dance.")
parser.add_argument('--canonical', type=str, required=True, help="Path to the canonical dance CSV file.")
parser.add_argument('--countdown', type=int, default=3, help="Seconds to count down before starting capture.")
parser.add_argument('--display-width', type=int, default=960, help="Width (in px) for each video pane.")
args = parser.parse_args()

# --- Utilities ---
def _play_beep(frequency: int, duration_ms: int) -> None:
    if winsound:
        winsound.Beep(frequency, duration_ms)


def _run_countdown(seconds: int) -> None:
    if seconds <= 0:
        return
    print(f"Preparing to start. Countdown: {seconds} seconds.")
    for remaining in range(seconds, 0, -1):
        print(f"Starting capture in {remaining}...")
        _play_beep(880, 200)
        time.sleep(1)
    print("Go!")
    _play_beep(1200, 300)


def _resize_keep_aspect(frame: np.ndarray, width: int) -> np.ndarray:
    if frame is None or frame.size == 0:
        return frame
    h, w = frame.shape[:2]
    if w == 0:
        return frame
    scale = width / float(w)
    new_size = (width, max(1, int(round(h * scale))))
    return cv2.resize(frame, new_size)


# --- Main Comparison Logic ---
def main():
    print("Step: Loading canonical dance... (Why: Preparing for comparison)")
    canonical_df, meta = io_utils.read_angles_csv(args.canonical)
    if canonical_df is None:
        print("E101: Failed to load canonical dance file.")
        return
    
    print(f"Result: Loaded {meta.get('source_file', 'canonical dance')} with {meta.get('frame_count')} frames.")

    canonical_duration = float(canonical_df['timestamp'].iloc[-1]) if not canonical_df.empty else 0.0

    reference_cap = None
    reference_path = meta.get('source_file')
    reference_fps = meta.get('fps')
    if reference_path and os.path.exists(reference_path):
        reference_cap = cv2.VideoCapture(reference_path)
        if not reference_cap.isOpened():
            print(f"E110: Reference video open failed ({reference_path}).")
            reference_cap = None
        else:
            print(f"Result: Reference playback armed from {reference_path}.")
            if not reference_fps or reference_fps <= 0:
                reference_fps = reference_cap.get(cv2.CAP_PROP_FPS) or None
    else:
        if reference_path:
            print(f"E110: Reference video missing at {reference_path}.")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("E102: Webcam open failed.")
        if reference_cap:
            reference_cap.release()
        return

    webcam_fps = cap.get(cv2.CAP_PROP_FPS)
    if not webcam_fps or webcam_fps <= 0:
        webcam_fps = 30.0

    target_fps = reference_fps or webcam_fps
    frame_interval = 1.0 / max(target_fps, 1.0)
    last_frame_time = time.time()

    _run_countdown(args.countdown)
    cv2.namedWindow('Comparison View', cv2.WINDOW_NORMAL)

    # --- Matplotlib Setup for Live Plotting ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5.5, 4.0))
    live_line, = ax1.plot([], [], 'g-', label='Live Angle')
    canonical_line, = ax1.plot([], [], 'b-', label='Canonical Angle')
    diff_line, = ax2.plot([], [], 'r-', label='Difference')
    
    ax1.legend()
    ax2.legend()
    ax1.set_title("Joint Angle Comparison")
    ax1.set_ylabel("Angle (degrees)")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Absolute Difference")

    current_joint_idx = 0
    
    def update_plot(frame_data):
        live_angles, canonical_angles, diffs, timestamps = frame_data
        
        joint_name = config['joints'][current_joint_idx]
        ax1.set_title(f"Comparison for {joint_name}")

        live_joint_angles = [d.get(joint_name, np.nan) for d in live_angles]
        canonical_joint_angles = [d.get(joint_name, np.nan) for d in canonical_angles]
        
        live_line.set_data(timestamps, live_joint_angles)
        canonical_line.set_data(timestamps, canonical_joint_angles)
        diff_line.set_data(timestamps, [d.get(joint_name, np.nan) for d in diffs])
        
        ax1.relim()
        ax1.autoscale_view()
        ax2.relim()
        ax2.autoscale_view()
        return live_line, canonical_line, diff_line

    plt.ion()
    plt.show(block=False)
    fig.tight_layout()

    # --- Real-time Loop ---
    live_angles_history = []
    canonical_angles_history = []
    diff_history = []
    timestamp_history = []
    start_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        now_time = time.time()
        elapsed = now_time - last_frame_time
        if elapsed < frame_interval:
            time.sleep(frame_interval - elapsed)
        last_frame_time = time.time()
        
        frame = cv2.flip(frame, 1)
        
        # Live Pose
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = POSE.process(rgb_frame)
        
        timestamp = time.time() - start_time
        
        if results.pose_landmarks:
            live_angle_dict, pose_conf = angles.outputToAngleDict(results.pose_landmarks.landmark, mp_pose)
            
            # Find nearest canonical frame (loop when live session runs longer than reference)
            effective_time = timestamp
            if canonical_duration > 0:
                effective_time = timestamp % canonical_duration

            nearest_idx = (canonical_df['timestamp'] - effective_time).abs().idxmin()
            canonical_row = canonical_df.iloc[nearest_idx]
            
            canonical_angle_dict = {j: canonical_row.get(j, np.nan) for j in config['joints']}
            
            # Draw skeletons
            mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS, 
                                   landmark_drawing_spec=mp_draw.DrawingSpec(color=config['plotting']['live_color']))
            
            # (Optional) Draw canonical skeleton - would require storing landmarks
            
            # Update history for plotting
            live_angles_history.append(live_angle_dict)
            canonical_angles_history.append(canonical_angle_dict)
            diff_history.append({j: abs(live_angle_dict.get(j, 0) - canonical_angle_dict.get(j, 0)) for j in config['joints']})
            timestamp_history.append(timestamp)

            # Limit history size
            if len(timestamp_history) > 100:
                live_angles_history.pop(0)
                canonical_angles_history.pop(0)
                diff_history.pop(0)
                timestamp_history.pop(0)

            update_plot((live_angles_history, canonical_angles_history, diff_history, timestamp_history))
            fig.canvas.draw_idle()
            plt.pause(0.001)

        display_width = max(320, args.display_width)
        live_display = _resize_keep_aspect(frame, display_width)

        if reference_cap:
            ref_ok, ref_frame = reference_cap.read()
            if not ref_ok:
                # Loop the reference video so the visual guide keeps moving.
                reference_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ref_ok, ref_frame = reference_cap.read()

            if ref_ok:
                reference_display = _resize_keep_aspect(ref_frame, display_width)
                if target_fps and reference_cap:
                    ref_interval = 1.0 / max(target_fps, 1.0)
                    time.sleep(max(ref_interval - (time.time() - last_frame_time), 0))
            else:
                reference_cap.release()
                reference_cap = None
                reference_display = None
        else:
            reference_display = None

        if reference_display is None:
            reference_display = np.zeros_like(live_display)
            cv2.putText(reference_display, 'Reference video unavailable', (20, reference_display.shape[0] // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        pad_height = abs(reference_display.shape[0] - live_display.shape[0])
        if pad_height > 0:
            if reference_display.shape[0] < live_display.shape[0]:
                reference_display = np.pad(reference_display, ((0, pad_height), (0, 0), (0, 0)), mode='constant')
            else:
                live_display = np.pad(live_display, ((0, pad_height), (0, 0), (0, 0)), mode='constant')

        combined = np.vstack([reference_display, live_display])

        joint_name = config['joints'][current_joint_idx]
        last_diff = diff_history[-1].get(joint_name, np.nan) if diff_history else np.nan
        overlay_text = f"Joint: {joint_name} | Diff: {last_diff:.1f} deg"
        cv2.putText(combined, overlay_text, (20, combined.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(combined, 'Top: Reference  |  Bottom: Live Feed (press J to cycle joints, Q to quit)',
                    (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow('Comparison View', combined)
        cv2.resizeWindow('Comparison View', combined.shape[1], combined.shape[0])

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('j'):
            current_joint_idx = (current_joint_idx + 1) % len(config['joints'])
            print(f"Switched to joint: {config['joints'][current_joint_idx]}")

    cap.release()
    if reference_cap:
        reference_cap.release()
    cv2.destroyAllWindows()
    plt.ioff()

if __name__ == "__main__":
    main()
