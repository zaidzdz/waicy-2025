import cv2
import mediapipe as mp
import numpy as np
import argparse
import json
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

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

mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils
POSE = mp_pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# --- Argument Parsing ---
parser = argparse.ArgumentParser(description="Compare live webcam feed to a canonical dance.")
parser.add_argument('--canonical', type=str, required=True, help="Path to the canonical dance CSV file.")
args = parser.parse_args()

# --- Main Comparison Logic ---
def main():
    print("Step: Loading canonical dance... (Why: Preparing for comparison)")
    canonical_df, meta = io_utils.read_angles_csv(args.canonical)
    if canonical_df is None:
        print("E101: Failed to load canonical dance file.")
        return
    
    print(f"Result: Loaded {meta.get('source_file', 'canonical dance')} with {meta.get('frame_count')} frames.")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("E102: Webcam open failed.")
        return

    # --- Matplotlib Setup for Live Plotting ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
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

    ani = FuncAnimation(fig, update_plot, blit=True, interval=50)
    plt.ion()
    plt.show()

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
        
        frame = cv2.flip(frame, 1)
        
        # Live Pose
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = POSE.process(rgb_frame)
        
        timestamp = time.time() - start_time
        
        if results.pose_landmarks:
            live_angle_dict, pose_conf = angles.outputToAngleDict(results.pose_landmarks.landmark, mp_pose)
            
            # Find nearest canonical frame
            nearest_idx = (canonical_df['timestamp'] - timestamp).abs().idxmin()
            canonical_row = canonical_df.iloc[nearest_idx]
            
            canonical_angle_dict = {j: canonical_row[f"{j}_angle"] for j in config['joints']}
            
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

        cv2.imshow('Live Comparison', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('j'):
            current_joint_idx = (current_joint_idx + 1) % len(config['joints'])
            print(f"Switched to joint: {config['joints'][current_joint_idx]}")

    cap.release()
    cv2.destroyAllWindows()
    plt.ioff()

if __name__ == "__main__":
    main()
