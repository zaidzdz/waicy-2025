import cv2
import time
import mediapipe as mp
import numpy as np
import angles

# This script remains as a lightweight demonstration of live angle calculation.
# For recording and comparison, please use recorder.py and live_compare.py.

mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
prev_time = 0

print("Running lightweight demo. This will print live angles to the console.")
print("For full functionality (recording, comparison), use recorder.py and live_compare.py.")

with mp_pose.Pose(static_image_mode=False,
                  model_complexity=1,
                  enable_segmentation=False,
                  min_detection_confidence=0.5,
                  min_tracking_confidence=0.5) as pose:

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        if results.pose_landmarks:
            mp_draw.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                connection_drawing_spec=mp_draw.DrawingSpec(color=(0, 200, 255), thickness=2)
            )

            landmarks = results.pose_landmarks.landmark
            
            # Using the new, robust angle calculation
            angle_dict, pose_confidence = angles.outputToAngleDict(landmarks, mp_pose)
            
            print(f"Pose Confidence: {pose_confidence:.2f}", end=' | ')
            for name, angle in angle_dict.items():
                print(f"{name}: {angle:.1f}", end=' | ')
            print()

        # FPS display
        now = time.time()
        fps = 1 / (now - prev_time)
        prev_time = now
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Lightweight Pose Demo", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
