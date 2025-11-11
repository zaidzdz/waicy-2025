import cv2
import time
import mediapipe as mp
import numpy as np
import angles
import csv


def csvToArrays (CSV):
    data = []
    with open(CSV, newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            # Skip empty rows
            if not row:
                continue
            # Convert all values *after* the first one to floats
            numbers = [float(x) for x in row[1:]]
            data.append(numbers)
    return data

dance1data = csvToArrays("dances/output/Dance1.csv")

mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)




angle_map = {
    "LEFT_ELBOW":     ("LEFT_SHOULDER", "LEFT_ELBOW", "LEFT_WRIST"),
    "RIGHT_ELBOW":    ("RIGHT_SHOULDER", "RIGHT_ELBOW", "RIGHT_WRIST"),
    "LEFT_KNEE":      ("LEFT_HIP", "LEFT_KNEE", "LEFT_ANKLE"),
    "RIGHT_KNEE":     ("RIGHT_HIP", "RIGHT_KNEE", "RIGHT_ANKLE"),
    "LEFT_SHOULDER":  ("LEFT_ELBOW", "LEFT_SHOULDER", "LEFT_HIP"),
    "RIGHT_SHOULDER": ("RIGHT_ELBOW", "RIGHT_SHOULDER", "RIGHT_HIP"),
    "LEFT_HIP":       ("LEFT_SHOULDER", "LEFT_HIP", "LEFT_KNEE"),
    "RIGHT_HIP":      ("RIGHT_SHOULDER", "RIGHT_HIP", "RIGHT_KNEE"),
}

prev = 0
framenum = 0
with mp_pose.Pose(static_image_mode=False,
                  model_complexity=1,
                  enable_segmentation=False,
                  min_detection_confidence=0.5,
                  min_tracking_confidence=0.5) as pose:

    while True:
        success, frame = cap.read()
        if not success:
            break
        
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        if results.pose_landmarks:
            # Make a shallow copy so we can modify visibility
            
            mp_draw.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_draw.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=3),
                connection_drawing_spec=mp_draw.DrawingSpec(color=(0,200,255), thickness=2)
            )

            landmarks = results.pose_landmarks.landmark

            
            camAngles = angles.outputToAngleArray(landmarks,mp_pose)
            danceAngles = dance1data[framenum]
            

            

        # FPS display
        now = time.time()
        fps = 1 / (now - prev)
        prev = now
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Difference: {angles.differenceAngleArrays(camAngles, danceAngles)}",
            (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("MediaPipe Pose (Body Only - No Head)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        framenum+=1

cap.release()
cv2.destroyAllWindows()
