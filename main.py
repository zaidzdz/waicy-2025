import cv2
import time
import mediapipe as mp
import numpy as np
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
prev = 0
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
angles = []
def calculate_angle(a, b, c):
    a = np.array(a)  # First point
    b = np.array(b)  # Middle point (joint)
    c = np.array(c)  # Last point

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.degrees(np.arccos(cosine_angle))
    
    return angle

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

            for name, (a, b, c) in angle_map.items():
                p1 = landmarks[mp_pose.PoseLandmark[a].value]
                p2 = landmarks[mp_pose.PoseLandmark[b].value]
                p3 = landmarks[mp_pose.PoseLandmark[c].value]

                angle = calculate_angle(
                    (p1.x, p1.y, p1.z),
                    (p2.x, p2.y, p2.z),
                    (p3.x, p3.y, p3.z)
                )
                angles.append(angle)
            print(angles)
            angles = []

            

        # FPS display
        now = time.time()
        fps = 1 / (now - prev)
        prev = now
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("MediaPipe Pose (Body Only - No Head)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
