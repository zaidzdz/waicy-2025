import cv2
import time
import mediapipe as mp

mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
prev = 0

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
