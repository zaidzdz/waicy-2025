import cv2
import mediapipe as mp
import csv
import angles
video_path = 'dances/train/Dance1.mp4'
output_csv = 'dances/output/Dance1.csv'

frame_number = 0
data = []

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(static_image_mode=False,
                  model_complexity=1,
                  enable_segmentation=False,
                  min_detection_confidence=0.5,
                  min_tracking_confidence=0.5)


# Open the video file
cap = cv2.VideoCapture(video_path)



frame_number = 0
    
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe Pose
    result = pose.process(frame_rgb)

    
    if result.pose_landmarks:
        

       angle_array = (angles.outputToAngleArray(result.pose_landmarks.landmark, mp_pose))
       data.append([frame_number] + angle_array)
    frame_number += 1

    # Display the frame

    # Exit if 'q' keypyt

with open(output_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(data)
