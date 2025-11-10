import numpy as np
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
def outputToAngleArray(landmarks,mp_pose):
    angles = []
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
                
    return angles
def differenceAngleArrays(arrayA, arrayB):
    newArray = []

    for i in arrayA:
        newArray.append(arrayA[i]-arrayB[i])
    return newArray