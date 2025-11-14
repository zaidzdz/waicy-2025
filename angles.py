import numpy as np

# This dictionary maps joint names to the three landmarks required to calculate the angle.
# The order of landmarks is crucial: (endpoint 1, center point (joint), endpoint 2).
angle_map = {
    "LEFT_SHOULDER":  ("LEFT_ELBOW", "LEFT_SHOULDER", "LEFT_HIP"),
    "RIGHT_SHOULDER": ("RIGHT_ELBOW", "RIGHT_SHOULDER", "RIGHT_HIP"),
    "LEFT_ELBOW":     ("LEFT_SHOULDER", "LEFT_ELBOW", "LEFT_WRIST"),
    "RIGHT_ELBOW":    ("RIGHT_SHOULDER", "RIGHT_ELBOW", "RIGHT_WRIST"),
    "LEFT_HIP":       ("LEFT_SHOULDER", "LEFT_HIP", "LEFT_KNEE"),
    "RIGHT_HIP":      ("RIGHT_SHOULDER", "RIGHT_HIP", "RIGHT_KNEE"),
    "LEFT_KNEE":      ("LEFT_HIP", "LEFT_KNEE", "LEFT_ANKLE"),
    "RIGHT_KNEE":     ("RIGHT_HIP", "RIGHT_KNEE", "RIGHT_ANKLE"),
}

def calculate_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """
    Calculates the angle at joint 'b' between vectors 'ba' and 'bc'.

    This function computes the angle in degrees. It is hardened against numerical
    instability by checking for near-zero vector norms (which occur when points
    are too close) and clamping the cosine value to the valid range [-1.0, 1.0]
    to prevent `np.arccos` from failing.

    Args:
        a (np.ndarray): Coordinates of the first point (e.g., wrist).
        b (np.ndarray): Coordinates of the middle point/joint (e.g., elbow).
        c (np.ndarray): Coordinates of the third point (e.g., shoulder).

    Returns:
        float: The calculated angle in degrees, or np.nan if the angle is undefined
               due to overlapping points.
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b

    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)

    if norm_ba < 1e-6 or norm_bc < 1e-6:
        return np.nan  # Return NaN if points are overlapping

    cosine_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
    
    # Clamp the value to handle potential floating-point inaccuracies
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    
    angle = np.degrees(np.arccos(cosine_angle))
    
    return angle

def outputToAngleDict(landmarks, mp_pose, visibility_threshold=0.3) -> tuple[dict, float]:
    """
    Converts MediaPipe's 33 landmarks into a dictionary of 8 named angles.

    This function iterates through the `angle_map`, calculates each angle, and
    returns a dictionary mapping the angle name to its value. It also computes
    an average pose confidence score based on the visibility of the landmarks
    used in the calculations. If a landmark's visibility is below the threshold,
    the corresponding angle is set to NaN.

    Args:
        landmarks: The pose landmarks detected by MediaPipe.
        mp_pose: The MediaPipe pose solution object.
        visibility_threshold (float): The minimum visibility for a landmark to be used.

    Returns:
        tuple: A tuple containing:
            - dict: A dictionary of {angle_name: angle_value_or_nan}.
            - float: The average pose confidence (visibility) of the used landmarks.
    """
    angle_dict = {}
    confidences = []
    
    for name, (p1_name, p2_name, p3_name) in angle_map.items():
        p1 = landmarks[mp_pose.PoseLandmark[p1_name].value]
        p2 = landmarks[mp_pose.PoseLandmark[p2_name].value]
        p3 = landmarks[mp_pose.PoseLandmark[p3_name].value]

        # Collect visibilities for confidence calculation
        vis1 = p1.visibility if hasattr(p1, 'visibility') else 1.0
        vis2 = p2.visibility if hasattr(p2, 'visibility') else 1.0
        vis3 = p3.visibility if hasattr(p3, 'visibility') else 1.0
        confidences.extend([vis1, vis2, vis3])

        if vis1 < visibility_threshold or vis2 < visibility_threshold or vis3 < visibility_threshold:
            angle_dict[name] = np.nan
            continue

        angle = calculate_angle(
            (p1.x, p1.y, p1.z),
            (p2.x, p2.y, p2.z),
            (p3.x, p3.y, p3.z)
        )
        angle_dict[name] = angle

    pose_confidence = np.mean(confidences) if confidences else 0.0
    return angle_dict, pose_confidence

def outputToAngleArray(landmarks, mp_pose, visibility_threshold=0.3) -> list:
    """
    Converts MediaPipe landmarks to a fixed-order list of 8 angles.

    This function provides backward compatibility with the original API. It calls
    `outputToAngleDict` and returns the angle values in a specific, documented order.
    The order is: LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_ELBOW, RIGHT_ELBOW,
    LEFT_HIP, RIGHT_HIP, LEFT_KNEE, RIGHT_KNEE.

    Args:
        landmarks: The pose landmarks detected by MediaPipe.
        mp_pose: The MediaPipe pose solution object.
        visibility_threshold (float): The minimum visibility for a landmark to be used.

    Returns:
        list: A list of 8 angle values (or NaN) in a fixed order.
    """
    angle_dict, _ = outputToAngleDict(landmarks, mp_pose, visibility_threshold)
    
    # Fixed order for backward compatibility
    fixed_order = [
        "LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_ELBOW", "RIGHT_ELBOW",
        "LEFT_HIP", "RIGHT_HIP", "LEFT_KNEE", "RIGHT_KNEE"
    ]
    
    return [angle_dict.get(name, np.nan) for name in fixed_order]

def differenceAngleArrays(arrayA, arrayB):
    newArray = []

    for i in range(len(arrayA)):
        newArray.append(arrayA[i]-arrayB[i])
    return newArray
