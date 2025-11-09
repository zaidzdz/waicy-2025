import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter

def resample_sequence(sequence: np.ndarray, target_len: int) -> np.ndarray:
    """
    Resamples a sequence to a target length using linear interpolation.

    Args:
        sequence (np.ndarray): The input sequence (e.g., a time series of angles).
        target_len (int): The desired length of the output sequence.

    Returns:
        np.ndarray: The resampled sequence.
    """
    if len(sequence) == target_len:
        return sequence
    
    original_indices = np.linspace(0, 1, len(sequence))
    target_indices = np.linspace(0, 1, target_len)
    
    interp_func = interp1d(original_indices, sequence, kind='linear', axis=0, fill_value="extrapolate")
    resampled_sequence = interp_func(target_indices)
    
    return resampled_sequence

def smooth_sequence(sequence: np.ndarray, window: int = 5, polyorder: int = 2) -> np.ndarray:
    """
    Smoothes a sequence using a Savitzky-Golay filter.

    This is useful for reducing jitter in angle data from frame to frame.

    Args:
        sequence (np.ndarray): The input sequence.
        window (int): The length of the filter window (must be odd).
        polyorder (int): The order of the polynomial used to fit the samples.

    Returns:
        np.ndarray: The smoothed sequence.
    """
    if window > len(sequence) or window % 2 == 0:
        # Window must be odd and smaller than the sequence length
        return sequence
    return savgol_filter(sequence, window, polyorder, axis=0)

def normalize_angles(angle_array: np.ndarray, landmarks=None) -> np.ndarray:
    """
    Normalizes angles. (Optional: Can be extended to normalize by torso size if needed).

    Currently, this is a placeholder. A potential implementation could scale angles
    based on the distance between shoulders to make them more invariant to camera distance.

    Args:
        angle_array (np.ndarray): The array of angles to normalize.
        landmarks: The full landmark set (for more advanced normalization).

    Returns:
        np.ndarray: The normalized angles.
    """
    # Placeholder for future normalization logic
    return angle_array
