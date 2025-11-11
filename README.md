# Dance Analysis Toolkit

This project provides a set of tools to record, analyze, and compare human dance movements from video sources using MediaPipe for pose estimation. It allows you to create "canonical" recordings of dances and then compare live performances against them in real-time.

## Core Features

- **Record Dances**: Capture pose angle data from either a pre-recorded video file or a live webcam feed.
- **Self-Describing Data**: Recordings are saved as a CSV file containing frame-by-frame angle data, along with a paired JSON metadata file that describes the recording context.
- **Live Comparison**: Compare your live webcam pose against a pre-recorded canonical dance, with visual feedback showing both skeletons and a live plot of angle differences.
- **Robust & Modular**: The code is designed to be resilient to errors (like poor landmark visibility) and is organized into modular scripts for clarity and extensibility.

## File Structure

```
.
├── angles.py           # Core angle calculation logic
├── config.json         # Configuration for joints, plotting, etc.
├── dances/
│   ├── output/         # Default location for saved recordings
│   └── train/          # Location for training videos
├── io_utils.py         # Helpers for reading/writing CSV and JSON data
├── live_compare.py     # Main script for real-time comparison
├── logs/               # Directory for log files
├── main.py             # A lightweight demo of live angle calculation
├── preprocess.py       # Functions for smoothing and resampling data
├── README.md           # This file
├── recorder.py         # Main script for recording dances
├── requirements.txt    # Python package dependencies
└── video-train.py      # Legacy script, now refactored as a simple example
```

## Installation

1.  **Clone the repository** (if you haven't already).
2.  **Install dependencies**: It is recommended to use a virtual environment.

    ```bash
    # Create and activate a virtual environment (optional but recommended)
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

    # Install the required packages
    pip install -r requirements.txt
    ```

## Usage

This toolkit provides two main workflows: recording a canonical dance and comparing against it.

### 1. Recording a Canonical Dance

You can record a dance from a video file or a live webcam stream. The output will be a `.csv` file with the angle data and a `_meta.json` file.

#### From a Video File

Use this to process an existing video. The script will process the entire video and save the output.

**Command:**

```bash
python recorder.py --source video --path "dances/train/Dance1.mp4" --out "dances/output/Dance1"
```

- `--source video`: Specifies that the input is a video file.
- `--path`: The path to your video file.
- `--out`: The desired output path and filename prefix. This will create `Dance1.csv` and `Dance1_meta.json`.

#### From a Webcam

Use this to record a new dance live.

**Command:**

```bash
python recorder.py --source webcam --out "dances/output/MyNewDance"
```

**Keybindings:**

-   **`r`**: Press to **start** or **stop** recording. A "REC" indicator will appear when recording is active.
-   **`s`**: Press to **save** the last recorded session to the output files specified by `--out`.
-   **`q`**: Press to **quit** the application.

### 2. Live Comparison

Once you have a canonical dance recording, you can use `live_compare.py` to see how a live performance matches up.

**Command:**

```bash
python live_compare.py --canonical "dances/output/Dance1.csv"
```

- `--canonical`: The path to the `.csv` file of the dance you want to compare against.

This will open two windows:
1.  **OpenCV Window (`Live Comparison`)**:
    -   Shows your live skeleton (in green) overlaid on the translucent canonical skeleton (in blue).
    -   Allows you to visually inspect alignment and posture differences.
2.  **Matplotlib Window (`Joint Angle Comparison`)**:
    -   A live plot showing the angle of a specific joint for both the live and canonical performance.
    -   The bottom plot shows the absolute difference, making it easy to spot deviations.

**Keybindings:**

-   **`j`**: Cycle through the different joints to change which one is being plotted.
-   **`q`**: Quit the application.

## How It Works

1.  **Pose Estimation**: We use Google's [MediaPipe Pose](https://google.github.io/mediapipe/solutions/pose.html) to detect 33 key body landmarks in each frame.
2.  **Angle Calculation (`angles.py`)**: From these landmarks, we calculate 8 key body angles (shoulders, elbows, hips, knees). The calculations are hardened to be numerically stable, returning `NaN` (Not a Number) if a landmark is not clearly visible.
3.  **Data Storage (`io_utils.py`)**: The calculated angles, along with a timestamp and confidence score, are stored in a standard CSV format. A JSON file is also saved to provide context, such as the source video, FPS, and date of recording.
4.  **Real-time Alignment**: During live comparison, the system aligns the live webcam feed with the canonical recording by finding the closest timestamp, ensuring the poses are compared at the same point in the dance.

## Error Codes

The scripts use a set of standardized error codes to help diagnose issues. These are printed to the console if an error occurs.

-   `E101`: Input video/file not found.
-   `E102`: Webcam failed to open.
-   `E103`: MediaPipe failed to initialize.
-   `E104`/`E107`: Failed to write CSV or metadata file (check permissions).
-   `E105`: Too many low-confidence frames detected (check lighting).
-   `E109`: An unknown exception occurred (check logs for details).
