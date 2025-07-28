# GCFive

GCFive provides a set of utilities for capturing high‑speed camera frames of a golf ball, extracting its spin from consecutive images, and simulating the resulting trajectory. The codebase grew out of experimentation with a HuaTeng industrial camera and the physics model from the open‑source JaySimG project. The repository is organised as a standalone Python package accompanied by helper scripts and Jupyter notebooks.

## Features

- **Camera capture** &ndash; Modules in `camera/` wrap the HuaTeng Vision SDK so that images can be streamed from the camera with minimal boilerplate. The `cv_grab_callback` helper manages frame acquisition threads and buffers.
- **Ball detection** &ndash; Image preprocessing routines live in `image_processing/`. They include traditional methods as well as a YOLO based detector (`ballDetectionyolo.py`) used by the main capture pipeline.
- **Spin estimation** &ndash; The `spin/` package searches a 3&ndash;D rotation space to align two ball images. It relies on Gabor filtering to emphasise dimple patterns and outputs side‑, back‑ and axial‑spin.
- **Trajectory physics** &ndash; Scripts in `trajectory_simulation/` are a Python approximation of the Godot scripts from JaySimG. They allow a shot’s flight to be simulated and metrics such as carry distance and apex to be derived.
- **Example data** &ndash; Small sample images are included under `data/Images/` and spin analysis output lives in `data/spin/`.

## Installation

1. Install the Python requirements:
   ```bash
   pip install -r requirements.txt
   ```
2. Install the HuaTeng camera SDK. Prebuilt binaries are provided under `sdk/`. Choose the package matching your architecture (see `uname -a`) and install it, for example on Ubuntu:
   ```bash
   sudo dpkg -i MVS-2.1.2_x86_64_20221024.deb
   ```
   Alternatively extract the `tar.gz` archive and run its `setup.sh` script.
3. (Optional) Review `calibration.json` and `hitting_zone_calibration.json` if you need to adjust the camera calibration or hitting zone samples.

## Quick start

- **Capture and detect**
  ```bash
  python main.py
  ```
  The script connects to the camera, waits for a ball to appear in the predefined zone and records a short burst of frames. Press `q` in the display window to exit.
- **Estimate spin between two images**
  ```bash
  python spin/GetBallRotation.py image1.png image2.png
  ```
  The command line interface prints the calculated RPM values. Alternatively import `get_fine_ball_rotation` from Python code.
- **Simulate a flight**
  ```bash
  python trajectory_simulation/flightDataCalculation.py
  ```
  This module exposes helpers such as `get_trajectory_metrics()` that take launch conditions and return carry distance, total distance and more.
- **Train the physics model**
  ```bash
  python trajectory_simulation/train_physics.py shots.csv
  ```
  The CSV should contain columns like `Carry`, `Ball Speed`, `Spin`, `VLA`, `Apex` and `Land Angle`. Optimiser output is written to `results/train_log/`.

## Running the tests

Unit tests cover the physics utilities and parts of the image pipeline. After installing the requirements simply execute:

```bash
pytest
```

Tests requiring OpenCV attempt to import `cv2`; if the dependency is unavailable they will be skipped.

## Repository layout

```
camera/                Camera and calibration helpers
image_processing/      Preprocessing and detection routines
spin/                  Ball rotation search and utility functions
trajectory_simulation/ Physics model translated from Godot
notebooks/             Example notebooks demonstrating the code
scripts/               Miscellaneous helpers
data/                  Sample images and spin results
sdk/                   HuaTeng Vision SDK binaries
```

## License

This project is released under the MIT License. See `LICENSE` for the full text.

## Development

The repository uses `pre-commit` to enforce code style. Black and isort
are run automatically before each commit. Set up the hooks with:

```bash
pip install pre-commit
pre-commit install
```

You can manually run the checks across the whole project using
`pre-commit run --all-files`.
