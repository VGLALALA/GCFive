# GCFive

GCFive contains a collection of Python utilities for detecting a golf ball in high‑speed camera images, estimating the ball's spin between frames, and simulating its resulting trajectory. The repository is organized as a set of standalone scripts together with a small physics module translated from the Godot game engine.

## Features

- **Camera capture** – Scripts in the `Camera/` folder interface with the HuaTeng Vision SDK to grab images from a high‑speed industrial camera.
- **Ball detection** – `ball_detection.py` provides a threaded capture pipeline that searches each frame for the golf ball and begins tracking once detected.
- **Spin estimation** – `GetBallRotation.py` isolates the ball from two frames, applies Gabor filtering to highlight dimples, and searches a 3‑D rotation space to find the best alignment.
- **Trajectory physics** – `trajectory_simulation/` contains a Python approximation of the JaySimG project’s physics scripts. `train_physics.py` tunes aerodynamic constants to match real shot data.
- **Example data** – The `data/Images/` folder includes sample frames for experimentation and the `data/spin/` folder holds analysis results.

## Installation

1. Install the Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Install the HuaTeng camera SDK. Binary packages for several architectures are provided. Choose the package matching your system (check `uname -a`). For example, on Ubuntu x86‑64:
   ```bash
   sudo dpkg -i MVS-2.1.2_x86_64_20221024.deb
   ```
   or extract the `.tar.gz` archive and run `setup.sh`.
   A Chinese/English package list is also available in the legacy `README` file for reference.

## Usage

- **Capturing frames** – Run `ball_detection.py` to connect to the camera, detect the ball, and start monitoring. Press `q` in the display window to stop.
- **Estimating spin** – After obtaining two cropped ball images, call `GetBallRotation.get_fine_ball_rotation(image1, image2)` to return the side‑, back‑ and axial‑spin angles. Running the script directly prompts for image paths and prints the spin in RPM.
- **Training the physics model** – Execute `train_physics.py <dataset.csv>` where the CSV contains columns such as `Carry`, `Ball Speed`, `Spin`, `VLA`, `Apex`, and `Land Angle`. The optimizer logs results to `train_logs/` by default.

## Repository layout

```
Camera/                 Camera grabbing helpers and SDK wrappers
trajectory_simulation/  Physics simulation utilities translated from Godot
data/                   Sample images and spin analysis output
lib/, sdk/              Prebuilt binaries for the HuaTeng Vision camera
```

## License

This project is released under the MIT License. See `LICENSE` for details.
