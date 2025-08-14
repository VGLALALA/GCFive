# Camera Module

This folder documents the modules under `camera/`.

- `600framesTest.py`: Captures 600 frames from the camera to verify throughput.
- `MVSCamera.py`: Wrapper around the MVS SDK for initializing and grabbing frames.
- `cv_grab_callback.py`: Provides an OpenCV callback for asynchronous frame capture.
- `focalPointCalibration.py`: Calibrates the camera's focal point using captured images.
- `get_fps.py`: Utility to measure frames-per-second from the camera.
- `hittingZoneCalibration.py`: Determines the camera's hitting zone region of interest.
- `mvsdk.py`: Low-level Python bindings for the MindVision SDK.
- `__init__.py`: Marks the package and exposes camera utilities.
