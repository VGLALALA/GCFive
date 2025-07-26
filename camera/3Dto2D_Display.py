#!/usr/bin/env python3
"""
3Dto2D_Display.py
------------------
Project 3D points (e.g., a golf ball flight path or a single 3D location) onto the
camera image using a simple pinhole model. Uses the focal length (in pixels)
produced by focalPointCalibration.py and assumes fx ≈ fy = focal_px with the
principal point at the ROI center by default.

Features
--------
- Load `calibration.json` to get focal_length_px.
- Build an intrinsic matrix K from focal_px and ROI geometry.
- Project arbitrary 3D points using OpenCV's `projectPoints`.
- Live viewer: grab frames from the mvsdk camera (preferred) or fall back to
  a standard webcam. Overlay projected points/paths.
- Optional YOLO detection overlay of the 2D ball center for visual reference.

Key bindings
------------
q   : quit
p   : toggle showing the projected demo trajectory
b   : toggle drawing of a synthetic 3D ball point (single point)
d   : toggle YOLO detection overlay (largest ball)

Notes
-----
- Coordinate convention for 3D points used here: X right, Y up, Z forward
  (away from camera). This matches a common computer-vision convention. You may
  adapt the handedness/axes to your own pipeline; only consistency matters.
- The demo generates a short parabolic arc in meters then converts to mm.
- Distortion is assumed zero. If you later estimate distortion coefficients,
  pass them into `project_points`.
"""
from __future__ import annotations

import os
import json
import math
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import cv2


from image_processing.ballDetection import detect_golfballs as yolo_detect  # type: ignore

CALIB_FILE = "calibration.json"
ROI_W, ROI_H = 640, 300
ROI_X, ROI_Y = 0, 100
EXPOSURE_US = 50

YOLO_CONF  = 0.25
YOLO_IMGSZ = 640

# -------------------------- Data structures --------------------------
@dataclass
class Intrinsics:
    K: np.ndarray                 # 3x3 camera matrix
    dist: Optional[np.ndarray]    # distortion coeffs (k1,k2,p1,p2,k3) if known

# -------------------------- Calibration I/O --------------------------
def load_focal_length_px(path: str = CALIB_FILE) -> Optional[float]:
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        data = json.load(f)
    return float(data.get("focal_length_px")) if "focal_length_px" in data else None

# -------------------------- Intrinsics builder --------------------------
def build_intrinsics(
    focal_px: float,
    width: int,
    height: int,
    cx: Optional[float] = None,
    cy: Optional[float] = None,
    dist: Optional[Sequence[float]] = None,
) -> Intrinsics:
    """Create intrinsics with fx=fy=focal_px and principal point (cx, cy).
    If cx/cy are None, use the image center.
    """
    if cx is None:
        cx = width * 0.5
    if cy is None:
        cy = height * 0.5
    K = np.array([[focal_px, 0.0, cx],
                  [0.0, focal_px, cy],
                  [0.0, 0.0, 1.0]], dtype=np.float64)
    d = None if dist is None else np.array(dist, dtype=np.float64).reshape(-1, 1)
    return Intrinsics(K=K, dist=d)

# -------------------------- Projection --------------------------
def project_points(
    pts_3d_mm: np.ndarray,
    intr: Intrinsics,
    rvec: Optional[np.ndarray] = None,
    tvec: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Project 3D points (in millimetres) to pixel coordinates.

    Args:
        pts_3d_mm: (N, 3) array of points in camera coordinates, mm.
        intr: intrinsics (K, dist).
        rvec: Rodrigues rotation vector (3,), default zero.
        tvec: translation vector (3,), default zero.

    Returns:
        pts_2d: (N, 2) array of pixel coordinates.
    """
    pts_3d_mm = np.asarray(pts_3d_mm, dtype=np.float64).reshape(-1, 3)
    if rvec is None:
        rvec = np.zeros((3, 1), dtype=np.float64)
    else:
        rvec = np.asarray(rvec, dtype=np.float64).reshape(3, 1)
    if tvec is None:
        tvec = np.zeros((3, 1), dtype=np.float64)
    else:
        tvec = np.asarray(tvec, dtype=np.float64).reshape(3, 1)

    pts_2d, _ = cv2.projectPoints(pts_3d_mm, rvec, tvec, intr.K, intr.dist)
    return pts_2d.reshape(-1, 2)

# -------------------------- Drawing helpers --------------------------
def draw_points(frame: np.ndarray, pts_px: np.ndarray, radius: int = 3):
    for p in pts_px:
        x, y = int(round(p[0])), int(round(p[1]))
        cv2.circle(frame, (x, y), radius, (0, 255, 0), -1, lineType=cv2.LINE_AA)

def draw_polyline(frame: np.ndarray, pts_px: np.ndarray, thickness: int = 2):
    if len(pts_px) < 2:
        return
    pts = np.round(pts_px).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(frame, [pts], isClosed=False, color=(255, 0, 0), thickness=thickness, lineType=cv2.LINE_AA)

def draw_text(frame: np.ndarray, text: str, org=(10, 20)):
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 255, 255), 2, cv2.LINE_AA)

# -------------------------- Demo trajectory --------------------------
def demo_arc_points_m(num: int = 40) -> np.ndarray:
    """Generate a tiny parabolic arc in metres: forward Z, up Y, right X.
    Returns (N,3) in metres.
    """
    # Parameters roughly resembling a short-launch segment
    z_max = 1.2   # forward reach (m)
    x_span = 0.10 # slight right curve (m)
    y_apex = 0.25 # peak height (m)

    z = np.linspace(0.10, z_max, num)
    x = np.linspace(0.0, x_span, num)
    # simple parabola peaking near mid-flight
    y = 4 * y_apex * (z / z_max) * (1 - (z / z_max))
    return np.stack([x, y, z], axis=1)

# -------------------------- Detection overlay --------------------------
def draw_largest_ball_detection(frame: np.ndarray):
    if yolo_detect is None:
        draw_text(frame, "YOLO not available", (10, 80))
        return
    dets = yolo_detect(frame, conf=YOLO_CONF, imgsz=YOLO_IMGSZ, display=False) or []
    if not dets:
        draw_text(frame, "No ball detected", (10, 80))
        return
    xc, yc, r = max(dets, key=lambda t: t[2])
    xi, yi, ri = int(round(xc)), int(round(yc)), int(round(r))
    cv2.circle(frame, (xi, yi), ri, (0, 255, 255), 2)
    cv2.rectangle(frame, (xi - ri, yi - ri), (xi + ri, yi + ri), (255, 0, 255), 1)
    draw_text(frame, f"Ball @ ({xi},{yi}) r={ri}px", (10, 100))

# -------------------------- Live viewer --------------------------
def open_camera_prefer_mvsdk() -> Tuple[Optional[object], Optional[cv2.VideoCapture], int, int]:
    """Try mvsdk first; fall back to cv2.VideoCapture(0).
    Returns (mvs_cam, cv_cap, width, height).
    """
    # Try mvsdk MVSCamera
    if MVSCamera is not None:
        try:
            mcam = MVSCamera(ROI_W, ROI_H, ROI_X, ROI_Y, EXPOSURE_US)
            mcam.open()
            return mcam, None, ROI_W, ROI_H
        except Exception as e:  # pragma: no cover - hardware dependent
            print(f"[WARN] MVSCamera open failed, falling back to cv2: {e}")
    # Fallback to OpenCV webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():  # pragma: no cover - hardware dependent
        raise RuntimeError("No camera available (mvsdk and cv2 both failed)")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return None, cap, w, h


def read_frame(mcam, cap) -> np.ndarray:
    if mcam is not None:
        frame = mcam.grab()
        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        return frame
    assert cap is not None
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError("Failed to read frame from cv2 camera")
    return frame


def release_camera(mcam, cap):
    try:
        if mcam is not None:
            mcam.close()
        if cap is not None:
            cap.release()
    finally:
        pass

# -------------------------- Main --------------------------
def main():
    focal_px = load_focal_length_px(CALIB_FILE)
    if focal_px is None:
        print("[WARN] No focal length in calibration.json; using 1000 px default.")
        focal_px = 1000.0

    mcam, cap, width, height = open_camera_prefer_mvsdk()
    intr = build_intrinsics(focal_px, width, height)
    print("Intrinsics K=\n", intr.K)

    # Demo geometry: create a 3D arc (metres → mm)
    arc_m = demo_arc_points_m(50)
    arc_mm = arc_m * 1000.0

    # Place the arc so it starts ~300 mm in front of the camera and goes forward.
    # Our convention: Z is forward, Y up, X right. Ensure Z>0 for visible points.
    offset_mm = np.array([0.0, 0.0, 300.0])
    arc_mm = arc_mm + offset_mm

    show_path = True
    show_ball_point = True
    show_detect = True

    cv2.namedWindow("3D→2D", cv2.WINDOW_NORMAL)

    try:
        while True:
            frame = read_frame(mcam, cap)

            overlays: List[str] = []
            overlays.append(f"focal_px={focal_px:.1f}")

            if show_path:
                pts2d = project_points(arc_mm, intr)
                draw_polyline(frame, pts2d)
                overlays.append("path:on")
            else:
                overlays.append("path:off")

            if show_ball_point:
                # Single 3D point 600 mm forward, 30 mm right, 30 mm up
                ball_3d = np.array([[30.0, 30.0, 600.0]])  # mm
                p2d = project_points(ball_3d, intr)
                draw_points(frame, p2d, radius=4)
                overlays.append("ball3D:on")
            else:
                overlays.append("ball3D:off")

            if show_detect:
                draw_largest_ball_detection(frame)
                overlays.append("detect:on")
            else:
                overlays.append("detect:off")

            draw_text(frame, "  |  ".join(overlays), (10, 22))
            draw_text(frame, "q:quit  p:toggle path  b:toggle 3D point  d:toggle detect", (10, height - 12))

            cv2.imshow("3D→2D", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('p'):
                show_path = not show_path
            elif key == ord('b'):
                show_ball_point = not show_ball_point
            elif key == ord('d'):
                show_detect = not show_detect
    finally:
        release_camera(mcam, cap)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
