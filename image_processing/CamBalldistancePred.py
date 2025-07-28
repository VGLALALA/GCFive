#!/usr/bin/env python3
# CamBalldistancePred.py with Distance Prediction Function and Camera Wrapper

import json
import os

from image_processing.ballDetection import detect_golfballs as yolo_detect

# -------- Calibration constants --------
CALIB_FILE = "calibration.json"
GOLF_BALL_DIAMETER_MM = 42.67
GOLF_BALL_RADIUS_MM = GOLF_BALL_DIAMETER_MM / 2.0  # 21.335 mm
YOLO_CONF = 0.3
YOLO_IMGSZ = 640
# -------- Camera (mvsdk) config --------
ROI_W, ROI_H = 640, 300
ROI_X, ROI_Y = 0, 100
EXPOSURE_US = 50  # 0.5 ms


# -------- Calibration helpers --------
def load_calibration(path=CALIB_FILE):
    if not os.path.exists(path):
        return None, []
    with open(path, "r") as f:
        data = json.load(f)
    return data.get("focal_length_px"), data.get("samples", [])


def calc_focal_length_px(radius_px, distance_mm):
    return (radius_px * distance_mm) / GOLF_BALL_RADIUS_MM


def estimate_distance_mm(focal_px, radius_px):
    print()
    return (focal_px * GOLF_BALL_RADIUS_MM) / radius_px


# -------- Distance prediction helper --------
def predict_distance_from_frame(frame):
    focal_px, _ = load_calibration()
    """
    Given a BGR frame and focal length (px), detect the largest golf ball
    and return estimated distance in millimeters (or None if no detection).
    """
    dets = yolo_detect(frame, conf=YOLO_CONF, imgsz=YOLO_IMGSZ, display=False)
    # print(dets)
    if not dets:
        return None
    # choose largest
    dets.sort(key=lambda t: t[2], reverse=True)
    _, _, r_px = dets[0]
    # print(f"Debug: Focal Length (px) = {focal_px}, Radius (px) = {r_px}")
    return estimate_distance_mm(focal_px, r_px)
