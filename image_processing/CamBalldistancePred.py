#!/usr/bin/env python3
# CamBalldistancePred.py with Distance Prediction Function and Camera Wrapper

import json
import os

from utility.config_reader import CONFIG
from image_processing.ballDetection import detect_golfballs as yolo_detect

# -------- Calibration constants --------
CALIB_FILE = CONFIG.get("Calibration", "calib_file", fallback="calibration.json")
GOLF_BALL_DIAMETER_MM = CONFIG.getfloat(
    "Calibration", "ball_diameter_mm", fallback=42.67
)
GOLF_BALL_RADIUS_MM = GOLF_BALL_DIAMETER_MM / 2.0  # 21.335 mm
YOLO_CONF = CONFIG.getfloat("YOLO", "conf", fallback=0.3)
YOLO_IMGSZ = CONFIG.getint("YOLO", "imgsz", fallback=640)
# -------- Camera (mvsdk) config --------
ROI_W = CONFIG.getint("Camera", "roi_w", fallback=640)
ROI_H = CONFIG.getint("Camera", "roi_h", fallback=300)
ROI_X = CONFIG.getint("Camera", "roi_x", fallback=0)
ROI_Y = CONFIG.getint("Camera", "roi_y", fallback=100)
EXPOSURE_US = CONFIG.getint("Camera", "exposure_us", fallback=500)  # 0.5 ms


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
