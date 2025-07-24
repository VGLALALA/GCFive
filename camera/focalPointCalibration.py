#!/usr/bin/env python3
# focalPointCalibration.py

import os
import json
import time
import ctypes
import numpy as np
import cv2

# -------- YOLO / detection --------
YOLO_CONF  = 0.25
YOLO_IMGSZ = 640
CLASS_ID   = 0        # golf ball class
MODEL_PATH = "data/model/golfballv4.pt"  # only used if you uncomment fallback

# Try to use your existing detector
from image_processing.ballDetection import detect_golfballs as yolo_detect
HAS_EXTERNAL_DETECTOR = True

# # ----- Fallback: inline detector (uncomment if needed) -----
# from ultralytics import YOLO
# model = YOLO(MODEL_PATH)
# def yolo_detect(image, conf=YOLO_CONF, imgsz=YOLO_IMGSZ, display=False):
#     res = model.predict(source=image, conf=conf, imgsz=imgsz, verbose=False)
#     if not res or len(res[0].boxes) == 0:
#         return []
#     dets = []
#     for box in res[0].boxes:
#         if hasattr(box, "cls") and int(box.cls.item()) != CLASS_ID:
#             continue
#         x1, y1, x2, y2 = box.xyxy.cpu().numpy().astype(int).flatten()
#         w, h = x2 - x1, y2 - y1
#         r = (w + h) / 4.0
#         xc, yc = x1 + w / 2.0, y1 + h / 2.0
#         dets.append((xc, yc, r))
#     dets.sort(key=lambda t: t[0])
#     return dets

# -------- Calibration constants --------
CALIB_FILE = "calibration.json"
GOLF_BALL_DIAMETER_MM = 42.67
GOLF_BALL_RADIUS_MM   = GOLF_BALL_DIAMETER_MM / 2.0

# -------- Camera (mvsdk) config --------
ROI_W, ROI_H = 640, 300
ROI_X, ROI_Y = 0, 100
EXPOSURE_US  = 500  # 0.5 ms

# Import mvsdk (try relative then absolute)
try:
    from . import mvsdk
except Exception:
    import mvsdk

# -------- UI helpers (Tkinter) --------
try:
    import tkinter as tk
    from tkinter import simpledialog
    HAS_TK = True
except Exception:
    HAS_TK = False

def ask_distance_mm(default_val=1000.0):
    """
    Show a tiny GUI prompt for the true distance (mm).
    Fallback to terminal input if Tk isn't available.
    """
    if HAS_TK:
        root = tk.Tk()
        root.withdraw()
        while True:
            val = simpledialog.askstring("Calibration", "Enter TRUE distance (mm):",
                                         initialvalue=str(default_val), parent=root)
            if val is None:     # user cancelled
                root.destroy()
                return None
            try:
                v = float(val)
                root.destroy()
                return v
            except ValueError:
                continue
    else:
        try:
            return float(input("Enter TRUE distance (mm): ").strip())
        except ValueError:
            return None

# ------------- mvsdk camera wrapper -------------
class MVSCamera:
    def __init__(self, roi_w, roi_h, roi_x, roi_y, exposure_us):
        self.roi_w = roi_w
        self.roi_h = roi_h
        self.roi_x = roi_x
        self.roi_y = roi_y
        self.exposure_us = exposure_us

        self.hCamera = None
        self.mono    = True
        self.buf     = None
        self.buf_sz  = 0

    def open(self):
        devs = mvsdk.CameraEnumerateDevice()
        if not devs:
            raise RuntimeError("No camera found")
        self.hCamera = mvsdk.CameraInit(devs[0], -1, -1)

        cap = mvsdk.CameraGetCapability(self.hCamera)
        self.mono = (cap.sIspCapacity.bMonoSensor != 0)

        fmt = mvsdk.CAMERA_MEDIA_TYPE_MONO8 if self.mono else mvsdk.CAMERA_MEDIA_TYPE_BGR8
        mvsdk.CameraSetIspOutFormat(self.hCamera, fmt)

        res = mvsdk.CameraGetImageResolution(self.hCamera)
        res.iIndex      = 0xFF
        res.iHOffsetFOV = self.roi_x
        res.iVOffsetFOV = self.roi_y
        res.iWidthFOV   = self.roi_w
        res.iHeightFOV  = self.roi_h
        res.iWidth      = self.roi_w
        res.iHeight     = self.roi_h
        mvsdk.CameraSetImageResolution(self.hCamera, res)

        mvsdk.CameraSetTriggerMode(self.hCamera, 0)
        mvsdk.CameraSetAeState(self.hCamera, 0)
        mvsdk.CameraSetExposureTime(self.hCamera, self.exposure_us)

        mvsdk.CameraPlay(self.hCamera)

    def grab(self, timeout_ms=2000):
        pRaw, head = mvsdk.CameraGetImageBuffer(self.hCamera, timeout_ms)

        if (self.buf is None) or (self.buf_sz != head.uBytes):
            if self.buf:
                mvsdk.CameraAlignFree(self.buf)
            self.buf = mvsdk.CameraAlignMalloc(head.uBytes, 16)
            self.buf_sz = head.uBytes

        mvsdk.CameraImageProcess(self.hCamera, pRaw, self.buf, head)
        mvsdk.CameraReleaseImageBuffer(self.hCamera, pRaw)

        frame_data = (ctypes.c_ubyte * head.uBytes).from_address(self.buf)
        if self.mono:
            frame = np.frombuffer(frame_data, dtype=np.uint8).reshape((head.iHeight, head.iWidth))
        else:
            frame = np.frombuffer(frame_data, dtype=np.uint8).reshape((head.iHeight, head.iWidth, 3))
        return frame

    def close(self):
        if self.buf:
            mvsdk.CameraAlignFree(self.buf)
            self.buf = None
        if self.hCamera:
            mvsdk.CameraUnInit(self.hCamera)
            self.hCamera = None


# ------------- Calibration helpers -------------
def load_calibration(path=CALIB_FILE):
    if not os.path.exists(path):
        return None, []
    with open(path, "r") as f:
        data = json.load(f)
    return data.get("focal_length_px"), data.get("samples", [])

def save_calibration(focal_px, samples, path=CALIB_FILE):
    data = {
        "focal_length_px": focal_px,
        "samples": [{"radius_px": r, "distance_mm": d} for r, d in samples]
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"💾 Saved calibration to {path}")

def calc_focal_length_px(radius_px, distance_mm):
    return (radius_px * distance_mm) / GOLF_BALL_RADIUS_MM

def estimate_distance_mm(focal_px, radius_px):
    return (focal_px * GOLF_BALL_RADIUS_MM) / radius_px


# ------------- Drawing helpers -------------
def draw_dets(frame, dets):
    """Draw circles/boxes directly on the frame (no extra window)."""
    for (xc, yc, r) in dets:
        cv2.circle(frame, (int(xc), int(yc)), int(r), (0, 255, 0), 2)
        x1, y1 = int(xc - r), int(yc - r)
        x2, y2 = int(xc + r), int(yc + r)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 1)
        cv2.putText(frame, "ball", (x1, max(0, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)


def overlay_text(frame, mode, focal_px):
    txt1 = f"Mode: {mode}  | ENTER=Capture  m=ToggleMode  q=Quit"
    txt2 = f"Focal(px): {focal_px:.2f}" if focal_px else "Focal(px): N/A"
    cv2.putText(frame, txt1, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50,255,255), 1)
    cv2.putText(frame, txt2, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50,255,255), 1)


# ------------- Main interactive loop -------------
def main():
    focal_px, samples = load_calibration(CALIB_FILE)
    if focal_px:
        print(f"Loaded focal length: {focal_px:.2f} px ({len(samples)} samples)")

    cam = MVSCamera(ROI_W, ROI_H, ROI_X, ROI_Y, EXPOSURE_US)
    cam.open()
    print("Camera started.")

    mode = "CALIB" if focal_px is None else "ESTIMATE"
    print(f"Mode: {mode}")
    print("Keys: ENTER=capture | m=toggle mode | q=quit")

    cv2.namedWindow("Cam", cv2.WINDOW_NORMAL)

    ENTER_KEYS = {13, 10}  # CR/LF

    try:
        last_dist_default = 1000.0
        while True:
            frame = cam.grab()
            if frame.ndim == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

            overlay_text(frame, mode, focal_px if focal_px else 0.0)

            key = cv2.waitKey(1) & 0xFF

            if key in ENTER_KEYS:
                # Capture & process
                dets = yolo_detect(frame, conf=YOLO_CONF, imgsz=YOLO_IMGSZ, display=False)
                if not dets:
                    print("⚠ No ball detected on capture.")
                    continue

                dets.sort(key=lambda t: t[2], reverse=True)
                _, _, r_px = dets[0]
                draw_dets(frame, [dets[0]])  # show the chosen detection

                if mode == "CALIB":
                    print(f"\nDetected radius = {r_px:.2f}px")
                    dist_mm = ask_distance_mm(default_val=last_dist_default)
                    if dist_mm is None:
                        print("Calibration cancelled.")
                        continue
                    last_dist_default = dist_mm
                    samples.append((r_px, dist_mm))
                    focal_list = [calc_focal_length_px(r, d) for r, d in samples]
                    focal_px = float(np.mean(focal_list))
                    print(f"Samples: {len(samples)} | Current focal ~ {focal_px:.2f}px")
                    save_calibration(focal_px, samples)

                else:  # ESTIMATE
                    if focal_px is None:
                        print("No focal length. Switch to CALIB first.")
                        continue
                    dist = estimate_distance_mm(focal_px, r_px)
                    print(f"📐 Estimated Distance: {dist:.1f} mm  ({dist/10:.1f} cm)")

            cv2.imshow("Cam", frame)

            if key == ord('m'):
                mode = "ESTIMATE" if mode == "CALIB" else "CALIB"
                print(f"Switched mode → {mode}")

            elif key == ord('q'):
                break

    finally:
        cam.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
