#!/usr/bin/env python3
# focalPointCalibration.py

import os
import json
import ctypes
import numpy as np
import cv2
import time

# -------- YOLO / detection --------
YOLO_CONF  = 0.25
YOLO_IMGSZ = 640
CLASS_ID   = 0        # golf ball class
MODEL_PATH = "data/model/golfballv4.pt"  # only used if you uncomment fallback

# Try to use your existing detector
from image_processing.ballDetection import detect_golfballs as yolo_detect
HAS_EXTERNAL_DETECTOR = True


# -------- Calibration constants --------
CALIB_FILE = "calibration.json"
GOLF_BALL_DIAMETER_MM = 42.67
GOLF_BALL_RADIUS_MM   = GOLF_BALL_DIAMETER_MM / 2.0  # 21.335 mm

# -------- Camera (mvsdk) config --------
ROI_W, ROI_H = 640, 300
ROI_X, ROI_Y = 0, 100
EXPOSURE_US  = 50  # 0.5 ms

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
    """Prompt for true distance (mm) via Tkinter, fallback to terminal."""
    if HAS_TK:
        root = tk.Tk()
        root.withdraw()
        while True:
            val = simpledialog.askstring("Calibration", "Enter TRUE distance (mm):",
                                         initialvalue=str(default_val), parent=root)
            if val is None:
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
    for (xc, yc, r) in dets:
        cv2.circle(frame, (int(xc), int(yc)), int(r), (0, 255, 0), 2)
        x1, y1 = int(xc - r), int(yc - r)
        x2, y2 = int(xc + r), int(yc + r)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 1)
        cv2.putText(frame, "ball", (x1, max(0, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

def overlay_text(frame, mode, focal_px, dist_mm_live=None):
    txt1 = f"Mode: {mode}  | ENTER=Capture  m=ToggleMode  q=Quit"
    txt2 = f"Focal(px): {focal_px:.2f}" if focal_px else "Focal(px): N/A"
    cv2.putText(frame, txt1, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50,255,255), 1)
    cv2.putText(frame, txt2, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50,255,255), 1)
    if dist_mm_live is not None:
        txt3 = f"Dist: {dist_mm_live:.1f} mm ({dist_mm_live/10:.1f} cm)"
        cv2.putText(frame, txt3, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,0), 1)

# --------- Simple EWMA smoother (optional) ---------
class EWMA:
    def __init__(self, alpha=0.25):
        self.alpha = alpha
        self.val = None
    def update(self, x):
        self.val = x if self.val is None else self.alpha * x + (1 - self.alpha) * self.val
        return self.val

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

    ENTER_KEYS = {13, 10}
    DETECT_EVERY = 3          # YOLO every N frames
    frame_idx = 0
    dist_filter = EWMA(alpha=0.25)
    live_dist = None

    try:
        last_dist_default = 1000.0
        while True:
            frame = cam.grab()
            if frame.ndim == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            frame_idx += 1

            # ---- LIVE ESTIMATE ----
            if mode == "ESTIMATE" and focal_px is not None:
                if frame_idx % DETECT_EVERY == 0:
                    dets = yolo_detect(frame, conf=YOLO_CONF, imgsz=YOLO_IMGSZ, display=False)
                    if dets:
                        dets.sort(key=lambda t: t[2], reverse=True)
                        xc, yc, r_px = dets[0]
                        draw_dets(frame, [dets[0]])
                        d_mm = estimate_distance_mm(focal_px, r_px)
                        live_dist = dist_filter.update(d_mm)
                    else:
                        live_dist = None
            else:
                live_dist = None

            overlay_text(frame, mode, focal_px if focal_px else 0.0, live_dist)

            key = cv2.waitKey(1) & 0xFF

            if key in ENTER_KEYS:
                dets = yolo_detect(frame, conf=YOLO_CONF, imgsz=YOLO_IMGSZ, display=False)
                if not dets:
                    print("⚠ No ball detected on capture.")
                    continue

                dets.sort(key=lambda t: t[2], reverse=True)
                _, _, r_px = dets[0]
                draw_dets(frame, [dets[0]])

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
                else:
                    dist = estimate_distance_mm(focal_px, r_px)
                    print(f"📐 Estimated Distance (snapshot): {dist:.1f} mm  ({dist/10:.1f} cm)")

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
