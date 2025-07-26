#coding=utf-8
import cv2
import numpy as np
import camera.mvsdk as mvsdk
import time
import queue
import traceback
from image_processing.ballDetectionyolo import detect_golfballs  # Import YOLO detection function

# Global buffer variable (allocated in main thread, used by processing thread)
pFrameBuffer_global = 0
BALL_DIAM_MM            = 42.67
GOLF_BALL_RADIUS_MM = 21.335
THRESHOLD_APART_MM      = 80.0      # Minimum distance in mm for capture pairing
DESIRED_EXPOSURE_US     = 50.0
DESIRED_ANALOG_GAIN     = 1000.0
DESIRED_GAMMA           = 0.25
FPS_NOMINAL             = 1300.0    # Nominal frames per second for trajectory simulation
DEBUG                   = True
MAX_CAPTURE_FRAMES      = 100       # Maximum frames to capture in hitting mode
SETUP_DET_INTERVAL      = 0.5       # Interval for detection in setup mode
WAIT_TO_CAPTURE         = 1.5       # Time to hold still before entering hitting mode
MOVEMENT_THRESHOLD_MM   = 2.0 
# Add new global variables
FRAMES_TO_CAPTURE = 30  # Number of frames to capture
TARGET_FPS = 2000  # Target FPS for camera
recorded_frames = []
is_recording = False

def setup_camera_and_buffer():
	global pFrameBuffer_global
	try:
		DevList = mvsdk.CameraEnumerateDevice()
		nDev = len(DevList)
		if nDev < 1:
			print("No camera was found!")
			return None, None

		for i, DevInfo in enumerate(DevList):
			print("{}: {} {}".format(i, DevInfo.GetFriendlyName(), DevInfo.GetPortType()))
		i = 0 if nDev == 1 else int(input("Select camera: "))
		DevInfo = DevList[i]
		print(DevInfo)

		hCamera = 0
		try:
			hCamera = mvsdk.CameraInit(DevInfo, -1, -1)
		except mvsdk.CameraException as e:
			print("CameraInit Failed({}): {}".format(e.error_code, e.message))
			return None, None

		cap = mvsdk.CameraGetCapability(hCamera)
		monoCamera = (cap.sIspCapacity.bMonoSensor != 0)

		if monoCamera:
			mvsdk.CameraSetIspOutFormat(hCamera, mvsdk.CAMERA_MEDIA_TYPE_MONO8)
		else:
			mvsdk.CameraSetIspOutFormat(hCamera, mvsdk.CAMERA_MEDIA_TYPE_BGR8)

		# Set camera parameters for proper image capture
		mvsdk.CameraSetTriggerMode(hCamera, 0)
		mvsdk.CameraSetAeState(hCamera, 0)
		mvsdk.CameraSetExposureTime(hCamera, DESIRED_EXPOSURE_US)
		gmin, gmax, _ = mvsdk.CameraGetAnalogGainXRange(hCamera)
		mvsdk.CameraSetAnalogGainX(hCamera, max(gmin, min(DESIRED_ANALOG_GAIN, gmax)))
		gamma_max = cap.sGammaRange.iMax
		mvsdk.CameraSetGamma(hCamera, int(DESIRED_GAMMA * gamma_max))

		# --- Set custom image resolution as requested ---
		image_resolution = mvsdk.tSdkImageResolution()
		image_resolution.iIndex = 0  # Custom index, or use an unused one
		image_resolution.iWidth = 640
		image_resolution.iHeight = 300
		image_resolution.iWidthFOV = 640
		image_resolution.iHeightFOV = 300
		image_resolution.iOffsetX = 0
		image_resolution.iOffsetY = 90
		mvsdk.CameraSetImageResolution(hCamera, image_resolution)
		print(mvsdk.CameraGetImageResolution(hCamera))
		
		# ------------------------------------------------

		mvsdk.CameraPlay(hCamera)

		# Get actual resolution from camera
		actual_resolution = mvsdk.CameraGetImageResolution(hCamera)
		actual_width = actual_resolution.iWidth
		actual_height = actual_resolution.iHeight
		print(f"Actual camera resolution: {actual_width}x{actual_height}")

		# Calculate buffer size using actual resolution
		FrameBufferSize = actual_width * actual_height * (1 if monoCamera else 3)
		pFrameBuffer_global = mvsdk.CameraAlignMalloc(FrameBufferSize, 16)

		return hCamera, monoCamera
	except Exception as e:
		print("Exception in setup_camera_and_buffer:", e)
		traceback.print_exc()
		return None, None

def release_camera_and_buffer(hCamera):
	global pFrameBuffer_global
	try:
		if hCamera:
			mvsdk.CameraUnInit(hCamera)
		if pFrameBuffer_global:
			mvsdk.CameraAlignFree(pFrameBuffer_global)
			pFrameBuffer_global = 0 # Reset global buffer
	except Exception as e:
		print("Exception in release_camera_and_buffer:", e)
		traceback.print_exc()

def acquire_frames(hCamera, frame_queue, stop_event):
	print("Acquisition thread started.")
	try:
		while not stop_event.is_set():
			try:
				pRawData, FrameHead = mvsdk.CameraGetImageBuffer(hCamera, 1250)
				# Put raw data and header into the queue
				frame_queue.put((pRawData, FrameHead))
			except mvsdk.CameraException as e:
				if e.error_code != mvsdk.CAMERA_STATUS_TIME_OUT:
					print(f"Acquisition thread camera error: {e.error_code} - {e.message}")
					# Continue loop on timeout or other errors
					pass
	except Exception as e:
		print("Exception in acquire_frames:", e)
		traceback.print_exc()
	print("Acquisition thread stopped.")



# —————————————————————————————————————————————————————————————
# globals (must be defined once elsewhere in your module):
#   pFrameBuffer_global  # mvsdk image buffer pointer
#   recorded_frames      # list to hold captured gray frames
#   is_recording         # bool flag
#   FRAMES_TO_CAPTURE    # int, how many frames to record
#   GOLF_BALL_RADIUS_MM  = 21.335  # mm
#   FRAME_APART_MM       = 80.0    # target separation in mm
# —————————————————————————————————————————————————————————————
import math
from camera.focalPointCalibration import load_calibration
def process_frames(hCamera,
                   monoCamera,
                   detected_circle,
                   original_cropped_roi,
                   frame_queue,
                   stop_event,
                   interactive=True):
    """
    Pull frames, watch ROI for motion, record a burst, then:

     • if interactive: step through newest→oldest with YOLO overlays
     • always: auto‑select the frame whose real‑world ball displacement
               is closest to FRAME_APART_MM

    """
    print("Processing thread started.")
    global recorded_frames, is_recording

    try:
        # 1) Unpack initial detection
        first_x_px, first_y_px, first_r_px = detected_circle

        # 2) Build square ROI around that point for motion detection
        h_roi, w_roi = original_cropped_roi.shape[:2]
        half_crop = 100 // 2
        x1 = max(0, first_x_px - half_crop)
        y1 = max(0, first_y_px - half_crop)
        x2 = min(x1 + w_roi, 640)
        y2 = min(y1 + h_roi, 300)

        movement_threshold = 12
        print("Monitoring movement in the detected area…")

        frame_count = 0
        start_time = time.time()

        # ——— Phase A: wait for motion, then record FRAMES_TO_CAPTURE frames ———
        while not stop_event.is_set() or not frame_queue.empty():
            try:
                pRaw, head = frame_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            # convert raw buffer → numpy frame
            mvsdk.CameraImageProcess(hCamera, pRaw, pFrameBuffer_global, head)
            mvsdk.CameraReleaseImageBuffer(hCamera, pRaw)
            arr = (mvsdk.c_ubyte * head.uBytes).from_address(pFrameBuffer_global)
            frame = np.frombuffer(arr, dtype=np.uint8)
            try:
                frame = frame.reshape((
                    head.iHeight,
                    head.iWidth,
                    1 if head.uiMediaType == mvsdk.CAMERA_MEDIA_TYPE_MONO8 else 3
                ))
            except Exception:
                continue

            gray = (cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    if (not monoCamera and frame.ndim == 3) else frame)

            # crop + resize to match original_cropped_roi
            x1c = max(0, min(first_x_px - half_crop, gray.shape[1]-1))
            y1c = max(0, min(first_y_px - half_crop, gray.shape[0]-1))
            x2c = max(x1c+1, min(first_x_px + half_crop, gray.shape[1]))
            y2c = max(y1c+1, min(first_y_px + half_crop, gray.shape[0]))
            crop = gray[y1c:y2c, x1c:x2c]
            if crop.shape[:2] != original_cropped_roi.shape[:2] and crop.size:
                try:
                    crop = cv2.resize(crop, (w_roi, h_roi))
                except Exception:
                    continue

            # compare & trigger
            if original_cropped_roi.size and crop.size:
                diff = cv2.absdiff(original_cropped_roi, crop)
                md = np.mean(diff)
                frame_count += 1
                fps = frame_count / (time.time() - start_time)
                print(f"Comparison FPS: {fps:.1f}, Mean Difference: {md:.2f}")

                if md > movement_threshold and not is_recording:
                    print("BALL MOVED → starting recording")
                    is_recording = True
                    recorded_frames = []

                if is_recording:
                    recorded_frames.append(gray.copy())
                    print(f"Recording FPS: {fps:.1f}")
                    if len(recorded_frames) >= FRAMES_TO_CAPTURE:
                        print(f"Recording complete: {len(recorded_frames)} frames")
                        is_recording = False
                        stop_event.set()
                        break

        # ——— Reference real‑world coords from the *first* frame ———
        focal_px, _ = load_calibration()
        if focal_px is None:
            raise RuntimeError("Camera must be calibrated first")

        # estimate first frame depth:
        z0_mm = (GOLF_BALL_RADIUS_MM * focal_px) / first_r_px

        # ——— 1) Optional interactive pass ———
        if interactive:
            print("\n--- Interactive inspection (newest→oldest) ---")
            for idx in range(len(recorded_frames)-1, -1, -1):
                frm = recorded_frames[idx]
                disp = frm.copy()

                # prep BGR for YOLO
                if frm.ndim == 2:
                    bgr = cv2.cvtColor(frm, cv2.COLOR_GRAY2BGR)
                else:
                    bgr = frm
                if bgr.shape[2] != 3:
                    bgr = cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)

                try:
                    circles = detect_golfballs(bgr, conf=0.7, imgsz=640, display=False)
                except Exception as e:
                    print("YOLO error:", e)
                    circles = []

                if circles:
                    print(f"Frame {idx}: {len(circles)} detections")
                    # annotate all
                    for (cx, cy, rr) in circles:
                        cv2.circle(disp, (int(cx),int(cy)), int(rr), (0,255,0), 2)
                    # also draw reference center
                    cv2.circle(disp, (first_x_px, first_y_px), 3, (0,0,255), -1)
                else:
                    print(f"Frame {idx}: no detection")

                cv2.imshow(f"Orig {idx}", frm)
                cv2.imshow(f"Detect {idx}", disp)
                if cv2.waitKey(0) == ord('q'):
                    break
                cv2.destroyWindow(f"Orig {idx}")
                cv2.destroyWindow(f"Detect {idx}")
            print("Interactive inspection done.")

        # ——— 2) Auto best‑match pass: find frame ≈ FRAME_APART_MM apart ———
        print("\n=== Auto best‑match for separation ≈", THRESHOLD_APART_MM, "mm ===")
        best_idx, best_score = None, float('inf')
        best_info = None

        for idx, frm in enumerate(recorded_frames):
            # prep for YOLO
            if frm.ndim == 2:
                bgr = cv2.cvtColor(frm, cv2.COLOR_GRAY2BGR)
            else:
                bgr = frm
            if bgr.shape[2] != 3:
                bgr = cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)

            try:
                circles = detect_golfballs(bgr, conf=0.7, imgsz=640, display=False)
            except Exception as e:
                print(f"YOLO error in frame {idx}:", e)
                continue

            for (cx, cy, rr) in circles:
                # pixel‐space separation:
                pd = math.hypot(cx - first_x_px, cy - first_y_px)
                # convert px → mm (approx horizontal only)
                sep_mm = pd * (z0_mm / focal_px)
                score = abs(sep_mm - THRESHOLD_APART_MM)
                print(f"  Frame {idx}: {sep_mm:.1f} mm (score {score:.1f})")
                if score < best_score:
                    best_score = score
                    best_idx = idx
                    best_info = (int(cx), int(cy), int(rr), sep_mm)

        if best_idx is not None:
            cx, cy, rr, sep_mm = best_info
            print(f"\nBest match: frame {best_idx} → {sep_mm:.1f} mm apart")
            disp = recorded_frames[best_idx].copy()
            cv2.circle(disp, (cx, cy), rr, (0,255,0), 2)
            cv2.line(disp, (cx, cy), (first_x_px, first_y_px), (255,0,0), 2)
            cv2.putText(disp, f"{sep_mm:.1f} mm", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            cv2.imshow(f"Best Match {best_idx}", disp)
            cv2.waitKey(0)
            cv2.destroyWindow(f"Best Match {best_idx}")
        else:
            print("No valid detection found at the desired separation.")

        print("Processing complete.")
        cv2.destroyAllWindows()

    except Exception as e:
        print("Exception in process_frames:", e)
        traceback.print_exc()
        stop_event.set()
    finally:
        try: cv2.destroyAllWindows()
        except: pass
        print("Processing thread stopped.")
