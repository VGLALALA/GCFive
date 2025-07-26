#coding=utf-8
import cv2
import numpy as np
import mvsdk
import platform
import time
import threading
import queue
from datetime import datetime
import sys
import traceback
from ballDetectionyolo import detect_golfballs  # Import YOLO detection function

# Global buffer variable (allocated in main thread, used by processing thread)
pFrameBuffer_global = 0
BALL_DIAM_MM            = 42.67
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

def process_frames(hCamera, monoCamera, detected_circle, original_cropped_roi, frame_queue, stop_event):
	print("Processing thread started.")
	global recorded_frames, is_recording
	first_frame_ball_pos = None
	try:
		x, y, r = detected_circle

		# Calculate crop coordinates using original ROI dimensions
		h_roi, w_roi = original_cropped_roi.shape[:2]
		half_crop = 100 // 2
		x1 = max(0, x - half_crop)
		y1 = max(0, y - half_crop)
		x2 = min(x1 + w_roi, 640)
		y2 = min(y1 + h_roi, 300)

		movement_threshold = 12

		print("Monitoring movement in the detected area.")

		frame_count = 0
		start_time = time.time()

		while not stop_event.is_set() or not frame_queue.empty():
			try:
				frame_data_tuple = frame_queue.get(timeout=0.1)
				if frame_data_tuple is None:
					break

				pRawData, FrameHead = frame_data_tuple
				mvsdk.CameraImageProcess(hCamera, pRawData, pFrameBuffer_global, FrameHead)
				mvsdk.CameraReleaseImageBuffer(hCamera, pRawData)

				if platform.system() == "Windows":
					mvsdk.CameraFlipFrameBuffer(pFrameBuffer_global, FrameHead, 1)

				frame = np.frombuffer((mvsdk.c_ubyte * FrameHead.uBytes).from_address(pFrameBuffer_global), dtype=np.uint8)
				try:
					frame = frame.reshape((FrameHead.iHeight, FrameHead.iWidth, 1 if FrameHead.uiMediaType == mvsdk.CAMERA_MEDIA_TYPE_MONO8 else 3))
				except Exception as e:
					print("Error reshaping frame:", e)
					traceback.print_exc()
					continue

				if not monoCamera and frame.ndim == 3:
					gray_current = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
				else:
					gray_current = frame

				x1_curr = max(0, min(x - half_crop, frame.shape[1] - 1))
				y1_curr = max(0, min(y - half_crop, frame.shape[0] - 1))
				x2_curr = max(x1_curr + 1, min(x + half_crop, frame.shape[1]))
				y2_curr = max(y1_curr + 1, min(y + half_crop, frame.shape[0]))

				current_cropped = gray_current[y1_curr:y2_curr, x1_curr:x2_curr]

				if current_cropped.shape[:2] != original_cropped_roi.shape[:2] and current_cropped.size > 0:
					try:
						current_cropped_resized = cv2.resize(current_cropped, (w_roi, h_roi))
					except Exception as e:
						print("Error resizing cropped region:", e)
						traceback.print_exc()
						continue
				else:
					current_cropped_resized = current_cropped

				if original_cropped_roi.size > 0 and current_cropped_resized.size > 0:
					try:
						difference = cv2.absdiff(original_cropped_roi, current_cropped_resized)
						mean_difference = np.mean(difference)
					except Exception as e:
						print("Error computing difference:", e)
						traceback.print_exc()
						continue

					frame_count += 1
					current_time = time.time()
					elapsed_time = current_time - start_time
					fps = frame_count / elapsed_time if elapsed_time > 0 else 0
					print(f"Comparison FPS: {fps:.1f}, Mean Difference: {mean_difference:.2f}")

					if mean_difference > movement_threshold and not is_recording:
						print("BALL MOVED - Starting frame recording")
						is_recording = True
						recorded_frames = []
						try:
							cv2.circle(current_cropped_resized, (x - x1_curr, y - y1_curr), r, (0, 0, 255), 2)
						except Exception as e:
							print("Error drawing circle:", e)
							traceback.print_exc()

					# Record frames if we're in recording mode
					if is_recording:
						recorded_frames.append(gray_current.copy())
						
						# Detect ball position in first frame for reference
						
						# Print FPS instead of "Captured frame"
						current_time = time.time()
						elapsed_time = current_time - start_time
						fps = frame_count / elapsed_time if elapsed_time > 0 else 0
						print(f"Recording FPS: {fps:.1f}")

						if len(recorded_frames) >= FRAMES_TO_CAPTURE:
							is_recording = False
							print(f"Recording complete. Captured {len(recorded_frames)} frames")

							# Stop the acquisition thread before processing frames
							stop_event.set()

							# Use first frame ball position for distance calculation
							if first_frame_ball_pos is not None:
								first_x, first_y, first_r = first_frame_ball_pos
								print(f"First frame ball position: ({first_x}, {first_y})")
							else:
								# Fallback to original detection position
								first_x, first_y, first_r = detected_circle
								print(f"Using original ball position: ({first_x}, {first_y})")
							
							# Process frames one at a time
							for i in range(len(recorded_frames) - 1, -1, -1):
								frame = recorded_frames[i]
								print(f"\nProcessing frame {i}")

								display_frame = frame.copy()

								try:
									# Convert grayscale frame to BGR for YOLO detection
									if len(frame.shape) == 2:
										frame_bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
									else:
										frame_bgr = frame
									
									# Ensure frame is 3-channel BGR
									if frame_bgr.shape[2] != 3:
										print(f"Warning: Frame has {frame_bgr.shape[2]} channels, converting to BGR")
										frame_bgr = cv2.cvtColor(frame_bgr, cv2.COLOR_GRAY2BGR)
									
									circles = detect_golfballs(frame_bgr, conf=0.7, imgsz=640, display=False)
								except Exception as e:
									print("Error running YOLO detection:", e)
									traceback.print_exc()
									circles = None

								if circles is not None and len(circles) > 0:
									print(f"Found {len(circles)} ball(s) in frame {i}")
									
									# Find the closest ball to original position
									closest_circle = None
									wanted_dist = 100
									min_distance = float('inf')
									
									for circle in circles:
										try:
											x, y, r = map(int, circle)
											
											# Calculate distance to first frame ball position
											distance = np.sqrt((x - first_x)**2 + (y - first_y)**2)
											
											if abs(wanted_dist-distance) < min_distance:
												min_distance = distance
												closest_circle = (x, y, r, distance)
												
											print(f"  Ball at ({x}, {y}) - Distance to original: {distance:.1f} pixels")
											
										except Exception as e:
											print("Error processing circle:", e)
											traceback.print_exc()
									
									# Draw the closest detected ball
									if closest_circle:
										x, y, r, distance = closest_circle
										cv2.circle(display_frame, (x, y), r, (0, 255, 0), 2)  # Green circle
										cv2.circle(display_frame, (x, y), 2, (0, 0, 255), 3)  # Red center
										
										# Draw line to first frame position
										cv2.line(display_frame, (x, y), (first_x, first_y), (255, 0, 0), 2)
										
										# Add distance text
										cv2.putText(display_frame, f"Distance: {distance:.1f}px", 
												   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
										cv2.putText(display_frame, f"Detected: ({x}, {y})", 
												   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
										cv2.putText(display_frame, f"First Frame: ({first_x}, {first_y})", 
												   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
										
										print(f"  Closest ball: ({x}, {y}) - Distance: {distance:.1f} pixels")
									else:
										print("  No valid balls found")
								else:
									print("  No balls detected in this frame")

								try:
									cv2.imshow(f"Original Frame {i}", frame)
									if circles is not None and len(circles) > 0:
										cv2.imshow(f"Frame {i} with YOLO detection", display_frame)
								except Exception as e:
									print("Error displaying frame:", e)
									traceback.print_exc()

								print("Press any key to process next frame, or 'q' to quit")
								key = cv2.waitKey(0)
								if key == ord('q'):
									break

								try:
									cv2.destroyWindow(f"Original Frame {i}")
									if circles is not None and len(circles) > 0:
										cv2.destroyWindow(f"Frame {i} with YOLO detection")
								except Exception as e:
									print("Error destroying window:", e)
									traceback.print_exc()
							# Find frame with ball closest to first frame position
							print("\n=== Finding frame with ball closest to first frame position ===")
							closest_idx = None
							min_distance = float('inf')
							closest_circle = None

							for i, frame in enumerate(recorded_frames):
								try:
									# Convert grayscale frame to BGR for YOLO detection
									if len(frame.shape) == 2:
										frame_bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
									else:
										frame_bgr = frame
									
									# Ensure frame is 3-channel BGR
									if frame_bgr.shape[2] != 3:
										print(f"Warning: Frame has {frame_bgr.shape[2]} channels, converting to BGR")
										frame_bgr = cv2.cvtColor(frame_bgr, cv2.COLOR_GRAY2BGR)
									
									circles = detect_golfballs(frame_bgr, conf=0.7, imgsz=640, display=False)
								except Exception as e:
									print(f"Error running YOLO detection in frame {i}:", e)
									continue
									
								if circles is not None and len(circles) > 0:
									for circle in circles:
										try:
											x, y, r = map(int, circle)
											distance = np.sqrt((x - first_x)**2 + (y - first_y)**2)
											
											if distance < min_distance:
												min_distance = distance
												closest_idx = i
												closest_circle = (x, y, r, distance)
												
											print(f"  Frame {i}: Ball at ({x}, {y}) - Distance: {distance:.1f}px")
											
										except Exception as e:
											print(f"Error processing circle in frame {i}:", e)
											continue

							if closest_idx is not None and closest_circle is not None:
								try:
									x, y, r, distance = closest_circle
									print(f"\nBest match: Frame {closest_idx} with ball at ({x}, {y}) - Distance: {distance:.1f}px")
									
									# Show first frame
									cv2.imshow(f"First Frame {closest_idx}", recorded_frames[closest_idx])
									
									# Show frame with detection and distance info
									disp = recorded_frames[closest_idx].copy()
									cv2.circle(disp, (x, y), r, (0, 255, 0), 2)  # Green circle
									cv2.circle(disp, (x, y), 2, (0, 0, 255), 3)  # Red center
									
									# Draw line to first frame position
									cv2.line(disp, (x, y), (first_x, first_y), (255, 0, 0), 2)
									
									# Add distance text
									cv2.putText(disp, f"Distance: {distance:.1f}px", 
											   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
									cv2.putText(disp, f"Detected: ({x}, {y})", 
											   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
									cv2.putText(disp, f"First Frame: ({first_x}, {first_y})", 
											   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
									cv2.putText(disp, f"Frame: {closest_idx}", 
											   (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
									
									cv2.imshow(f"Best Match - Frame {closest_idx}", disp)
									print("Press any key to continue...")
									cv2.waitKey(0)
									cv2.destroyWindow(f"Original Frame {closest_idx}")
									cv2.destroyWindow(f"Best Match - Frame {closest_idx}")
								except Exception as e:
									print("Error displaying best match frame:", e)
									traceback.print_exc()
							else:
								print("No balls detected in any frame.")
							print("Frame processing complete")
							try:
								cv2.destroyAllWindows()
							except Exception as e:
								print("Error destroying all windows:", e)
								traceback.print_exc()
							return

				else:
					print("Warning: Invalid ROI detected during monitoring.")

			except queue.Empty:
				pass
			except Exception as e:
				print(f"Processing thread error: {e}")
				traceback.print_exc()
				stop_event.set()
				break

	except Exception as e:
		print("Exception in process_frames:", e)
		traceback.print_exc()
	finally:
		try:
			cv2.destroyAllWindows()
		except Exception as e:
			print("Error destroying all windows in finally:", e)
			traceback.print_exc()
	print("Processing thread stopped.")

# Note: main() and standalone execution block are removed
# This file is now intended to be imported and its functions called by ball_detection.py

# --------- TESTING CODE TO OPEN AND CAPTURE WITH THIS CAMERA ---------
def test_camera_capture(num_frames=100):
	"""
	Test function to open the camera, capture a number of frames, and display them.
	"""
	hCamera, monoCamera = setup_camera_and_buffer()
	if hCamera is None:
		print("Failed to initialize camera.")
		return

	print("Starting camera capture test...")
	captured = 0
	start_time = time.time()
	try:
		while captured < num_frames:
			try:
				pRawData, FrameHead = mvsdk.CameraGetImageBuffer(hCamera, 1000)
			except mvsdk.CameraException as e:
				if e.error_code == mvsdk.CAMERA_STATUS_TIME_OUT:
					print("Timeout waiting for frame...")
					continue
				else:
					print(f"Camera error: {e.error_code} - {e.message}")
					break

			try:
				mvsdk.CameraImageProcess(hCamera, pRawData, pFrameBuffer_global, FrameHead)
				mvsdk.CameraReleaseImageBuffer(hCamera, pRawData)
			except Exception as e:
				print("Error processing/releasing image buffer:", e)
				traceback.print_exc()
				break

			if platform.system() == "Windows":
				try:
					mvsdk.CameraFlipFrameBuffer(pFrameBuffer_global, FrameHead, 1)
				except Exception as e:
					print("Error flipping frame buffer:", e)
					traceback.print_exc()

			try:
				frame = np.frombuffer(
					(mvsdk.c_ubyte * FrameHead.uBytes).from_address(pFrameBuffer_global),
					dtype=np.uint8
				)
				if monoCamera:
					frame = frame.reshape((FrameHead.iHeight, FrameHead.iWidth))
				else:
					frame = frame.reshape((FrameHead.iHeight, FrameHead.iWidth, 3))
			except Exception as e:
				print("Error reshaping frame in test_camera_capture:", e)
				traceback.print_exc()
				break

			try:
				cv2.imshow("Camera Test Frame", frame)
				key = cv2.waitKey(1)
				if key == ord('q'):
					print("User requested exit.")
					break
			except Exception as e:
				print("Error displaying test frame:", e)
				traceback.print_exc()
				break

			captured += 1
			# Print FPS instead of "Captured frame"
			current_time = time.time()
			elapsed_time = current_time - start_time
			fps = captured / elapsed_time if elapsed_time > 0 else 0
			print(f"Test capture FPS: {fps:.1f}")

		print("Camera capture test complete.")
	except Exception as e:
		print("Exception in test_camera_capture:", e)
		traceback.print_exc()
	finally:
		try:
			release_camera_and_buffer(hCamera)
		except Exception as e:
			print("Error releasing camera and buffer in finally:", e)
			traceback.print_exc()
		try:
			cv2.destroyAllWindows()
		except Exception as e:
			print("Error destroying all windows in finally:", e)
			traceback.print_exc()

if __name__ == "__main__":
	try:
		test_camera_capture(num_frames=200)
	except Exception as e:
		print("Unhandled exception in __main__:", e)
		traceback.print_exc()
