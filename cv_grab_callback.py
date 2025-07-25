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

# Global buffer variable (allocated in main thread, used by processing thread)
pFrameBuffer_global = 0

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

		# Set camera parameters for high-speed capture
		mvsdk.CameraSetTriggerMode(hCamera, 0)  # Continuous mode
		mvsdk.CameraSetFrameSpeed(hCamera, 2)   # Highest frame speed
		mvsdk.CameraSetAeState(hCamera, 0)      # Manual exposure
		mvsdk.CameraSetExposureTime(hCamera, 1)  # 1ms exposure for 800 FPS
		mvsdk.CameraSetGain(hCamera, 100, 100, 100)  # Set RGB gains to 100
		mvsdk.CameraSetAntiFlick(hCamera, 0)
		# Set camera contrast to a higher value (example: 180)
		mvsdk.CameraSetContrast(hCamera, 100 )

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

		# Calculate buffer size using fixed resolution (640x300)
		FrameBufferSize = 640 * 480 * (1 if monoCamera else 3)
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

							# Process frames one at a time
							for i in range(len(recorded_frames) - 1, -1, -1):
								frame = recorded_frames[i]
								print(f"\nProcessing frame {i}")

								display_frame = frame.copy()

								try:
									circles = cv2.HoughCircles(
										frame,
										cv2.HOUGH_GRADIENT,
										dp=1,
										minDist=50,
										param1=50,
										param2=30,
										minRadius=20,
										maxRadius=100
									)
								except Exception as e:
									print("Error running HoughCircles:", e)
									traceback.print_exc()
									circles = None

								if circles is not None:
									print(f"Found circle in frame {i}")
									for circle in circles[0]:
										try:
											x, y, r = map(int, circle)
											cv2.circle(display_frame, (x, y), r, (0, 255, 0), 2)
											cv2.circle(display_frame, (x, y), 2, (0, 0, 255), 3)
										except Exception as e:
											print("Error drawing detected circle:", e)
											traceback.print_exc()

								try:
									cv2.imshow(f"Original Frame {i}", frame)
									if circles is not None:
										cv2.imshow(f"Frame {i} with circle", display_frame)
								except Exception as e:
									print("Error displaying frame:", e)
									traceback.print_exc()

								print("Press any key to process next frame, or 'q' to quit")
								key = cv2.waitKey(0)
								if key == ord('q'):
									break

								try:
									cv2.destroyWindow(f"Original Frame {i}")
									if circles is not None:
										cv2.destroyWindow(f"Frame {i} with circle")
								except Exception as e:
									print("Error destroying window:", e)
									traceback.print_exc()
							# INSERT_YOUR_CODE
							# Find, for each frame, the detected circle whose x is closest to 50, and display that frame and the original
							closest_idx = None
							closest_diff = None
							closest_circle = None

							for i, frame in enumerate(recorded_frames):
								try:
									circles = cv2.HoughCircles(
										frame,
										cv2.HOUGH_GRADIENT,
										dp=1,
										minDist=50,
										param1=50,
										param2=30,
										minRadius=20,
										maxRadius=100
									)
								except Exception as e:
									print("Error running HoughCircles in closest search:", e)
									traceback.print_exc()
									circles = None
								if circles is not None:
									for circle in circles[0]:
										try:
											x, y, r = map(int, circle)
											diff = abs(x - 100)
											if closest_diff is None or diff < closest_diff:
												closest_diff = diff
												closest_idx = i
												closest_circle = (x, y, r)
										except Exception as e:
											print("Error processing circle in closest search:", e)
											traceback.print_exc()

							if closest_idx is not None and closest_circle is not None:
								try:
									cv2.imshow(f"Original Frame (closest x to 50) idx={closest_idx}", recorded_frames[closest_idx])
									disp = recorded_frames[closest_idx].copy()
									x, y, r = closest_circle
									cv2.circle(disp, (x, y), r, (0, 255, 0), 2)
									cv2.circle(disp, (x, y), 2, (0, 0, 255), 3)
									cv2.imshow(f"Frame with circle x closest to 50 (x={x})", disp)
									print(f"Displayed frame {closest_idx} with circle x={x} closest to 50")
									cv2.waitKey(0)
									cv2.destroyWindow(f"Original Frame (closest x to 50) idx={closest_idx}")
									cv2.destroyWindow(f"Frame with circle x closest to 50 (x={x})")
								except Exception as e:
									print("Error displaying/destroying closest frame:", e)
									traceback.print_exc()
							else:
								print("No circles detected in any frame for x closest to 50.")
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
