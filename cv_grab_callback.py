#coding=utf-8
import cv2
import numpy as np
import mvsdk
import platform
import time
import threading
import queue
from datetime import datetime

# Global buffer variable (allocated in main thread, used by processing thread)
pFrameBuffer_global = 0

def setup_camera_and_buffer():
	global pFrameBuffer_global
	# This function will handle camera init and buffer allocation
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

	# Set camera parameters
	mvsdk.CameraSetTriggerMode(hCamera, 0)
	mvsdk.CameraSetFrameSpeed(hCamera, 2)
	mvsdk.CameraSetAeState(hCamera, 0)
	mvsdk.CameraSetExposureTime(hCamera, 1250)  # Set exposure to 5ms
	mvsdk.CameraSetGain(hCamera, 100, 100, 100)  # Set RGB gains to 100
	mvsdk.CameraSetAntiFlick(hCamera, 0)

	mvsdk.CameraPlay(hCamera)

	# Calculate buffer size using fixed resolution (640x480)
	FrameBufferSize = 640 * 480 * (1 if monoCamera else 3)
	pFrameBuffer_global = mvsdk.CameraAlignMalloc(FrameBufferSize, 16)

	return hCamera, monoCamera

def release_camera_and_buffer(hCamera):
	global pFrameBuffer_global
	if hCamera:
		mvsdk.CameraUnInit(hCamera)
	if pFrameBuffer_global:
		mvsdk.CameraAlignFree(pFrameBuffer_global)
		pFrameBuffer_global = 0 # Reset global buffer

# Thread function for acquiring frames
def acquire_frames(hCamera, frame_queue, stop_event):
	print("Acquisition thread started.")
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
		print("Acquisition thread stopped.")

# Thread function for processing frames
def process_frames(hCamera, monoCamera, detected_circle, original_cropped_roi, frame_queue, stop_event):
	print("Processing thread started.")

	x, y, r = detected_circle
	
	# Calculate crop coordinates using original ROI dimensions
	h_roi, w_roi = original_cropped_roi.shape[:2]
	half_crop = 100 // 2 # Use the fixed crop size as in ball_detection
	x1 = max(0, x - half_crop)
	y1 = max(0, y - half_crop)
	x2 = min(x1 + w_roi, 640)  # Ensure crop stays within original ROI dimensions and frame bounds
	y2 = min(y1 + h_roi, 480)  # Ensure crop stays within original ROI dimensions and frame bounds

	movement_threshold = 12

	print("Monitoring movement in the detected area.")

	# Display the original cropped ROI
	# cv2.imshow("Original Cropped ROI", original_cropped_roi)

	# Initialize variables for FPS calculation
	frame_count = 0
	start_time = time.time()

	while not stop_event.is_set() or not frame_queue.empty():
		try:
			# Get frame data from the queue with a timeout
			# Use a small timeout so the thread can check the stop_event
			frame_data_tuple = frame_queue.get(timeout=0.1)
			if frame_data_tuple is None: # Check for sentinel value
				break
			
			pRawData, FrameHead = frame_data_tuple

			# Process the raw data into the global frame buffer
			mvsdk.CameraImageProcess(hCamera, pRawData, pFrameBuffer_global, FrameHead)

			# Release the raw data buffer back to the SDK (do this after processing)
			mvsdk.CameraReleaseImageBuffer(hCamera, pRawData)

			if platform.system() == "Windows":
				mvsdk.CameraFlipFrameBuffer(pFrameBuffer_global, FrameHead, 1)

			# Convert the global frame buffer to OpenCV format and grayscale if needed
			frame = np.frombuffer((mvsdk.c_ubyte * FrameHead.uBytes).from_address(pFrameBuffer_global), dtype=np.uint8)
			frame = frame.reshape((FrameHead.iHeight, FrameHead.iWidth, 1 if FrameHead.uiMediaType == mvsdk.CAMERA_MEDIA_TYPE_MONO8 else 3))
			
			if not monoCamera and frame.ndim == 3:
				gray_current = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
			else:
				gray_current = frame

			# Ensure crop coordinates are within frame bounds and apply crop
			# Recalculate crop coordinates based on the current frame's center
			# Note: Assuming ball center doesn't move significantly from initial detection crop region
			# If ball moves out of the initial crop region, this logic would need adjustment
			x1_curr = max(0, min(x - half_crop, frame.shape[1] - 1))
			y1_curr = max(0, min(y - half_crop, frame.shape[0] - 1))
			x2_curr = max(x1_curr + 1, min(x + half_crop, frame.shape[1]))
			y2_curr = max(y1_curr + 1, min(y + half_crop, frame.shape[0]))

			current_cropped = gray_current[y1_curr:y2_curr, x1_curr:x2_curr]

			# Resize current_cropped to match original_cropped_roi size if necessary
			if current_cropped.shape[:2] != original_cropped_roi.shape[:2] and current_cropped.size > 0:
				current_cropped_resized = cv2.resize(current_cropped, (w_roi, h_roi))
			else:
				current_cropped_resized = current_cropped # Use as is if matching or empty

			# Verify we have valid ROIs before processing
			if original_cropped_roi.size > 0 and current_cropped_resized.size > 0:
				# Compare the raw cropped images
				difference = cv2.absdiff(original_cropped_roi, current_cropped_resized)
				mean_difference = np.mean(difference)

				# Display the current cropped ROI just before comparison
				# cv2.imshow("Current Cropped ROI", current_cropped_resized)
				
				# Display the original cropped ROI again in the loop to keep it visible
				# cv2.imshow("Original Cropped ROI", original_cropped_roi)

				# Print the mean difference and FPS
				frame_count += 1
				current_time = time.time()
				elapsed_time = current_time - start_time
				fps = frame_count / elapsed_time if elapsed_time > 0 else 0
				print(f"Comparison FPS: {fps:.1f}, Mean Difference: {mean_difference:.2f}")


				# Check for significant change
				if mean_difference > movement_threshold:
					print("BALL MOVED")
					cv2.circle(current_cropped_resized, (x - x1_curr, y - y1_curr), r, (0, 0, 255), 2)  # Red circle if moved

					# --- Stop and Display Last Comparison ---
					print("Ball movement detected. Stopping monitoring.")

					# Ensure the last frame is displayed
					cv2.imshow("Original Cropped ROI", original_cropped_roi)
					cv2.imshow("Current Cropped ROI", current_cropped_resized)

					print("Displaying last compared images. Press any key to exit.")
					cv2.waitKey(0) # Wait indefinitely for a key press

					# Set stop event to signal acquisition thread
					stop_event.set()
					break # Exit the processing loop

			else:
				# Print a warning if ROI is invalid
				print("Warning: Invalid ROI detected during monitoring.")

		except queue.Empty:
			# If queue is empty and stop_event is not set, just continue waiting
			pass # Use pass to keep correct indentation
		except Exception as e:
			print(f"Processing thread error: {e}")
			stop_event.set()
			break

	print("Processing thread stopped.")
	# Close windows when processing stops
	cv2.destroyAllWindows()
	

# Note: main() and standalone execution block are removed
# This file is now intended to be imported and its functions called by ball_detection.py
