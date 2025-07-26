import cv2
import numpy as np
import mvsdk
import platform
import time
from ballDetectionyolo import detect_golfballs
from ultralytics import YOLO
ROI_W, ROI_H            = 640, 280
ROI_X, ROI_Y            =   0, 120

# Ball and capture settings
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
def debug_yolo_detection():
    print("Debug YOLO detection - showing bounding boxes and details")
    
    try:
        # Setup camera
        DevList = mvsdk.CameraEnumerateDevice()
        if len(DevList) < 1:
            print("No camera found!")
            return
        
        DevInfo = DevList[0]
        hCamera = mvsdk.CameraInit(DevInfo, -1, -1)
        
        cap = mvsdk.CameraGetCapability(hCamera)
        monoCamera = (cap.sIspCapacity.bMonoSensor != 0)
        
        if monoCamera:
            mvsdk.CameraSetIspOutFormat(hCamera, mvsdk.CAMERA_MEDIA_TYPE_MONO8)
        else:
            mvsdk.CameraSetIspOutFormat(hCamera, mvsdk.CAMERA_MEDIA_TYPE_BGR8)
        
        # Set camera parameters
        mvsdk.CameraSetTriggerMode(hCamera, 0)
        mvsdk.CameraSetAeState(hCamera, 0)
        mvsdk.CameraSetExposureTime(hCamera, DESIRED_EXPOSURE_US)
        gmin, gmax, _ = mvsdk.CameraGetAnalogGainXRange(hCamera)
        mvsdk.CameraSetAnalogGainX(hCamera, max(gmin, min(DESIRED_ANALOG_GAIN, gmax)))
        gamma_max = cap.sGammaRange.iMax
        mvsdk.CameraSetGamma(hCamera, int(DESIRED_GAMMA * gamma_max))
        mvsdk.CameraPlay(hCamera)
        
        # Get resolution
        actual_resolution = mvsdk.CameraGetImageResolution(hCamera)
        actual_width = actual_resolution.iWidth
        actual_height = actual_resolution.iHeight
        print(f"Camera resolution: {actual_width}x{actual_height}")
        
        # Allocate buffer
        FrameBufferSize = actual_width * actual_height * (1 if monoCamera else 3)
        pFrameBuffer = mvsdk.CameraAlignMalloc(FrameBufferSize, 16)
        
        # Load YOLO model
        model = YOLO("data/model/golfballv4.pt")
        
        print("Press 'q' to quit, 's' to save frame, any other key to continue...")
        
        frame_count = 0
        while True:
            try:
                pRawData, FrameHead = mvsdk.CameraGetImageBuffer(hCamera, 1000)
                mvsdk.CameraImageProcess(hCamera, pRawData, pFrameBuffer, FrameHead)
                mvsdk.CameraReleaseImageBuffer(hCamera, pRawData)
                
                if platform.system() == "Windows":
                    mvsdk.CameraFlipFrameBuffer(pFrameBuffer, FrameHead, 1)
                
                # Convert to OpenCV format
                frame_data = (mvsdk.c_ubyte * FrameHead.uBytes).from_address(pFrameBuffer)
                frame = np.frombuffer(frame_data, dtype=np.uint8)
                frame = frame.reshape((FrameHead.iHeight, FrameHead.iWidth, 1 if FrameHead.uiMediaType == mvsdk.CAMERA_MEDIA_TYPE_MONO8 else 3))
                
                # Convert to BGR for YOLO
                if monoCamera:
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                else:
                    frame_bgr = frame
                
                # Run YOLO detection with detailed output
                results = model.predict(source=frame_bgr, conf=0.25, imgsz=640, verbose=False)
                
                # Create display frame
                display_frame = frame_bgr.copy()
                
                if results and len(results) > 0:
                    boxes = results[0].boxes
                    if boxes is not None and len(boxes) > 0:
                        print(f"\nFrame {frame_count} - Found {len(boxes)} detection(s):")
                        
                        for i, box in enumerate(boxes):
                            # Get box coordinates
                            x_min, y_min, x_max, y_max = box.xyxy.cpu().numpy().astype(int).flatten()
                            confidence = float(box.conf.item())
                            class_id = int(box.cls.item())
                            
                            print(f"  Detection {i+1}:")
                            print(f"    Class ID: {class_id}")
                            print(f"    Confidence: {confidence:.3f}")
                            print(f"    Bounding Box: ({x_min}, {y_min}) to ({x_max}, {y_max})")
                            print(f"    Width: {x_max - x_min}, Height: {y_max - y_min}")
                            
                            # Calculate center and radius
                            width, height = x_max - x_min, y_max - y_min
                            x_center = x_min + width // 2
                            y_center = y_min + height // 2
                            radius = int((width + height) / 4)
                            
                            print(f"    Center: ({x_center}, {y_center})")
                            print(f"    Radius: {radius}")
                            
                            # Draw bounding box
                            cv2.rectangle(display_frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                            
                            # Draw center point
                            cv2.circle(display_frame, (x_center, y_center), 3, (0, 0, 255), -1)
                            
                            # Draw radius circle
                            cv2.circle(display_frame, (x_center, y_center), radius, (255, 0, 0), 2)
                            
                            # Add label
                            label = f"Ball {i+1}: conf={confidence:.2f}, class={class_id}"
                            cv2.putText(display_frame, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    else:
                        print(f"Frame {frame_count} - No detections")
                else:
                    print(f"Frame {frame_count} - No results")
                
                # Add frame info
                cv2.putText(display_frame, f"Frame: {frame_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(display_frame, f"Press 'q' to quit, 's' to save", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Display frame
                cv2.imshow("Debug YOLO Detection", display_frame)
                
                frame_count += 1
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    # Save current frame
                    filename = f"debug_frame_{frame_count}.png"
                    cv2.imwrite(filename, display_frame)
                    print(f"Saved frame as {filename}")
                    
            except mvsdk.CameraException as e:
                if e.error_code != mvsdk.CAMERA_STATUS_TIME_OUT:
                    print(f"Camera error: {e.error_code} - {e.message}")
                continue
            except Exception as e:
                print(f"Error: {e}")
                break
        
        # Cleanup
        mvsdk.CameraAlignFree(pFrameBuffer)
        mvsdk.CameraUnInit(hCamera)
        cv2.destroyAllWindows()
        print("Debug completed")
        
    except Exception as e:
        print(f"Setup error: {e}")

if __name__ == "__main__":
    debug_yolo_detection() 