import cv2
import numpy as np
import mvsdk
import platform
import time

def simple_camera_test():
    print("Simple camera test - just display camera feed")
    
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
        
        # Set basic camera parameters
        mvsdk.CameraSetTriggerMode(hCamera, 0)  # Continuous mode
        mvsdk.CameraSetFrameSpeed(hCamera, 2)   # High frame speed
        mvsdk.CameraSetAeState(hCamera, 1)      # Auto exposure ON
        mvsdk.CameraSetExposureTime(hCamera, 10000)  # 10ms exposure (in microseconds)
        mvsdk.CameraSetGain(hCamera, 200, 200, 200)  # Higher gain
        mvsdk.CameraSetContrast(hCamera, 150)   # Higher contrast
        
        mvsdk.CameraPlay(hCamera)
        
        # Get actual resolution
        actual_resolution = mvsdk.CameraGetImageResolution(hCamera)
        actual_width = actual_resolution.iWidth
        actual_height = actual_resolution.iHeight
        print(f"Camera resolution: {actual_width}x{actual_height}")
        
        # Allocate buffer
        FrameBufferSize = actual_width * actual_height * (1 if monoCamera else 3)
        pFrameBuffer = mvsdk.CameraAlignMalloc(FrameBufferSize, 16)
        
        print("Press 'q' to quit, any other key to continue...")
        
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
                
                # Convert to BGR for display
                if monoCamera:
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                else:
                    frame_bgr = frame
                
                # Test: Create a colored pattern if frame is all black
                if frame_bgr.max() == 0:
                    # Create a test pattern
                    test_pattern = np.zeros((480, 640, 3), dtype=np.uint8)
                    test_pattern[:, :, 0] = 128  # Blue channel
                    test_pattern[:, :, 1] = 128  # Green channel  
                    test_pattern[:, :, 2] = 128  # Red channel
                    # Add some colored rectangles
                    cv2.rectangle(test_pattern, (100, 100), (200, 200), (0, 255, 0), -1)  # Green
                    cv2.rectangle(test_pattern, (300, 100), (400, 200), (0, 0, 255), -1)  # Red
                    cv2.rectangle(test_pattern, (500, 100), (600, 200), (255, 0, 0), -1)  # Blue
                    frame_bgr = test_pattern
                    print("Using test pattern - camera frame is black")
                
                # Add text to frame
                cv2.putText(frame_bgr, f"Frame: {frame_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame_bgr, f"Shape: {frame_bgr.shape}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame_bgr, f"Min: {frame_bgr.min()}, Max: {frame_bgr.max()}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Display frame
                cv2.imshow("Simple Camera Test", frame_bgr)
                
                frame_count += 1
                if frame_count % 30 == 0:
                    print(f"Captured {frame_count} frames")
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                    
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
        print("Camera test completed")
        
    except Exception as e:
        print(f"Setup error: {e}")

if __name__ == "__main__":
    simple_camera_test() 