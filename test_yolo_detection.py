import cv2
import numpy as np
import mvsdk
import platform
import time
from ballDetectionyolo import detect_golfballs

def test_camera_and_yolo():
    print("Testing camera and YOLO detection...")
    
    # Test 1: Check if YOLO model loads
    print("1. Testing YOLO model loading...")
    try:
        from ultralytics import YOLO
        model = YOLO("data/model/golfballv4.pt")
        print("✅ YOLO model loaded successfully")
    except Exception as e:
        print(f"❌ YOLO model loading failed: {e}")
        return
    
    # Test 2: Test camera setup
    print("\n2. Testing camera setup...")
    try:
        DevList = mvsdk.CameraEnumerateDevice()
        nDev = len(DevList)
        if nDev < 1:
            print("❌ No camera found!")
            return
        
        print(f"✅ Found {nDev} camera(s)")
        for i, DevInfo in enumerate(DevList):
            print(f"   {i}: {DevInfo.GetFriendlyName()}")
        
        # Use first camera
        DevInfo = DevList[0]
        hCamera = mvsdk.CameraInit(DevInfo, -1, -1)
        
        cap = mvsdk.CameraGetCapability(hCamera)
        monoCamera = (cap.sIspCapacity.bMonoSensor != 0)
        print(f"✅ Camera initialized (Mono: {monoCamera})")
        
        # Set camera format
        if monoCamera:
            mvsdk.CameraSetIspOutFormat(hCamera, mvsdk.CAMERA_MEDIA_TYPE_MONO8)
        else:
            mvsdk.CameraSetIspOutFormat(hCamera, mvsdk.CAMERA_MEDIA_TYPE_BGR8)
        
        mvsdk.CameraPlay(hCamera)
        print("✅ Camera started")
        
    except Exception as e:
        print(f"❌ Camera setup failed: {e}")
        return
    
    # Test 3: Capture and display frames
    print("\n3. Testing frame capture and display...")
    try:
        # Allocate buffer
        FrameBufferSize = 640 * 480 * (1 if monoCamera else 3)
        pFrameBuffer = mvsdk.CameraAlignMalloc(FrameBufferSize, 16)
        
        frame_count = 0
        start_time = time.time()
        
        while frame_count < 30:  # Capture 30 frames
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
                
                # Convert to BGR for display and YOLO
                if monoCamera:
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                else:
                    frame_bgr = frame
                
                # Display frame
                cv2.imshow("Camera Test", frame_bgr)
                key = cv2.waitKey(1)
                if key == ord('q'):
                    break
                
                frame_count += 1
                if frame_count % 10 == 0:
                    print(f"   Captured {frame_count} frames...")
                
            except mvsdk.CameraException as e:
                if e.error_code != mvsdk.CAMERA_STATUS_TIME_OUT:
                    print(f"   Camera error: {e.error_code} - {e.message}")
                continue
        
        print("✅ Frame capture test completed")
        
        # Test 4: Test YOLO detection on a frame
        print("\n4. Testing YOLO detection...")
        try:
            detected_balls = detect_golfballs(frame_bgr, conf=0.25, imgsz=640, display=False)
            print(f"✅ YOLO detection completed. Found {len(detected_balls)} ball(s)")
            
            if detected_balls:
                for i, (x, y, r) in enumerate(detected_balls):
                    print(f"   Ball {i+1}: center=({x}, {y}), radius={r}")
                    cv2.circle(frame_bgr, (x, y), r, (0, 255, 0), 2)
                    cv2.circle(frame_bgr, (x, y), 2, (0, 0, 255), 3)
                
                cv2.imshow("YOLO Detection Result", frame_bgr)
                print("Press any key to continue...")
                cv2.waitKey(0)
            else:
                print("   No balls detected in this frame")
                
        except Exception as e:
            print(f"❌ YOLO detection failed: {e}")
        
        # Cleanup
        mvsdk.CameraAlignFree(pFrameBuffer)
        mvsdk.CameraUnInit(hCamera)
        cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"❌ Frame capture test failed: {e}")
        try:
            mvsdk.CameraUnInit(hCamera)
        except:
            pass

if __name__ == "__main__":
    test_camera_and_yolo() 