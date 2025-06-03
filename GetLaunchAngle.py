import cv2
import numpy as np
import math

def calculate_golf_ball_trajectory(image1, image2, fps):
    """
    Calculate launch angle and ball speed from two consecutive golf ball images.
    
    Parameters:
    - image1: First frame (numpy array or image path)
    - image2: Second frame (numpy array or image path)
    - fps: Frame rate from getFPS.py
    
    Returns:
    - dict: Contains ball_speed_mps, ball_speed_mph, launch_angle_degrees, 
            and visualization image
    """
    
    # Golf ball specifications
    BALL_DIAMETER_MM = 42.67
    
    # Load images if paths are provided
    if isinstance(image1, str):
        img1 = cv2.imread(image1)
    else:
        img1 = image1.copy()
        
    if isinstance(image2, str):
        img2 = cv2.imread(image2)
    else:
        img2 = image2.copy()
    
    # Function to detect golf ball in image
    def detect_golf_ball(image):
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        
        # Use HoughCircles to detect circular objects
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=50,
            param1=100,
            param2=30,
            minRadius=10,
            maxRadius=100
        )
        
        if circles is not None:
            circles = np.uint16(np.around(circles))
            # Return the first detected circle (center_x, center_y, radius)
            return circles[0, 0]
        return None
    
    # Detect golf ball in both images
    ball1 = detect_golf_ball(img1)
    ball2 = detect_golf_ball(img2)
    
    if ball1 is None or ball2 is None:
        raise ValueError("Could not detect golf ball in one or both images")
    
    # Extract positions and radius
    x1, y1, r1 = ball1
    x2, y2, r2 = ball2
    
    # Calculate scale factor (pixels per mm)
    # Average radius in pixels
    avg_radius_pixels = (r1 + r2) / 2
    pixels_per_mm = (avg_radius_pixels * 2) / BALL_DIAMETER_MM
    
    # Calculate displacement in pixels
    dx_pixels = x2 - x1
    dy_pixels = y2 - y1  # Note: In image coordinates, y increases downward
    
    # Convert to real-world coordinates (mm)
    # Flip y-axis for conventional coordinate system
    dx_mm = dx_pixels / pixels_per_mm
    dy_mm = -dy_pixels / pixels_per_mm  # Negative because image y-axis is inverted
    
    # Calculate time between frames
    dt = 1 / fps  # seconds
    
    # Calculate velocities (mm/s)
    vx = dx_mm / dt
    vy = dy_mm / dt
    
    # Calculate ball speed
    speed_mm_per_s = math.sqrt(vx**2 + vy**2)
    speed_m_per_s = speed_mm_per_s / 1000
    speed_mph = speed_m_per_s * 2.237  # Convert m/s to mph
    
    # Calculate launch angle
    launch_angle_rad = math.atan2(vy, vx)
    launch_angle_deg = math.degrees(launch_angle_rad)
    
    # Create visualization
    vis_img = create_trajectory_visualization(
        img1, img2, ball1, ball2, launch_angle_deg, speed_mph
    )
    
    return {
        'ball_speed_mps': speed_m_per_s,
        'ball_speed_mph': speed_mph,
        'launch_angle_degrees': launch_angle_deg,
        'horizontal_velocity_mps': vx / 1000,
        'vertical_velocity_mps': vy / 1000,
        'ball_position_1': (x1, y1),
        'ball_position_2': (x2, y2),
        'pixels_per_mm': pixels_per_mm,
        'visualization': vis_img
    }

def create_trajectory_visualization(img1, img2, ball1, ball2, angle, speed):
    """Create a visualization showing the ball trajectory"""
    # Create side-by-side visualization
    h, w = img1.shape[:2]
    vis = np.zeros((h, w*2 + 10, 3), dtype=np.uint8)
    
    # Place images side by side
    vis[:, :w] = img1
    vis[:, w+10:] = img2
    
    # Draw detected balls
    cv2.circle(vis, (int(ball1[0]), int(ball1[1])), int(ball1[2]), (0, 255, 0), 2)
    cv2.circle(vis, (int(ball2[0] + w + 10), int(ball2[1])), int(ball2[2]), (0, 255, 0), 2)
    
    # Draw trajectory line
    cv2.line(vis, 
             (int(ball1[0]), int(ball1[1])), 
             (int(ball2[0] + w + 10), int(ball2[1])), 
             (255, 0, 0), 2)
    
    # Add text annotations
    cv2.putText(vis, f"Launch Angle: {angle:.1f} degrees", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(vis, f"Ball Speed: {speed:.1f} mph", 
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return vis

# Alternative implementation using blob detection for better accuracy
def calculate_golf_ball_trajectory_advanced(image1, image2, fps):
    """
    Advanced version using blob detection for more accurate ball detection.
    """
    
    BALL_DIAMETER_MM = 42.67
    
    # Load images
    if isinstance(image1, str):
        img1 = cv2.imread(image1)
    else:
        img1 = image1.copy()
        
    if isinstance(image2, str):
        img2 = cv2.imread(image2)
    else:
        img2 = image2.copy()
    
    def detect_ball_blob(image):
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Set up blob detector parameters
        params = cv2.SimpleBlobDetector_Params()
        params.filterByArea = True
        params.minArea = 100
        params.maxArea = 10000
        params.filterByCircularity = True
        params.minCircularity = 0.7
        params.filterByConvexity = True
        params.minConvexity = 0.8
        params.filterByInertia = True
        params.minInertiaRatio = 0.7
        
        # Create detector
        detector = cv2.SimpleBlobDetector_create(params)
        
        # Detect blobs
        keypoints = detector.detect(gray)
        
        if keypoints:
            # Get the largest blob (assuming it's the golf ball)
            largest = max(keypoints, key=lambda x: x.size)
            return (largest.pt[0], largest.pt[1], largest.size/2)
        return None
    
    # Detect balls
    ball1 = detect_ball_blob(img1)
    ball2 = detect_ball_blob(img2)
    
    if ball1 is None or ball2 is None:
        # Fallback to Hough circles
        return calculate_golf_ball_trajectory(image1, image2, fps)
    
    # Continue with same calculations as before...
    x1, y1, r1 = ball1
    x2, y2, r2 = ball2
    
    avg_radius_pixels = (r1 + r2) / 2
    pixels_per_mm = (avg_radius_pixels * 2) / BALL_DIAMETER_MM
    
    dx_mm = (x2 - x1) / pixels_per_mm
    dy_mm = -(y2 - y1) / pixels_per_mm
    
    dt = 1 / fps
    
    vx = dx_mm / dt
    vy = dy_mm / dt
    
    speed_mm_per_s = math.sqrt(vx**2 + vy**2)
    speed_m_per_s = speed_mm_per_s / 1000
    speed_mph = speed_m_per_s * 2.237
    
    launch_angle_rad = math.atan2(vy, vx)
    launch_angle_deg = math.degrees(launch_angle_rad)
    
    return {
        'ball_speed_mps': speed_m_per_s,
        'ball_speed_mph': speed_mph,
        'launch_angle_degrees': launch_angle_deg,
        'horizontal_velocity_mps': vx / 1000,
        'vertical_velocity_mps': vy / 1000,
        'ball_position_1': (x1, y1),
        'ball_position_2': (x2, y2),
        'pixels_per_mm': pixels_per_mm
    }

# Example usage:
if __name__ == "__main__":
    # Example with getFPS.py
    import getFPS  # Assuming getFPS.py provides a function to get FPS
    
    # Get frame rate
    fps = 790  # Or however getFPS.py provides the frame rate
    
    # Load images
    image1 = cv2.imread("frame1.jpg")
    image2 = cv2.imread("frame2.jpg")
    
    # Calculate trajectory
    results = calculate_golf_ball_trajectory(image1, image2, fps)
    
    print(f"Ball Speed: {results['ball_speed_mph']:.1f} mph ({results['ball_speed_mps']:.1f} m/s)")
    print(f"Launch Angle: {results['launch_angle_degrees']:.1f}°")
    print(f"Horizontal Velocity: {results['horizontal_velocity_mps']:.2f} m/s")
    print(f"Vertical Velocity: {results['vertical_velocity_mps']:.2f} m/s")
    
    # Show visualization
    cv2.imshow("Trajectory Analysis", results['visualization'])
    cv2.waitKey(0)
    cv2.destroyAllWindows()