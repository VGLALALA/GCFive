import cv2
import numpy as np
from IsolateCode import isolate_ball
from GolfBall import GolfBall

def test_isolate_ball():
    # Load the test image
    image_path = r"C:\Users\theka\Downloads\GCFive\Images\gs_log_img__log_ball_final_found_ball_img.png"
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if image is None:
        print("Error: Could not load image")
        return
    
    # Create a dummy GolfBall object with necessary attributes
    ball = GolfBall(
        x=image.shape[1]//2,
        y=image.shape[0]//2,
        measured_radius_pixels=50,
        angles_camera_ortho_perspective=(0.0, 0.0, 0.0)
    )
    
    # Test the isolate_ball function
    ball_image, local_ball = isolate_ball(image, ball)
    
    # Display results
    cv2.imshow("Original Image", image)
    cv2.imshow("Isolated Ball", ball_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Print information about the isolated ball
    print(f"Isolated ball shape: {ball_image.shape}")
    print(f"Local ball center: ({local_ball.x}, {local_ball.y})")
    print(f"Local ball radius: {local_ball.measured_radius_pixels}")

if __name__ == "__main__":
    test_isolate_ball()

# Quick test
image = cv2.imread(r"C:\Users\theka\Downloads\GCFive\Images\gs_log_img__log_ball_final_found_ball_img.png", cv2.IMREAD_GRAYSCALE)
ball = GolfBall(
    x=image.shape[1]//2,
    y=image.shape[0]//2,
    measured_radius_pixels=50,
    angles_camera_ortho_perspective=(0.0, 0.0, 0.0)
)
ball_image, _ = isolate_ball(image, ball)
cv2.imshow("Quick Test", ball_image)
cv2.waitKey(0)
cv2.destroyAllWindows() 