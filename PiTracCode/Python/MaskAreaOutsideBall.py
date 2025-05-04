import cv2
import numpy as np
from GolfBall import GolfBall

def mask_area_outside_ball(
    ball_image: np.ndarray, 
    ball: GolfBall,
    mask_reduction_factor: float,
    ignore_value: int
) -> np.ndarray:
    """
    Masks out everything outside a reduced circle around the ball, filling the outside with ignore_value.
    
    Args:
        ball_image: Input image (grayscale or BGR) as a NumPy array
        ball: GolfBall object containing center coordinates and radius
        mask_reduction_factor: Factor [0..1] to shrink the mask circle relative to ball radius
        ignore_value: Value to fill outside the ball area
        
    Returns:
        A new image where pixels outside the reduced circle are set to ignore_value
    """
    # Calculate the reduced radius
    reduced_radius = int(ball.measured_radius_pixels * mask_reduction_factor)
    
    # Create a mask with the same dimensions as the ball_image
    mask = np.full(ball_image.shape[:2], ignore_value, dtype=ball_image.dtype)
    
    # Draw a filled circle on the mask with the reduced radius
    cv2.circle(mask, (int(ball.x), int(ball.y)), reduced_radius, (255,), thickness=-1)
    
    # Apply the mask to the ball_image
    masked_image = cv2.bitwise_and(ball_image, ball_image, mask=mask)
    
    # Show the debug window with the masked image and green circle
    debug_image = masked_image.copy()
    cv2.circle(debug_image, (int(ball.x), int(ball.y)), reduced_radius, (0, 255, 0), thickness=2)
    cv2.imshow("Masked Ball Image", debug_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return masked_image
