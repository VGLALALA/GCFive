import cv2
import numpy as np
from GolfBall import GolfBall
# Commenting out the import that causes the error
from ballDetection import run_hough_with_radius, auto_determine_circle_radius
from Convert_Canny import convert_to_canny
def format_image_to_golfball(image_path: str) -> GolfBall:
    """
    Formats an image to a GolfBall class instance by detecting the golf ball outline
    and determining its radius using the Hough Circle Transform.

    Args:
        image_path: Path to the image file.

    Returns:
        A GolfBall instance with detected x, y coordinates and measured radius.
    """
    # Read the image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    

    # Automatically determine the radius of the circle
    radius = auto_determine_circle_radius(image_path)
    canny = convert_to_canny(image_path)
    # Run Hough Circle Transform to detect circles
    circles = run_hough_with_radius(canny, radius)
    
    # Assuming the first detected circle is the golf ball
    if circles is not None and len(circles) > 0:
        x, y, r = circles[0]
        return GolfBall(x=x, y=y, measured_radius_pixels=r, angles_camera_ortho_perspective=(0.0, 0.0, 0.0))
    else:
        raise ValueError("No golf ball detected in the image.")
