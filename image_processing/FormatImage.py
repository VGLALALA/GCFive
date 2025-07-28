import cv2
import numpy as np

from spin.GolfBall import GolfBall

from .ballDetection import detect_golfballs


def format_image_to_golfball(image_path: str) -> GolfBall:
    """
    Detect a golf ball in ``image_path`` using the YOLO model and return it as a
    :class:`GolfBall` instance.

    Args:
        image_path: Path to the image file.

    Returns:
        A ``GolfBall`` with detected center coordinates and radius in pixels.
    """

    # Load the image in BGR so the detector can run on colour data
    img = cv2.imread(image_path)

    circles = detect_golfballs(img, display=False)

    if circles:
        x, y, r = circles[0]
        return GolfBall(
            x=x,
            y=y,
            measured_radius_pixels=r,
            angles_camera_ortho_perspective=(0.0, 0.0, 0.0),
        )
    raise ValueError("No golf ball detected in the image.")
