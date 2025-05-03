import cv2
import numpy as np

def mask_area_outside_ball(
    ball_image: np.ndarray,
    ball_center: tuple[int, int],
    measured_radius_pixels: float,
    mask_reduction_factor: float,
    mask_value: tuple[int, int, int] | int
) -> np.ndarray:
    """
    Masks out everything outside a reduced circle around the ball, filling the outside with mask_value.
    
    Args:
        ball_image:       Input image (grayscale or BGR) as a NumPy array.
        ball_center:      (x, y) center coordinates of the ball.
        measured_radius_pixels:  Detected radius of the ball in pixels.
        mask_reduction_factor:   Factor [0..1] to shrink the mask circle relative to measured_radius_pixels.
        mask_value:       Scalar value or BGR tuple to fill outside the ball area.
    
    Returns:
        A new image where pixels outside the reduced circle are set to mask_value.
    """
    h, w = ball_image.shape[:2]
    mask_radius = int(measured_radius_pixels * mask_reduction_factor)
    cx, cy = ball_center

    # 1) Create a black mask and draw a filled white circle where the ball is
    mask = np.zeros((h, w), dtype=np.uint8 if ball_image.ndim == 2 else np.uint8)
    cv2.circle(mask, (cx, cy), mask_radius, 255, thickness=-1)

    # 2) Preserve the ball area by ANDing
    if ball_image.ndim == 2:
        result = cv2.bitwise_and(ball_image, ball_image, mask=mask)
    else:
        # for BGR images, apply the single-channel mask to all three channels
        result = cv2.bitwise_and(ball_image, ball_image, mask=mask)

    # 3) Build an inverse mask filled with mask_value
    #    First, create a full-image of mask_value
    if isinstance(mask_value, int):
        fill = np.full_like(ball_image, mask_value)
    else:
        fill = np.full_like(ball_image, mask_value)

    #    Then carve out the circle to be black so XOR will leave the circle area intact
    inv_mask = fill.copy()
    cv2.circle(inv_mask, (cx, cy), mask_radius, (0, 0, 0) if ball_image.ndim == 3 else 0, thickness=-1)

    # 4) XOR the preserved-ball image with the inverse mask
    result = cv2.bitwise_xor(result, inv_mask)

    return result
