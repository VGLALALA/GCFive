import cv2
import numpy as np

def mask_area_outside_ball(ball_image: np.ndarray,
                           ball,
                           mask_reduction_factor: float,
                           mask_value: tuple[int, int, int]) -> np.ndarray:
    """
    Masks everything outside a reduced-radius circle around the ball,
    painting the outside region with mask_value.
    
    :param ball_image:     Input image (e.g. edge-detected dimples), H×W×C
    :param ball:           GolfBall-like object with .measured_radius_pixels, .x, .y
    :param mask_reduction_factor: Fraction to shrink the mask circle (e.g. 0.92)
    :param mask_value:     BGR color to paint outside (e.g. (128,128,128))
    :returns:              New image with outside masked
    """
    # 1) compute reduced mask radius
    mask_radius = int(ball.measured_radius_pixels * mask_reduction_factor)

    # 2) first mask: white circle on black
    mask = np.zeros_like(ball_image)
    cv2.circle(mask,
               center=(int(ball.x), int(ball.y)),
               radius=mask_radius,
               color=(255, 255, 255),
               thickness=-1)

    # 3) keep only inside-circle pixels
    result = cv2.bitwise_and(ball_image, mask)

    # 4) build inverted mask: rectangle of mask_value with black circle punched out
    cv2.rectangle(mask,
                  pt1=(0, 0),
                  pt2=(ball_image.shape[1], ball_image.shape[0]),
                  color=mask_value,
                  thickness=cv2.FILLED)
    cv2.circle(mask,
               center=(int(ball.x), int(ball.y)),
               radius=mask_radius,
               color=(0, 0, 0),
               thickness=-1)

    # 5) XOR in the outside region => paints everything outside the circle
    result = cv2.bitwise_xor(result, mask)

    return result
