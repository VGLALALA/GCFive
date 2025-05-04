import cv2
import numpy as np
import copy
from typing import Tuple
from GolfBall import GolfBall
from MaskAreaOutsideBall import mask_area_outside_ball

def isolate_ball(
    img: np.ndarray,
    ball: GolfBall
) -> Tuple[np.ndarray, GolfBall]:
    """
    Crops out a square region around the detected ball, recenters the ball metadata,
    equalizes (optional), then masks out pixels just outside the ball edge.
    If ball.x, ball.y, or ball.measured_radius_pixels are 0 or None, auto-detects them.

    Args:
        img:   Full grayscale image.
        ball:  GolfBall instance with .x, .y, .measured_radius_pixels.

    Returns:
        ball_crop:   Cropped & masked image of the ball.
        ball_local:  A copy of `ball` whose x,y coords are relative to the crop.
    """
    # Auto-detect if needed
    if not getattr(ball, 'x', None) or not getattr(ball, 'y', None) or not getattr(ball, 'measured_radius_pixels', None):
        print("[isolate_ball] Ball parameters missing, running HoughCircles...")
        circles = cv2.HoughCircles(
            img, cv2.HOUGH_GRADIENT, dp=1.2, minDist=50,
            param1=100, param2=30, minRadius=30, maxRadius=70
        )
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            print(f"[isolate_ball] All detected circles: {circles}")
            x, y, r = circles[0]
            print(f"[isolate_ball] Detected ball at (x={x}, y={y}), radius={r}")
            ball.x = x
            ball.y = y
            ball.measured_radius_pixels = r
        else:
            print("[isolate_ball] No ball detected in image!")
            raise ValueError("No ball detected in image!")
    else:
        print(f"[isolate_ball] Using provided ball: x={ball.x}, y={ball.y}, radius={ball.measured_radius_pixels}")

    # Make a local copy so we don't mutate the caller's ball
    ball_local = copy.deepcopy(ball)

    # 1) Compute a slightly larger radius to include a tiny border
    surround_mult = 1.05
    r1  = int(round(ball_local.measured_radius_pixels * surround_mult))
    rInc = r1 - ball_local.measured_radius_pixels

    # 2) Determine top-left corner of crop
    x1 = int(ball_local.x - r1)
    y1 = int(ball_local.y - r1)
    w  = h = 2 * r1
    print(f"[isolate_ball] Crop top-left: ({x1}, {y1}), size: {w}x{h}")

    # 3) Clip to image bounds
    x1 = max(0, min(x1, img.shape[1] - w - 1))
    y1 = max(0, min(y1, img.shape[0] - h - 1))

    # 4) Crop
    ball_crop = img[y1 : y1 + h, x1 : x1 + w].copy()

    # 5) Recenter ball coordinates in the cropped image
    new_center = int(round(rInc + ball_local.measured_radius_pixels))
    ball_local.x = new_center
    ball_local.y = new_center
    print(f"[isolate_ball] Local ball center in crop: ({ball_local.x}, {ball_local.y}), radius={ball_local.measured_radius_pixels}")

    # 6) (Optional) equalize histogram if desired
    # ball_crop = cv2.equalizeHist(ball_crop)

    # 7) Mask out just outside the ball to remove edge artifacts
    #    reference reduction factor from C++
    reference_mask_factor = 0.995
    ball_crop = mask_area_outside_ball(
        ball_crop,
        (ball_local.x, ball_local.y),
        ball_local.measured_radius_pixels,
        reference_mask_factor,
        0
    )

    return ball_crop, ball_local

