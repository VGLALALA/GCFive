
import cv2
from typing import Tuple
from image_processing.SimilarityCalculation import delta_similarity

FRAME_SIMILARITY_DELTA = 0.01

def has_ball_moved(prev_frame: cv2.Mat, curr_frame: cv2.Mat, bbox: Tuple[int, int, int, int]) -> Tuple[bool, float]:
    """Return True if the region defined by bbox changed more than FRAME_SIMILARITY_DELTA."""
    x1, y1, x2, y2 = bbox
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(curr_frame.shape[1], x2)
    y2 = min(curr_frame.shape[0], y2)
    roi_prev = prev_frame[y1:y2, x1:x2]
    roi_curr = curr_frame[y1:y2, x1:x2]
    if roi_prev.size == 0 or roi_curr.size == 0:
        return False, 0.0
    _, delta, _ = delta_similarity(roi_prev, roi_curr)
    return delta > FRAME_SIMILARITY_DELTA, delta

