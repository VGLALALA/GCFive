"""Utilities for detecting motion of the golf ball between frames."""

import cv2
from typing import Tuple

from .SimilarityCalculation import delta_similarity

FRAME_SIMILARITY_DELTA = 0.0015


def detect_ball_movement(
    prev_frame: "cv2.Mat",
    curr_frame: "cv2.Mat",
    bbox: Tuple[int, int, int, int],
    threshold: float = FRAME_SIMILARITY_DELTA,
) -> Tuple[float, bool]:
    """Return the similarity delta and whether movement is detected.

    Parameters
    ----------
    prev_frame : ndarray
        Previous video frame in BGR.
    curr_frame : ndarray
        Current video frame in BGR.
    bbox : tuple
        Bounding box ``(x, y, w, h)`` around the ball in ``prev_frame``.
    threshold : float
        Delta threshold signalling that the ball has moved.

    Returns
    -------
    (delta, moved)
        ``delta`` is the frame difference returned by
        :func:`SimilarityCalculation.delta_similarity` and ``moved`` is ``True``
        if ``delta`` exceeds ``threshold``.
    """

    x, y, w, h = bbox
    roi_prev = prev_frame[y : y + h, x : x + w]
    roi_curr = curr_frame[y : y + h, x : x + w]

    if roi_prev.size == 0 or roi_curr.size == 0:
        return 0.0, False

    _, delta, _ = delta_similarity(roi_prev, roi_curr)
    return delta, delta > threshold

