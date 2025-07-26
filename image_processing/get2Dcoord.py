import math
from image_processing.CamBalldistancePred import predict_distance_from_frame, load_calibration
from image_processing.ballDetection import detect_golfballs

def get_ball_xz(frame, conf=0.25, imgsz=640, display=False):
    """
    Given a BGR frame returns:
      • x_mm: horizontal offset (right-positive) relative to camera optical axis
      • z_mm: forward distance along optical axis
      • d_mm: true Euclidean distance from camera to ball center

    Uses:
      - predict_distance_from_frame(frame) → z_mm
      - detect_golfballs(frame, conf, imgsz, display) → [(x_px, y_px, r_px), ...]

    Requires calibration.json to exist so load_calibration() yields focal length.
    """
    # 1) Estimate forward-axis distance (Z) in mm
    z_mm = predict_distance_from_frame(frame)
    if z_mm is None:
        return None  # no ball detected

    # 2) Find the ball in pixels
    dets = detect_golfballs(frame, conf=conf, imgsz=imgsz, display=display)
    if not dets:
        return None

    # pick the largest detection
    dets.sort(key=lambda t: t[2], reverse=True)
    x_px, y_px, r_px = dets[0]

    # 3) Load calibrated focal length (in px)
    focal_px, _ = load_calibration()
    if focal_px is None:
        raise RuntimeError("Camera must be calibrated first")

    # 4) Convert pixel offset to mm in the object plane
    h, w = frame.shape[:2]
    cx = w / 2.0
    pixel_offset_x = x_px - cx
    x_mm = pixel_offset_x * (z_mm / focal_px)

    # 5) Compute true (straight-line) distance from camera to ball:
    #    d = sqrt(z^2 + x^2)
    d_mm = math.hypot(z_mm, x_mm)

    return x_mm, z_mm
