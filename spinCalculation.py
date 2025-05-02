import cv2
import numpy as np
from typing import Tuple, List, Optional


def _detect_ball(frame: np.ndarray, dp: float = 1.2, min_dist: float = 30,
                 param1: int = 100, param2: int = 30,
                 min_radius: int = 10, max_radius: int = 300) -> Optional[Tuple[int, int, int]]:
    """Detect the golf ball via HoughCircles (on a grayscale frame).

    Returns (x, y, r) in pixel units or None if detection failed.
    """
    gray = cv2.medianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 5)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp, min_dist,
                               param1=param1, param2=param2,
                               minRadius=min_radius, maxRadius=max_radius)
    if circles is None:
        return None
    circles = np.uint16(np.around(circles))[0]
    # pick the strongest circle (largest accumulator value is first)
    x, y, r = circles[0]
    return int(x), int(y), int(r)


def _crop_ball(frame: np.ndarray, ball: Tuple[int, int, int], margin: float = 0.1) -> np.ndarray:
    """Crop a square ROI centred on the ball with a small margin."""
    h, w = frame.shape[:2]
    x, y, r = ball
    m = int(r * (1 + margin))
    x0 = max(x - m, 0)
    y0 = max(y - m, 0)
    x1 = min(x + m, w)
    y1 = min(y + m, h)
    return frame[y0:y1, x0:x1].copy(), (x0, y0)


def _extract_keypoints(roi: np.ndarray, max_kp: int = 500):
    """Detect ORB keypoints & descriptors inside ROI."""
    orb = cv2.ORB_create(nfeatures=max_kp)
    kp, des = orb.detectAndCompute(roi, None)
    return kp, des


def _match_keypoints(des1, des2, ratio_thresh: float = 0.75):
    """Brute‑force Hamming matching with Lowe ratio test."""
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(des1, des2, k=2)
    good = [m for m, n in matches if m.distance < ratio_thresh * n.distance]
    return good


def _pixel_to_sphere(kp: List[cv2.KeyPoint], center: Tuple[int, int], radius: int) -> np.ndarray:
    """Convert 2‑D keypoints to 3‑D unit vectors on the ball surface (assuming perfect sphere)."""
    cx, cy = center
    vectors = []
    for k in kp:
        dx = (k.pt[0] - cx) / radius
        dy = (k.pt[1] - cy) / radius
        r2 = dx * dx + dy * dy
        if r2 >= 1:
            continue  # outside sphere
        dz = np.sqrt(1 - r2)
        vec = np.array([dx, -dy, dz])  # image y axis downward → invert sign
        vec /= np.linalg.norm(vec)
        vectors.append(vec)
    return np.array(vectors)


def _estimate_rotation(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """Kabsch algorithm: find R s.t. R·P ~= Q."""
    assert P.shape == Q.shape and P.shape[1] == 3
    P_mean = P.mean(axis=0)
    Q_mean = Q.mean(axis=0)
    P_centered = P - P_mean
    Q_centered = Q - Q_mean
    H = P_centered.T @ Q_centered
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = Vt.T @ U.T
    return R


def _rotation_to_axis_angle(R: np.ndarray) -> Tuple[np.ndarray, float]:
    """Return unit axis (xyz) and angle (rad) from rotation matrix."""
    angle = np.arccos(np.clip((np.trace(R) - 1) / 2.0, -1, 1))
    if abs(angle) < 1e-6:
        return np.array([1, 0, 0]), 0.0
    rx = R[2, 1] - R[1, 2]
    ry = R[0, 2] - R[2, 0]
    rz = R[1, 0] - R[0, 1]
    axis = np.array([rx, ry, rz])
    axis /= (2 * np.sin(angle))
    axis /= np.linalg.norm(axis)
    return axis, angle


def calculate_spin(image1: np.ndarray, image2: np.ndarray, delta_t_seconds: float,
                   *, debug: bool = False, debug_dir: str = "debug") -> Tuple[float, float, float]:
    """Estimate golf‑ball spin parameters from two consecutive images.

    Args:
        image1, image2: BGR frames (np.ndarray) of identical resolution.
        delta_t_seconds: Time between frames (s).
        debug: If True, writes annotated frames & intermediate images.
        debug_dir: Folder where debug images are stored.

    Returns:
        backspin_rpm, axis_deg, sidespin_rpm
    """
    # 1. Ball detection
    ball1 = _detect_ball(image1)
    ball2 = _detect_ball(image2)
    if ball1 is None or ball2 is None:
        raise ValueError("Ball detection failed in one of the frames")

    # 2. Crop ROIs
    roi1, offset1 = _crop_ball(image1, ball1)
    roi2, offset2 = _crop_ball(image2, ball2)

    # 3. Keypoint extraction
    kp1, des1 = _extract_keypoints(roi1)
    kp2, des2 = _extract_keypoints(roi2)
    if des1 is None or des2 is None:
        raise ValueError("No descriptors found")

    # 4. Match keypoints
    good = _match_keypoints(des1, des2)
    if len(good) < 6:
        raise ValueError("Not enough good matches: {}".format(len(good)))

    # 5. Build corresponding 3‑D vectors on sphere surface
    c1 = (ball1[0] - offset1[0], ball1[1] - offset1[1])
    c2 = (ball2[0] - offset2[0], ball2[1] - offset2[1])
    r1 = ball1[2]
    r2 = ball2[2]

    pts1 = []
    pts2 = []
    for m in good:
        pts1.append(kp1[m.queryIdx])
        pts2.append(kp2[m.trainIdx])

    V1 = _pixel_to_sphere(pts1, c1, r1)
    V2 = _pixel_to_sphere(pts2, c2, r2)

    # Keep only pairs where mapping succeeded for both points
    min_len = min(len(V1), len(V2))
    if min_len < 6:
        raise ValueError("Too few valid 3‑D pairs after projection")
    V1 = V1[:min_len]
    V2 = V2[:min_len]

    # 6. Estimate rotation
    R = _estimate_rotation(V1, V2)
    axis, angle_rad = _rotation_to_axis_angle(R)

    # 7. Spin calculations
    omega_rad_s = angle_rad / delta_t_seconds
    rpm_total = omega_rad_s * 60.0 / (2 * np.pi)

    # Convention: x = backspin axis (pitch), y = sidespin axis (yaw)
    backspin_rpm = rpm_total * axis[0]
    sidespin_rpm = rpm_total * axis[1]
    axis_deg = np.degrees(np.arctan2(abs(axis[0]), abs(axis[1] + 1e-12)))  # elevation

    # 8. Optional debug visualisations
    if debug:
        import os
        os.makedirs(debug_dir, exist_ok=True)
        canny1 = cv2.Canny(cv2.cvtColor(roi1, cv2.COLOR_BGR2GRAY), 50, 150)
        canny2 = cv2.Canny(cv2.cvtColor(roi2, cv2.COLOR_BGR2GRAY), 50, 150)
        cv2.imwrite(f"{debug_dir}/canny1.png", canny1)
        cv2.imwrite(f"{debug_dir}/canny2.png", canny2)

        # Draw matches
        debug_img = cv2.drawMatches(roi1, kp1, roi2, kp2, good, None,
                                    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        cv2.imwrite(f"{debug_dir}/matches.png", debug_img)

    return float(backspin_rpm), float(axis_deg), float(sidespin_rpm)


__all__ = ["calculate_spin"]
if __name__ == "__main__":
    import cv2

    # Paths to the two test images
    img_path1 = "/home/vglalala/GCFive/Images/spin_ball_1_gray_image1.png"
    img_path2 = "/home/vglalala/GCFive/Images/spin_ball_2_gray_image1.png"

    # Load frames
    frame1 = cv2.imread(img_path1)
    frame2 = cv2.imread(img_path2)
    if frame1 is None or frame2 is None:
        raise FileNotFoundError(f"Could not load one of the test images:\n  {img_path1}\n  {img_path2}")

    # Duration between frames in seconds
    delta_t = 1.0 / 3000.0

    # Run spin calculation
    backspin_rpm, axis_deg, sidespin_rpm = calculate_spin(frame1, frame2, delta_t)

    # Display results
    print("=== Spin Calculator Test ===")
    print(f"Backspin   : {backspin_rpm:.2f} RPM")
    print(f"Axis angle : {axis_deg:.2f}°")
    print(f"Sidespin   : {sidespin_rpm:.2f} RPM")
