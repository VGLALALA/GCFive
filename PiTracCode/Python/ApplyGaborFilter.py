import cv2
import numpy as np
import math

# Constants (tweak as needed)
K_GABOR_MAX_WHITE_PERCENT = 44
K_GABOR_MIN_WHITE_PERCENT = 38

def create_gabor_kernel(ks: int, sigma: float, theta_deg: float,
                        lambd: float, gamma: float, psi_deg: float) -> np.ndarray:
    """
    Port of CreateGaborKernel:
      ks    – kernel size
      sigma – filter sigma
      theta – orientation in degrees
      lambd – wavelength of the sinusoidal factor
      gamma – spatial aspect ratio
      psi   – phase offset in degrees
    """
    theta = np.deg2rad(theta_deg)
    psi   = np.deg2rad(psi_deg)
    # Note: OpenCV expects (ks, ks), sigma, theta, lambd, gamma, psi
    return cv2.getGaborKernel((ks, ks), sigma, theta, lambd, gamma, psi, ktype=cv2.CV_32F)


def apply_test_gabor_filter(img_f32: np.ndarray,
                            kernel_size: int, sigma: float,
                            lambd: float, theta_deg: float,
                            psi_deg: float, gamma: float,
                            binary_threshold: float
                           ) -> Tuple[np.ndarray,int]:
    """
    Port of ApplyTestGaborFilter:
      img_f32          – float32 image normalized to [0,1]
      kernel_size      – spatial size of Gabor kernel
      sigma, lambd, theta_deg, psi_deg, gamma – Gabor params
      binary_threshold – threshold *10 for final binarization
    Returns:
      dimple_edges     – binary (0/255) edge image
      white_percent    – percentage of white pixels in that image
    """
    rows, cols = img_f32.shape[:2]
    accum = np.zeros_like(img_f32, dtype=np.float32)
    dest  = np.zeros_like(accum)

    # Sweep orientations
    theta_inc = 11.25
    theta = 0.0
    while theta < 360.0 + 1e-6:
        kernel = create_gabor_kernel(kernel_size, sigma, theta, lambd, gamma, psi_deg)
        cv2.filter2D(img_f32, cv2.CV_32F, kernel, dst=dest)
        np.maximum(accum, dest, out=accum)
        theta += theta_inc

    # Scale to 0–255 and convert to uint8
    accum_gray = cv2.convertScaleAbs(accum, alpha=255.0)

    # Threshold
    edge_lo = int(round(binary_threshold * 10.0))
    _, dimple_edges = cv2.threshold(accum_gray, edge_lo, 255, cv2.THRESH_BINARY)

    # Compute white percentage
    white_pixels = cv2.countNonZero(dimple_edges)
    total_pixels = rows * cols
    white_percent = int(round((white_pixels * 100.0) / total_pixels))

    return dimple_edges, white_percent


def apply_gabor_filter_to_ball(image_gray: np.ndarray,
                               prior_binary_threshold: float = -1.0
                              ) -> Tuple[np.ndarray, float]:
    """
    Port of ApplyGaborFilterToBall:
      image_gray             – uint8 single-channel image
      prior_binary_threshold – if >0, reuse as starting threshold
    Returns:
      dimple_edges           – final binary edge image
      calibrated_threshold   – the binary_threshold used in final pass
    """
    # Convert to float32 [0,1]
    img_f32 = image_gray.astype(np.float32) / 255.0

    # Default Gabor parameters (non-equalized branch)
    kernel_size = 21
    pos_sigma  = 2
    pos_lambda = 6
    pos_gamma  = 4
    pos_th     = 60
    pos_psi    = 27
    binary_threshold = 8.5

    # Override if provided
    if prior_binary_threshold > 0:
        binary_threshold = prior_binary_threshold

    # Compute derived params
    sigma = pos_sigma / 2.0
    lambd = float(pos_lambda)
    theta = float(pos_th) * 2.0
    psi   = float(pos_psi) * 10.0
    gamma = pos_gamma / 20.0

    # First Gabor pass
    dimple_edges, white_percent = apply_test_gabor_filter(
        img_f32, kernel_size, sigma, lambd, theta, psi, gamma, binary_threshold
    )

    # Adjust threshold if too few/too many white pixels
    calibrated_threshold = binary_threshold
    if prior_binary_threshold < 0 and (
        white_percent < K_GABOR_MIN_WHITE_PERCENT
        or white_percent >= K_GABOR_MAX_WHITE_PERCENT
    ):
        ratchet_down = (white_percent < K_GABOR_MIN_WHITE_PERCENT)
        # Loop until within desired white-range or limits reached
        while (white_percent < K_GABOR_MIN_WHITE_PERCENT
               or white_percent >= K_GABOR_MAX_WHITE_PERCENT):
            # Adjust by 1.0 or 0.5 depending on distance
            delta = 1.0 if abs(white_percent - (K_GABOR_MIN_WHITE_PERCENT if ratchet_down else K_GABOR_MAX_WHITE_PERCENT)) > 5 else 0.5
            calibrated_threshold += -delta if ratchet_down else delta

            # Re-run
            dimple_edges, white_percent = apply_test_gabor_filter(
                img_f32, kernel_size, sigma, lambd, theta, psi, gamma, calibrated_threshold
            )

            # Break if threshold out of bounds
            if not (2.0 <= calibrated_threshold <= 30.0):
                print(f"Warning: Gabor binary_threshold reached limit: {calibrated_threshold}")
                break

    return dimple_edges, calibrated_threshold
