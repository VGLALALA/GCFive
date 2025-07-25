import cv2
import numpy as np

# Gabor white‐pixel percentage limits
K_GABOR_MIN_WHITE_PERCENT = 38
K_GABOR_MAX_WHITE_PERCENT = 44

def create_gabor_kernel(ks: int,
                        sigma: float,
                        theta_deg: float,
                        lambd: float,
                        gamma: float,
                        psi_deg: float) -> np.ndarray:
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
                            kernel_size: int,
                            sig: float,
                            lm: float,
                            th: float,
                            ps: float,
                            gm: float,
                            binary_threshold: float) -> (np.ndarray, int):
    """
    Apply a bank of rotated Gabor filters, threshold, and compute white‐pixel percentage.
    Returns (binary_edge_image, white_percent).
    """
    h, w = img_f32.shape[:2]
    accum = np.zeros_like(img_f32, dtype=np.float32)
    theta_increment = 11.25

    for theta in np.arange(0, 360.0 + 1e-6, theta_increment):
        kernel = create_gabor_kernel(kernel_size, sig, theta, lm, gm, ps)
        dest = cv2.filter2D(img_f32, cv2.CV_32F, kernel)
        np.maximum(accum, dest, out=accum)

    # Convert accum to 8-bit
    accum_gray = np.clip(accum * 255.0, 0, 255).astype(np.uint8)
    edge_threshold_low = int(round(binary_threshold * 10.0))
    _, dimple_edges = cv2.threshold(
        accum_gray, edge_threshold_low, 255, cv2.THRESH_BINARY
    )

    # Compute white‐pixel percentage
    white_percent = int(round(
        100.0 * cv2.countNonZero(dimple_edges) / (h * w)
    ))
    return dimple_edges, white_percent


def apply_gabor_filter_image(image_gray: np.ndarray,
                               prior_binary_threshold: float = -1.0) -> (np.ndarray, float):
    """
    Mimics ApplyGaborFilterToBall. Returns (edge_image, calibrated_threshold).
    """
    # ensure grayscale
    assert image_gray.ndim == 2 and image_gray.dtype == np.uint8

    # normalize to [0,1]
    img_f32 = image_gray.astype(np.float32) / 255.0

    # default "pos_" parameters (tweak as desired)
    kernel_size = 21
    pos_sigma  = 2
    pos_lambda = 6
    pos_gamma  = 4
    pos_th     = 60
    pos_psi    = 27
    binary_threshold = 3

    if prior_binary_threshold > 0:
        binary_threshold = prior_binary_threshold

    sig = pos_sigma / 2.0
    lm  = float(pos_lambda)
    th  = float(pos_th) * 2.0
    ps  = float(pos_psi) * 10.0
    gm  = pos_gamma / 20.0

    # first pass
    dimple_img, white_percent = apply_test_gabor_filter(
        img_f32, kernel_size, sig, lm, th, ps, gm, binary_threshold
    )

    # if out of bounds and no prior override, adjust threshold
    if prior_binary_threshold < 0 and (
        white_percent < K_GABOR_MIN_WHITE_PERCENT
        or white_percent >= K_GABOR_MAX_WHITE_PERCENT
    ):
        ratchet_down = (white_percent < K_GABOR_MIN_WHITE_PERCENT)

        while (
            white_percent < K_GABOR_MIN_WHITE_PERCENT
            or white_percent >= K_GABOR_MAX_WHITE_PERCENT
        ):
            # step size
            delta = 1.0 if abs((ratchet_down and K_GABOR_MIN_WHITE_PERCENT - white_percent) or
                               (not ratchet_down and white_percent - K_GABOR_MAX_WHITE_PERCENT)) > 5 else 0.5
            binary_threshold += -delta if ratchet_down else +delta

            # clamp to avoid infinite loop
            if not (2.0 <= binary_threshold <= 30.0):
                break

            dimple_img, white_percent = apply_test_gabor_filter(
                img_f32, kernel_size, sig, lm, th, ps, gm, binary_threshold
            )

        calibrated_binary_threshold = binary_threshold
    else:
        calibrated_binary_threshold = binary_threshold

    return dimple_img, calibrated_binary_threshold
