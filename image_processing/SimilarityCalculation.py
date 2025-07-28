import cv2
import numpy as np


def delta_similarity(f1, f2, down=4, blur=3, px_thresh=6):
    # 1) gray + shrink for speed/noise
    g1 = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(f2, cv2.COLOR_BGR2GRAY)
    if down > 1:
        g1 = cv2.resize(
            g1, (g1.shape[1] // down, g1.shape[0] // down), interpolation=cv2.INTER_AREA
        )
        g2 = cv2.resize(
            g2, (g2.shape[1] // down, g2.shape[0] // down), interpolation=cv2.INTER_AREA
        )
    if blur > 0:
        g1 = cv2.GaussianBlur(g1, (blur, blur), 0)
        g2 = cv2.GaussianBlur(g2, (blur, blur), 0)

    diff = cv2.absdiff(g1, g2).astype(np.float32)
    # ignore tiny flicker
    changed = diff > px_thresh

    delta = diff.mean() / 255.0  # 0..1 motion magnitude
    changed_ratio = changed.mean()  # fraction of pixels over px_thresh
    similarity = 1.0 - delta  # 1==identical

    return similarity, delta, changed_ratio


if __name__ == "__main__":
    img1 = cv2.imread("data/Images/frame_00000.jpg")
    img2 = cv2.imread("data/Images/frame_00001.jpg")
    similarity, delta, changed_ratio = delta_similarity(img1, img2)
    print(f"Similarity: {similarity}, Delta: {delta}, Changed Ratio: {changed_ratio}")
