import cv2
import numpy as np

# Load the Canny edge image
img = cv2.imread("/home/vglalala/GCFive/Images/canny.png", cv2.IMREAD_GRAYSCALE)

# Find contours
contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

output = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)  # For drawing

for cnt in contours:
    area = cv2.contourArea(cnt)
    if area < 50 or area > 3000:
        continue  # Skip tiny or huge blobs

    perimeter = cv2.arcLength(cnt, True)
    circularity = 4 * np.pi * area / (perimeter ** 2 + 1e-6)  # Avoid division by zero

    if circularity > 0.5:  # 1.0 is a perfect circle
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        if 5 < radius < 50:  # Approximate radius of a golf ball in image
            cv2.circle(output, (int(x), int(y)), int(radius), (0, 255, 0), 2)

cv2.imwrite("golf_ball_detected.png", output)
print("Detection done. Saved as golf_ball_detected.png")
