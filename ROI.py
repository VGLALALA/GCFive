import cv2
import numpy as np
import os
import json
from GolfBall import GolfBall

def calculate_overlap_score(circle, other_circles):
    x, y, r = circle
    overlap_score = 1.0
    for ox, oy, oradius in other_circles:
        distance = np.sqrt((x - ox) ** 2 + (y - oy) ** 2)
        if distance < r + oradius:
            overlap_score -= 0.5  # Penalize for overlap
    return max(0, overlap_score)

def calculate_clarity_score(cropped_img):
    # Use variance of Laplacian to determine clarity
    if len(cropped_img.shape) == 3:
        gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = cropped_img
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    clarity_score = min(1.0, variance / 100.0)  # Normalize clarity score
    return clarity_score

def run_hough_with_radius(image):
    # image: input image (grayscale or BGR)
    blurred = cv2.GaussianBlur(image, (9, 9), 2)
    circles = cv2.HoughCircles(blurred, 
                               cv2.HOUGH_GRADIENT, 
                               dp=1.2, 
                               minDist=40,
                               param1=100, 
                               param2=30, 
                               minRadius=30, 
                               maxRadius=70)

    box_coords = []
    scores_info = []
    best_balls = []
    display_img = image.copy() if len(image.shape) == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        # Sort circles by x-coordinate to label them from left to right
        circles = sorted(circles, key=lambda c: c[0])
        h, w = image.shape[:2]
        best = sorted(circles, key=lambda c: (c[0] - w//2)**2 + (c[1] - h//2)**2)[:2]
        for idx, (x, y, r) in enumerate(circles):
            # Calculate the top-left and bottom-right points of the bounding box
            top_left = (x - r, y - r)
            bottom_right = (x + r, y + r)
            box_coords.append((top_left, bottom_right))
            # Draw the rectangle around the detected circle
            cv2.rectangle(display_img, top_left, bottom_right, (0, 255, 0), 2)
            # Label the rectangle with its index
            cv2.putText(display_img, str(idx + 1), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            # Print the coordinates of the bounding box
            print(f"Circle {idx + 1}: Top-left {top_left}, Bottom-right {bottom_right}")
            
            # Crop the detected circle from the original image
            cropped_img = image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
            
            # Calculate scores
            overlap_score = calculate_overlap_score((x, y, r), [c for i, c in enumerate(circles) if i != idx])
            clarity_score = calculate_clarity_score(cropped_img)
            total_score = (overlap_score + clarity_score) / 2
            print(f"Circle {idx + 1} Score: {total_score:.2f} (Overlap: {overlap_score:.2f}, Clarity: {clarity_score:.2f})")
            
            # Collect score information
            scores_info.append({
                "circle_index": idx + 1,
                "overlap_score": overlap_score,
                "clarity_score": clarity_score,
                "total_score": total_score,
            })
            # Save the best balls as GolfBall objects (relative to crop)
            for bx, by, br in best:
                if x == bx and y == by and r == br:
                    best_balls.append(GolfBall(x=x, y=y, measured_radius_pixels=r, angles_camera_ortho_perspective=(0.0, 0.0, 0.0)))
                    break
        print(f"Detected {len(circles)} circle(s).")
    else:
        print("No circles detected.")
    
    return best_balls if len(best_balls) == 2 else None, display_img
if __name__ == '__main__':
    test_img_path = "data/Images/12283_2023_442_Fig2_HTML.png"
    test_img = cv2.imread(test_img_path, cv2.IMREAD_GRAYSCALE)
    data, displayimg = run_hough_with_radius(test_img)
    cv2.imshow("test", displayimg)
    cv2.waitKey(0)
