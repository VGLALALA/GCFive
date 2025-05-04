import cv2
import numpy as np
import os

def convert_to_canny(image_path):
    # Read the image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Apply GaussianBlur to reduce noise and improve edge detection
    blurred = cv2.GaussianBlur(img, (5, 5), 1.4)
    
    # Use Canny edge detection
    edges = cv2.Canny(blurred, 100, 200)
    
    return edges

image_path = "/home/vglalala/GCFive/Images/log_cam2_last_strobed_img.png"
canny_image = convert_to_canny(image_path)
config_path = "/home/vglalala/GCFive/detection.json"
original_img = cv2.imread(image_path)
display_img = original_img.copy()

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
    gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    clarity_score = min(1.0, variance / 100.0)  # Normalize clarity score
    return clarity_score

def run_hough_with_radius(radius):
    blurred = cv2.GaussianBlur(canny_image, (9, 9), 2)
    circles = cv2.HoughCircles(blurred, 
                               cv2.HOUGH_GRADIENT, 
                               dp=1.2, 
                               minDist=40,
                               param1=100, 
                               param2=30, 
                               minRadius=radius - 5, 
                               maxRadius=radius + 5)

    box_coords = []
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        # Sort circles by x-coordinate to label them from left to right
        circles = sorted(circles, key=lambda c: c[0])
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
            cropped_img = original_img[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
            
            # Save the cropped image
            cropped_dir = "/home/vglalala/GCFive/Images/CroppedBalls"
            os.makedirs(cropped_dir, exist_ok=True)
            cropped_path = os.path.join(cropped_dir, f"cropped_circle_{idx + 1}.png")
            cv2.imwrite(cropped_path, cropped_img)
            print(f"Cropped image saved to {cropped_path}")
            
            # Calculate scores
            overlap_score = calculate_overlap_score((x, y, r), [c for i, c in enumerate(circles) if i != idx])
            clarity_score = calculate_clarity_score(cropped_img)
            total_score = (overlap_score + clarity_score) / 2
            print(f"Circle {idx + 1} Score: {total_score:.2f} (Overlap: {overlap_score:.2f}, Clarity: {clarity_score:.2f})")
            
        print(f"Detected {len(circles)} circle(s).")
    else:
        print("No circles detected.")
    
    return box_coords

coords = run_hough_with_radius(50)