import cv2
import numpy as np
import json
import os
from Convert_Canny import convert_to_canny
points = []
def auto_determine_circle_radius(image_path):
    """
    Automatically determines the radius of a circle in the image using Hough Circle Transform.

    Args:
        image_path: Path to the image file.

    Returns:
        The estimated radius of the detected circle.
    """
    # Read the image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Apply GaussianBlur to reduce noise and improve edge detection
    blurred = cv2.GaussianBlur(img, (9, 9), 2)
    
    # Use Hough Circle Transform to detect circles
    circles = cv2.HoughCircles(blurred, 
                               cv2.HOUGH_GRADIENT, 
                               dp=1.2, 
                               minDist=40,
                               param1=100, 
                               param2=30, 
                               minRadius=0, 
                               maxRadius=0)

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        # Assuming the first detected circle is the desired one
        x, y, r = circles[0]
        return r
    else:
        raise ValueError("No circle detected in the image.")
    
def run_hough_with_radius(canny_image, radius):
    blurred = cv2.GaussianBlur(canny_image, (9, 9), 2)
    circles = cv2.HoughCircles(blurred, 
                               cv2.HOUGH_GRADIENT, 
                               dp=1.2, 
                               minDist=40,
                               param1=100, 
                               param2=30, 
                               minRadius=radius - 5, 
                               maxRadius=radius + 5)

    circle_data = []
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        # Sort circles by x-coordinate to label them from left to right
        circles = sorted(circles, key=lambda c: c[0])
        for idx, (x, y, r) in enumerate(circles):
            # Calculate the top-left and bottom-right points of the bounding box
            top_left = (x - r, y - r)
            bottom_right = (x + r, y + r)
            # Append circle data
            circle_data.append((x, y, r))
        print(f"Detected {len(circles)} circle(s).")
    else:
        print("No circles detected.")
    
    return circle_data

def click_event(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        if len(points) == 2:
            radius = int(np.linalg.norm(np.array(points[0]) - np.array(points[1])) / 2)
            print(f"Estimated radius: {radius} px")

            # Save radius to config
            with open(config_path, 'w') as f:
                json.dump({"radius": radius}, f, indent=4)

            box_coords = run_hough_with_radius(radius)
            print("Bounding box coordinates:", box_coords)
