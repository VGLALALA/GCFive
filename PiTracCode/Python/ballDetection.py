import cv2
import numpy as np
import json
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
        print(f"Detected {len(circles)} circle(s).")
    else:
        print("No circles detected.")
    
    return box_coords

def click_event(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        if len(points) == 2:
            cv2.circle(display_img, points[0], 3, (0, 255, 0), -1)
            cv2.circle(display_img, points[1], 3, (0, 255, 0), -1)
            cv2.line(display_img, points[0], points[1], (255, 0, 0), 2)
            radius = int(np.linalg.norm(np.array(points[0]) - np.array(points[1])) / 2)
            print(f"Estimated radius: {radius} px")

            # Save radius to config
            with open(config_path, 'w') as f:
                json.dump({"radius": radius}, f, indent=4)

            box_coords = run_hough_with_radius(radius)
            print("Bounding box coordinates:", box_coords)
            cv2.imshow("Manual + Hough Detection", display_img)

# --- Main Entry ---
print("Choose detection mode:\n1. Manual set with click\n2. Load radius from config (detection.json)")
mode = input("Mode [1/2]: ").strip()

if mode == "1":
    print("Click two points across a golf ball to set diameter.")
    cv2.imshow("Manual + Hough Detection", display_img)
    cv2.setMouseCallback("Manual + Hough Detection", click_event)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

elif mode == "2":
    if not os.path.exists(config_path):
        print("Config file detection.json not found.")
    else:
        with open(config_path, 'r') as f:
            config = json.load(f)
            radius = config.get("radius")
            if radius:
                print(f"Using radius from config: {radius}px")
                box_coords = run_hough_with_radius(radius)
                print("Bounding box coordinates:", box_coords)
                cv2.imshow("Config-based Hough Detection", display_img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            else:
                print("No 'radius' key found in config file.")

else:
    print("Invalid option. Please choose 1 or 2.")
