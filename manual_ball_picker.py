import cv2
import numpy as np

image_path = r"C:\Users\theka\Downloads\GCFive\Images\gs_log_img__log_ball_final_found_ball_img.png"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
clone = image.copy()
points = []

cv2.namedWindow("Select Ball Center and Edge", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Select Ball Center and Edge", image.shape[1], image.shape[0])

def click_event(event, x, y, flags, param):
    global points, clone
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Clicked: ({x}, {y})")
        points.append((x, y))
        cv2.circle(clone, (x, y), 3, (255, 0, 0), -1)
        cv2.imshow("Select Ball Center and Edge", clone)
        if len(points) == 2:
            # Draw line between points
            cv2.line(clone, points[0], points[1], (255, 0, 0), 1)
            # Calculate radius
            center = points[0]
            edge = points[1]
            radius = int(np.linalg.norm(np.array(center) - np.array(edge)))
            print(f"Ball center: {center}, radius: {radius}")
            print("You can use these values in your GolfBall object.")
            # Draw the computed circle for feedback
            cv2.circle(clone, center, radius, (255, 255, 255), 2)
            cv2.imshow("Select Ball Center and Edge", clone)

cv2.imshow("Select Ball Center and Edge", clone)
cv2.setMouseCallback("Select Ball Center and Edge", click_event)
cv2.waitKey(0)
cv2.destroyAllWindows()