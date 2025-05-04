import cv2
def convert_to_canny(image_path):
    # Read the image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Apply GaussianBlur to reduce noise and improve edge detection
    blurred = cv2.GaussianBlur(img, (5, 5), 1.4)
    
    # Use Canny edge detection
    edges = cv2.Canny(blurred, 100, 200)
    
    return edges