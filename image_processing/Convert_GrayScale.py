import cv2


def convert_to_grayscale(image_path):
    """
    Converts an image to grayscale.

    Args:
        image_path: Path to the image file.

    Returns:
        Grayscale image as a numpy array.
    """
    # Read the image
    img = cv2.imread(image_path)

    # Convert the image to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return gray_img
