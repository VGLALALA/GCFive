import cv2


def match_ball_image_sizes(img1, img2, ball1=None, ball2=None):
    """
    Crops or pads two images to the same size, centered on their respective balls if provided.
    Returns the two new images (and optionally the updated ball objects).
    """
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    target_h = max(h1, h2)
    target_w = max(w1, w2)

    def pad_to_shape(img, target_h, target_w):
        h, w = img.shape[:2]
        pad_top = (target_h - h) // 2
        pad_bottom = target_h - h - pad_top
        pad_left = (target_w - w) // 2
        pad_right = target_w - w - pad_left
        return cv2.copyMakeBorder(
            img, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0
        )

    img1_padded = pad_to_shape(img1, target_h, target_w)
    img2_padded = pad_to_shape(img2, target_h, target_w)

    # If you want to update the ball objects' coordinates, do it here
    if ball1 is not None and ball2 is not None:
        ball1.x += (target_w - w1) // 2
        ball1.y += (target_h - h1) // 2
        ball2.x += (target_w - w2) // 2
        ball2.y += (target_h - h2) // 2
        return img1_padded, img2_padded, ball1, ball2
    else:
        return img1_padded, img2_padded
