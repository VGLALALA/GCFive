import numpy as np
def unproject_3d_ball_to_2d_image(src3d, ball):
    """
    src3d: output of project_2d_image_to_3d_ball (rows,cols,2)
    ball: used only if you want to re‐mask or postprocess
    returns: 2D np.ndarray (uint8) where each pixel is channel‐1 from src3d
    """
    # simply take the “pixel” channel back out
    dest = src3d[...,1].astype(np.uint8)
    return dest