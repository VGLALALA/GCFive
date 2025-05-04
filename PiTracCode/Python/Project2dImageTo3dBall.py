import numpy as np
from ProjectOp import ProjectionOp

K_PIXEL_IGNORE_VALUE = 128
def project_2d_image_to_3d_ball(image_gray, ball, rotation_angles_degrees):
    """
    image_gray: 2D np.ndarray (dtype=uint8) with values 0,255 or K_PIXEL_IGNORE_VALUE
    ball: GolfBall‐like object (see ProjectionOp.__init__ doc)
    rotation_angles_degrees: (angle_x, angle_y, angle_z)
    returns: 3D image as np.ndarray shape (rows,cols,2), dtype=int32
             channel 0 = depth (Z), channel 1 = pixel or ignore flag
    """
    rows, cols = image_gray.shape
    proj = np.zeros((rows, cols, 2), dtype=np.int32)
    proj[...,1] = K_PIXEL_IGNORE_VALUE

    op = ProjectionOp(ball, proj, rotation_angles_degrees)
    for row in range(rows):
        for col in range(cols):
            pv = int(image_gray[row, col])
            op.project_pixel(col, row, pv)

    return proj




