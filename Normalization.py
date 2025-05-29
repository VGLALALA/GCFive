    # print("step 11")
    #     # Convert the best rotation angles from degrees to radians
    # spin_offset_angle_radians_x = np.radians(best_rot_x)
    # spin_offset_angle_radians_y = np.radians(best_rot_y)
    # spin_offset_angle_radians_z = np.radians(best_rot_z)

    # # Perform the normalization to the real-world axes
    # normalized_rot_x = int(round(best_rot_x * np.cos(spin_offset_angle_radians_y) + best_rot_z * np.sin(spin_offset_angle_radians_y)))
    # normalized_rot_y = int(round(best_rot_y * np.cos(spin_offset_angle_radians_x) - best_rot_z * np.sin(spin_offset_angle_radians_x)))
    # normalized_rot_z = int(round(best_rot_z * np.cos(spin_offset_angle_radians_x) * np.cos(spin_offset_angle_radians_y)))
    # normalized_rot_z -= int(round(best_rot_y * np.sin(spin_offset_angle_radians_x)))
    # normalized_rot_z -= int(round(best_rot_x * np.sin(spin_offset_angle_radians_y)))

    # # Looks like golf folks consider the X (side) spin to be positive if the surface is
    # # going from right to left. So we negate it here.
    # normalized_rot_x = -normalized_rot_x

    # print("step 12")
    # result_bball2d_image = get_rotated_image(
    #     ball_image1,
    #     best_ball1,
    #     (normalized_rot_x,normalized_rot_y,-normalized_rot_z)
    # )

    # cv2.imshow("Actual", ball_image2)
    # cv2.imshow("Final rotated-by-best-angle originalBall1", result_bball2d_image)
    # cv2.waitKey(0)
