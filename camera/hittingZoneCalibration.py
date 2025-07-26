import cv2
import numpy as np
import json
from camera.MVSCamera import MVSCamera
from image_processing.get2Dcoord import get_ball_xz
from image_processing.CamBalldistancePred import ROI_W, ROI_H, ROI_X, ROI_Y, EXPOSURE_US
import math
import matplotlib.pyplot as plt

def order_points_ccw(points):
    """
    Order a list of (x, z) points in counter-clockwise order around their centroid.
    """
    cx = sum(p[0] for p in points) / len(points)
    cz = sum(p[1] for p in points) / len(points)
    return sorted(points, key=lambda p: math.atan2(p[1] - cz, p[0] - cx))

def plot_ordered_quadrilateral(points, filename='calibrated_hitting_zone.png'):
    """
    Given a list of [x, z] points, orders them CCW and saves the closed quadrilateral plot.
    
    Parameters:
    -----------
    points : list of [x, z]
        The captured points to connect.
    filename : str
        The filename to save the plot as.
    """
    # Convert to tuples
    pts = [tuple(pt) for pt in points]
    # Order and close
    ordered = order_points_ccw(pts)
    polygon = ordered + [ordered[0]]
    
    # Extract coordinates
    xs, zs = zip(*polygon)
    
    # Plot
    plt.figure()
    plt.plot(xs, zs)
    plt.scatter(xs, zs)
    plt.xlabel('X (mm)')
    plt.ylabel('Z (mm)')
    plt.title('Ordered Hitting Zone Quadrilateral')
    plt.axis('equal')
    plt.savefig(filename)
    plt.close()

# Example usage:
# captured_points = [[-96.8, 464.2], [-10.3, 448.2], [-115.8, 553.1], [-1.3, 541.6]]
# plot_ordered_quadrilateral(captured_points)

def calibrate_hitting_zone_stream(num_points=4, calibration_file='hitting_zone_calibration.json'):
    """
    Opens a live camera stream, lets you place the ball at each of `num_points`
    calibration positions. Press Enter to capture each point, 'q' to quit early.
    Saves the raw calibration data to a file.
    Returns:
      samples: list of (x_mm, z_mm) coordinates
    """
    cam = MVSCamera(ROI_W, ROI_H, ROI_X, ROI_Y, EXPOSURE_US, 1000, 0.25)
    cam.open()
    cv2.namedWindow('Calibration Stream', cv2.WINDOW_NORMAL)

    samples = []
    point_idx = 0
    print(f"\n--- Hitting Zone Calibration ({num_points} points) ---")
    print("Place ball in view. Press Enter to capture each point, 'q' to quit.")

    while point_idx < num_points:
        frame = cam.grab()
        # Convert frame to 3 channels if it has only 1 channel
        if len(frame.shape) == 2 or frame.shape[2] == 1:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        # Annotate live feed
        disp = frame.copy()
        cv2.putText(disp,
                    f'Point {point_idx+1}/{num_points}: Enter to capture',
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        cv2.imshow('Calibration Stream', disp)

        key = cv2.waitKey(1) & 0xFF
        if key in (13, 10):  # Enter key
            coords = get_ball_xz(frame)
            if coords is None:
                print("⚠ No ball detected, reposition and try again.")
                continue
            x_mm, z_mm = coords
            print(f"Captured point {point_idx+1}: x={x_mm:.1f} mm, z={z_mm:.1f} mm")
            samples.append((x_mm, z_mm))
            point_idx += 1

        elif key == ord('q'):
            print("Calibration aborted by user.")
            break

    cam.close()
    cv2.destroyAllWindows()

    if len(samples) < 2:
        print("Not enough points captured for calibration.")
        return None

    # Save raw calibration data to a file
    calibration_data = {
        "samples": samples
    }
    with open(calibration_file, 'w') as f:
        json.dump(calibration_data, f, indent=4)
    print(f"Calibration data saved to {calibration_file}")

    return samples

if __name__ == "__main__":
    samples = calibrate_hitting_zone_stream()
    plot_ordered_quadrilateral(samples)
    print("\nCalibration result:")
    print("Samples:", samples)
