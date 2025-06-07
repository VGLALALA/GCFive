import argparse
import csv
import os
import datetime
from typing import List, Dict, Optional

import numpy as np
from scipy.optimize import minimize
from trajectory_simulation.ball import Ball


def calculate_descending_angle(positions: np.ndarray, landing_index: int) -> float:
    if landing_index < 2:
        return 0.0
    prev_pos = positions[landing_index - 1]
    prev_prev = positions[landing_index - 2]
    delta_xz = prev_pos[[0, 2]] - prev_prev[[0, 2]]
    delta_y = prev_pos[1] - prev_prev[1]
    angle_rad = np.arctan2(-delta_y, np.linalg.norm(delta_xz))
    return np.degrees(angle_rad)


def simulate_shot(ball: Ball, data: Dict[str, float], dt: float = 0.01, max_t: float = 10.0):
    ball.reset()
    shot_data = {
        "Speed": data["Ball Speed"],
        "VLA": data["VLA"],
        "HLA": 0.0,
        "TotalSpin": data["Spin"],
        "SpinAxis": 0.0,
    }
    ball.hit_from_data(shot_data)
    t = 0.0
    positions = [ball.position.copy()]
    while t < max_t:
        ball.update(dt)
        t += dt
        positions.append(ball.position.copy())
        if ball.position[1] <= 0.0 and t > dt:
            break
    pos_arr = np.array(positions)
    landing_index = next(i for i, p in enumerate(pos_arr) if p[1] <= 0.0)
    carry = np.linalg.norm(pos_arr[landing_index][[0, 2]]) * 1.09361
    apex = pos_arr[:, 1].max() * 3.28084
    land_angle = calculate_descending_angle(pos_arr, landing_index)
    return carry, apex, land_angle


def load_dataset(path: str) -> List[Dict[str, Optional[float]]]:
    shots = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                shot = {
                    "Carry": float(row["Carry"]),
                    "Ball Speed": float(row["Ball Speed"]),
                    "Spin": float(row["Spin"]),
                    "VLA": float(row["VLA"]),
                    "Apex": float(row["Apex"]),
                    "Land Angle": float(row["Land Angle"]) if row.get("Land Angle") not in ("", None) else None,
                }
            except ValueError:
                continue
            shots.append(shot)
    return shots


PARAM_NAMES = [
    "drag_scale", "lift_scale",
    "u_k", "u_kr",
    "rho", "mu", "nu", "nu_g",
    "cd_mid_a", "cd_mid_b", "cd_mid_c",
    "cd_high_a", "cd_high_b", "cd_high_c",
]


def loss(params: np.ndarray, shots: List[Dict[str, Optional[float]]]) -> float:
    (
        drag_scale, lift_scale,
        u_k, u_kr,
        rho, mu, nu, nu_g,
        cd_mid_a, cd_mid_b, cd_mid_c,
        cd_high_a, cd_high_b, cd_high_c,
    ) = params

    ball = Ball(
        drag_scale=drag_scale,
        lift_scale=lift_scale,
        u_k=u_k,
        u_kr=u_kr,
        rho=rho,
        mu=mu,
        nu=nu,
        nu_g=nu_g,
        cd_mid_a=cd_mid_a,
        cd_mid_b=cd_mid_b,
        cd_mid_c=cd_mid_c,
        cd_high_a=cd_high_a,
        cd_high_b=cd_high_b,
        cd_high_c=cd_high_c,
    )
    total = 0.0
    for shot in shots:
        carry, apex, angle = simulate_shot(ball, shot)
        err = (carry - shot["Carry"]) ** 2 + (apex - shot["Apex"]) ** 2
        if shot["Land Angle"] is not None:
            err += (angle - shot["Land Angle"]) ** 2
        total += err
    return total / len(shots)


def main():
    parser = argparse.ArgumentParser(description="Tune physics model parameters using real shots")
    parser.add_argument("csv_file", help="CSV file with shot data")
    parser.add_argument("--log-dir", default="train_logs", help="Directory to store training logs")
    args = parser.parse_args()

    shots = load_dataset(args.csv_file)
    if not shots:
        print("No valid shots found in dataset")
        return

    initial = np.array([
        1.0, 1.0,  # drag_scale, lift_scale
        0.4, 0.2,  # u_k, u_kr
        1.225, 0.00001802, 0.00001470, 0.0012,  # rho, mu, nu, nu_g
        0.000000000129, -0.0000225, 1.50,  # cd_mid
        0.00000000001925, -0.0000052, 0.56,  # cd_high
    ])

    bounds = [
        (0.5, 1.5), (0.5, 1.5),  # scales
        (0.1, 0.6), (0.05, 0.4),  # friction
        (1.0, 1.4), (1e-5, 3e-5), (1e-5, 2e-5), (0.0005, 0.0025),  # air
        (6e-11, 2e-10), (-5e-5, 0.0), (0.5, 2.0),  # cd_mid
        (1e-11, 5e-11), (-1e-4, 0.0), (0.3, 0.8),  # cd_high
    ]

    result = minimize(loss, x0=initial, args=(shots,), method="Powell", bounds=bounds)

    print("Optimization result:", result.x)
    print("Final loss:", result.fun)

    os.makedirs(args.log_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(args.log_dir, f"train_{timestamp}.txt")
    with open(log_path, "w") as f:
        f.write(f"Final loss: {result.fun}\n")
        for name, value in zip(PARAM_NAMES, result.x):
            f.write(f"{name}: {value}\n")
    print("Log saved to", log_path)


if __name__ == "__main__":
    main()
