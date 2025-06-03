import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# (Paste the corrected GolfBallFlightModel here, or import it if you placed it in another file.)
class GolfBallFlightModel:
    def __init__(self):
        self.mass = 0.04593        # kg
        self.diameter = 0.04267    # m
        self.radius = self.diameter / 2
        self.area = np.pi * self.radius**2
        
        self.air_density = 1.225   # kg/m³
        self.kinematic_viscosity = 1.004e-6  # m²/s
        self.g = 9.81              # m/s²
        
        # Lift‐coefficient polynomial coefficients
        self.lift_coeffs = {'a': -3.25, 'b': 1.99, 'c': 0.0}

    def compute_reynolds_number(self, V):
        return (V * self.diameter) / self.kinematic_viscosity

    def compute_spin_factor(self, omega, V):
        return 0.0 if V == 0 else (omega * self.radius) / V

    def get_drag_coefficient(self, Re):
        if Re <= 1e5:
            a, b, c = 2.1e-11, -4.2e-6, 0.47
        else:
            a, b, c = 1.8e-12, -1.1e-6, 0.28
        return a * Re**2 + b * Re + c

    def get_lift_coefficient(self, S):
        a, b, c = self.lift_coeffs['a'], self.lift_coeffs['b'], self.lift_coeffs['c']
        return a * S**2 + b * S + c

    def flight_equations(self, t, state, spin_rate_rad_s):
        x, y, vx, vy = state
        v_mag = np.hypot(vx, vy)
        if v_mag == 0:
            return [0, 0, 0, -self.g]
        
        Re = self.compute_reynolds_number(v_mag)
        S  = self.compute_spin_factor(spin_rate_rad_s, v_mag)
        cd = self.get_drag_coefficient(Re)
        cl = self.get_lift_coefficient(S)
        
        q = 0.5 * self.air_density * v_mag**2
        drag_force = cd * self.area * q
        lift_force = cl * self.area * q
        
        drag_x = -drag_force * (vx / v_mag)
        drag_y = -drag_force * (vy / v_mag)
        lift_x = -lift_force * (vy / v_mag)
        lift_y =  lift_force * (vx / v_mag)
        
        ax = (drag_x + lift_x) / self.mass
        ay = (drag_y + lift_y) / self.mass - self.g
        
        return [vx, vy, ax, ay]

    def simulate_flight(self, ball_speed_mps, launch_angle_deg, spin_rate_rpm):
        launch_rad = np.radians(launch_angle_deg)
        spin_rad_s = spin_rate_rpm * 2 * np.pi / 60
        
        vx0 = ball_speed_mps * np.cos(launch_rad)
        vy0 = ball_speed_mps * np.sin(launch_rad)
        initial_state = [0, 0, vx0, vy0]
        
        def ground_contact(t, state, spin_rate_rad_s):
            return state[1]  # y = 0
        ground_contact.terminal = True
        ground_contact.direction = -1
        
        t_span = (0, 20)
        sol = solve_ivp(
            self.flight_equations,
            t_span,
            initial_state,
            args=(spin_rad_s,),
            events=ground_contact,
            dense_output=True,
            rtol=1e-8,
            atol=1e-10
        )
        
        if sol.t_events[0].size > 0:
            t_ground = sol.t_events[0][0]
            x_ground = sol.sol(t_ground)[0]
        else:
            t_ground = sol.t[-1]
            x_ground = sol.y[0, -1]
        
        t_detailed = np.linspace(0, t_ground, 500)
        traj = sol.sol(t_detailed)
        
        max_h_idx = np.argmax(traj[1])
        max_height = traj[1][max_h_idx]
        peak_vel  = np.hypot(traj[2, max_h_idx], traj[3, max_h_idx])
        
        rep_Re = self.compute_reynolds_number(peak_vel)
        rep_S  = self.compute_spin_factor(spin_rad_s, peak_vel)
        rep_cd = self.get_drag_coefficient(rep_Re)
        rep_cl = self.get_lift_coefficient(rep_S)
        
        return {
            'time_of_flight'      : t_ground,
            'carry_distance_m'    : x_ground,
            'carry_distance_yards': x_ground / 0.9144,
            'max_height_m'        : max_height,
            'max_height_time'     : t_detailed[max_h_idx],
            'rep_reynolds_number' : rep_Re,
            'rep_spin_factor'     : rep_S,
            'rep_drag_coefficient': rep_cd,
            'rep_lift_coefficient': rep_cl,
            'trajectory_x'        : traj[0],
            'trajectory_y'        : traj[1]
        }


# -------------- USAGE EXAMPLE --------------

if __name__ == "__main__":
    # 1) Create the model object
    model = GolfBallFlightModel()
    
    # 2) Provide shot parameters:
    #    - ball_speed_mps: initial speed in meters/second
    #    - launch_angle_deg: launch angle in degrees
    #    - spin_rate_rpm: backspin in rpm
    #
    # Example: 150 mph ≈ 150 * 0.44704 m/s, launch 14°, spin 2500 rpm
    ball_speed_mps = 150 * 0.44704
    launch_angle_deg = 14
    spin_rate_rpm = 2500
    
    # 3) Run the simulation
    results = model.simulate_flight(ball_speed_mps, launch_angle_deg, spin_rate_rpm)
    
    # 4) Print numerical outputs
    print(f"Time of Flight:        {results['time_of_flight']:.2f} s")
    print(f"Carry Distance:        {results['carry_distance_m']:.1f} m  "
          f"({results['carry_distance_yards']:.1f} yd)")
    print(f"Maximum Height:        {results['max_height_m']:.1f} m  "
          f"at t = {results['max_height_time']:.2f} s")
    print(f"Re (peak):            {results['rep_reynolds_number']:.0f}")
    print(f"S (peak spin factor): {results['rep_spin_factor']:.3f}")
    print(f"Cd (peak):            {results['rep_drag_coefficient']:.3f}")
    print(f"Cl (peak):            {results['rep_lift_coefficient']:.3f}")
    
    # 5) (Optional) Plot the trajectory
    x = results['trajectory_x']
    y = results['trajectory_y']
    
    plt.figure(figsize=(8, 6))
    plt.plot(x, y, 'b-', linewidth=2)
    plt.xlabel("Horizontal Distance (m)")
    plt.ylabel("Height (m)")
    plt.title(f"Trajectory: {launch_angle_deg}°, {spin_rate_rpm} rpm Backspin")
    plt.grid(True, alpha=0.3)
    
    # Add yard‐markers every 50 yd
    yard_to_m = 0.9144
    x_max = x.max()
    for yard in range(50, int(x_max / yard_to_m) + 50, 50):
        xm = yard * yard_to_m
        if xm <= x_max:
            plt.axvline(x=xm, color='gray', linestyle='--', alpha=0.4)
            plt.text(xm, y.max() * 0.1, f"{yard} yd", ha='center', fontsize=8, color='gray')
    
    plt.tight_layout()
    plt.show()
