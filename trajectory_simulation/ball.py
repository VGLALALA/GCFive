from dataclasses import dataclass, field
from typing import Tuple
import numpy as np
from .vector import vec3, length, normalized, cross, dot

@dataclass
class Ball:
    position: np.ndarray = field(default_factory=lambda: vec3())
    velocity: np.ndarray = field(default_factory=lambda: vec3())
    omega: np.ndarray = field(default_factory=lambda: vec3())

    mass: float = 0.04592623  # mass of the ball in kilograms
    radius: float = 0.021335  # radius of the ball in meters
    A: float = field(init=False)  # cross-sectional area
    I: float = field(init=False)  # moment of inertia
    u_k: float = 0.4  # kinetic friction coefficient
    u_kr: float = 0.2  # rolling friction coefficient

    rho: float = 1.225  # air density in kg/m^3
    mu: float = 0.00001802  # dynamic viscosity of air in kg/(m·s)
    nu: float = 0.00001470  # kinematic viscosity of air in m^2/s
    nu_g: float = 0.0012  # kinematic viscosity for ground interaction in m^2/s

    # Coefficients for drag curve
    cd_mid_a: float = 0.000000000129
    cd_mid_b: float = -0.0000225
    cd_mid_c: float = 1.50
    cd_high_a: float = 0.00000000001925
    cd_high_b: float = -0.0000052
    cd_high_c: float = 0.56

    # Tuning factors for aerodynamic model
    drag_scale: float = 1.0
    lift_scale: float = 1.0

    position_list: list = field(default_factory=list)  # list to store position history
    total_position_list: list = field(default_factory=list)

    def __post_init__(self):    
        self.A = np.pi * self.radius ** 2
        self.I = 0.4 * self.mass * self.radius ** 2

    def reset(self):
        self.position[:] = vec3(0.0, 0.1, 0.0)
        self.velocity[:] = vec3()
        self.omega[:] = vec3()
        self.position_list.clear()
        self.total_position_list.clear()

    def hit(self):
        self.position[:] = vec3()
        v = 44.7
        self.velocity[:] = vec3(
            v * np.cos(np.radians(20.8)) * np.cos(np.radians(1.7)),
            v * np.sin(np.radians(20.8)),
            v * np.sin(np.radians(1.7)),
        )
        self.omega[:] = vec3(
            0.0,
            784.0 * np.sin(np.radians(2.7)),
            784.0 * np.cos(np.radians(2.7)),
        )

    def hit_from_data(self, data: dict):
        self.position[:] = vec3(0.0, 0.05, 0.0)
        v = vec3(data["Speed"] * 0.44704, 0.0, 0.0)
        # vertical launch angle
        rot_v = np.radians(data["VLA"])
        rot_h = np.radians(data["HLA"])
        # rotate around z then x
        Rz = rotation_matrix(vec3(0.0, 0.0, 1.0), rot_v)
        Rx = rotation_matrix(vec3(1.0, 0.0, 0.0), rot_h)
        self.velocity[:] = Rx @ (Rz @ v)
        spin = vec3(0.0, 0.0, data["TotalSpin"] * 0.10472)
        axis = rotation_matrix(vec3(1.0, 0.0, 0.0), -np.radians(data["SpinAxis"]))
        self.omega[:] = axis @ spin

    def update(self, delta: float):
        on_ground = self.position[1] < 0.022

        F_g = vec3(0.0, -9.81 * self.mass, 0.0)
        F_m = vec3()
        F_d = vec3()
        F_f = vec3()
        F_gd = vec3()

        T_d = vec3()
        T_f = vec3()
        T_g = vec3()

        if on_ground:
            F_gd = -6 * np.pi * self.radius * self.nu_g * self.velocity
            F_gd[1] = 0.0
            b_vel = cross(vec3(0.0, 1.0, 0.0), self.omega) * self.radius
            b_vel = b_vel + self.velocity
            if length(b_vel) < 0.05:
                b_dir = normalized(self.velocity)
                F_f = -self.u_k * self.mass * 9.81 * b_dir
            else:
                b_dir = normalized(b_vel)
                F_f = -self.u_k * self.mass * 9.81 * b_dir
                T_f = cross(vec3(0.0, -self.radius, 0.0), F_f)
            if length(self.omega) != 0:
                T_g = -6.0 * np.pi * self.radius * self.nu_g * normalized(self.omega)
        else:
            speed = length(self.velocity)
            spin = 0.0
            if speed > 0.5:
                spin = length(self.omega) * self.radius / speed
            Re = self.rho * speed * self.radius * 2.0 / self.mu
            S = 0.5 * self.rho * self.A * self.radius * (-3.25 * spin + 1.99)
            if Re < 50000.0:
                Cd = 0.6
            elif Re < 87500.0:
                Cd = self.cd_mid_a * Re ** 2 + self.cd_mid_b * Re + self.cd_mid_c
            else:
                Cd = self.cd_high_a * Re ** 2 + self.cd_high_b * Re + self.cd_high_c
            Cd *= self.drag_scale
            S *= self.lift_scale
            F_m = cross(self.omega, self.velocity) * S
            T_d = -8.0 * np.pi * self.mu * self.radius ** 3 * self.omega
            F_d = -self.velocity * speed * Cd * self.rho * self.A / 2.0

        F = F_g + F_d + F_m + F_f + F_gd
        T = T_d + T_f + T_g

        self.velocity += (F / self.mass) * delta
        self.omega += (T / self.I) * delta

        # simple ground collision
        next_pos = self.position + self.velocity * delta
        self.total_position_list.append(next_pos.copy())
        if next_pos[1] < 0.0 and self.velocity[1] < 0:
            self.position_list.append(self.position.copy())
            self.velocity = self.bounce(self.velocity, vec3(0.0, 1.0, 0.0))
            next_pos[1] = 0.0
            if abs(self.velocity[1]) < 0.05:
                self.velocity[1] = 0.0
                if length(self.velocity) < 0.1:
                    self.velocity[:] = vec3()
        self.position[:] = next_pos

    def bounce(self, vel, normal):
        vel_norm = project(vel, normal)
        speed_norm = length(vel_norm)
        vel_orth = vel - vel_norm
        speed_orth = length(vel_orth)
        omg_norm = project(self.omega, normal)
        omg_orth = self.omega - omg_norm

        speed = length(self.velocity)
        theta_1 = angle_between(self.velocity, normal)
        theta_c = 15.4 * speed * theta_1 / 18.6 / 44.4
        v2_orth = 5.0/7.0*speed*np.sin(theta_1-theta_c) - 2.0*self.radius*length(omg_norm)/7.0
        if speed_orth < 0.01:
            vel_orth = vec3()
        else:
            vel_orth = limit_length(vel_orth, v2_orth)
        w2h = v2_orth/self.radius
        if length(omg_orth) < 0.1:
            omg_orth = vec3()
        else:
            omg_orth = limit_length(omg_orth, w2h)
        if speed_norm < 20.0:
            e = 0.12
        else:
            e = 0.510 - 0.0375*speed_norm + 0.000903*speed_norm*speed_norm
        vel_norm = vel_norm * -e
        self.omega = omg_norm + omg_orth
        return vel_norm + vel_orth

def project(v, n):
    n_norm = normalized(n)
    return n_norm * dot(v, n_norm)

def angle_between(a, b):
    a_n = normalized(a)
    b_n = normalized(b)
    dot_val = np.clip(dot(a_n, b_n), -1.0, 1.0)
    return np.arccos(dot_val)

def limit_length(v, l):
    cur = length(v)
    if cur == 0:
        return v
    if cur > abs(l):
        return normalized(v) * l
    return v

def rotation_matrix(axis, angle):
    axis = normalized(axis)
    x, y, z = axis
    c = np.cos(angle)
    s = np.sin(angle)
    C = 1 - c
    return np.array([
        [x*x*C + c,   x*y*C - z*s, x*z*C + y*s],
        [y*x*C + z*s, y*y*C + c,   y*z*C - x*s],
        [z*x*C - y*s, z*y*C + x*s, z*z*C + c  ],
    ])
