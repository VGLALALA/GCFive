import numpy as np


def vec3(x=0.0, y=0.0, z=0.0):
    return np.array([x, y, z], dtype=float)


def length(v):
    return float(np.linalg.norm(v))


def normalized(v):
    l = length(v)
    if l == 0:
        return vec3()
    return v / l


def cross(a, b):
    return np.cross(a, b)


def dot(a, b):
    return float(np.dot(a, b))
