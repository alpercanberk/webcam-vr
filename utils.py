import math
import numpy as np

def collision_matrix(theta):
    return np.array([[np.cos(theta)**2 - np.sin(theta)**2, 2*np.sin(theta)*np.cos(theta)],
                     [2*np.sin(theta)*np.cos(theta), np.sin(theta)**2 - np.cos(theta)**2]])

def vector_angle(v):
    unit_v = v/np.linalg.norm(v)
    unit_v[1] = -1*unit_v[1]
    angle = np.arctan(unit_v[1]/unit_v[0])
    if(unit_v[0] < 0):
        angle = np.pi + angle
    return angle, np.degrees(angle)

def pnt2line(pnt, start, end):
    start_arr = np.array(start)
    end_arr = np.array(end)
    pnt_arr = np.array(pnt)

    pnt_vec = pnt_arr - start_arr
    line_vec = end_arr - start_arr
    line_len = np.linalg.norm(line_vec)
    line_unitvec = line_vec/(line_len+0.01)

    pnt_vec_scaled = pnt_vec * 1.0/(line_len + 0.01)

    t = np.dot(line_unitvec, pnt_vec_scaled)

    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0

    nearest = line_vec * t
    dist = np.linalg.norm(nearest - pnt_vec)
    nearest = nearest + start_arr
    return dist

def nearest2line(pnt, start, end):
    start_arr = np.array(start)
    end_arr = np.array(end)
    pnt_arr = np.array(pnt)

    pnt_vec = pnt_arr - start_arr
    line_vec = end_arr - start_arr
    line_len = np.linalg.norm(line_vec)
    line_unitvec = line_vec/(line_len+0.01)

    pnt_vec_scaled = pnt_vec * 1.0/(line_len + 0.01)

    t = np.dot(line_unitvec, pnt_vec_scaled)

    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0

    nearest = line_vec * t
    dist = np.linalg.norm(nearest - pnt_vec)
    nearest = nearest + start_arr
    return nearest


def parse_coord(numpy_array):
    return [int(x) for x in numpy_array]


if __name__ == "__main__":
    np.set_printoptions(suppress=True)
    theta = np.pi/2
    m = collision_matrix(theta)
    print("collision matrix for", theta)
    v = [1, 0]
    print(m)
    print(np.matmul(m, v))
