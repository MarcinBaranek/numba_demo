import numba.cuda as cuda
import math

from core import multiply_matrix_by_scalar, add_inplace, multiply_matrix,\
    gen_normal_float32
from config import D, M


@cuda.jit(device=True)
def drift_step(point, time, dt, a_func, result):
    a_func(time, point, result)
    multiply_matrix_by_scalar(result, dt)


@cuda.jit(device=True)
def diffusion_step(point, time, dt, b_func, result, state):
    temp = cuda.local.array(shape=(D, M), dtype=point.dtype)
    dw = cuda.local.array(shape=(M, 1), dtype=point.dtype)
    gen_normal_float32(dw, state)
    multiply_matrix_by_scalar(dw, math.sqrt(dt))
    b_func(time, point, temp)
    multiply_matrix(temp, dw, result)


@cuda.jit(device=True)
def euler_step(point, time, dt, a_func, b_func, state):
    temp_drift = cuda.local.array(shape=(D, 1), dtype=point.dtype)
    temp_diffusion = cuda.local.array(shape=(D, 1), dtype=point.dtype)
    drift_step(point, time, dt, a_func, temp_drift)
    diffusion_step(point, time, dt, b_func, temp_diffusion, state)
    add_inplace(point, temp_drift)
    add_inplace(point, temp_diffusion)


@cuda.jit(device=True)
def euler_path(initial_point, t_0, end_time, n, a_func, b_func, state):
    dt = (end_time - t_0) / n
    time = t_0
    while time < end_time:
        # override initial_point
        euler_step(initial_point, time, dt, a_func, b_func, state)
        time += dt


# ===============================
# With error
# ===============================

@cuda.jit(device=True)
def diffusion_step_with_w(point, time, dt, b_func, result, wiener, state):
    temp = cuda.local.array(shape=(D, M), dtype=point.dtype)
    gen_normal_float32(wiener, state)
    multiply_matrix_by_scalar(wiener, math.sqrt(dt))
    b_func(time, point, temp)
    multiply_matrix(temp, wiener, result)


@cuda.jit(device=True)
def euler_step_with_w(point, time, dt, a_func, b_func, wiener, state):
    temp_drift = cuda.local.array(shape=(D, 1), dtype=point.dtype)
    temp_diffusion = cuda.local.array(shape=(D, 1), dtype=point.dtype)
    drift_step(point, time, dt, a_func, temp_drift)
    diffusion_step_with_w(point, time, dt, b_func,
                          temp_diffusion, wiener, state)
    point[1, 0] = temp_drift[1, 0]
    point[2, 0] = temp_diffusion[1, 0]
    # add_inplace(point, temp_drift)
    # add_inplace(point, temp_diffusion)


@cuda.jit(device=True)
def sse(array_a, array_b):
    total = 0
    for i in range(array_a.shape[0]):
        for j in range(array_a.shape[1]):
            total += (array_a[i, j] - array_b[i, j])**2
    return total


@cuda.jit(device=True)
def strong_euler_path(initial_point, t_0, end_time, n, a_func,
                      b_func, exact, state):
    wiener = cuda.local.array(shape=(M, 1), dtype=initial_point.dtype)
    dw = cuda.local.array(shape=(M, 1), dtype=initial_point.dtype)
    temp = cuda.local.array(shape=(D, 1), dtype=initial_point.dtype)
    temp_point = cuda.local.array(shape=(D, 1), dtype=initial_point.dtype)

    for i in range(D):
        temp_point[i, 0] = initial_point[i, 0]
        temp[i, 0] = initial_point[i, 0]
    for i in range(M):
        wiener[i, 0] = 0.0
    dt = (end_time - t_0) / n
    time = t_0 + dt
    total_err = 0

    while time < end_time:
        euler_step_with_w(temp_point, time, dt, a_func, b_func, dw, state)
        # add_inplace(wiener, dw)
        # exact(initial_point, time, wiener, temp)
        # total_err += sse(temp, temp_point)
        # time += dt
        initial_point[0, 0] = wiener[0, 0]
        initial_point[1, 0] = wiener[1, 0]
        initial_point[2, 0] = dw[2, 0]
        break
    # return total_err / n

    return wiener[4, 0]

