import math

import numba.cuda as cuda
from numba.cuda.random import xoroshiro128p_uniform_float32

from core import multiply_matrix, add_inplace
from config import M, D


@cuda.jit(device=True)
def a(t, x, result):
    # mu = [0.5, 0.7, 1.2]
    # for i in range(D):
    #     result[i, 0] = x[i, 0] * mu[i]
    result[0, 0] = x[0, 0] * 0.5
    result[1, 0] = x[1, 0] * 0.7
    result[2, 0] = x[2, 0] * 1.2


@cuda.jit(device=True)
def pa(t, x, delta, result, state):
    thread_id = cuda.grid(1)
    a(t, x, result)
    for i in range(D):
        result[i, 0] +=\
            delta * xoroshiro128p_uniform_float32(state, thread_id)


@cuda.jit(device=True)
def b(t, x, result):
    # sigma = [
    #     [0.5, 0.7, 0.2, 0.1],
    #     [-0.5, -0.7, -0.2, -0.1],
    #     [-0.5, -0.7, -0.2, 0.3]
    # ]
    # for i in range(D):
    #     for j in range(M):
    #         result[i, j] = x[i, j] * sigma[i][j]
    result[0, 0] = x[0, 0] * 0.5
    result[0, 1] = x[0, 1] * 0.7
    result[0, 2] = x[0, 2] * 0.2
    result[0, 3] = x[0, 3] * 0.1
    result[1, 0] = x[1, 0] * (-0.5)
    result[1, 1] = x[1, 1] * (-0.7)
    result[1, 2] = x[1, 2] * (-0.2)
    result[1, 3] = x[1, 3] * (-0.1)
    result[2, 0] = x[2, 0] * (-0.5)
    result[2, 1] = x[2, 1] * (-0.7)
    result[2, 2] = x[2, 3] * (-0.2)
    result[2, 3] = x[2, 3] * 0.3



@cuda.jit(device=True)
def pb(t, x, delta, result, state):
    thread_id = cuda.grid(1)
    b(t, x, result)
    for i in range(D):
        for j in range(M):
            result[i, j] +=\
                delta * xoroshiro128p_uniform_float32(state, thread_id)


@cuda.jit(device=True)
def exact(initial_point, t, wiener, result):
    temp = cuda.local.array(shape=(D, M), dtype=initial_point.dtype)
    b(t, initial_point, temp)
    for i in range(D):
        result[i, 0] = 1
        for j in range(M):
            temp[i, j] *= temp[i, j]
    a(t, initial_point, result)
    for i in range(D):
        total = 0
        for j in range(M):
            total += temp[i, j]
        result[i, 0] -= total / 2
        result[i, 0] *= t
    temp_point = cuda.local.array(shape=(D, 1), dtype=initial_point.dtype)
    for i in range(D):
        for j in range(M):
            temp_point[i, j] = 1
    b(t, initial_point, temp)
    multiply_matrix(temp, wiener, temp_point)
    add_inplace(result, temp_point)
    for i in range(D):
        result[i, 0] = initial_point[i, 0] * math.exp(result[i, 0])
