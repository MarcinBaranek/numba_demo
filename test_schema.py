import numba.cuda as cuda
import numpy as np
import pytest
from numba.cuda.random import create_xoroshiro128p_states

from schema import drift_step, diffusion_step, euler_step,\
    diffusion_step_with_w, euler_step_with_w


@cuda.jit(device=True)
def a(t, p, res):
    for i in range(p.shape[0]):
        res[i, 0] = p[i, 0] * (i + 1) * t


@cuda.jit
def kernel_drift_step(point, time, dt, result):
    drift_step(point, time, dt, a, result)


def test_drift_step():
    point = np.ones(shape=(3, 1)).astype('float32')
    result = np.zeros(shape=(3, 1)).astype('float32')

    d_point = cuda.to_device(point)
    d_result = cuda.to_device(result)

    kernel_drift_step[1, 1](d_point, 1, 1.e-2, d_result)

    d_result.copy_to_host(result)
    exp = np.array([[0.01], [0.02], [0.03]]).astype('float32')
    for c_, e_ in zip(result, exp):
        assert all(c_ == e_)


@cuda.jit(device=True)
def b(t, p, res):
    for i in range(res.shape[0]):
        for j in range(res.shape[1]):
            res[i, j] = p[i, 0] * (i + 1) / (t + j)


@cuda.jit
def kernel_diffusion_step(point, time, dt, result, state):
    diffusion_step(point, time, dt, b, result, state)


def test_diffusion_step():
    point = np.ones(shape=(3, 1)).astype('float32')
    result = np.zeros(shape=(3, 1)).astype('float32')

    d_point = cuda.to_device(point)
    d_result = cuda.to_device(result)
    state = create_xoroshiro128p_states(1, seed=2)
    kernel_diffusion_step[1, 1](d_point, 1, 1.e-2, d_result, state)

    d_result.copy_to_host(result)
    exp = np.array([[-0.1878498],
                    [-0.3756996],
                    [-0.56354946]]).astype('float32')
    for c_, e_ in zip(result, exp):
        assert all(c_ == e_)


@cuda.jit
def kernel_euler_step(point, time, dt, state):
    euler_step(point, time, dt, a, b, state)


def test_euler_step():
    point = np.ones(shape=(3, 1)).astype('float32')

    d_point = cuda.to_device(point)
    state = create_xoroshiro128p_states(1, seed=2)
    kernel_euler_step[1, 1](d_point, 1, 1.e-2, state)

    d_point.copy_to_host(point)
    exp = np.array([[0.8221502],
                    [0.64430034],
                    [0.4664505]]).astype('float32')
    for c_, e_ in zip(point, exp):
        assert all(c_ == e_)

# ==============
# With error
# ==============


@cuda.jit
def kernel_diffusion_step_with_w(point, time, dt, result, wiener, state):
    diffusion_step_with_w(point, time, dt, b, result, wiener, state)


def test_diffusion_step_with_w():
    point = np.ones(shape=(3, 1)).astype('float32')
    result = np.zeros(shape=(3, 1)).astype('float32')
    wiener = np.zeros(shape=(4, 1)).astype('float32')

    d_point = cuda.to_device(point)
    d_wiener = cuda.to_device(wiener)
    d_result = cuda.to_device(result)
    state = create_xoroshiro128p_states(1, seed=2)
    kernel_diffusion_step_with_w[1, 1](d_point, 1, 1.e-2, d_result,
                                       d_wiener, state)

    d_result.copy_to_host(result)
    d_wiener.copy_to_host(wiener)

    exp = np.array([[-0.1878498],
                    [-0.3756996],
                    [-0.56354946]]).astype('float32')
    for c_, e_ in zip(result, exp):
        assert all(c_ == e_)
    for c_, e_ in zip(wiener,
                      np.array([[-0.15159574], [-0.06456007],
                               [0.08825855], [-0.1335742]]).astype('float32')):
        assert all(c_ == e_)


@cuda.jit
def kernel_euler_step_with_w(point, time, dt, wiener, state):
    euler_step_with_w(point, time, dt, a, b, wiener, state)


def test_euler_step_with_w():
    point = np.ones(shape=(3, 1)).astype('float32')
    result = np.zeros(shape=(3, 1)).astype('float32')
    wiener = np.zeros(shape=(4, 1)).astype('float32')

    d_point = cuda.to_device(point)
    d_wiener = cuda.to_device(wiener)
    d_result = cuda.to_device(result)
    state = create_xoroshiro128p_states(1, seed=2)
    kernel_euler_step_with_w[1, 1](d_point, 1, 1.e-2, d_wiener, state)

    d_result.copy_to_host(result)
    d_wiener.copy_to_host(wiener)

    print(result)
    print(wiener)
    # exp = np.array([[-0.1878498],
    #                 [-0.3756996],
    #                 [-0.56354946]]).astype('float32')
    # for c_, e_ in zip(result, exp):
    #     assert all(c_ == e_)
    # for c_, e_ in zip(wiener,
    #                   np.array([[-0.15159574], [-0.06456007],
    #                            [0.08825855], [-0.1335742]]).astype('float32')):
    #     assert all(c_ == e_)
