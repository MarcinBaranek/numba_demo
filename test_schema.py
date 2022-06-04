import numba
import numba.cuda as cuda
import numpy as np
import pytest
from numba.cuda.random import create_xoroshiro128p_states

from schema import drift_step, diffusion_step, euler_step, euler_path,\
    get_kernel_euler_path, diffusion_step_with_w, euler_step_with_w,\
    strong_euler_path


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


@cuda.jit
def _kernel_euler_path(point, state, result):
    tx = cuda.threadIdx.x
    ty = cuda.blockIdx.x
    bw = cuda.blockDim.x
    pos = tx + ty * bw
    if pos < result.size:
        tmp = cuda.local.array(shape=(3, 1),
                               dtype=result.dtype)
        euler_path(point, np.float32(0.1), np.float32(1.1), 10, a, b, state,
                   tmp)
        for i in range(result.shape[0]):
            result[i, pos] = tmp[i, 0]


def test_euler_path():
    point = np.ones(shape=(3, 1)).astype('float32')
    result = np.zeros(shape=(3, 8)).astype('float32')

    d_point = cuda.to_device(point)
    d_result = cuda.to_device(result)
    state = create_xoroshiro128p_states(8, seed=2)
    _kernel_euler_path[2, 4](d_point, state, d_result)

    d_result.copy_to_host(result)
    exp = np.array([[-2.16325741e+01, -1.53962732e-03, 5.12758270e-02,
                     3.36964059e+00, -5.33286095e-01, -1.14022836e-01,
                     -3.62935219e+01, -1.15846052e+01],
                    [-3.64749603e+01, -6.35675862e-02, -3.69485855e+01,
                     1.10444534e+02, -8.86483002e+00, -3.47404861e+00,
                     -2.25641647e+02, -1.52339411e+01],
                    [3.65963593e+01,  2.22158004e-02, -4.44564819e+02,
                     -1.35478210e+02, -8.81883545e+02, -1.90537682e+01,
                     9.25933304e+01,  3.24392456e+02]]).astype('float32')
    assert result == pytest.approx(exp)


def test_kernel_euler_path():
    @cuda.jit(device=True)
    def a_(t, p, res):
        for i in range(p.shape[0]):
            res[i, 0] = p[i, 0] * 1.1

    @cuda.jit(device=True)
    def b_(t, p, res):
        for i in range(res.shape[0]):
            for j in range(res.shape[1]):
                res[i, j] = p[i, 0] * 0.9

    ths, blocks = 2, 1
    n_paths = ths * blocks
    n_points = 10
    point = np.array([[1], [2], [3]]).astype('float32')
    result = np.zeros(shape=(3 * n_paths, n_points)).astype('float32')
    time = np.reshape(np.linspace(0., 1., n_points).astype('float32'),
                      (-1, 1))
    d_time = cuda.to_device(time)
    d_point = cuda.to_device(point)
    d_result = cuda.to_device(result)
    state = create_xoroshiro128p_states(n_paths, seed=6)
    kernel = get_kernel_euler_path(a_, b_)
    kernel[blocks, ths](d_point, d_time, state, d_result)

    d_result.copy_to_host(result)

    # import matplotlib.pyplot as plt
    # for i in range(3 * n_paths):
    #     plt.plot(result[i, :].T, label=f'dim:{i%3}, path={i //3}')
    # plt.legend()
    # plt.show()

    exp = np.array([
        [1., 1.2369545, 0.48043966, 0.58112085, 0.49400002,
         0.16248724, 0.35357204, 0.4989446, 0.27714694, 0.22419173],
        [2., 2.473909, 0.9608793, 1.1622417, 0.98800004,
         0.32497448, 0.7071441, 0.9978892, 0.5542939, 0.44838345],
        [3., 3.7108636, 1.4413185, 1.7433621, 1.4819996,
         0.48746157, 1.0607157, 1.4968331, 0.8314404, 0.6725749],
        [1., 0.5835043, 0.5893528, 0.7172467, 1.2116382,
         0.5545416, 0.9431203, 0.41237712, 0.16462699, 0.04587987],
        [2., 1.1670086, 1.1787056, 1.4344934, 2.4232764,
         1.1090832, 1.8862406, 0.82475424, 0.32925397, 0.09175974],
        [3., 1.7505128, 1.7680582, 2.1517398, 3.634914,
         1.663624, 2.8293598, 1.237131, 0.49388075, 0.13763952]],
      dtype='float32')
    assert result == pytest.approx(exp)

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
    wiener = np.zeros(shape=(4, 1)).astype('float32')

    d_point = cuda.to_device(point)
    d_wiener = cuda.to_device(wiener)
    state = create_xoroshiro128p_states(1, seed=2)
    kernel_euler_step_with_w[1, 1](d_point, 1, 1.e-2, d_wiener, state)

    d_point.copy_to_host(point)
    d_wiener.copy_to_host(wiener)

    exp = np.array([[0.8221502], [0.64430034], [0.4664505]]).astype('float32')
    for c_, e_ in zip(point, exp):
        assert all(c_ == e_)
    for c_, e_ in zip(wiener,
                      np.array([[-0.15159574], [-0.06456007],
                               [0.08825855], [-0.1335742]]).astype('float32')):
        assert all(c_ == e_)


@cuda.jit(device=True)
def dummy_exact(p, t, w, temp):
    for i in range(p.shape[0]):
        temp[i, 0] = p[i, 0]


@cuda.jit
def kernel_strong_euler_path(point, state, result):
    tx = cuda.threadIdx.x
    ty = cuda.blockIdx.x
    bw = cuda.blockDim.x
    pos = tx + ty * bw
    if pos < result.size:
        result[pos] = strong_euler_path(point, np.float32(0.1),
                                        np.float32(1.1), 10, a,
                                        b, dummy_exact, state)


def test_strong_euler_path():
    point = np.ones(shape=(3, 1)).astype('float32')
    result = np.zeros(shape=(8,)).astype('float32')

    d_point = cuda.to_device(point)
    d_result = cuda.to_device(result)
    state = create_xoroshiro128p_states(8, seed=2)
    kernel_strong_euler_path[2, 4](d_point, state, d_result)

    d_result.copy_to_host(result)

    exp = np.array([5.9028809e+02, 2.6327908e+00, 5.0061813e+02, 6.2225164e+04,
                    7.2004858e+02, 1.3157059e+02, 6.9894859e+04, 8.0639951e+03]
                   ).astype('float32')
    assert result == pytest.approx(exp)
