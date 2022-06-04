import numpy as np
import numba.cuda as cuda
from numba.cuda.random import create_xoroshiro128p_states
import pytest
from collections.abc import Callable

from core import add, add_inplace, multiply_matrix, multiply_matrix_by_scalar,\
    norm, gen_normal_float32

np.random.seed(123)


@cuda.jit
def kernel_add(a, b, c):
    add(a, b, c)


@cuda.jit
def kernel_add_inplace(a, b):
    add_inplace(a, b)


@cuda.jit
def kernel_multiply_matrix(a, b, c):
    multiply_matrix(a, b, c)


@cuda.jit
def kernel_multiply_matrix_by_scalar(a, b):
    multiply_matrix_by_scalar(a, b)


@pytest.mark.parametrize('kernel, args, exp, res_idx', [
    (
            kernel_add,
            (
                    np.random.randn(5, 3).astype('float32'),
                    np.random.randn(5, 3).astype('float32'),
                    np.random.randn(5, 3).astype('float32')
            ),
            lambda a, b, c: a + b,
            2
    ),
    (
            kernel_add_inplace,
            (
                    np.random.randn(5, 3).astype('float32'),
                    np.random.randn(5, 3).astype('float32')
            ),
            lambda a, b: a + b,
            0
    ),
    (
            kernel_multiply_matrix,
            (
                    np.random.randn(5, 3).astype('float32'),
                    np.random.randn(3, 8).astype('float32'),
                    np.random.randn(5, 8).astype('float32')
            ),
            lambda a, b, c: a @ b,
            2
    ),
])
def test_common(kernel, args, exp, res_idx):
    if isinstance(exp, Callable):
        exp = exp(*args)
    d_args = tuple(cuda.to_device(arg) for arg in args)
    kernel[1, 1](*d_args)
    d_args[res_idx].copy_to_host(args[res_idx])

    for c_, e_ in zip(args[res_idx], exp):
        assert c_ == pytest.approx(e_)


def test_multiply_matrix_by_scalar():
    @cuda.jit
    def kernel(a, b):
        multiply_matrix_by_scalar(a, b)

    a = np.random.randn(5, 3).astype('float32')

    exp = 3. * a

    d_a = cuda.to_device(a)

    kernel[1, 1](d_a, np.float32(3))

    d_a.copy_to_host(a)

    for c_, e_ in zip(a, exp):
        assert all(c_ == e_)


def test_norm():
    @cuda.jit
    def kernel(a):
        a[0] = norm(a)

    a = np.random.randn(5, 1).astype('float32')

    exp = np.sqrt((a**2).sum())

    d_a = cuda.to_device(a)

    kernel[1, 1](d_a)

    d_a.copy_to_host(a)

    assert a[0] == exp


def test_gen_normal_float32():
    @cuda.jit
    def kernel(a, st):
        gen_normal_float32(a, st)

    a = np.random.randn(5, 3).astype('float32')
    state = create_xoroshiro128p_states(1, seed=2)
    d_a = cuda.to_device(a)

    kernel[1, 1](d_a, state)

    d_a.copy_to_host(a)

    exp = np.array([
        [-1.5159574, -0.6456007,   0.8825855],
        [-1.335742,   1.3315016,   -1.0989239],
        [-2.2282648,  1.9315921,   0.51445764],
        [-1.0922,     0.06489223,  0.683121],
        [-0.8037137,  0.91763496, -0.07555987],
    ])

    for c_, e_ in zip(a, exp):
        assert c_ == pytest.approx(e_)
