import numpy as np

THREADS = 8
BLOCKS = 2

D = 3   # wymiar procesu
M = 4   # wymiar wienera
K = THREADS * BLOCKS   # trajektorie
N = 2   # wymiar N


x_0 = np.reshape(np.array([1, 1, 1]), newshape=(3, 1))
