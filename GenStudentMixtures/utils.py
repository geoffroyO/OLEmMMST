import numpy as np
from numba import jit


@jit(nopython=True)
def batch_diagonal(A):
    N = A.shape[1]
    A = np.expand_dims(A, axis=1)
    return A * np.eye(N)
