import numpy as np


def batch_diagonal(A):
    N = A.shape[1]
    A = np.expand_dims(A, axis=1)
    return A * np.eye(N)
