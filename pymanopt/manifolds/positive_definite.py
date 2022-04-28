import numpy as np
from numpy import linalg as la
from numpy import random as rnd
from scipy.linalg import expm

from pymanopt.manifolds.manifold import EuclideanEmbeddedSubmanifold
from pymanopt.tools.multi import multilog, multiprod, multisym, multitransp


class SymmetricPositiveDefinite(EuclideanEmbeddedSubmanifold):
    """Manifold of symmetric positive definite matrices.

    Notes:
        The geometry is based on the discussion in chapter 6 of [Bha2007]_.
        Also see [SH2015]_ for more details.
    """

    def __init__(self, n, k=1):
        self._n = n
        self._k = k

        if k == 1:
            name = f"Manifold of positive definite {n}x{n} matrices"
        else:
            name = (
                f"Product manifold of {k} positive definite {n}x{n} matrices"
            )
        dimension = int(k * n * (n + 1) / 2)
        super().__init__(name, dimension)

    @property
    def typicaldist(self):
        return np.sqrt(self.dim)

    def dist(self, point_a, point_b):
        # Adapted from equation (6.13) of [Bha2007].
        c = la.cholesky(point_a)
        c_inv = la.inv(c)
        logm = multilog(
            multiprod(multiprod(c_inv, point_b), multitransp(c_inv)),
            pos_def=True,
        )
        return la.norm(logm)

    def inner(self, point, tangent_vector_a, tangent_vector_b):
        p_inv_tv_a = la.solve(point, tangent_vector_a)
        if tangent_vector_a is tangent_vector_b:
            p_inv_tv_b = p_inv_tv_a
        else:
            p_inv_tv_b = la.solve(point, tangent_vector_b)
        return np.tensordot(
            p_inv_tv_a, multitransp(p_inv_tv_b), axes=tangent_vector_a.ndim
        )

    def proj(self, point, vector):
        return multisym(vector)

    def egrad2rgrad(self, point, euclidean_gradient):
        # TODO: Check that this is correct
        return multiprod(multiprod(point, multisym(euclidean_gradient)), point)

    def ehess2rhess(
        self, point, euclidean_gradient, euclidean_hvp, tangent_vector
    ):
        # TODO: Check that this is correct
        return multiprod(
            multiprod(point, multisym(euclidean_hvp)), point
        ) + multisym(
            multiprod(
                multiprod(tangent_vector, multisym(euclidean_gradient)), point
            )
        )

    def norm(self, point, tangent_vector):
        return np.sqrt(self.inner(point, tangent_vector, tangent_vector))

    def rand(self):
        # Generate eigenvalues between 1 and 2.
        d = np.ones((self._k, self._n, 1)) + rnd.rand(self._k, self._n, 1)

        # Generate an orthogonal matrix.
        u = np.zeros((self._k, self._n, self._n))
        for i in range(self._k):
            u[i], _ = la.qr(rnd.randn(self._n, self._n))

        if self._k == 1:
            return multiprod(u, d * multitransp(u))[0]
        return multiprod(u, d * multitransp(u))

    def randvec(self, point):
        k = self._k
        n = self._n
        if k == 1:
            tangent_vector = multisym(rnd.randn(n, n))
        else:
            tangent_vector = multisym(rnd.randn(k, n, n))
        return tangent_vector / self.norm(point, tangent_vector)

    def transp(self, point_a, point_b, tangent_vector_b):
        return tangent_vector_b

    def exp(self, point, tangent_vector):
        p_inv_tv = la.solve(point, tangent_vector)
        if self._k > 1:
            e = np.zeros(np.shape(point))
            for i in range(self._k):
                e[i] = expm(p_inv_tv[i])
        else:
            e = expm(p_inv_tv)
        return multiprod(point, e)

    retr = exp

    def log(self, point_a, point_b):
        c = la.cholesky(point_a)
        c_inv = la.inv(c)
        logm = multilog(
            multiprod(multiprod(c_inv, point_b), multitransp(c_inv)),
            pos_def=True,
        )
        return multiprod(multiprod(c, logm), multitransp(c))

    def zerovec(self, point):
        k = self._k
        n = self._n
        if k == 1:
            return np.zeros((n, n))
        return np.zeros((k, n, n))
