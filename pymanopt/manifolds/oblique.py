import numpy as np
import numpy.linalg as la
import numpy.random as rnd

from pymanopt.manifolds.manifold import EuclideanEmbeddedSubmanifold


class Oblique(EuclideanEmbeddedSubmanifold):
    """Manifold of matrices w/ unit-norm columns.

    Oblique manifold: deals with matrices of size m-by-n such that each column
    has unit 2-norm, i.e., is a point on the unit sphere in R^m. The metric
    is such that the oblique manifold is a Riemannian submanifold of the
    space of m-by-n matrices with the usual trace inner product, i.e., the
    usual metric.
    """

    def __init__(self, m, n):
        self._m = m
        self._n = n
        name = f"Oblique manifold OB({m},{n})"
        dimension = (m - 1) * n
        super().__init__(name, dimension)

    @property
    def typicaldist(self):
        return np.pi * np.sqrt(self._n)

    def inner(self, point, tangent_vector_a, tangent_vector_b):
        return np.tensordot(
            tangent_vector_a, tangent_vector_b, axes=tangent_vector_a.ndim
        )

    def norm(self, point, tangent_vector):
        return la.norm(tangent_vector)

    def dist(self, point_a, point_b):
        XY = (point_a * point_b).sum(0)
        XY[XY > 1] = 1
        return la.norm(np.arccos(XY))

    def proj(self, point, vector):
        return vector - point * ((point * vector).sum(0)[np.newaxis, :])

    def ehess2rhess(
        self, point, euclidean_gradient, euclidean_hvp, tangent_vector
    ):
        # TODO(nkoep): Implement 'weingarten' instead.
        PXehess = self.proj(point, euclidean_hvp)
        return PXehess - tangent_vector * (
            (point * euclidean_gradient).sum(0)[np.newaxis, :]
        )

    def exp(self, point, tangent_vector):
        norm = np.sqrt((tangent_vector**2).sum(0))[np.newaxis, :]
        target_point = point * np.cos(norm) + tangent_vector * np.sinc(
            norm / np.pi
        )
        return target_point

    def retr(self, point, tangent_vector):
        return self._normalize_columns(point + tangent_vector)

    def log(self, point_a, point_b):
        vector = self.proj(point_a, point_b - point_a)
        distances = np.arccos((point_a * point_b).sum(0))
        norms = np.sqrt((vector**2).sum(0)).real
        # Try to avoid zero-division when both distances and norms are almost
        # zero.
        epsilon = np.finfo(np.float64).eps
        factors = (distances + epsilon) / (norms + epsilon)
        return vector * factors

    def rand(self):
        return self._normalize_columns(rnd.randn(self._m, self._n))

    def randvec(self, point):
        vector = rnd.randn(*point.shape)
        tangent_vector = self.proj(point, vector)
        return tangent_vector / self.norm(point, tangent_vector)

    def transp(self, point_a, point_b, tangent_vector_a):
        return self.proj(point_b, tangent_vector_a)

    def pairmean(self, point_a, point_b):
        return self._normalize_columns(point_a + point_b)

    def zerovec(self, point):
        return np.zeros((self._m, self._n))

    def _normalize_columns(self, array):
        return array / la.norm(array, axis=0)[np.newaxis, :]
