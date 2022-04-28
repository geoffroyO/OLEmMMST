import warnings

import numpy as np
import numpy.linalg as la
import numpy.random as rnd

from pymanopt.manifolds.manifold import EuclideanEmbeddedSubmanifold


class _SphereBase(EuclideanEmbeddedSubmanifold):
    """Base class for tensors with unit Frobenius norm.

    Notes:
        The implementation of the Weingarten map is taken from [AMT2013]_.
    """

    def __init__(self, *shape, name, dimension):
        if len(shape) == 0:
            raise TypeError("Need shape parameters.")
        self._shape = shape
        super().__init__(name, dimension)

    @property
    def typicaldist(self):
        return np.pi

    def inner(self, point, tangent_vector_a, tangent_vector_b):
        return np.tensordot(
            tangent_vector_a, tangent_vector_b, axes=tangent_vector_a.ndim
        )

    def norm(self, point, tangent_vector):
        return la.norm(tangent_vector)

    def dist(self, point_a, point_b):
        inner = max(min(self.inner(point_a, point_a, point_b), 1), -1)
        return np.arccos(inner)

    def proj(self, point, vector):
        return vector - self.inner(point, point, vector) * point

    def weingarten(self, point, tangent_vector, normal_vector):
        return -self.inner(point, point, normal_vector) * tangent_vector

    def exp(self, point, tangent_vector):
        norm = self.norm(point, tangent_vector)
        return point * np.cos(norm) + tangent_vector * np.sinc(norm / np.pi)

    def retr(self, point, tangent_vector):
        return self._normalize(point + tangent_vector)

    def log(self, point_a, point_b):
        vector = self.proj(point_a, point_b - point_a)
        distance = self.dist(point_a, point_b)
        epsilon = np.finfo(np.float64).eps
        factor = (distance + epsilon) / (self.norm(point_a, vector) + epsilon)
        return factor * vector

    def rand(self):
        point = rnd.randn(*self._shape)
        return self._normalize(point)

    def randvec(self, point):
        vector = rnd.randn(*self._shape)
        return self._normalize(self.proj(point, vector))

    def transp(self, point_a, point_b, tangent_vector_a):
        return self.proj(point_b, tangent_vector_a)

    def pairmean(self, point_a, point_b):
        return self._normalize(point_a + point_b)

    def zerovec(self, point):
        return np.zeros(self._shape)

    def _normalize(self, array):
        return array / la.norm(array)


class Sphere(_SphereBase):
    r"""The sphere manifold.

    Manifold of shape :math:`n_1 \times n_2 \times \ldots \times n_k` tensors
    with unit 2-norm.
    The metric is such that the sphere is a Riemannian submanifold of Euclidean
    space.
    """

    def __init__(self, *shape):
        if len(shape) == 0:
            raise TypeError("Need shape parameters.")
        if len(shape) == 1:
            (n1,) = shape
            name = f"Sphere manifold of {n1}-vectors"
        elif len(shape) == 2:
            n1, n2 = shape
            name = f"Sphere manifold of {n1}x{n2} matrices"
        else:
            name = f"Sphere manifold of shape {shape} tensors"
        dimension = np.prod(shape) - 1
        super().__init__(*shape, name=name, dimension=dimension)


class _SphereSubspaceIntersectionManifold(_SphereBase):
    def __init__(self, projector, name, dimension):
        m, n = projector.shape
        assert m == n, "projection matrix is not square"
        if dimension == 0:
            warnings.warn(
                "Intersected subspace is 1-dimensional. The manifold "
                "therefore has dimension 0 as it only consists of isolated "
                "points"
            )
        self._subspace_projector = projector
        super().__init__(n, name=name, dimension=dimension)

    def _validate_span_matrix(self, matrix):
        if len(matrix.shape) != 2:
            raise ValueError("Input array must be 2-dimensional")
        num_rows, num_columns = matrix.shape
        if num_rows < num_columns:
            raise ValueError(
                "The span matrix cannot have fewer rows than columns"
            )

    def proj(self, point, vector):
        return self._subspace_projector @ super().proj(point, vector)

    def rand(self):
        point = super().rand()
        return self._normalize(self._subspace_projector @ point)

    def randvec(self, point):
        vector = super().randvec(point)
        return self._normalize(self._subspace_projector @ vector)


class SphereSubspaceIntersection(_SphereSubspaceIntersectionManifold):
    r"""Sphere-subspace intersection manifold.

    Manifold of n-dimensional unit 2-norm vectors intersecting the
    :math:`r`-dimensional subspace of :math:`\R^n` spanned by the columns of
    the matrix ``matrix`` of size :math:`n \times r`.
    """

    def __init__(self, matrix):
        self._validate_span_matrix(matrix)
        m = matrix.shape[0]
        q, _ = la.qr(matrix)
        projector = q @ q.T
        subspace_dimension = la.matrix_rank(projector)
        name = (
            f"Sphere manifold of {m}-dimensional vectors intersecting a "
            f"{subspace_dimension}-dimensional subspace"
        )
        dimension = subspace_dimension - 1
        super().__init__(projector, name, dimension)


class SphereSubspaceComplementIntersection(
    _SphereSubspaceIntersectionManifold
):
    r"""Sphere-subspace compliment intersection manifold.

    Manifold of n-dimensional unit 2-norm vectors which are orthogonal to
    the :math:`r`-dimensional subspace of :math:`\R^n` spanned by columns of
    the matrix ``matrix``.
    """

    def __init__(self, matrix):
        self._validate_span_matrix(matrix)
        m = matrix.shape[0]
        q, _ = la.qr(matrix)
        projector = np.eye(m) - q @ q.T
        subspace_dimension = la.matrix_rank(projector)
        name = (
            f"Sphere manifold of {m}-dimensional vectors orthogonal "
            f"to a {subspace_dimension}-dimensional subspace"
        )
        dimension = subspace_dimension - 1
        super().__init__(projector, name, dimension)
