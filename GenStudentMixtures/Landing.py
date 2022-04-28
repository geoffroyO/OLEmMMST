import numpy as np
from geomstats.geometry.stiefel import Stiefel

"""
Implementation of the landing algorithm in Numpy, and using geomstats for projection
In construction - not tested
"""


class Landing:
    def __init__(self, egrad, maxiter=20000, learning_rate=1e-3, lambda_regul=1, eps=0.5, prec=1e-6):
        self.egrad = egrad
        self.maxiter = maxiter
        self.learning_rate = learning_rate
        self.lambda_regul = lambda_regul
        self.eps = eps
        self.prec = prec

    @staticmethod
    def relative_grad(point, egrad):
        grad = egrad + point
        grad = grad @ point.T
        grad -= grad.T
        return grad / 2

    def _safe_step_size(self, d, a):
        alpha = 2 * (self.lambda_regul * d - a * d - 2 * self.lambda_regul * d)
        beta = a ** 2 + self.lambda_regul ** 2 * d ** 3 + 2 * self.lambda_regul * a * d ** 2 + a ** 2 * d
        sol = (alpha + np.sqrt(alpha ** 2 + 4 * beta * (self.eps - d))) / 2 / beta
        return sol

    def _landing_direction(self, point, rgrad):
        m = len(point)
        distance = point @ point.T - np.eye(m)
        landing_field = (rgrad + self.lambda_regul * distance) @ point

        d = np.linalg.norm(distance)
        a = np.linalg.norm(rgrad)
        step_size = min(self._safe_step_size(d, a), self.learning_rate)
        return point - step_size * landing_field

    def optimize(self, D_init):
        point = D_init.copy()
        manifold = Stiefel(*point.shape)

        rgrad_norm = 1
        iter = 0
        while rgrad_norm > self.prec and iter <= self.maxiter:
            grad = self.egrad(point)
            rgrad = self.relative_grad(point, grad)
            point = self._landing_direction(point, rgrad)
            rgrad_norm = np.linalg.norm(rgrad)
            iter += 1
        return manifold.projection(point)
