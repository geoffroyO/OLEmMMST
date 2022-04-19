import numpy as np
from numpy.random import multivariate_normal

from scipy.special import loggamma

from GenStudentMixtures.utils import batch_diagonal

import torch


class MST:
    def __init__(self, mu, A, D, nu):
        self.mu = mu
        self.D = D
        self.A = A
        self.nu = nu

    def pdf(self, y):
        th2 = self.A * self.nu
        th1 = np.log(1 + (np.swapaxes(self.D, 1, 2) @ np.expand_dims((y - self.mu), -1))[..., 0] ** 2 / th2)
        exponent = - (self.nu + 1) / 2

        main = exponent * th1

        gam1 = loggamma((self.nu + 1) / 2)
        gam2 = loggamma(self.nu / 2)
        th2 = gam1 - (gam2 + 0.5 * np.log(np.pi * th2))

        main += th2

        return np.exp(main.sum(1))

    def sample(self, N):
        batch, M = self.mu.shape
        X = multivariate_normal(np.zeros(M), cov=np.diag(np.ones(M)), size=(batch, N,))

        # TODO comment tirer en batch sur numpy ?????
        W = torch.distributions.Gamma(torch.tensor(self.nu) / 2, torch.tensor(self.nu) / 2).sample((N,)).numpy()
        W = np.swapaxes(W, 0, 1)

        X /= np.sqrt(W)

        matA = batch_diagonal(np.sqrt(self.A))
        coef = self.D @ matA

        gen = np.expand_dims(self.mu, 1) + np.swapaxes(coef @ np.swapaxes(X, 2, 1), 1, 2)

        return gen