from GenStudentMixtures import utils
from GenStudentMixtures.Multivariate_Student_Generalized import MST
from GenStudentMixtures.Mixture_Multivariate_Student_Generalized import MMST

import numpy as np
import autograd.numpy as autonp

from scipy.special import digamma
from scipy.optimize import brentq

import pymanopt
from pymanopt.manifolds import Stiefel
from pymanopt.solvers import TrustRegions

import multiprocessing as mp

from tqdm import tqdm

class GenStudentMixtures:
    def __init__(self, piinit, muinit, Ainit, Dinit, nuinit):

        self.pi = piinit
        self.pi_hist = [self.pi]
        self.mu = muinit
        self.mu_hist = [self.mu]
        self.D = Dinit
        self.D_hist = [self.D]
        self.A = Ainit
        self.A_hist = [self.A]
        self.nu = nuinit
        self.nu_hist = [self.nu]

        self.s0 = np.zeros(len(self.pi))
        self.s1 = np.zeros(self.D.shape)
        self.S2 = np.zeros((*self.D.shape, self.mu.shape[-1]))
        self.s3 = np.zeros(self.A.shape)
        self.s4 = np.zeros(self.A.shape)

    def _alpha_beta(self, y):
        tmp = self.nu / 2
        alpha = tmp + 0.5
        beta = tmp + (np.swapaxes(self.D, 1, 2) @ np.expand_dims((y - self.mu), -1))[..., 0] ** 2 / (2 * self.A)
        return alpha, beta

    def _U(self, alpha, beta):
        return alpha / beta

    def _Utilde(self, alpha, beta):
        return digamma(alpha) - np.log(beta)

    # Update nu
    # TODO to parallelize
    def _fun_nu(self, nukm, s3km, s4km):
        return s4km - s3km - digamma(nukm / 2) + np.log(nukm / 2) + 1

    def update_nu(self, s3, s4):
        K, M = s3.shape
        new_nu = np.zeros((K, M))
        for k in range(K):
            for m in range(M):
                s3km, s4km = s3[k, m], s4[k, m]
                fun = lambda x: self._fun_nu(x, s3km, s4km)
                new_nu[k, m] = brentq(fun, .001, 100)
        return np.array(new_nu, dtype=np.float64)

    # Update μ
    def update_mu(self, s1, s3):
        S3_inv = utils.batch_diagonal(1 / s3)
        v = np.expand_dims(np.diagonal(np.swapaxes(self.D, 1, 2) @ np.swapaxes(s1, 1, 2), 0, -2, -1), -1)
        return (self.D @ (S3_inv @ v))[..., 0], v[..., 0]

    # Update A
    def update_A(self, v, S2, s3):
        tmp = np.swapaxes(self.D[:, None, ...], -2, -1) @ S2
        tmp = tmp @ self.D[:, None, ...]
        tmp = np.diagonal(tmp, 0, -2, -1)
        return np.diagonal(tmp, 0, -2, -1) - v ** 2 / s3

    # Update pi
    def update_pi(self, s0):
        return s0

    # Update D
    def update_D(self, s1, S2, s3, solver=TrustRegions()):
        manifold = Stiefel(self.D.shape[1], self.D.shape[2])

        def find_cost(k, s1, S2, s3):
            @pymanopt.function.autograd(manifold)
            def cost(D):
                sum_all = 0
                M = len(D[0])
                for m in range(M):
                    tmp = s1[k, m] / s3[k, m]
                    matQuad = (S2[k, m] - np.expand_dims(tmp, -1) @ np.expand_dims(s1[k, m], -1).T)
                    quadForm = D[:, m].T @ matQuad @ D[:, m]
                    sum_all += autonp.log(quadForm)
                return sum_all

            return cost

        D_new = np.zeros(self.D.shape)
        for k in range(len(self.D)):
            cost = find_cost(k, s1, S2, s3)
            problem = pymanopt.Problem(manifold, cost, verbosity=0)
            D_new[k] = solver.solve(problem)
        return D_new

    def updateStat(self, y, r, gam):
        self.s0 = gam * r + (1 - gam) * self.s0

        alpha, beta = self._alpha_beta(y)
        u, utilde = self._U(alpha, beta), self._Utilde(alpha, beta)
        r = np.expand_dims(r, -1)
        ru, rutilde = r * u, r * utilde

        y_unsqueeze = np.expand_dims(y, -1)
        ymat = y_unsqueeze @ y_unsqueeze.T

        self.s1 = gam * np.einsum('ij,k->ijk', ru, y) + (1 - gam) * self.s1
        self.S2 = gam * np.einsum('ij,kl->ijkl', ru, ymat) + (1 - gam) * self.S2
        self.s3 = gam * ru + (1 - gam) * self.s3
        self.s4 = gam * rutilde + (1 - gam) * self.s4

    def updateParams(self):
        s0 = self.s0
        s1 = self.s1 / s0[:, None, None]
        S2 = self.S2 / s0[:, None, None, None]
        s3 = self.s3 / np.expand_dims(s0, -1)
        s4 = self.s4 / np.expand_dims(s0, -1)

        self.pi = self.update_pi(s0)
        self.pi_hist.append(self.pi)
        self.D = self.update_D(s1, S2, s3)
        self.D_hist.append(self.D)
        self.mu, v = self.update_mu(s1, s3)
        self.mu_hist.append(self.mu)
        self.A = self.update_A(v, S2, s3)
        self.A_hist.append(self.A)
        self.nu = self.update_nu(s3, s4)
        self.nu_hist.append(self.nu)

    def fit(self, X):
        gam_vec = (1 - 10e-10) * np.array([k for k in range(1, len(X) + 1)]) ** (-6 / 10)

        for i in range(500):
            y = X[i]
            mst = MST(self.mu, self.A, self.D, self.nu).pdf(y)
            r = self.pi * mst / MMST(self.pi).pdf(mst)
            self.updateStat(y, r, gam_vec[i])

        for i in tqdm(range(500, 1200)):
            y = X[i]
            mst = MST(self.mu, self.A, self.D, self.nu).pdf(y)
            r = self.pi * mst / MMST(self.pi).pdf(mst)
            self.updateParams()
            self.updateStat(y, r, gam_vec[i])

    def predict(self, X):
        cluster_lab = np.zeros(len(X))

        for i, y in enumerate(X):
            mst = MST(self.mu, self.A, self.D, self.nu).pdf(y)
            r = self.pi * mst / MMST(self.pi).pdf(mst)
            cluster_lab[i] = np.argmax(r)
        return cluster_lab