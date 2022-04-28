import numpy as np

from scipy.special import digamma
from scipy.optimize import brentq

import multiprocessing
from joblib import Parallel, delayed

from itertools import permutations

from numba import jit
import copy
from tqdm import tqdm

from GenStudentMixtures.utils import batch_diagonal

from GenStudentMixtures.Multivariate_Student_Generalized import MST
from GenStudentMixtures.Mixture_Multivariate_Student_Generalized import MMST

import pymanopt
from pymanopt.manifolds import Stiefel
from pymanopt.solvers import ConjugateGradient


class GenStudentMixtures:
    def __init__(self, pi, mu, A, D, nu):
        self.pi = pi
        self.mu = mu
        self.A = A
        self.D = D
        self.nu = nu

        self.pi_hist = []
        self.mu_hist = []
        self.A_hist = []
        self.D_hist = []
        self.nu_hist = []

    ########################
    ###### Statistics ######
    ########################

    def _compute_alpha_beta(self, y):
        tmp = self.nu / 2
        alpha = tmp + 0.5
        beta = tmp + (np.swapaxes(self.D, 1, 2) @ np.expand_dims((y - self.mu), -1))[..., 0] ** 2 / (2 * self.A)
        return alpha, beta

    @staticmethod
    @jit(nopython=True)
    def _U(alpha, beta):
        return alpha / beta

    @staticmethod
    def _Utilde(alpha, beta):
        return digamma(alpha) - np.log(beta)

    def updateStat(self, y, r, gam, stat):
        alpha, beta = self._compute_alpha_beta(y)
        u, utilde = self._U(alpha, beta), self._Utilde(alpha, beta)
        r_expand = np.expand_dims(r, -1)
        ru, rutilde = r_expand * u, r_expand * utilde

        y_unsqueeze = np.expand_dims(y, -1)
        ymat = y_unsqueeze @ y_unsqueeze.T
        stat_update = {'s0': gam * r + (1 - gam) * stat['s0'],
                       's1': gam * np.einsum('ij,k->ijk', ru, y, optimize=True) + (1 - gam) * stat['s1'],
                       'S2': gam * np.einsum('ij,kl->ijkl', ru, ymat, optimize=True) + (1 - gam) * stat['S2'],
                       's3': gam * ru + (1 - gam) * stat['s3'], 's4': gam * rutilde + (1 - gam) * stat['s4']}

        return stat_update

    ########################
    ###### Parameters ######
    ########################

    # Update pi
    @staticmethod
    def _update_pi(s0):
        return s0  # / s0.sum() depends on initialization

    # Update mu
    def _update_mu(self, s1, s3):
        S3_inv = batch_diagonal(1 / s3)
        v = np.expand_dims(np.diagonal(np.swapaxes(self.D, 1, 2) @ np.swapaxes(s1, 1, 2), 0, -2, -1), -1)
        return (self.D @ (S3_inv @ v))[..., 0], v[..., 0]

    # Update A
    def _update_A(self, v, S2, s3):
        tmp = np.swapaxes(self.D[:, None, ...], -2, -1) @ S2
        tmp = tmp @ self.D[:, None, ...]
        tmp = np.diagonal(tmp, 0, -2, -1)
        return np.diagonal(tmp, 0, -2, -1) - v ** 2 / s3

    # Update D
    @staticmethod
    def _loss(D, matQuadk):
        tmp = np.swapaxes(D, -2, -1) @ matQuadk
        tmp = tmp @ D
        tmp = np.diagonal(tmp, 0, -2, -1)
        quadForm = np.diagonal(tmp, 0, -2, -1)
        return np.sum(quadForm)

    @staticmethod
    def _compute_matQuad(s1, S2, s3):
        tmp = s1 / np.expand_dims(s3, -1)
        return S2 - np.expand_dims(tmp, -1) @ s1[:, :, None, :]

    def _best_permutation(self, D, matQuad):
        # TODO to optimize
        D_opt = np.zeros(D.shape)
        for k in range(len(D)):
            minim_permuted = np.inf
            matQuadk = matQuad[k]
            for e in permutations(list(D[k].T)):
                D_permuted = np.vstack(e).T
                cost = self._loss(D_permuted, matQuadk)
                if cost < minim_permuted:
                    D_opt[k] = D_permuted.copy()
                    minim_permuted = cost
        return D_opt

    def _update_D(self, s1, S2, s3):
        def find_cost(matQuadk, manifold):
            @pymanopt.function.numpy(manifold)
            def cost(D):
                tmp = np.swapaxes(D, -2, -1) @ matQuadk
                tmp = tmp @ D
                tmp = np.diagonal(tmp, 0, -2, -1)
                quadForm = np.diagonal(tmp, 0, -2, -1)
                return np.sum(quadForm)

            @pymanopt.function.numpy(manifold)
            def grad(D):
                # TODO try to avoid the loop even if M is small
                grad = np.zeros(D.shape)
                M = len(D)
                for m in range(M):
                    grad[m] = 2 * matQuadk[m] @ D[:, m]
                return grad.T

            return cost, grad

        def opti_D(matQuadk):
            manifold = Stiefel(*matQuadk[0].shape)
            solver = ConjugateGradient(maxiter=4000)
            cost, grad = find_cost(matQuadk, manifold)
            problem = pymanopt.Problem(manifold, cost, egrad=grad, verbosity=0)
            return solver.solve(problem)

        matQuad = self._compute_matQuad(s1, S2, s3)
        d = (delayed(opti_D)(matQuad[k]) for k in range(len(s1)))
        D_tmp = np.array(Parallel(n_jobs=multiprocessing.cpu_count())(d))

        return self._best_permutation(D_tmp, matQuad)

    # Update nu
    @staticmethod
    def _fun_nu(nukm, s3km, s4km):
        return s4km - s3km - digamma(nukm / 2) + np.log(nukm / 2) + 1

    def _update_nu(self, s3, s4):
        K, M = s3.shape
        new_nu = np.zeros((K, M))
        for k in range(K):
            for m in range(M):
                s3km, s4km = s3[k, m], s4[k, m]
                fun = lambda x: self._fun_nu(x, s3km, s4km)
                new_nu[k, m] = brentq(fun, .01, 100)
        return new_nu.astype(np.float64)

    def _updateParams(self, stat):
        s0 = stat['s0']
        s1 = stat['s1'] / s0[:, None, None]
        S2 = stat['S2'] / s0[:, None, None, None]
        s3 = stat['s3'] / np.expand_dims(s0, -1)
        s4 = stat['s4'] / np.expand_dims(s0, -1)

        self.pi = self._update_pi(s0)
        self.D = self._update_D(s1, S2, s3)
        self.mu, v = self._update_mu(s1, s3)
        self.A = self._update_A(v, S2, s3)
        self.nu = self._update_nu(s3, s4)

        self.pi_hist.append(copy.deepcopy(self.pi))
        self.mu_hist.append(copy.deepcopy(self.mu))
        self.A_hist.append(copy.deepcopy(self.A))
        self.D_hist.append(copy.deepcopy(self.D))
        self.nu_hist.append(copy.deepcopy(self.nu))

    def fit(self, X, gam, mini_batch=50):
        stat = {'s0': np.zeros(len(self.pi)),
                's1': np.zeros(self.D.shape),
                'S2': np.zeros((*self.D.shape, self.mu.shape[-1])),
                's3': np.zeros(self.A.shape),
                's4': np.zeros(self.A.shape)}
        for i in tqdm(range(0, len(X) - mini_batch, mini_batch)):
            stat_new = {'s0': np.zeros(len(self.pi)),
                        's1': np.zeros(self.D.shape),
                        'S2': np.zeros((*self.D.shape, self.mu.shape[-1])),
                        's3': np.zeros(self.A.shape),
                        's4': np.zeros(self.A.shape)}
            for k in range(mini_batch):
                y = X[i + k]
                mst = MST(self.mu, self.A, self.D, self.nu).pdf(y)
                r = self.pi * mst / MMST(self.pi).pdf(mst, y)
                stat_tmp = self.updateStat(y, r, gam[i // mini_batch], stat)
                stat_new['s0'] += stat_tmp['s0'] / mini_batch
                stat_new['s1'] += stat_tmp['s1'] / mini_batch
                stat_new['S2'] += stat_tmp['S2'] / mini_batch
                stat_new['s3'] += stat_tmp['s3'] / mini_batch
                stat_new['s4'] += stat_tmp['s4'] / mini_batch
            if (i // mini_batch) % 500 == 0:
                print(self.pi)
            stat = copy.deepcopy(stat_new)
            self._updateParams(stat)
