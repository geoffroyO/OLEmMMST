import numpy as np
from scipy.optimize import root_scalar
from scipy.special import polygamma, digamma, gamma as Gamma
from numpy import linalg as LA

import pymanopt
from pymanopt.manifolds import Stiefel
from pymanopt import Problem
from pymanopt.solvers import SteepestDescent
from tqdm import tqdm
import warnings

import rpy2.robjects as robjects
import rpy2.robjects.numpy2ri

rpy2.robjects.numpy2ri.activate()

r = robjects.r
r['source']('TKM.R')
trimkmeans = robjects.globalenv['trimkmeans']


class MixtureMultipleScaleDistribution:
    DEBUG = 3
    INFO = 2
    WARNING = 1
    ERROR = 0

    def __init__(self, n_components=1, tol=1e-5, max_iter=100, verbose=0):
        # parameters
        self.pi = None
        self.mu = None
        self.A = None
        self.D = None
        self.nu = None

        # Internal variables
        self.__p = None
        self.__mp = None
        self.__omega = None
        self.__v = None

        self.n_components = n_components
        self.tol = tol
        self.max_iter = max_iter
        self.verbose = verbose

        self.__nu_min = 0
        self.__nu_max = 60
        self.__converged = False

    def _initalization(self, X, n_first=400):
        res = trimkmeans(X[:n_first], self.n_components)
        clusters, self.mu = res[0], res[1]

        self.pi = np.array([(clusters == k).sum() / (clusters != 0).sum() for k in range(1, self.n_components + 1)],
                           dtype=np.float64)
        self.D = np.array([np.eye(len(X[0]))] * self.n_components, dtype=np.float64)
        self.A = np.ones((self.n_components, len(X[0])), dtype=np.float64)
        self.nu = np.ones((self.n_components, len(X[0])), dtype=np.float64)

    def fit(self, X):
        n_samples, n_features = X.shape
        self._initalization(X)

        # Initialize internal variables
        self.__p = 0
        self.__mp = np.empty((self.n_components, n_samples, 1))
        self.__omega = np.empty((self.n_components, n_samples, n_features))
        self.__v = np.empty((self.n_components, n_samples, n_features, n_features))

        self.__update_log_likelihood(X)

        # EM loop
        self.__converged = False
        for i in tqdm(range(self.max_iter)):
            last_log_likelihood = self.log_likelihood

            self.message(self.DEBUG, "Iteration %d (LL = %f)" % (i, self.log_likelihood))

            self.__e_step(X)
            self.__m_step(X)
            self.__update_log_likelihood(X)

            if np.isnan(self.log_likelihood):
                self.message(self.ERROR, "Log-likelihood is NaN, aborting")
                break

            d = self.log_likelihood - last_log_likelihood
            if abs(d / last_log_likelihood) < self.tol:
                self.__converged = True
                self.message(self.INFO, "Converged in %d iterations with a maximum log-likelihood of %f" % (
                    i + 1, self.log_likelihood))
                break

            if d < 0:
                self.message(self.WARNING, "Iteration %d decreased log-likelihood" % (i + 1))

    def predict(self, X):
        n_samples, _ = X.shape

        self.__mp = np.empty((self.n_components, n_samples, 1))

        self.__compute_mp(X)

        return self.__mp.argmax(axis=0)

    @staticmethod
    def ms(X, mu, D, A, nu):
        t1 = A * nu
        t2 = (D.T @ (X - mu).T).T
        n = Gamma((nu + 1) / 2) / (Gamma(nu / 2) * np.sqrt(t1 * np.pi))

        return np.prod(n * np.power(1 + (t2 ** 2) / t1, -(nu + 1) / 2), axis=1)

    def __compute_mp(self, X):
        for k in range(len(self.pi)):
            self.__mp[k] = self.pi[k] * MixtureMultipleScaleDistribution.ms(X, self.mu[k], self.D[k], self.A[k],
                                                                            self.nu[k]).reshape((X.shape[0], 1))

        self.__p = np.sum(self.__mp, axis=0)

    def __e_step(self, X):
        # Compute tau and the log-likelihood in the same loop
        self.__tau = self.__mp / self.__p

        # Compute omega
        for k in range(self.n_components):
            normalized = np.square((self.D[k].T @ (X - self.mu[k]).T).T) / self.A[k]
            self.__omega[k] = (self.nu[k] + 1) / (self.nu[k] + normalized)

    def __update_log_likelihood(self, X):
        self.__compute_mp(X)
        self.log_likelihood = np.sum(np.log(self.__p))
        n, d = X.shape
        dof = self.n_components * (1 + 2 * d + d * (d + 1) / 2) - 1
        self.bic = dof * np.log(n) - 2 * self.log_likelihood

    def __m_step(self, X):
        # Update pi
        n_samples, n_features = X.shape

        n = np.sum(self.__tau, axis=1)

        self.pi = n / n_samples

        # Update mu
        tau_omega = self.__tau * self.__omega
        d = np.sum(tau_omega, axis=1)

        for k in range(self.n_components):
            t = self.D[k].T @ X.T
            t = self.__omega[k] * t.T
            t = (self.D[k] @ t.T).T

            m = np.sum(self.__tau[k] * t, axis=0)
            self.mu[k] = m / d[k]

        # Update D
        previous_D = self.D

        for k in range(self.n_components):
            t = (X - self.mu[k]).reshape((n_samples, n_features, 1))
            t2 = t @ t.transpose((0, 2, 1))
            self.__v[k] = self.__tau[k].reshape(n_samples, 1, 1) * t2

        manifold = Stiefel(n_features, n_features)
        solver = SteepestDescent(max_iterations=1000, verbosity=0)

        for k in range(self.n_components):
            delta_a = (self.__omega[k] / self.A[k]).reshape(n_samples, n_features, 1)

            @pymanopt.function.numpy(manifold)
            def cost1(D):
                return np.trace(np.sum(D @ (delta_a * (D.T @ self.__v[k])), axis=0))

            @pymanopt.function.numpy(manifold)
            def egrad1(D):
                return 2 * np.sum(self.__v[k] @ (D.T * delta_a).transpose(0, 2, 1), axis=0)

            problem = Problem(manifold=manifold, cost=cost1, egrad=egrad1)
            self.D[k] = solver.solve(problem, self.D[k].copy())

        # Update A
        previous_A = self.A

        for k in range(self.n_components):
            rotated = np.square((self.D[k].T @ (X - self.mu[k]).T).T)
            self.A[k] = np.sum(tau_omega[k] * rotated, axis=0) / n[k]

        # Reorder eigenvectors and eigenvalues if necessary
        for k in range(self.n_components):
            X = self.D[k] @ np.diag(np.sqrt(self.A[k]))
            Y = previous_D[k] @ np.diag(np.sqrt(previous_A[k]))
            dist = np.empty((n_features, n_features))

            for i in range(n_features):
                for j in range(n_features):
                    dist[i, j] = min(LA.norm(X[:, i] - Y[:, j]), LA.norm(X[:, i] + Y[:, j]))

            permutation = np.argmin(dist, axis=0)

            if np.any(permutation != np.arange(n_features)):
                self.message(self.INFO, "Reordering eigenvectors")

                if np.any(np.sort(permutation) == np.arange(n_features)):
                    self.D[k] = self.D[k, :, permutation]
                    self.A[k] = self.A[k, :, permutation]
                else:
                    self.message(self.WARNING, "Degenerated permutation")

        # Updating nu
        C0 = (self.nu + 1) / 2
        C1 = np.sum(self.__tau * (np.log(self.__omega) - self.__omega), axis=1) / n.reshape(self.n_components, 1)
        C = 1 + C1 + digamma(C0) - np.log(C0)

        for k in range(self.n_components):
            for m in range(n_features):
                def f(nu):
                    return np.log(nu / 2) - digamma(nu / 2) + C[k, m]

                def fprime(nu):
                    return 1 / nu - polygamma(1, nu / 2) / 2

                def fprime2(nu):
                    return -1 / (nu * nu) - polygamma(2, nu / 2) / 4

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    r = root_scalar(f, x0=self.nu[k, m], fprime=fprime, fprime2=fprime2, method="halley")

                if not r.converged:
                    self.message(self.WARNING, "Nu estimation did not converge")
                else:
                    self.nu[k, m] = r.root

        np.clip(self.nu, self.__nu_min, self.__nu_max, out=self.nu)

    def get_parameters(self):
        return self.pi, self.mu, self.D, self.A, self.nu

    def message(self, level, msg):
        if self.verbose >= level:
            print(msg)
