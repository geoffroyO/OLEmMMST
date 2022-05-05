import numpy as np
from numpy.random import choice, shuffle, permutation

from GenStudentMixtures.Multivariate_Student_Generalized import MST


class MMST:
    def __init__(self, pi, mu=None, A=None, D=None, nu=None):
        self.pi = pi
        self.mu = mu
        self.D = D
        self.A = A
        self.nu = nu

    def pdf(self, mst=None, y=None):
        if mst is not None:
            return (self.pi * mst).sum()
        else:
            return (self.pi * MST(self.mu, self.A, self.D, self.nu).pdf(y)).sum()

    def sample(self, N):
        classes = choice(len(self.pi), N, p=self.pi)

        gen = MST(self.mu, self.A, self.D, self.nu).sample(N)
        gen_mix = np.zeros((1, len(self.mu[0])))
        clusters = []
        for k in range(len(self.pi)):
            gen_mix = np.concatenate((gen_mix, gen[k, classes == k, :]))
            N_k = (classes == k).sum()
            clusters += [k] * N_k
        gen_mix = gen_mix[1:]
        permute = permutation(len(clusters))
        return gen_mix[permute], np.array(clusters)[permute]
