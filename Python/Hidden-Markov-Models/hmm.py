#!/usr/bin/python

from __future__ import print_function
import numpy as np
import argparse
# import matplotlib.pyplot as plt
from numpy.linalg import inv, det


def load_data(filename):
    res = []
    with open(filename, "r") as f:
        for line in f:
            res.append(np.array([float(x) for x in line.strip().split()]))

    # plt.scatter([x[0] for x in res], [x[1] for x in res])
    # plt.show()

    return np.array(res)


def gaussian(x, miu, sig):
    d = x - miu
    return np.exp(-0.5 * d.dot(inv(sig).dot(d))) / (((2 * np.pi) ** (len(x) / 2)) * np.sqrt(det(sig)))


class GaussianHMM:

    def __init__(self, x, k=4, epochs=10):
        train, dev = x[:int(len(x) * 0.9)], x[int(len(x) * 0.9):]
        n_sample, n_feature = train.shape
        self.train_X = train
        self.dev_X = dev
        self.n_sample = n_sample
        self.n_feature = n_feature
        self.K = k
        self.epochs = epochs
        self.mu = np.random.rand(self.K, self.n_feature)  # mean for gaussian
        self.sigma = np.array([np.eye(self.n_feature) for _ in range(self.K)])  # cov matrix for gaussian
        self.pi = np.ones(self.K) / float(self.K)  # Initial probabilities
        self.A = np.ones((self.K, self.K)) / float(self.K)  # transition prob
        self.current_epoch = 0
        self.record_train = []
        self.record_dev = []

    def emission_prob(self, x):
        return np.array([gaussian(x, self.mu[i], self.sigma[i]) for i in range(self.K)])

    def forward(self):
        alpha = np.zeros((self.n_sample, self.K))
        alpha[0] = self.pi * self.emission_prob(self.train_X[0])
        c = np.zeros(self.n_sample)
        c[0] = np.sum(alpha[0])
        alpha[0] /= c[0]

        for i in range(1, self.n_sample):
            alpha[i] = self.emission_prob(self.train_X[i]) * np.dot(alpha[i - 1], self.A)
            c[i] = np.sum(alpha[i])
            if c[i] == 0:
                c[i] = 1.0
            alpha[i] /= c[i]

        return alpha, c

    def backward(self, c):
        beta = np.zeros((self.n_sample, self.K))
        beta[-1] = np.ones(self.K)

        for i in reversed(range(self.n_sample - 1)):
            # TODO: beta keeps decreasing and leads to nan
            beta[i] = np.dot(beta[i + 1] * self.emission_prob(self.train_X[i + 1]), self.A.T)
            beta[i] /= c[i + 1]

        return beta

    def e_step(self):
        alpha, c = self.forward()
        beta = self.backward(c)
        gamma = alpha * beta
        ksi = np.zeros((self.K, self.K))
        for i in range(1, self.n_sample):
            ksi += np.outer(alpha[i - 1], self.emission_prob(self.train_X[i]) * beta[i]) * self.A / c[i]

        return gamma, ksi

    def m_step(self, gamma, ksi):
        self.pi = gamma[0] / np.sum(gamma[0])
        sum_gamma = np.sum(gamma, axis=0)
        for k in range(self.K):
            self.A[k] = ksi[k] / np.sum(ksi[k])
            self.sigma[k] = np.dot(gamma[:, k] * (self.train_X - self.mu[k]).T,
                                   self.train_X - self.mu[k]) / sum_gamma[k]
        self.mu = np.dot(gamma.T, self.train_X) / sum_gamma[:, np.newaxis]

    def train(self):
        print("Total Epochs: %d" % self.epochs)
        for i in range(self.epochs):
            self.current_epoch = i + 1
            gamma, ksi = self.e_step()
            self.m_step(gamma, ksi)
            # print("Epoch:", self.current_epoch)
            self.record_train.append(self.likelihood(gamma, ksi, dev=False))
            self.record_dev.append(self.likelihood(gamma, ksi, dev=True))
            # if i > 1:
            #     if abs(self.record_train[-1] - self.record_train[-2]) < 0.0001:
            #         print("Converge at %d epoch!" % self.current_epoch)
            #         break
        print("Final likelihood on train:", self.record_train[-1], "\non dev:", self.record_dev[-1])

    def likelihood(self, gamma, ksi, dev=False):

        first = np.dot(gamma[0], np.log(self.pi))
        second = np.sum(ksi * self.A)
        length = len(self.dev_X) if dev else self.n_sample
        if dev:
            third = np.sum([np.dot(gamma[n], np.log(self.emission_prob(self.dev_X[n]))) for n in range(length)])
        else:
            third = np.sum([np.dot(gamma[n], np.log(self.emission_prob(self.train_X[n]))) for n in range(length)])

        return (first + second + third) / length


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("-K", type=int, default=4)
    args = vars(parser.parse_args())

    Epochs = args['epochs']
    K = args['K']
    X = load_data("points.dat")

    hmm = GaussianHMM(X, K, Epochs)
    hmm.train()
