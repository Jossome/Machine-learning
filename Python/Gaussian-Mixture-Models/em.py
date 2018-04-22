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


def N(X, miu, sig):
    d = X - miu
    return np.exp(-0.5 * d.dot(inv(sig).dot(d))) / (((2 * np.pi) ** (len(X) / 2)) * np.sqrt(det(sig)))


def EM(X, K=4, epochs=10, tied=False):

    train, dev = X[:int(len(X) * 0.9)], X[int(len(X) * 0.9):]

    n_sample, n_feature = train.shape

    # Initialize
    lam = np.random.dirichlet(np.ones(K), size=1)[0]
    miu = np.random.rand(K, 2)
    if tied:
        sig = np.eye(2)
    else:
        sig = [np.eye(2) for _ in range(K)]

    z = np.empty((n_sample, K))
    last = 0
    record_train = []
    record_dev = []

    for i in range(epochs):
        # E-step
        for n in range(n_sample):
            if tied:
                denom = np.sum(lam[k] * N(X[n], miu[k], sig) for k in range(K))
            else:
                denom = np.sum(lam[k] * N(X[n], miu[k], sig[k]) for k in range(K))

            for k in range(K):
                if tied:
                    z[n, k] = lam[k] * N(X[n], miu[k], sig) / denom
                else:
                    z[n, k] = lam[k] * N(X[n], miu[k], sig[k]) / denom

        # M-step
        total = z.sum(axis=0)
        lam = total / n_sample
        miu = np.dot(z.T, train) / total[:, np.newaxis]

        if tied:
            sig = (np.dot(train.T, train) - np.dot(total * miu.T, miu)) / total.sum()
        else:
            for k in range(K):
                sig[k] = np.dot(z[:, k] * (train - miu[k]).T, train - miu[k]) / total[k]

        now = likelihood(train, lam, miu, sig, K=K, tied=tied)
        record_train.append(now)
        record_dev.append(likelihood(dev, lam, miu, sig, K=K, tied=tied))
        if abs(now - last) < 0:
            break
        else:
            last = now

    return lam, miu, sig, record_train, record_dev


def likelihood(X, lam, miu, sig, K=4, tied=False):
    n_sample, _ = X.shape
    if tied:
        return np.sum(np.log(np.sum(lam[k] * N(X[n], miu[k], sig) for k in range(K)))
                      for n in range(n_sample)) / n_sample
    else:
        return np.sum(np.log(np.sum(lam[k] * N(X[n], miu[k], sig[k]) for k in range(K)))
                      for n in range(n_sample)) / n_sample


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--tied", type=bool, default=False)
    parser.add_argument("-K", type=int, default=4)
    args = vars(parser.parse_args())
    epochs = args['epochs']
    K = args['K']
    tied = args['tied']

    X = load_data("points.dat")
    _, _, _, _, record_dev = EM(X, K=K, epochs=epochs, tied=tied)
    print("Likelihood on dev:", record_dev[-1])
