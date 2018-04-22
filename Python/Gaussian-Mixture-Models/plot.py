#!/usr/bin/python

from __future__ import print_function
from Wang_Shaojie_hw7 import *
import matplotlib.pyplot as plt


def differentK():
    X = load_data("points.dat")

    res_train = []
    res_dev = []

    for K in range(1, 7):
        _, _, _, record_train, record_dev = EM(X, K=K, epochs=50, tied=False)
        res_train.append(record_train)
        res_dev.append(record_dev)

    plt.subplot(2, 2, 1)
    for K in range(1, 7):
        plt.plot(res_train[K - 1], label="K = " + str(K))
    plt.title("Likelihood on train set, separate")
    plt.legend()

    plt.subplot(2, 2, 2)
    for K in range(1, 7):
        plt.plot(res_dev[K - 1], label="K = " + str(K))
    plt.title("Likelihood on dev set, separate")
    plt.legend()

    res_train_t = []
    res_dev_t = []

    for K in range(1, 7):
        _, _, _, record_train, record_dev = EM(X, K=K, epochs=50, tied=True)
        res_train_t.append(record_train)
        res_dev_t.append(record_dev)

    plt.subplot(2, 2, 3)
    for K in range(1, 7):
        plt.plot(res_train_t[K - 1], label="K = " + str(K))
    plt.title("Likelihood on train set, tied")
    plt.legend()

    plt.subplot(2, 2, 4)
    for K in range(1, 7):
        plt.plot(res_dev_t[K - 1], label="K = " + str(K))
    plt.title("Likelihood on dev set, tied")
    plt.legend()
    plt.show()

    return res_train, res_dev, res_train_t, res_dev_t


def isTied(res_train, res_dev, res_train_t, res_dev_t):

    for K in range(1, 7):
        plt.subplot(2, 6, 2 * K - 1)
        plt.plot(res_train[K - 1], label="separate")
        plt.plot(res_train_t[K - 1], label="tied")
        plt.title("Likelihood on train set, K = " + str(K))
        plt.legend()
        plt.subplot(2, 6, 2 * K)
        plt.plot(res_dev[K - 1], label="separate")
        plt.plot(res_dev_t[K - 1], label="tied")
        plt.title("Likelihood on dev set, K = " + str(K))
        plt.legend()

    plt.show()


res_train, res_dev, res_train_t, res_dev_t = differentK()
isTied(res_train, res_dev, res_train_t, res_dev_t)

