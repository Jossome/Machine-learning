#!/usr/bin/python

from __future__ import print_function
from Wang_Shaojie_hw8 import *
import matplotlib.pyplot as plt


def experiment():
    data = load_data("points.dat")

    res_train = []
    res_dev = []

    for k in range(1, 10):

        m = GaussianHMM(data, k, 50)
        m.train()
        res_train.append(m.record_train)
        res_dev.append(m.record_dev)

    plt.subplot(1, 2, 1)
    for k in range(1, 10):
        plt.plot(res_train[k - 1], label="k = " + str(k))
    plt.title("Likelihood on train set")
    plt.legend()

    plt.subplot(1, 2, 2)
    for k in range(1, 10):
        plt.plot(res_dev[k - 1], label="k = " + str(k))
    plt.title("Likelihood on dev set")
    plt.legend()

    plt.show()


experiment()

