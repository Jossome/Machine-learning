#!/usr/bin/python

from __future__ import print_function
import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt
import copy


def load_data(filename):
    df = pd.read_csv(filename, header = None, sep = " ")
    y = df[0]
    df = df.drop([0], axis = 1)
    x = []
    for index, row in df.iterrows():
        this = np.zeros(124)
        this[-1] = 1
        for each in list(row):
            if type(each) == str:
                tmp = each.split(":")
                i, xi = int(tmp[0]), int(tmp[1])
                this[i - 1] = xi  # should minus 1 on i
        x.append(this)
    
    return x, y


def sign(x):
    if x < 0: return -1
    elif x == 0: return 0
    else: return 1


def train(x, y, iterations, noDev):
    w = np.zeros(124)
    cnt = 0
    w_list = []
    acc_list = []
    while True:
        for n in range(len(x)):
            tn = sign(np.dot(w, x[n]))
            if tn != y[n]:
                w += (y[n] * x[n])
        
        cnt += 1

        if noDev:
            if cnt == iterations:
                break

        else:
            xdev, ydev = load_data("/u/cs246/data/adult/a7a.dev")
            acc = test(w, xdev, ydev)
            acc_list.append(acc)
            w_list.append(copy.deepcopy(w))
            if cnt == iterations:
                # Get the best of these iterations
                w = w_list[acc_list.index(max(acc_list))]
                
                # Plot the accuracy by iterations
                plt.plot(range(iterations), acc_list)
                plt.show()

                break

    return w


def test(w, x, y):
    cnt = 0
    for n in range(len(x)):
        pred = sign(np.dot(w, x[n]))
        if pred == y[n]:
            cnt += 1

    return float(cnt) / len(x)


parser = argparse.ArgumentParser()
parser.add_argument("--iterations", type = int, default = 10)
parser.add_argument("--noDev", action = "store_true", default = False)
args = vars(parser.parse_args())
iterations = args['iterations']
noDev = args['noDev']


x, y = load_data("/u/cs246/data/adult/a7a.train")
w = train(x, y, iterations, noDev)

x, y = load_data("/u/cs246/data/adult/a7a.test")
acc = test(w, x, y)

print("Test accuracy:", acc)
print("Feature weights (bias last):", (" ").join([str(n) for n in w]))
