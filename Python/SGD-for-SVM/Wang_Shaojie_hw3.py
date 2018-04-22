#!/usr/bin/python

from __future__ import print_function
import pandas as pd
import numpy as np
import argparse
import copy


def load_data(filename):
    df = pd.read_csv(filename, header = None, sep = " ")
    y = df[0]
    df = df.drop([0], axis = 1)
    x = []
    for index, row in df.iterrows():
        this = np.zeros(123)
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


def train(x, y, c, epochs):
    w = np.zeros(123)
    b = 0
    cnt = 0
    w_list = []
    acc_list = []
    lr = 0.1
    while True:
        cnt += 1
        for n in range(len(x)):
            tn = np.dot(w, x[n])
            if y[n] * (tn + b) >= 1:
                w -= lr * (1.0 / len(x)) * w
            else:
                w -= lr * ((1.0 / len(x)) * w - c * y[n] * x[n])
                b += lr * y[n] * c
        

        if cnt == epochs:
            break

    return w, b


def test(w, b, x, y):
    cnt = 0
    for n in range(len(x)):
        pred = sign(np.dot(w, x[n]) + b)
        if pred == y[n]:
            cnt += 1

    return float(cnt) / len(x)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type = int, default = 1)
    parser.add_argument("--capacity", type = float, default = 0.01)
    args = vars(parser.parse_args())
    epochs = args['epochs']
    capacity = args['capacity']
    
    
    x, y = load_data("/u/cs246/data/adult/a7a.train")
    w, b = train(x, y, capacity, epochs)
    
    acc_train = test(w, b, x, y)
    
    x_test, y_test = load_data("/u/cs246/data/adult/a7a.test")
    acc_test = test(w, b, x_test, y_test)
    
    x_dev, y_dev = load_data("/u/cs246/data/adult/a7a.dev")
    acc_dev = test(w, b, x_dev, y_dev)
    
    print("EPOCHS:", epochs)
    print("CAPACITY:", capacity)
    print("TRAINING_ACCURACY:", acc_train) 
    print("TEST_ACCURACY:", acc_test)
    print("DEV_ACCURACY:", acc_dev)
    print("FINAL_SVM:", [b] + list(w))
