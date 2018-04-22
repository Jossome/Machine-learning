#!/usr/bin/python

from __future__ import print_function
from Wang_Shaojie_hw3 import *
import matplotlib.pyplot as plt
import numpy as np

def experiment():
    acc_test_list = []
    acc_dev_list = []
    c_list = np.logspace(-3, 4, 20, endpoint = True)
    epochs = 5
    
    for c in c_list:

        x, y = load_data("/u/cs246/data/adult/a7a.train")
        w, b = train(x, y, c, epochs)

        xdev, ydev = load_data("/u/cs246/data/adult/a7a.dev")
        acc_dev = test(w, b, xdev, ydev)
        acc_dev_list.append(acc_dev)
        
        xtest, ytest = load_data("/u/cs246/data/adult/a7a.test")
        acc_test = test(w, b, xtest, ytest)
        acc_test_list.append(acc_test)
    
    # Plot the accuracy by epochs
    plt.plot(c_list, acc_dev_list, label = "dev_accuracy")
    plt.plot(c_list, acc_test_list, label = "test_accuracy")
    plt.xscale("log")
    plt.legend()
    plt.show()


experiment()
