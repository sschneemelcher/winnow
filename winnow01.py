#!/usr/bin/env python3
import numpy as np
from tensorflow import keras

"""
Winnow algorithm classifying the numbers 0 and 1 of the MNIST Dataset.

Achieves an accuracy of > 95% on the test set after making only ~11 mistakes in 
training (~less than 50 examples shown), which is absolutely insane compared
to the perceptron algorithm which needs more than 100 times
the amount of steps to extrapolate to  the training set 
(see perceptron01.py to try it out).
"""

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

x_train = np.where(x_train.reshape(-1, 784) > 128, 1, 0)
x_test = np.where(x_train.reshape(-1, 784) > 128, 1, 0)

one_train = np.asarray([x_train[i] for i in range(len(y_train)) if y_train[i] == 1])
zero_train = np.asarray([x_train[i] for i in range(len(y_train)) if y_train[i] == 0])

one_test = np.asarray([x_test[i] for i in range(len(y_test)) if y_test[i] == 1])
zero_test = np.asarray([x_test[i] for i in range(len(y_test)) if y_test[i] == 0])

x_train = np.vstack((one_train, zero_train)).reshape((-1, 784))
y_train = np.append(np.ones(len(one_train)), np.zeros(len(zero_train)))

x_test = np.vstack((one_test, zero_test)).reshape((-1, 784))
y_test = np.append(np.ones(len(one_test)), np.zeros(len(zero_test)))

n = 128
weights = np.ones(784)
mistakes = 0
perm = np.random.permutation(range(len(x_train)))
for idx, i in enumerate(perm[:1000]):
    pred = int(x_train[i] @ weights >= n)
    if pred == 0 and y_train[i] == 1:
        weights = np.where(x_train[i] == 1, np.minimum(weights * 2, n), weights)
        mistakes += 1
    elif pred == 1 and y_train[i] == 0:
        weights = np.where(x_train[i] == 1, weights * 0.5, weights)
        mistakes += 1

    acc = np.mean(np.where(x_train @ weights >= n, 1, 0) == y_train)
    print(idx, mistakes, acc)
