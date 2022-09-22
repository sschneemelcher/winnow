#!/usr/bin/env python3
import numpy as np
from tensorflow import keras


(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

x_train = x_train.reshape(-1, 784) / 255
x_test = x_test.reshape(-1, 784) / 255

n = 64
weights = np.ones((784, 10))
mistakes = 0
lam = 1.001
thresh = 0.75
for i in range(len(x_train)):

    output = x_train[i] @ weights
    pred = np.argmax(output)

    if pred != y_train[i]:
        mistakes += 1
        weights[:, pred] = np.where(x_train[i] >= thresh, weights[:, pred] * (1/lam), weights[:, pred])
        weights[:, y_train[i]] = np.where(x_train[i] >= thresh, np.minimum(weights[:, y_train[i]] * lam, n), weights[:, y_train[i]])
    
acc = np.mean(np.argmax(x_train @ weights, axis=-1) == y_train)
print(mistakes, acc)
