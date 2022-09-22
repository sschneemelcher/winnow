#!/usr/bin/env python3
import numpy as np
from tensorflow import keras


predict = lambda weights, x: np.argmax(x @ weights, axis=-1)

def train(weights, x, y_pred, y_true, n, lr=2.0, thresh=0.8):
        weights[:, y_pred] = np.where(x >= thresh, weights[:, y_pred] / lr, weights[:, y_pred])
        weights[:, y_true] = np.where(x >= thresh, np.minimum(weights[:, y_true] * lr, n), weights[:, y_true])
        return weights


(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

x_train = x_train.reshape(-1, 784) / 255
x_test = x_test.reshape(-1, 784) / 255

n = 64
weights = np.ones((784, 10))
mistakes = 0
lr = 1.001
thresh = 0.75
epochs = 5

for e in range(epochs):
    for i in range(len(x_train)):
        x, y = x_train[i], y_train[i]
        pred = predict(weights, x)
    
        if pred != y:
            mistakes += 1
            weights = train(weights, x, pred, y, n, lr, thresh)
        
    acc = np.mean(predict(weights, x_test) == y_test)
    print(e, mistakes, acc)
