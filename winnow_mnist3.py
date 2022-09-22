#!/usr/bin/env python3
import numpy as np
from tensorflow import keras


def softmax(x):
    x -= np.max(x, axis=-1, keepdims=True)
    return np.exp(x) / np.sum(np.exp(x))

predict = lambda weights, x: softmax(x @ weights)


def train(weights, x, y_pred, y_true, n, lr=2.0, thresh=0.5):
        weights[:, y_pred] = np.where(x >= thresh, weights[:, y_pred] / lr, weights[:, y_pred])
        weights[:, y_true] = np.where(x >= thresh, np.minimum(weights[:, y_true] * lr, n), weights[:, y_true])
        return weights


def train2(weights, x, y_pred, y_true, n, lr=2.0, thresh=0.8):
        losses = np.subtract(y_pred, y_true)
        l_abs = np.minimum(np.abs(losses) * lr, 0.99)
        for i in range(len(losses)):
            if losses[i] > 0.5:
                weights[:, i] = np.where(x >= thresh, weights[:, i] * l_abs[i], weights[:, i])
            elif losses[i] < -0.5:
                weights[:, i] = np.where(x >= thresh, np.minimum(weights[:, i] / l_abs[i], n), weights[:, i])
        return weights


(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

x_train = x_train.reshape(-1, 784) / 255
x_test = x_test.reshape(-1, 784) / 255

y_test_max = y_test

y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)


n = 64
epochs = 5

weights = np.ones((784, 10))

mistakes = 0

for e in range(epochs):
    for i in range(len(x_train)):
        x, y = x_train[i], y_train[i]
        output = predict(weights, x)
        pred = np.argmax(output)
        if pred != np.argmax(y):
            weights = train2(weights, x, output, y, n, 4)
            mistakes += 1
            
        if (i+1) % 20000 == 0:
            print(e+1, (e+1) * i, mistakes, np.mean(np.argmax(predict(weights, x_test), axis=-1) == y_test_max))
