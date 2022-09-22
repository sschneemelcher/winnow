#!/usr/bin/env python3
import numpy as np
from tensorflow import keras
from keras import layers

num_classes = 10
input_shape = (28, 28, 1)

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()


one_train = np.asarray([x_train[i] for i in range(len(y_train)) if y_train[i] == 1])
zero_train = np.asarray([x_train[i] for i in range(len(y_train)) if y_train[i] == 0])

one_test = np.asarray([x_test[i] for i in range(len(y_test)) if y_test[i] == 1])
zero_test = np.asarray([x_test[i] for i in range(len(y_test)) if y_test[i] == 0])

x_train = np.vstack((one_train, zero_train))
y_train = np.append(np.ones(len(one_train)), np.zeros(len(zero_train)))

x_test = np.vstack((one_test, zero_test))
y_test = np.append(np.ones(len(one_test)), np.zeros(len(zero_test)))

x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

n=int(len(x_train) * 0.5)
perm = np.random.permutation(n)
x_train = x_train[perm]
y_train = y_train[perm]

x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255


model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.Flatten(),
        layers.Dense(num_classes, activation="softmax"),
    ]
)

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

model.fit(x_train, y_train, batch_size=1, epochs=1)

score = model.evaluate(x_test, y_test, verbose=0)
print("Test accuracy:", score[1])
