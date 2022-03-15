from operator import mod
import tensorflow as tf
from tensorflow.keras import datasets, layers, models, losses
import os

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train = tf.pad(x_train, [[0,0], [2,2], [2,2]]) / 255
x_test = tf.pad(x_test, [[0,0], [2,2], [2,2]]) / 255

x_train = tf.expand_dims(x_train, axis=3, name=None)
x_test = tf.expand_dims(x_test, axis=3, name=None)


x_val = x_train[-2000:,:,:,:]
y_val = y_train[-2000:]
x_train = x_train[-2000:,:,:,:]
y_train = y_train[-2000:]

model = models.Sequential()
model.add(layers.Conv2D(6, 5, activation='tanh', input_shape=x_train.shape[1:]))
model.add(layers.AveragePooling2D(2))
model.add(layers.Activation('sigmoid'))
model.add(layers.Conv2D(16, 5, activation='tanh'))
model.add(layers.AveragePooling2D(2))
model.add(layers.Activation('sigmoid'))
model.add(layers.Conv2D(120, 5, activation='tanh'))
model.add(layers.Flatten())
model.add(layers.Dense(84, activation='tanh'))
model.add(layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss=losses.sparse_categorical_crossentropy, metrics=['accuracy'])
history = model.fit(x_train, y_train, batch_size=64, epochs=40, validation_data=(x_val, y_val))

model.evaluate(x_test, y_test)

model_json = model.to_json()
with open("mnist_lenet.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("mnist_lenet_weights.h5")