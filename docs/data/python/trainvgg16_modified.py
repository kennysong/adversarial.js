import tensorflowjs as tfjs 
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras import models, layers, applications
from tensorflow import keras

def preprocess_data(X, Y):
    X = X.astype('float32')
    X_p = keras.applications.vgg16.preprocess_input(X)
    Y_p = keras.utils.to_categorical(Y, 10)
    return(X_p, Y_p)

if __name__ == "__main__":
    (Xt, Yt), (X, Y) = keras.datasets.cifar10.load_data()
    X_p, Y_p = preprocess_data(Xt, Yt)
    Xv_p, Yv_p = preprocess_data(X, Y)
    base_model = keras.applications.vgg16.VGG16(include_top=False,
                                            weights='imagenet',
                                            pooling='avg'
                                            )
    model= keras.Sequential()
    model.add(keras.layers.UpSampling2D())
    model.add(base_model)
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(512, activation=('relu'))) 
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Dense(256, activation=('relu')))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Dense(10, activation=('softmax')))
    callback = []
    def decay(epoch):
        return 0.001 / (1 + 1 * 30)
    callback += [keras.callbacks.LearningRateScheduler(decay, verbose=1)]
    callback += [keras.callbacks.ModelCheckpoint('cifar10.h5',
                                                  save_best_only=True,
                                                  mode='min'
                                                  )]
    model.compile(optimizer='adam', loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(x=X_p, y=Y_p,
              batch_size=128,
              validation_data=(Xv_p, Yv_p),
              epochs=10, shuffle=True,
              callbacks=callback,
              verbose=1
              )
    
    tfjs.converters.save_keras_model(model, '../cifar10/vgg16')
    model.save('../cifar10/vgg16/vgg-cifar10.h5')