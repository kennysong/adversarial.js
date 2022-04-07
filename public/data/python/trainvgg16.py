import tensorflowjs as tfjs
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras import models, layers, applications
from tensorflow import keras

(x_train,y_train),(x_test,y_test)= keras.datasets.mnist.load_data()
# expand new axis, channel axis 
x_train = np.expand_dims(x_train, axis=-1)

# [optional]: we may need 3 channel (instead of 1)
x_train = np.repeat(x_train, 3, axis=-1)

# it's always better to normalize 
x_train = x_train.astype('float32') / 255

# resize the input shape , i.e. old shape: 28, new shape: 32
x_train = tf.image.resize(x_train, [32,32]) # if we want to resize 

# one hot 
y_train = tf.keras.utils.to_categorical(y_train , num_classes=10)

input = keras.Input(shape=(32,32,3))
model = applications.VGG16(weights='imagenet', include_top = False, input_tensor = input)

global_pool = layers.GlobalMaxPooling2D()(model.output)
out = layers.Dense(10, activation='softmax', use_bias=True)(global_pool)

v16 = keras.Model(model.input, out)


v16.compile(loss = keras.losses.CategoricalCrossentropy(), metrics = keras.metrics.CategoricalAccuracy(), optimizer = keras.optimizers.Adam())

v16.fit(x_train, y_train, batch_size=128, epochs=10, verbose=2)

tfjs.converters.save_keras_model(v16, '../mnist/vgg16')
#model.save('./js/vgg16/vgg16.h5')