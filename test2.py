import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense, Flatten, Conv2D
from keras import Model
import os

inputs = tf.keras.Input(shape=(3, 3, 2))
x = tf.keras.layers.Dense(4, activation=tf.nn.relu)(inputs)
outputs = tf.keras.layers.Dense(5, activation=tf.nn.softmax)(x)
model = tf.keras.Model(inputs=inputs, outputs=outputs)  
model.summary()


directory = 'model'
if not os.path.exists(directory):
    os.makedirs(directory)
model.save('model/saved_model')