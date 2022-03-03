# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 22:44:51 2022

@author: nawaz
"""

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution3D, MaxPooling3D, ZeroPadding3D
from tensorflow.keras.layers import BatchNormalization

def get_model(summary=True):
    """ Return the Keras model of the network
    """
    model = Sequential()
    # 1st Volumetric Convolutional block
    model.add(Convolution3D(8, (3, 3, 3), activation='relu', padding='same', input_shape=(64, 64, 48, 1)))
    model.add(Convolution3D(8, (3, 3, 3), activation='relu', padding='same'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    # 2nd Volumetric Convolutional block
    model.add(Convolution3D(16, (3, 3, 3), activation='relu', padding='same'))
    model.add(Convolution3D(16, (3, 3, 3), activation='relu', padding='same'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    # 3rd Volumetric Convolutional block
    model.add(Convolution3D(32, (3, 3, 3), activation='relu', padding='same'))
    model.add(Convolution3D(32, (3, 3, 3), activation='relu', padding='same'))
    model.add(Convolution3D(32, (3, 3, 3), activation='relu', padding='same'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    # 4th Volumetric Convolutional block
    model.add(Convolution3D(64, (3, 3, 3), activation='relu', padding='same'))
    model.add(Convolution3D(64, (3, 3, 3), activation='relu', padding='same'))
    model.add(Convolution3D(64, (3, 3, 3), activation='relu', padding='same'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    model.add(Flatten())
    # 1th Deconvolutional layer with batchnorm and dropout for regularization
    model.add(Dense(128, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.7))
    # 2th Deconvolutional layer
    model.add(Dense(64, activation='relu'))
    #model.add(BatchNormalization())
    #model.add(Dropout(0.7))
    # Output with softmax nonlinearity for classification
    model.add(Dense(5, activation='softmax'))
    if summary:
        print(model.summary())
    return model