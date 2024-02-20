import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import json
import os
import math
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras
import tensorflow as tf
import structlog
import librosa
import matplotlib.pyplot as plt
import sounddevice as sd
from scipy.io.wavfile import write

DATA_PATH = "data1.json"
with open(DATA_PATH, "r") as fp:
        data = json.load(fp)

X = np.array(data["mfcc"])
y = np.array(data["labels"])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=3)
X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=2)

X_train = X_train[..., np.newaxis]
X_validation = X_validation[..., np.newaxis]
X_test = X_test[..., np.newaxis]

input_shape = (X_train.shape[1], X_train.shape[2], 1)


model = keras.Sequential()
    
model.add(keras.layers.Conv2D(128, (3, 3), activation='relu', input_shape=input_shape))
model.add(keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'))
model.add(keras.layers.BatchNormalization()) 

model.add(keras.layers.Conv2D(32, (2, 2), activation='relu', input_shape=input_shape))
model.add(keras.layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same'))
model.add(keras.layers.BatchNormalization())

model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(64, activation='relu'))
model.add(keras.layers.Dropout(0.3))

model.add(keras.layers.Dense(3, activation='softmax'))

optimiser = keras.optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer=optimiser,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(X_train, y_train, validation_data=(X_validation, y_validation), batch_size=32, epochs=30)
model.save('weights.h5')