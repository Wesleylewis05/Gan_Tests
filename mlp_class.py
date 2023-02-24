import tensorflow as tf
from tensorflow import keras
from keras.models import Model, load_model, Sequential # for assembling a Neural Network model
from keras.layers import ZeroPadding1D, Input, Dense, Embedding, Reshape, Concatenate, Flatten, Dropout, Conv1DTranspose # for adding layers
from keras.layers import Conv2D, Conv2DTranspose, MaxPool2D, ReLU, LeakyReLU, Conv1D, Flatten, MaxPooling1D, BatchNormalization # for adding layers
from keras.utils import plot_model # for plotting model diagram
from keras.optimizers import Adam, SGD # for model optimization 
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from keras.models import load_model
import pydot
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn import metrics
# Data manipulation
import numpy as np # for data manipulation
import pandas as  pd
import numpy.matlib
from numpy.random import randn
from numpy.random import randint
from numpy import zeros
# Visualization
import matplotlib 
import matplotlib.pyplot as plt # for data visualization
import os
from collections import defaultdict
from typing import Dict, List, Tuple
from SlurpData import GenerateDataSet

class MLP:
    def __init__(self, classes, dataset: pd.DataFrame):
        self.classes = classes
        self.model = keras.Model()
        self.dataset = dataset
        self.X = self.dataset.drop(["labels"], axis=1)
        self.y = self.dataset["labels"]
        self.train_values = List
        self.test_values = List
        self.scaler = MinMaxScaler()
        self.train_test_split()
        self.input_len = (self.train_values[0].shape[1],)
        

    def train_test_split(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.10, stratify=self.y)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.3, stratify=y_train)
        # Fit scaler to X_train
        self.scaler.fit(X_train)
        # Scale X
        X_train = self.scaler.transform(X_train)
        X_val = self.scaler.transform(X_val)
        X_test = self.scaler.transform(X_test)
        # Store train and test values
        self.train_values = [X_train, X_val, y_train, y_val]
        self.test_values = [X_test, y_test]

    def create_mlp(self):
        self.model = Sequential()
        self.model.add(Dense(200, activation="relu", input_shape=self.input_len))
        self.model.add(Dense(100, activation="relu"))
        self.model.add(Dense(25, activation="relu"))
        
        self.model.add(Dense(self.classes, activation = 'softmax'))
        opt = SGD(learning_rate=0.01, momentum=0.9)
        self.model.compile(loss = 'sparse_categorical_crossentropy', optimizer=opt,metrics = ['accuracy'])
        return self.model

    def model_summary(self):
        self.model.summary()
        plot_model(self.model,show_shapes=True)

    
    def train_model(self, b_size, e):
        self.model.fit(self.train_values[0], self.train_values[2],\
             batch_size=b_size,epochs=e, verbose=1, validation_data=(self.train_values[1], self.train_values[3]))
        acc = self.model.evaluate(self.test_values[0], self.test_values[1])
        print("Loss:", acc[0], " Accuracy:", acc[1])
        self.model.save("mlpmodel.h5")

    def evaluate_model(self, X, true):
        pred = self.model.predict(X)
        pred_y = pred.argmax(axis=-1)
        accuracy = np.sum(np.equal(true,pred_y))/len(true)
        print("Accuracy: ", accuracy)

    def get_test_values(self):
        return self.test_values[0], self.test_values[1]