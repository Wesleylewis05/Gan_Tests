import tensorflow as tf
from tensorflow import keras
from keras.models import Model, load_model, Sequential # for assembling a Neural Network model
from keras.layers import ZeroPadding1D, Input, Dense, Embedding, Reshape, Concatenate, Flatten, Dropout, Conv1DTranspose # for adding layers
from keras.layers import Conv2D, Conv2DTranspose, MaxPool2D, ReLU, LeakyReLU, Conv1D, Flatten, MaxPooling1D, BatchNormalization # for adding layers
from tensorflow.keras.utils import plot_model # for plotting model diagram
from tensorflow.keras.optimizers import Adam, SGD # for model optimization 
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from keras.models import load_model
import pydot
from sklearn.preprocessing import MinMaxScaler
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
print(tf.__version__)

G = GenerateDataSet()
spectrometer_collected_data = G.read_data()
calibrated_data = G.calibrate_data(spectrometer_collected_data)
all_data, labels = G.generate_data_labels(calibrated_data)
all_data = np.hstack((all_data,np.gradient(all_data,axis=1)))

print(np.unique(labels[:,0]))

# Fit labels to model
le_contents = preprocessing.LabelEncoder()
le_containers = preprocessing.LabelEncoder()
lcontents = le_contents.fit(labels[:,0])
labels_contents = le_contents.transform(labels[:,0])
lcontainers = le_containers.fit(labels[:,1])
labels_containers = le_containers.transform(labels[:,1])
encoded_labels = np.vstack((labels_containers,labels_contents)).T


data = pd.DataFrame(all_data)
data["labels"] = labels_contents
classes = len(np.unique(labels_contents))
classesx = (np.unique(labels_contents))
print(classesx)

X = data.drop(["labels"], axis=1)
y = data["labels"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.3, stratify=y_train)


newdata_x = X_train.copy()
newdata_y = y_train.copy()
newdata = pd.DataFrame(newdata_x)
newdata["labels"] = (newdata_y)


size_ = data.shape[1] 


def MLP_():
    model = Sequential()
    model.add(Dense(200, activation="relu", input_shape=(X_train.shape[1],)))
    model.add(Dense(100, activation="relu"))
    model.add(Dense(25, activation="relu"))
    
    model.add(Dense(classes, activation = 'softmax'))
    opt = SGD(lr=0.01, momentum=0.9)
    model.compile(loss = 'sparse_categorical_crossentropy', optimizer=opt,metrics = ['accuracy'])
    model.summary()
    return model

mlp_model = MLP_()
mlp_model.summary()
plot_model(mlp_model,show_shapes=True)


mlp_model.fit(X_train, y_train, batch_size=64,epochs=500, verbose=1, validation_data=(X_val, y_val))
mlp_model.save("mlpmodel.h5")
acc = mlp_model.evaluate(X_test, y_test)
print("Loss:", acc[0], " Accuracy:", acc[1])
pred = mlp_model.predict(X_test)
pred_y = pred.argmax(axis=-1)
temp = sum(pred_y == y_test)
temp = 10*(temp/len(y))
print(temp)

demo_collected_data = G.read_numpy_data()
spectrometer_collected_data = G.read_numpy_data()
calibrated_data = G.calibrate_data(spectrometer_collected_data)
all_data, labels = G.generate_data_labels(calibrated_data)
all_data = np.hstack((all_data,np.gradient(all_data,axis=1)))
labels_contents = le_contents.transform(labels[:,0])
print(labels[:,0], labels_contents)
pred = mlp_model.predict(all_data)
pred_y = pred.argmax(axis=-1)


print(pred_y,np.unique(pred_y),len(np.unique(pred_y)), labels_contents,sep="\n")

correct = (labels_contents == pred_y)
accuracy = correct.sum() / correct.size
print(accuracy)

confusion_matrix = metrics.confusion_matrix(labels_contents, pred_y)
print(confusion_matrix)
