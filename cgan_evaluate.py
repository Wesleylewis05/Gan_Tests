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
print(tf.__version__)
from SlurpData import GenerateDataSet
from mlp_class import MLP


G = GenerateDataSet()
spectrometer_collected_data = G.read_data()
calibrated_data = G.calibrate_data(spectrometer_collected_data)
all_data, labels = G.generate_data_labels(calibrated_data)
print(np.unique(labels[:,0]))

G.fit_encoding(labels)
encoded_labels, labels_contents, labels_containers = G.transform_encoding(labels)

data = pd.DataFrame(all_data)
data["labels"] = labels_contents
classes = len(np.unique(labels_contents))
classesx = (np.unique(labels_contents))
print(classesx)


def generate_latent_points(latent_dim, n_samples, n_classes=classes):
    x_input = randn(latent_dim * n_samples)
    # reshape into a batch of inputs for the network
    z_input = x_input.reshape(n_samples, latent_dim)
    # generate labels
    labels = randint(0, n_classes, n_samples)
    return [z_input, labels]

# Generate fake data
model = load_model('trained_generated_model.h5')

latent_points, labels = generate_latent_points(100, 16900)

labels = np.asarray([x for _ in range(classes*100) for x in range(classes)])


Q  = model.predict([latent_points, labels])
Q = np.squeeze(Q, axis=2)

# Fake Data DataFrame
fdata = pd.DataFrame(Q)
fdata["labels"] = labels
fdata = fdata.sample(frac = 1)

xX_train, xX_test, xy_train, xy_test = train_test_split(data.iloc[:,0:606], data.iloc[:,-1], test_size=0.10, stratify=data.iloc[:,-1])
xX_train, xX_val, xy_train, xy_val = train_test_split(xX_train, xy_train, test_size=0.30, stratify=xy_train)

# append the fake data to x_train and y_train
xframes = [xX_train, fdata.iloc[:,0:606]]
yframes = [xy_train, fdata.iloc[:,-1]]
xresult = pd.concat(xframes)
yresult = pd.concat(yframes)

def MLP_():
    model = Sequential()
    model.add(Dense(200, activation="relu", input_shape=(xX_train.shape[1],)))
    model.add(Dense(100, activation="relu"))
    model.add(Dense(25, activation="relu"))
    
    model.add(Dense(classes, activation = 'softmax'))
    opt = SGD(learning_rate=0.01, momentum=0.9)
    model.compile(loss = 'sparse_categorical_crossentropy', optimizer=opt,metrics = ['accuracy'])
    model.summary()
    return model

mlp_model = MLP_()
mlp_model.summary()
plot_model(mlp_model,show_shapes=True)

mlp_model.fit(xresult, yresult, batch_size=64,epochs=500, verbose=1, validation_data=(xX_val, xy_val))
acc = mlp_model.evaluate(xX_test, xy_test)
print("Loss:", acc[0], " Accuracy:", acc[1])

# read demo data
spectrometer_collected_data = G.read_numpy_data()
calibrated_data = G.calibrate_data(spectrometer_collected_data)
demo_data, demo_labels = G.generate_data_labels(calibrated_data)
demo_encoded_labels, demo_labels_contents, demo_labels_containers = G.transform_encoding(demo_labels)

demo_df = pd.DataFrame(demo_data)
demo_df["labels"] = demo_labels_contents
X_ = demo_df.drop(["labels"], axis=1)
y_ = demo_df["labels"]

pred = mlp_model.predict(X_)
pred_y = pred.argmax(axis=-1)
accuracy = np.sum(np.equal(y_,pred_y))/len(y_)
print("Accuracy: ", accuracy)


