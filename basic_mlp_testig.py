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
from mlp_class import MLP
print(tf.__version__)

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

M = MLP(classes, data)
X_test, y_test = M.get_test_values()
M.create_mlp()
M.model_summary()
M.train_model(64, 500)
M.evaluate_model(X_test, y_test)
# read demo data
spectrometer_collected_data = G.read_numpy_data()
calibrated_data = G.calibrate_data(spectrometer_collected_data)
demo_data, demo_labels = G.generate_data_labels(calibrated_data)
demo_encoded_labels, demo_labels_contents, demo_labels_containers = G.transform_encoding(demo_labels)

print(demo_labels[:,0], demo_labels_contents)

demo_df = pd.DataFrame(demo_data)
demo_df["labels"] = demo_labels_contents
X_ = demo_df.drop(["labels"], axis=1)
y_ = demo_df["labels"]


#X_ = M.scaler.transform(X_)
M.evaluate_model(demo_data , demo_labels_contents)

# confusion_matrix = metrics.confusion_matrix(labels_contents, pred_y)
# print(confusion_matrix)
