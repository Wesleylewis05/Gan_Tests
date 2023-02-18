import tensorflow as tf
from tensorflow import keras
from keras.models import Model, load_model, Sequential # for assembling a Neural Network model
from keras.layers import ZeroPadding1D, Input, Dense, Embedding, Reshape, Concatenate, Flatten, Dropout, Conv1DTranspose # for adding layers
from keras.layers import Conv2D, Conv2DTranspose, MaxPool2D, ReLU, LeakyReLU, Conv1D, Flatten, MaxPooling1D, BatchNormalization # for adding layers
from tensorflow.keras.utils import plot_model # for plotting model diagram
from tensorflow.keras.optimizers import Adam # for model optimization 
from tensorflow.keras.models import load_model
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from keras.models import load_model
import pydot
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
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
spectrometer_collected_data = G.read_numpy_data()
calibrated_data = G.calibrate_data(spectrometer_collected_data)
all_data, labels = G.generate_data_labels(calibrated_data)
all_data = np.hstack((all_data,np.gradient(all_data,axis=1)))
data = pd.DataFrame(all_data)
data["labels"] = labels[:,0]

"""
['Almond Milk' 'Coke' 'Empty' 'Ibuprofen' 'Ketchup' 'Olive oil'
 'Orange Juice' 'Salt' 'Soy Sauce' 'Sugar' 'Tylenol (PM)' 'Vegetable Oil'
 'Water']
[ 0  1  2  3  4  5  6  7  8  9 10 11 12]
"""

data['labels'] = data['labels'].replace([1], 'Coke')
data['labels'] = data['labels'].replace([3], 'Ibuprofen')
data['labels'] = data['labels'].replace([7], 'Salt')
data['labels'] = data['labels'].replace([9], 'Sugar')
data['labels'] = data['labels'].replace([10], 'Tylenol (PM)')
data['labels'] = data['labels'].replace([8], 'Soy Sauce')


#print(np.unique(labels[:,0])) 





X = data.drop(["labels"], axis=1)
y = data["labels"]
classes = len(np.unique(y))
classesx = (np.unique(y))


scalar = MinMaxScaler()
scalar.fit(all_data)
X = scalar.transform(X)

model = load_model('mlpmodel.h5')
pred = model.predict(X)
pred_y = pred.argmax(axis=-1)

print(pred_y)
temp = sum(pred_y == y)
temp = temp/len(y)
print(temp)