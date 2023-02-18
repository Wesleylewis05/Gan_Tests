import tensorflow as tf
from tensorflow import keras
from keras.models import Model, load_model, Sequential # for assembling a Neural Network model
from keras.layers import ZeroPadding1D, Input, Dense, Embedding, Reshape, Concatenate, Flatten, Dropout, Conv1DTranspose # for adding layers
from keras.layers import Conv2D, Conv2DTranspose, MaxPool2D, ReLU, LeakyReLU, Conv1D, Flatten, MaxPooling1D, BatchNormalization # for adding layers
from tensorflow.keras.utils import plot_model # for plotting model diagram
from tensorflow.keras.optimizers import Adam # for model optimization 
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
from .basic_cgan_testig import generate_latent_points

# Read in all data
read_files = os.listdir('data/container_content')
readings = {z.split('.')[0]:pd.read_csv(os.path.join('./data/container_content',z)) for z in read_files}
# Remove extra data fields from the readings
for key in readings.keys():
    try:
        readings[key] = readings[key].loc[:,list(readings[key].columns)[:-4]].to_numpy().astype(np.int32)
    except Exception as e:
        print(str(e))

# Read in calibration data
hamamatsu_dark = np.median(pd.read_csv('./calibration/hamamatsu_black_ref.csv').to_numpy().astype(np.int32), axis=0)
hamamatsu_white = np.median(pd.read_csv('./calibration/hamamatsu_white_ref.csv').to_numpy().astype(np.int32), axis=0)
mantispectra_dark = np.median(pd.read_csv('./calibration/mantispectra_black_ref.csv').to_numpy()[:,:-5].astype(np.int32), axis=0)
mantispectra_white = np.median(pd.read_csv('./calibration/mantispectra_white_ref.csv').to_numpy()[:,:-5].astype(np.int32), axis=0)

# Create composite calibration file
white_ref = np.concatenate((hamamatsu_white, mantispectra_white))[1:]
dark_ref = np.concatenate((hamamatsu_dark, mantispectra_dark))[1:]

# Create calibration function
def spectral_calibration(reading):
    t = np.divide((reading-dark_ref), (white_ref-dark_ref), where=(white_ref-dark_ref)!=0)
    # Handle cases where there is null division, which casts values as "None"
    if np.sum(t==None) > 0:
        print('Null readings!')
    t[t== None] = 0
    # Handle edge cases with large spikes in data, clip to be within a factor of the white reference to avoid skewing the model
    t = np.clip(t,-2.5,2.5)
    return t

# Calibrate all the data
readings_cal = {}
for key in readings.keys():
    readings_cal[key] = np.apply_along_axis(spectral_calibration,1,readings[key])


# Read in the container-substrate pairings
pairings = pd.read_csv('./data/container_substrate.csv',header=1, keep_default_na=False)
# Remove blank data rows
pairings = pairings.loc[:18,(pairings.columns)[:20]]
# Unique substances
contents = list(pairings.columns[1:])



# Containers to exclude - wood, stainless steel, aluminum
exclude_containers = ['O','P','Q','I','K','M']
exclude_contents = [15,2,0,7,10]
# Generalized function to group data by the contents type

def random_scale(reading: np.array) -> np.array:
    return reading * np.random.default_rng().normal(1.0,0.05,1)[0]

def generate_data_labels(readings: Dict) -> defaultdict:
    data_by_contents = np.array([])
    labels_by_contents = np.array([])

    # Iterate over all data_frames types
    for key in readings.keys():
        # Iterate over all containers, but skip Aluminum (P), Stainless Steel (Q), and Wood (R)
        if key[0] in exclude_containers or (len(key) > 1 and int(key[1:]) in exclude_contents): #:or int(key[1:]) in exclude_contents:
            continue
        for index, val in enumerate(contents):
            if key not in list(pairings[val]):
                continue
            # Otherwise the data is useful to use, let's proceed with the data wrangling
            useData = readings[key]
            # ADD SCALING NOISE TO THE DATA HERE
            useData = np.matlib.repmat(useData,3,1)
            useData = np.apply_along_axis(random_scale,1,useData)
            # Get the plain name of the container
            useContainer = pairings[np.equal.outer(pairings.to_numpy(copy=True),  [key]).any(axis=1).all(axis=1)]['container / substrate'].iloc[0]
            # Add the index as the key value
            data_by_contents = np.vstack((data_by_contents, useData)) if data_by_contents.size else useData
            labels_by_contents = np.vstack((labels_by_contents, np.matlib.repmat([val,useContainer],useData.shape[0],1))) if labels_by_contents.size else np.matlib.repmat([val,useContainer],useData.shape[0],1)
    return data_by_contents, labels_by_contents



all_data, labels = generate_data_labels(readings_cal)
all_data = np.hstack((all_data,np.gradient(all_data,axis=1)))


print(np.unique(labels[:,0]))



# Fit labels to model
le_contents = preprocessing.LabelEncoder()
le_containers = preprocessing.LabelEncoder()
labels_contents = le_contents.fit_transform(labels[:,0])
labels_containers = le_containers.fit_transform(labels[:,1])
encoded_labels = np.vstack((labels_containers,labels_contents)).T


data = pd.DataFrame(all_data)
data["labels"] = labels_contents
classes = len(np.unique(labels_contents))
classesx = (np.unique(labels_contents))

X = data.drop(["labels"], axis=1)
y = data["labels"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.3, stratify=y_train)


newdata_x = X_train.copy()
newdata_y = y_train.copy()
newdata = pd.DataFrame(newdata_x)
newdata["labels"] = (newdata_y)


size_ = data.shape[1] 

model = load_model('trained_generated_model.h5')
latent_points, labels = generate_latent_points(100, 16900)
labels = np.asarray([x for _ in range(classes*100) for x in range(classes)])


# generate data and labels
Q  = model.predict([latent_points, labels])
Q = (Q + 1) / 2.0
Q = np.squeeze(Q, axis=2)

fdata = pd.DataFrame(Q)
fdata["labels"] = labels
fdata = fdata.sample(frac = 1)


xX_train, xX_test, xy_train, xy_test = train_test_split(data.iloc[:,0:606], data.iloc[:,-1], test_size=0.10, stratify=data.iloc[:,-1])
xX_train, xX_val, xy_train, xy_val = train_test_split(xX_train, xy_train, test_size=0.30, stratify=xy_train)


xframes = [xX_train, fdata.iloc[:,0:606]]
yframes = [xy_train, fdata.iloc[:,-1]]
xresult = pd.concat(xframes)
yresult = pd.concat(yframes)