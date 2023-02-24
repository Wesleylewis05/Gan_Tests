import tensorflow as tf
from tensorflow import keras
from keras.models import Model, load_model, Sequential # for assembling a Neural Network model
from keras.layers import ZeroPadding1D, Input, Dense, Embedding, Reshape, Concatenate, Flatten, Dropout, Conv1DTranspose # for adding layers
from keras.layers import Conv2D, Conv2DTranspose, MaxPool2D, ReLU, LeakyReLU, Conv1D, Flatten, MaxPooling1D, BatchNormalization # for adding layers
from keras.utils import plot_model # for plotting model diagram
from keras.optimizers import Adam # for model optimization 
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

class GenerateDataSet:

    def __init__(self):
        # Read in the container-substrate pairings
        pairings = pd.read_csv('./data/container_substrate.csv',header=1, keep_default_na=False)
        # Remove blank data rows
        self.pairings = pairings.loc[:18,(pairings.columns)[:20]]
        # Unique substances
        self.contents = list(pairings.columns[1:])
        # Containers to exclude - wood, stainless steel, aluminum
        self.exclude_containers = ['O','P','Q','I','K','M']
        self.exclude_contents = [15,2,0,7,10]
        self.le_contents = preprocessing.LabelEncoder()
        self.le_containers = preprocessing.LabelEncoder()

    def read_data(self):
        read_files = os.listdir('data/container_content')
        readings = {z.split('.')[0]:pd.read_csv(os.path.join('./data/container_content',z)) for z in read_files}
        # Remove extra data fields from the readings
        for key in readings.keys():
            try:
                readings[key] = readings[key].loc[:,list(readings[key].columns)[:-4]].to_numpy().astype(np.int32)

            except Exception as e:
                print(str(e))
        return readings

    def read_numpy_data(self):
        read_files = os.listdir('data/demo_data')
        readings = {z.split('.')[0]:np.load(os.path.join('./data/demo_data',z)) for z in read_files}            
        
        for key in readings.keys():
            try:
                if len(np.shape(readings[key])) > 2:
                    readings[key] = np.squeeze(readings[key])
                readings[key] = pd.DataFrame(readings[key])
                readings[key] = readings[key].loc[:,list(readings[key].columns)[:-1]].tail(1).to_numpy().astype(np.int32)
            except Exception as e:
                print(str(e))
        return readings
    
    def calibrate_data(self, readings):
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
        return readings_cal

    # Generalized function to group data by the contents type
    def random_scale(self, reading: np.array) -> np.array:
        return reading * np.random.default_rng().normal(1.0,0.05,1)[0]

    def generate_data_labels(self, readings: Dict) -> defaultdict:
        data_by_contents = np.array([])
        labels_by_contents = np.array([])

        # Iterate over all data_frames types
        for key in readings.keys():
            # Iterate over all containers, but skip Aluminum (P), Stainless Steel (Q), and Wood (R)
            if key[0] in self.exclude_containers or (len(key) > 1 and int(key[1:]) in self.exclude_contents): #:or int(key[1:]) in exclude_contents:
                continue
            for index, val in enumerate(self.contents):
                if key not in list(self.pairings[val]):
                    continue
                # Otherwise the data is useful to use, let's proceed with the data wrangling
                useData = readings[key]
                # ADD SCALING NOISE TO THE DATA HERE
                useData = np.matlib.repmat(useData,3,1)
                useData = np.apply_along_axis(self.random_scale,1,useData)
                # Get the plain name of the container
                useContainer = self.pairings[np.equal.outer(self.pairings.to_numpy(copy=True),  [key]).any(axis=1).all(axis=1)]['container / substrate'].iloc[0]
                # Add the index as the key value
                data_by_contents = np.vstack((data_by_contents, useData)) if data_by_contents.size else useData
                labels_by_contents = np.vstack((labels_by_contents, np.matlib.repmat([val,useContainer],useData.shape[0],1))) if labels_by_contents.size else np.matlib.repmat([val,useContainer],useData.shape[0],1)

        # apply gradient to data
        data_by_contents = np.hstack((data_by_contents,np.gradient(data_by_contents,axis=1)))

        return data_by_contents, labels_by_contents

    def fit_encoding(self, l):
        # Fit labels to model
        # fit contents
        self.le_contents.fit(l[:,0])
        # fit containers
        self.le_containers.fit(l[:,1])

    def transform_encoding(self, l):
        # Fit labels to model
        labels_contents = self.le_contents.transform(l[:,0])
        labels_containers = self.le_containers.transform(l[:,1])
        encoded_labels = np.vstack((labels_containers,labels_contents)).T
        return encoded_labels, labels_contents, labels_containers
                

# G = GenerateDataSet()
# spectrometer_collected_data = G.read_numpy_data()
# calibrated_data = G.calibrate_data(spectrometer_collected_data)
# all_data, labels = G.generate_data_labels(calibrated_data)
# all_data = np.hstack((all_data,np.gradient(all_data,axis=1)))
# print(all_data)