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
from SlurpData import GenerateDataSet
print(tf.__version__)


G = GenerateDataSet()
spectrometer_collected_data = G.read_data()
calibrated_data = G.calibrate_data(spectrometer_collected_data)
all_data, labels = G.generate_data_labels(calibrated_data)
all_data = np.hstack((all_data,np.gradient(all_data,axis=1)))

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


def generate_latent_points(latent_dim, n_samples, n_classes=classes):
    x_input = randn(latent_dim * n_samples)
    # reshape into a batch of inputs for the network
    z_input = x_input.reshape(n_samples, latent_dim)
    # generate labels
    labels = randint(0, n_classes, n_samples)
    return [z_input, labels]


# use the generator to generate n fake examples, with class labels
def generate_fake_samples(generator, latent_dim, n_samples):
    # generate points in latent space
    z_input, labels_input = generate_latent_points(latent_dim, n_samples)
    # predict outputs
    images = generator.predict([z_input, labels_input])
    # create class labels
    y = zeros((n_samples, 1))
    return [images, labels_input], y


# generate n real samples with class labels; We randomly select n samples from the real data
def generate_real_samples(n):
    X = data.sample(n)
    x = X.iloc[:,0:606]
    labels = X.iloc[:,-1]
    y = np.ones((n, 1))    
    return [x, labels], y

def define_generator(latent_dim, c=classes):
    # label input
    in_label = Input(shape=(1,))
    label_embedding = Embedding(c, latent_dim)(in_label)
    n_nodes = 1 * X_train.shape[1]
    li = Dense(n_nodes)(label_embedding)
    li = Reshape((X_train.shape[1], 1))(li)
    # image generator input
    n_nodes = 1 * X_train.shape[1]
    in_lat = Input(shape=(latent_dim,))
    gen = Dense(n_nodes)(in_lat)
    gen = LeakyReLU(alpha=0.2)(gen)
    gen = Reshape((X_train.shape[1], 1))(gen)
    # merge image gen and label input
    merge = Concatenate()([gen, li])
    
    gen = Conv1DTranspose(32, 4, padding='same')(merge)
    gen = BatchNormalization(momentum=0.8)(gen)
    gen = LeakyReLU(alpha=0.2)(gen)
    
    gen = Conv1DTranspose(64, 4, padding='same')(gen)
    gen = BatchNormalization(momentum=0.8)(gen)
    gen = LeakyReLU(alpha=0.2)(gen)
    
    gen = Conv1DTranspose(128, 4, padding='same')(gen)
    gen = BatchNormalization(momentum=0.8)(gen)
    gen = LeakyReLU(alpha=0.2)(gen)
    gen = Dropout(0.7)(gen)
    
    # output
    out_layer = Conv1DTranspose(1, 1, activation='tanh', padding='same')(gen)
    model = Model([in_lat, in_label], out_layer)
    return model


def define_discriminator(n_inputs=X_train.shape[1], n_c = classes):
    in_shape = (X_train.shape[1], 1)
    # label input
    in_label = Input(shape=(1,))
    label_embedding = Embedding(n_c, n_inputs)(in_label)
    n_nodes = X_train.shape[1] * 1
    li = Dense(n_nodes)(label_embedding)
    li = Reshape((X_train.shape[1], 1))(li)
    # image input
    in_image = Input(shape=in_shape)
    # concat label as a channel
    merge = Concatenate()([in_image, li])
    fe = Conv1D(filters=128, kernel_size=4)(merge)
    #fe = BatchNormalization()(fe)
    fe = LeakyReLU(alpha=0.2)(fe)
    
    fe = Conv1D(64, kernel_size=4)(fe) 
    #fe = BatchNormalization()(fe)
    fe = LeakyReLU(alpha=0.2)(fe)
    fe = ZeroPadding1D()(fe)
    
    fe = Conv1D(128, kernel_size=4)(fe)
    #fe = BatchNormalization()(fe)
    fe = LeakyReLU(alpha=0.2)(fe)
    fe = ZeroPadding1D()(fe)
    
    fe = Conv1D(256, kernel_size=4)(fe)
    #fe = BatchNormalization()(fe)
    fe = LeakyReLU(alpha=0.2)(fe)
    fe = ZeroPadding1D()(fe)
    
    fe = Conv1D(512, kernel_size=4)(fe)
    #fe = BatchNormalization()(fe)
    fe = LeakyReLU(alpha=0.2)(fe)
    fe = ZeroPadding1D()(fe)
    
    fe = Flatten()(fe)
    fe = Dropout(0.1)(fe)
    # output
    out_layer = Dense(1, activation='sigmoid')(fe)
    # define model
    model = Model([in_image, in_label], out_layer)
    opt = "SGD"
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model

# define the combined generator and discriminator model, for updating the generator
def define_gan(generator, discriminator):
    # make weights in the discriminator not trainable
    discriminator.trainable = False    
    # get noise and label inputs from generator model
    gen_noise, gen_label = generator.input
    # get image output from the generator model
    gen_output = generator.output
    # connect image output and label input from generator as inputs to discriminator
    gan_output = discriminator([gen_output, gen_label])
    # define gan model as taking noise and label and outputting a classification
    model = Model([gen_noise, gen_label], gan_output)
    # compile model
    opt = Adam(learning_rate=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt) 
    return model

# create a line plot of loss for the gan and save to file
def plot_history(d_hist, g_hist):
    # plot loss
    plt.subplot(1, 1, 1)
    plt.plot(d_hist, label='d')
    plt.plot(g_hist, label='gen')
    plt.show()
    plt.close()


# train the generator and discriminator
def train(g_model, d_model, gan_model, latent_dim, n_epochs=30, n_batch=32, dataset=data):    # determine half the size of one batch, for updating the  discriminator
    half_batch = int(n_batch / 2)
    bat_per_epo = int(dataset[0].shape[0] / n_batch)
    d_history = []
    g_history = []    # manually enumerate epochs
    for epoch in range(n_epochs):
        # enumerate batches over the training set
        for j in range(bat_per_epo):
            [X_real, labels_real], y_real = generate_real_samples(half_batch)    # prepare fake examples
            # generate 'fake' examples
            [X_fake, labels], y_fake = generate_fake_samples(g_model, latent_dim, half_batch)    
            # update discriminator
            d_loss_real, d_real_acc = d_model.train_on_batch([X_real, labels_real], y_real)
            d_loss_fake, d_fake_acc = d_model.train_on_batch([X_fake, labels], y_fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)    
            # prepare points in latent space as input for the generator
            [z_input, labels_input] = generate_latent_points(latent_dim, n_batch) 
            # create inverted labels for the fake samples
            y_gan = np.ones((n_batch, 1))    
            # update the generator via the discriminator's error
            g_loss_fake = gan_model.train_on_batch([z_input, labels_input], y_gan)   
            print('>%d, d1=%.3f, d2=%.3f d=%.3f g=%.3f' % (epoch+1, d_loss_real, d_loss_fake, d_loss,  g_loss_fake))    
            d_history.append(d_loss)
            g_history.append(g_loss_fake)    
            #plot_history(d_history, g_history)    
            g_model.save('trained_generated_model.h5')


# size of the latent space
latent_dim = 100 # create the discriminator
discriminator = define_discriminator()# create the generator
generator = define_generator(latent_dim)# create the gan
gan_model = define_gan(generator, discriminator)# train model
train(generator, discriminator, gan_model, latent_dim)