import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pylab as plt
from scipy.io import loadmat

import model as m
import params
import util

#Import data
raw_data = util.import_data()
#Remove nans
util.remove_nans(raw_data)
#Normalize
raw_data = util.normalize(raw_data)

#Load in nan_map
nan_map = loadmat(params.NAN_MAP_PATH)['nan_map'].astype(bool)

#Convert nans back to 0
raw_data = util.reset_nan_values(raw_data, nan_map)

#For each window, run the algorithm
#for i in range(int(params.TV_SPLIT * len(raw_data)) - 1 - params.PREDICT_WINDOW_SIZE):
for i in range(1):
    print("SIMULATION WINDOW %d OUT OF %d:" % ((i + 1), int(params.TV_SPLIT * len(raw_data)) - 1 - params.PREDICT_WINDOW_SIZE))
    
    #Get empty model
    seq = m.get_model()
    
    #Get training and validation data and remove nans
    xtrain, ytrain, xval, yval = util.split_data(raw_data, shift=i)
    
    #Network training
    m.train(seq, xtrain, ytrain, params.NUM_TRAIN_EPOCHS)
    
    #Predict the window
    print("PREDICTING:")
    preds, error = m.predict_window(seq, xtrain, ytrain, xval, yval, nan_map, params.PREDICT_WINDOW_SIZE, params.NUM_UPDATE_EPOCHS, verbose=True)
    
    #DEBUG
    print("RMSE of %d Weeks" % (params.PREDICT_WINDOW_SIZE))
    print(error)
    

    