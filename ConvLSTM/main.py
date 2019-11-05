import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pylab as plt

import model as m
import params
import util

#Import data
raw_data = util.import_data()

#For each window, run the algorithm
#for i in range(int(params.TV_SPLIT * len(raw_data)) - 1 - params.PREDICT_WINDOW_SIZE):
for i in range(1):
    print("SIMULATION WINDOW %d OUT OF %d:" % ((i + 1), int(params.TV_SPLIT * len(raw_data)) - 1 - params.PREDICT_WINDOW_SIZE))
    
    #Get empty model
    seq = m.get_model()
    
    #Get training and validation data and remove nans
    xtrain, ytrain, xval, yval = util.split_data(util.remove_nans(raw_data), shift=i)
    
    #Network training
    m.train(seq, xtrain, ytrain, params.NUM_TRAIN_EPOCHS)
    
    #Predict the window
    preds, error = m.predict_window(seq, xtrain, ytrain, xval, yval, params.PREDICT_WINDOW_SIZE, params.NUM_UPDATE_EPOCHS)
    
    #DEBUG
    print("RMSE of %d Weeks" % (params.PREDICT_WINDOW_SIZE))  
    print(error)
    
    
    