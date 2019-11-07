import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

import params
import util

def get_model():
    seq = keras.models.Sequential()
    #Add initial layer
    seq.add(layers.ConvLSTM2D(filters=params.NUM_FILTERS[0], kernel_size=params.KERNEL_SIZES[0],
                                      batch_input_shape=(1, 1, params.X_SHAPE, params.Y_SHAPE, len(params.USE_VARIABLES)),
                                      padding=params.PADDING, return_sequences=True, stateful=True))
    
    #Add hidden layers
    for i in range(1, params.NUM_LAYERS):
        if (len(params.KERNEL_SIZES[i]) == 2):
            #ConvLSTM Layer
            seq.add(layers.ConvLSTM2D(filters=params.NUM_FILTERS[i], kernel_size=params.KERNEL_SIZES[i],
                                      padding=params.PADDING, return_sequences=True, stateful=True))
        else:
            #Conv3D Layer
            seq.add(layers.Conv3D(filters=params.NUM_FILTERS[i], kernel_size=params.KERNEL_SIZES[i],
                                  activation='tanh', padding=params.PADDING, data_format=params.DATA_FORMAT))
    
    #Compile
    seq.compile(loss=params.LOSS, optimizer=params.OPTIM)
    
    return seq

def train(seq, xtrain, ytrain, epochs, verbose=True, reset_states=True):
    if verbose: print("TRAINING:")
    for epoch in range(epochs):
        if verbose: print("Epoch %d:" % (epoch + 1))
        for i in range(len(xtrain)):
            #Print on last iteration
            if i == len(xtrain) - 1 and verbose:
                seq.fit(xtrain[i:i+1], ytrain[i:i+1], batch_size=1, epochs=1, shuffle=False)
            else:
                seq.fit(xtrain[i:i+1], ytrain[i:i+1], batch_size=1, epochs=1, shuffle=False, verbose=0)
        if epoch != epochs - 1 or (epoch == epochs - 1 and reset_states):
            seq.reset_states()
            
    if verbose: print("DONE")
    
def predict_window(seq, xtrain, ytrain, xval, yval, nan_map, window_length, update_epochs, verbose=False):
    x_hat = xtrain
    y_hat = ytrain
    pred = np.zeros((window_length, 1, 541, 385, xtrain.shape[4]))
    for i in range(window_length):
        if verbose: print("Week %d..." % (i + 1))
        #Update state
        train(seq, x_hat, y_hat, update_epochs, verbose=False, reset_states=False)
        #Predict
        cur_pred = seq.predict(y_hat[-1:])
        cur_pred = util.reset_nan_values(cur_pred, nan_map)
        pred[i] = cur_pred[0]
        #Dynamically add onto x_hat and y_hat
        x_hat = np.concatenate((x_hat, yval[-1:]))
        y_hat = np.concatenate((y_hat, cur_pred))
        #Reset state for next prediction
        seq.reset_states()
    #Find RMSE
    errors = (((pred - yval[:window_length])**2).mean((1, 2, 3, 4)))**(0.5)
    return (pred, errors)
        
            
    
    
    
    
    
    
    
    
            