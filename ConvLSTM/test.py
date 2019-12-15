import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pylab as plt


seq = keras.models.Sequential()
seq.add(layers.ConvLSTM2D(filters=36, kernel_size=(5, 5), 
                   batch_input_shape=(1, 1, 541, 385, 1),
                   padding='same', return_sequences=True, stateful=True))

seq.add(layers.ConvLSTM2D(filters=36, kernel_size=(5, 5),
                   padding='same', return_sequences=True, stateful=True))

seq.add(layers.Conv3D(filters=1, kernel_size=(2,2,2),
               padding='same', data_format='channels_last'))
seq.compile(loss='mean_squared_error', optimizer='adam')

#Import MovingMNIST data
#Current shape [batch, time, x, y]
DATA_PATH = 'C:\\Users\Justin Wang\Documents\Data\GoM LCS\\SSH_1992.mat'

from scipy.io import loadmat
data = loadmat(DATA_PATH)['templayer'][0]
temp = np.zeros((len(data), 541, 385))
for i in range(len(data)):
    temp[i,:,:] = data[i]
raw_data = temp
data = raw_data

data[np.isnan(data)] = 0



#Make data correspond to 366 batches
data = data[:,np.newaxis,:,:,np.newaxis]

#Subsample to weekly
data = data[0::7,:,:,:,:]

train = data[:40]
val = data[40:]

xtrain = train[:-1]
ytrain = train[1:]

xval = val[:-1]
yval = val[1:]

# Train the network
epochs = 100
for i in range(epochs):
    print("Epoch", (i + 1))
    for j in range(len(xtrain)):
        if (j == len(xtrain) - 1):
            seq.fit(xtrain[j:j+1], ytrain[j:j+1], batch_size=1, epochs=1, shuffle=False)
        else:  
            seq.fit(xtrain[j:j+1], ytrain[j:j+1], batch_size=1, epochs=1, shuffle=False, verbose = 0)
    seq.reset_states()

x_hat = xtrain
y_hat = ytrain
pred = np.zeros((len(xval), 541, 385))
for i in range(10):
    print("FORECASTING WEEK", (i + 1))
    print("Updating state...")
    for j in range(len(x_hat)):
        seq.fit(x_hat[j:j+1], y_hat[j:j+1], batch_size=1, epochs=1, shuffle=False, verbose=0)
    print("Predicting...")
    cur_pred = seq.predict(xval[i:i+1])
    pred[i] = cur_pred[0,0,:,:,0]
    x_hat = np.concatenate((x_hat, xval[i:i+1]))
    y_hat = np.concatenate((y_hat, cur_pred))
    print("Resetting state...")
    seq.reset_states()
        
error = (((pred[0:10, ~np.isnan(raw_data[0])] - yval[0:10,0,~np.isnan(raw_data[0]),0])**2).mean((1)))**(0.5)

    

