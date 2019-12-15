from scipy.io import loadmat
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import params

#Imports the data as specified in params.py
def import_data():
    
    total_days = 0
    for num_days in params.NUM_DAYS_IN_YEARS:
        total_days += num_days
        
    data = np.zeros((total_days, 1, params.X_SHAPE, params.Y_SHAPE, len(params.USE_VARIABLES)))
    cur_day = 0
    for i in range(len(params.USE_YEARS)):
        year = params.USE_YEARS[i]
        for j in range(len(params.USE_VARIABLES)):
            var = params.USE_VARIABLES[j]
            temp_data_path = os.path.join(params.DATA_FILES_PATH, var[0] + '_' + str(year) + '.mat')
            temp_data = loadmat(temp_data_path)[var[1]][0]
            #Load in all data
            for k in range(len(temp_data)):
                data[cur_day + k, 0, :, :, j] = temp_data[k]
            #Add up days added to data
            cur_day += len(temp_data)
    
    if params.WEEKLY_SUBSAMPLE:
        if params.USE_RANDOM and False:
            #TODO
            data = data #To not cause compiler error
        else:
            data = data[params.SUBSAMPLE_DAY::7,:,:,:,:]
    
    return data

def normalize(data):
    return (2 * (data - data.min()) / (data.max() - data.min())) - 1

#Splits the data as specified in params.py
def split_data(data, shift=0):
    num_split = int(params.TV_SPLIT * len(data))
    train = data[:num_split + shift]
    val = data[num_split + shift:]
    
    return (train[:-1], train[1:], val[:-1], val[1:])

#Replaces nans in data with 0
def remove_nans(data):
    data[np.isnan(data)] = 0
    return data

#Generates plots and saves predictions to folder
def output_window_results(preds, yval, error, output_folder):
    #Save raw predictions
    np.save(os.path.join(output_folder, 'preds'), preds)
    #Generate plots
    for i in range(len(preds)):
        fig = plt.figure()
        f, axarr = plt.subplots(2, 1, figsize=(7,5))
        #Ground truth
        axarr[0].set_title("Ground Truth - Week %d" % (i + 1))
        axarr[0].imshow(np.rot90(yval[i,0,:,:,0]))
        axarr[0].axis('off')
        
        #Prediction
        axarr[1].set_title("Prediction RMSE: %.4f" % (error[i]))
        axarr[1].imshow(np.rot90(preds[i,0,:,:,0]))
        axarr[1].axis('off')
        
        #Save figure
        plt.savefig(os.path.join(output_folder, str(i + 1)))
        
        #Close figure
        plt.close(fig)
    
#Resets all nan located values to 0
def reset_nan_values(data, nan_map):
    #Assume data is of shape (week, 0, x, y, channel)
    data[:,0,nan_map,:] = 0
    return data
    
    
    
    
    
    
    
    
    
    
    