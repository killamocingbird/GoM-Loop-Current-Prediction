from scipy.io import loadmat
import numpy as np
import os

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
    
    
    