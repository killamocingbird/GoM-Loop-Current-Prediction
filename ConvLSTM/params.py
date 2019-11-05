"""

Global parameters for simulation

"""


""" MODEL SPECIFICATIONS """

#The dimensions of NUM_FILTERS and KERNEL_SIZES has to equal NUM_LAYERS
NUM_LAYERS = 3
NUM_FILTERS = [36, 36, 1]

#For each ConvLSTM2D or Conv2D layer, specify a 2D kernel size, and for every Conv3D layer, specify a 3D kernel size
KERNEL_SIZES = [(5, 5), (5, 5), (3, 3, 3)]

LOSS = 'mean_squared_error'
OPTIM = 'adam'
ACTIVATION = 'sigmoid'

#DO NOT CHANGE
PADDING = 'same'


""" DATA SPECIFICATIONS """

#Shape of the data
X_SHAPE = 541
Y_SHAPE = 385

#Path to data files (please have them in the following format: {variable_name}_{year}.mat)
DATA_FILES_PATH = 'C:\\Users\\Justin Wang\\Documents\\Data\\GoM LCS\\'

#Variables to search for in DATA_FILES_PATH. The order will determine channel order for model input
#({Variable name}, {mat extract variable})
USE_VARIABLES = [('SSH', 'templayer')]

#Variables years to search for in DATA_FILES_PATH. The order will determine the time sequence
USE_YEARS = [1992]
#Denotes the number of days in the ith year (e.g. there are 366 days in 1992)
NUM_DAYS_IN_YEARS = [366]

#Subsample data from daily to weekly
WEEKLY_SUBSAMPLE = True

if WEEKLY_SUBSAMPLE:
    #Randomly weekly subsamplying
    USE_RANDOM = False
    
    if not USE_RANDOM:
        #A day from 0 to 6 specifying the day of the week to choose
        SUBSAMPLE_DAY = 0
        
DATA_FORMAT = 'channels_last'


""" TRAINING SPECIFICATIONS """

#Percentage of train / validation split
TV_SPLIT = 0.5

#Number of epochs to fit initial model
NUM_TRAIN_EPOCHS = 20

#Number of epochs to train on dynamic prediction
NUM_UPDATE_EPOCHS = 1


""" PREDICTION SPECIFICATIONS """

#Number of weeks ahead to predict
PREDICT_WINDOW_SIZE = 20

#TODO: Output path for results
OUTPUT_PATH = ''

#TODO: Draw error graphs
DRAW_ERROR = True

#TODO: Draw visual graphs of predictions
DRAW_PREDICTIONS = True










