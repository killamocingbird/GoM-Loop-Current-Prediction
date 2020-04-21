import numpy as np
import os
from scipy.io import loadmat

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader


test_folder = 'C:/Users/Justin Wang/Documents/Data/GoM LCS/'

# Pulls data and returns raw SSH without any preprocessing
class RawSSHDataset(Dataset):
    def __init__(self, data_folder, regex='SSH_', start_year=1992, end_year=2008, subsample_freq=7, shift=0, key='templayer', empty=False):
        files = os.listdir(data_folder)
        self.data = []
        
        # Empty dictates whether or not to make an empty class or load in data
        if not empty:
            for year in tqdm(range(start_year, end_year + 1)):
                if regex + str(year) + '.mat' in files:
                    self.data.append(np.stack(loadmat(os.path.join(data_folder, regex+str(year)+'.mat'))[key][0]).astype(np.float32))
                else:
                    raise Exception("Discontinuous range: %s%d.mat does not exist" % (regex, year))
                
            # Concatenate all data
            self.data = np.concatenate(self.data, 0)
            
            # Subsamples depending on shift
            self.data = self.data[shift%subsample_freq::subsample_freq]
            
            # Cast to tensor and float
            self.data = torch.as_tensor(self.data).float()
            
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    

# Pulls data that's preprocessed with EOF
# class PCASSHDataset(RawSSHDataset):
#     def __init__(self, data_folder, eof_path, regex='SSH_', start_year=1992, end_year=2008, subsample_freq=7, shift=0, key='templayer', empty=False):
#         # Super constructor
#         super().__init__(data_folder, regex='SSH_', start_year=1992, end_year=2008, subsample_freq=7, shift=0, key='templayer', empty=False)
        
        
    