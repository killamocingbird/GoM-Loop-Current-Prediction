import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from AdamHD import AdamHD
from lbfgs import Stoch_LBFGS
import model as m
import sys
sys.path.append('../')
import dataset as d
import util as u

""" Helper functions """

def loss_weight(x, shift=6, base=0.1, magnitude=3):
    #Assume x is of shape [1, L, time]
    weights =  magnitude / (1 + torch.exp(torch.arange(x.shape[1]).float() - shift)) + base
    
    return x * weights.expand(x.shape[2], len(weights)).t().unsqueeze(0).to(x.device)

def loss_f(y_hat, y, use_weight = True):
    # Find squared error
    loss = (y_hat - y)**2
    if use_weight:
        loss = loss_weight(loss)
    return loss.mean()

def reconstruct(data, injection_map, EOFs_path):
    # data is of shape [time, variable]
    # injection_map is of shape [x, y]
    # Multiply by EOFs
    eofs = torch.load(EOFs_path, map_location=data.device)
    temp = torch.matmul(data, eofs.t())
    ret = torch.zeros(len(data), injection_map.shape[0], injection_map.shape[1]).to(data.device)
    ret[injection_map.expand(len(data), injection_map.shape[0], injection_map.shape[1])] = temp.view(-1)
    return ret

def predict_forward(model, data, num_steps, injection_map, EOFs_path):
    # Load in EOFs
    eofs = torch.load(EOFs_path, map_location=data.device)
    # injection_map is of shape [x, y]
    ret = torch.zeros(num_steps, injection_map.shape[0], injection_map.shape[1]).to(data.device)
    with torch.no_grad():
        for i in range(num_steps):
            # Model data format [1, variables, time]
            pred = model(data)[:,:,-1:]
            
            # Reshape prediction and insert into ret
            ret[i][injection_map] = torch.matmul(pred.squeeze(), eofs.t()).view(-1)
            
            data = torch.cat((data, pred), 2)
    return ret

""" Parameters for script """
### General ###
device = 'cuda' if torch.cuda.is_available() else 'cpu'     # Device for training
header = 'TCN_Fine_'                                             # Header for run
seed = 0                                                    # Seeds for stochastic reproducability

### Training ###
DO_TRAIN = True
#checkpoint = header+'checkpoint.pth'                        # Checkpoint to load training from
checkpoint = None
#criteria = loss_f
criteria = nn.MSELoss()
criteria2 = nn.MSELoss()
epochs = 1000                                                # Number of epochs to train for
lr = 1e-3                                                   # Initali learning rate
op = optim.LBFGS                                             # Optimization algorithm
patience = 5                                                # Patience
sched = ReduceLROnPlateau                                   # Learning rate decay algorithm
split_ratio = 0.90                                           # Training validation split
verbose = 1
weight_decay = 5e-4

analysis_region = [slice(200,300), slice(200,300)]          # Region to do predictions in

""" Main script """
# Set seed
u.set_seed(seed, device=device)

# Load in and format data for training
u.bprint("Loading data...")
data = d.RawSSHDataset(d.test_folder, shift=1, end_year=2008)
data = data.data[[slice(None)] + analysis_region]
# Map to all the non-nan values
injection_map = ~torch.isnan(data)[0]

# Vectorize and remove nans from data
data = data[injection_map.expand(len(data), injection_map.shape[0], injection_map.shape[1])].view(len(data), -1)

# Perform SVD, save data as PCs and remove 
mu,s,v = torch.svd(data)
data = torch.matmul(mu, s.diag()).to(device)
torch.save(v, header+'EOFs')
del mu, s, v

# Split data into training and validation
train_data = data[:int(len(data) * split_ratio)].to(device)
test_data = data[int(len(data) * split_ratio):].to(device)

# Declare model
u.bprint("Declaring model...")
# 9 Layers equates to a receptive field of 2**9
model = m.ResidualTCN(data.shape[1], [128] + [data.shape[1]]).to(device)
#optimizer = op(model.parameters(), lr=lr, hypergrad_lr=1e-8)
#optimizer = optim.Adam(model.parameters(), lr=lr)
optimizer = op(model.parameters())
#optimizer = Stoch_LBFGS(model.parameters(), history_size=10, line_search_fn = True, batch_mode=False)
scheduler = sched(optimizer)

# Load checkpoint
if checkpoint is not None:
    u.bprint("Loading checkpoint...")
    checkpoint = torch.load(checkpoint, map_location='cpu')
    model.load(checkpoint)
    model = model.to(device)
    optimizer.load_state_dict(checkpoint['optimizer'])
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)

# Training Script
if DO_TRAIN:
    u.bprint("Optimizing %d parameters on %s" % (u.count_parameters(model), device))
    min_loss = 1e8
    if op is optim.Adam:
        for epoch in range(epochs):
            start_time = time.time()
            
            model.train()
            y_hat = model(train_data[:-1].transpose(1, 0).unsqueeze(0))
            loss = criteria(y_hat, train_data[1:].transpose(1, 0).unsqueeze(0))
            
            #Calculate Weight Decay
            l2_reg = 0.0
            for param in model.final.parameters():
                l2_reg += torch.sum(param**2)
            l2_norm = weight_decay * l2_reg / 2
            loss += l2_norm
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Validation
            model.eval()
            with torch.no_grad():
                y_hat = model(data[:-1].transpose(1, 0).unsqueeze(0))[:,:,-len(test_data):]
                val_loss = criteria(y_hat, test_data.transpose(1, 0).unsqueeze(0))
                real_val_loss = criteria2(y_hat, test_data.transpose(1, 0).unsqueeze(0))
            if (epoch+1)%verbose == 0:
                u.bprint("[%d]: train: %.8f | val: %.8f | %.4f | t: %.2fs" % (epoch + 1, loss.item(), val_loss.item(), real_val_loss.item(), verbose * (time.time() - start_time)))
            model.update_loss(loss, val_loss = val_loss)
            
            if val_loss < min_loss:
                min_loss = val_loss
                model.save(header=header, optimizer=optimizer)
            
            scheduler.step(loss)
    else:
        for epoch in range(epochs):
            start_time = time.time()
            
            model.train()
            def closure():
                if torch.is_grad_enabled():
                    optimizer.zero_grad()
                y_hat = model(train_data[:-1].transpose(1, 0).unsqueeze(0))
                loss = criteria(y_hat, train_data[1:].transpose(1, 0).unsqueeze(0))
                
                #Calculate Weight Decay
                l2_reg = 0.0
                for param in model.parameters():
                    l2_reg += torch.sum(param**2)
                l2_norm = weight_decay * l2_reg / 2
                loss += l2_norm
                
                if loss.requires_grad:
                    loss.backward()
                        
                return loss
            
            optimizer.step(closure)
            # Calculate loss again for book keeping
            with torch.no_grad():
                y_hat = model(train_data[:-1].transpose(1, 0).unsqueeze(0))
                loss = criteria(y_hat, train_data[1:].transpose(1, 0).unsqueeze(0))
            
            # Validation
            model.eval()
            with torch.no_grad():
                y_hat = model(data[:-1].transpose(1, 0).unsqueeze(0))[:,:,-len(test_data):]
                val_loss = criteria(y_hat, test_data.transpose(1, 0).unsqueeze(0))
                real_val_loss = criteria2(y_hat, test_data.transpose(1, 0).unsqueeze(0))
            if (epoch+1)%verbose == 0:   
                u.bprint("[%d]: train: %.8f | val: %.8f | %.4f | t: %.2fs" % (epoch + 1, loss.item(), val_loss.item(), real_val_loss.item(), verbose * (time.time() - start_time)))
            model.update_loss(loss, val_loss = val_loss)
            
            if val_loss < min_loss:
                min_loss = val_loss
                model.save(header=header, optimizer=optimizer)

    
# Load in best model
device = 'cpu'
pred_shift = 0
cut = len(train_data)
checkpoint = torch.load(header+'checkpoint.pth', map_location=device)
model.load(checkpoint)
model = model.to(device)
del checkpoint, optimizer, train_data, test_data

# Move everything to CPU
feed_in = data[:cut + pred_shift].to(device)
feed_out = data[cut + pred_shift:].to(device)

# Prediction
num_pred_steps = 20
prediction = predict_forward(model, feed_in.transpose(1, 0).unsqueeze(0), num_pred_steps, injection_map, header+'EOFs').to(device)
ground_truth = reconstruct(feed_out[:num_pred_steps], injection_map, header+'EOFs').to(device)

# Get RMSE
RMSEs = ((prediction[injection_map.expand(len(prediction), injection_map.shape[0], injection_map.shape[1])] - \
        ground_truth[injection_map.expand(len(prediction), injection_map.shape[0], injection_map.shape[1])])**2).view(num_pred_steps, -1).mean(1)**(1/2)

# Plot maps
import matplotlib.pyplot as plt
plt.ioff()
for i in range(num_pred_steps):
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.title("Week %d Ground Truth" % (i + 1))
    plt.imshow(ground_truth[i].cpu())
    
    plt.subplot(1, 2, 2)
    plt.title("Prediction RMSE: %.4f" % (RMSEs[i].cpu()))
    plt.imshow(prediction[i].cpu())
    
    plt.savefig(header+"week_%d_prediction.png" % (i + 1))
    
        
    
    
    
    












