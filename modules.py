import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm




class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.Tanh()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ELU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.Tanh()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class FinalBlock(TemporalBlock):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding):
        super().__init__(n_inputs, n_outputs, kernel_size, stride, dilation, padding)
        
        # No activation and no drop out for regression
        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.conv2, self.chomp2)

# Foundation to base all models on
class Foundation(nn.Module):
    def __init__(self):
        super().__init__()
        self.train_loss = []
        self.val_loss = []
    
    # Tracking losses
    def update_loss(self, train_loss, val_loss=None):
        self.train_loss.append(train_loss)
        if val_loss is not None:
            self.val_loss.append(val_loss)
            
    # Plots saved training curves
    def gen_train_plots(self, save_path='', header=''):
        if len(self.train_loss) < 1:
            raise "No losses to plot"
        
        plt.figure()
        plt.title('Training Curve')
        plt.plot([i for i in range(len(self.train_loss))], self.train_loss, linestyle='dashed',
                 label='Training')
        if len(self.val_loss) >= 1:
            plt.plot([i for i in range(len(self.val_loss))], self.val_loss,
                     label='Validation')
        plt.legend()
        
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        
        plt.savefig(os.path.join(save_path, header + 'train_curve.png'))
        plt.close()
    
    # Predict a whole dataset on a specified device
    def predict_data(self, x, aux=None, f=None, batch_size=-1, device='cpu'):
        with torch.no_grad():
            if batch_size == -1:
                if aux is not None:
                    if f is not None:
                        return self(x.float().to(device), aux.float().to(device), f)
                    else:
                        return self(x.float().to(device), aux.float().to(device))
                else:
                    return self(x.float().to(device))
            else:
                # Get shape of output
                if aux is not None:
                    if f is not None:
                        out_shape = self(x[:2].float().to(device),
                                         aux[:2].float().to(device), f).shape
                    else:
                        out_shape = self(x[:2].float().to(device),
                                         aux[:2].float().to(device)).shape
                else:
                    out_shape = self(x[:2].float().to(device)).shape
                out = torch.zeros(len(x), *out_shape[1:])
                for batch_idx in range(len(x) // batch_size + 1):
                    if batch_idx != (len(x) // batch_size) - 1:
                        xbatch = x[batch_idx*batch_size:(batch_idx+1)*batch_size].float().to(device)
                        if aux is not None:
                            auxbatch = aux[batch_idx*batch_size:(batch_idx+1)*batch_size].float().to(device)
                            if f is not None:
                                out[batch_idx*batch_size:(batch_idx+1)*batch_size] = self(xbatch, auxbatch, f).cpu()
                            else:
                                out[batch_idx*batch_size:(batch_idx+1)*batch_size] = self(xbatch, auxbatch).cpu()
                        else:
                            out[batch_idx*batch_size:(batch_idx+1)*batch_size] = self(xbatch).cpu()
                    else:
                        xbatch = x[batch_idx*batch_size:].float().to(device)
                        if aux is not None:
                            auxbatch = aux[batch_idx*batch_size:].float().to(device)
                            if f is not None:
                                out[batch_idx*batch_size:] = self(xbatch, auxbatch, f).cpu()
                            else:
                                out[batch_idx*batch_size:] = self(xbatch, auxbatch).cpu()
                        else:
                            out[batch_idx*batch_size:] = self(xbatch).cpu()
                return out
                
    
    def save(self, save_path='', header='', optimizer=None):
        checkpoint = {
            'state_dict': self.state_dict(),
            'train_loss': self.train_loss,
            'val_loss': self.val_loss
        }
        if optimizer is not None:
            checkpoint['optimizer'] = optimizer.state_dict()
        torch.save(checkpoint, save_path + header + 'checkpoint.pth')
        torch.save(self, save_path + header + 'model.pth')
    
    def load(self, checkpoint):
        self.load_state_dict(checkpoint['state_dict'])
        self.train_loss = checkpoint['train_loss']
        self.val_loss = checkpoint['val_loss']