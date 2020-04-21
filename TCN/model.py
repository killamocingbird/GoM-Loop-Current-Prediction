import torch.nn as nn

import sys
sys.path.append('../')
import modules as m



class TemporalConvNet(m.Foundation):
    # num_channels is a list
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [m.TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                       padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        # Add final layer
        #layers += [m.FinalBlock(num_channels[-2], num_channels[-1], kernel_size, stride=1, dilation=dilation_size, padding=(kernel_size-1)*dilation_size)]
        self.network = nn.Sequential(*layers)
        self.final = nn.Conv1d(num_channels[-1], num_inputs, 1)

    def forward(self, x):
        # [batch, features, time]
        return self.final(self.network(x))
        
    
class ResidualTCN(TemporalConvNet):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super().__init__(num_inputs, num_channels, kernel_size=2, dropout=0.2)
    
    def forward(self, x):
        return self.final(self.network(x) + x)
    

