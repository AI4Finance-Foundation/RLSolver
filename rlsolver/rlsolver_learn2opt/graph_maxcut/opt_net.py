import torch as th
import torch.nn as nn
from copy import deepcopy
import numpy as np
from torch import Tensor

class Opt_net(nn.Module):
    def __init__(self, N, hidden_layers):
        super(Opt_net, self).__init__()
        self.N = N
        self.hidden_layers = hidden_layers
        self.lstm = nn.LSTM(self.N, self.hidden_layers,1,batch_first=True)
        self.output = nn.Linear(hidden_layers, self.N)

    def forward(self, configuration, hidden_state, cell_state):
        x, (h, c) = self.lstm(configuration, (hidden_state, cell_state))
        return self.output(x).sigmoid(), h, c