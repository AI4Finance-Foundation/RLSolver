import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
import operator
from itertools import islice
import os
from collections import OrderedDict
import collections.abc as container_abcs
import functools


def to_var(x, requires_grad=True):
    # if torch.cuda.is_available():
    #     x = x.cuda()
    return Variable(x, requires_grad=requires_grad)


def calc_file_name(front: str, id2: int, val: int, end: str):
    return front + "_" + str(id2) + "_" + str(val) + end + "pkl"


def detach_var(v, device):
    var = Variable(v.data, requires_grad=True).to(device)
    var.retain_grad()
    return var


def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition('.')
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)


def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)

    return functools.reduce(_getattr, [obj] + attr.split('.'))



# choice 0: use Synthetic data with N and sparsity
# choice >= 1: use Gset with the ID choice
def load_test_data(choice: int, device: th.device, N: int=10, sparsity: float=0.5):
    sparsity = sparsity
    n = N
    if choice > 0:
        try:
            maxcut_gset2npy(choice)
            test_data = th.as_tensor(np.load(f"./data/maxcut/gset_{choice}.npy")).to(device)
        except Exception as e:
            test_data = th.zeros(n, n, device=device)
            upper_triangle = th.mul(th.ones(n, n).triu(diagonal=1), (th.rand(n, n) < sparsity).int().triu(diagonal=1))
            test_data = upper_triangle + upper_triangle.transpose(-1, -2)
            np.save(f'./data/N{n}Sparsity{sparsity}.npy', test_data.cpu().numpy())
    else:
        test_data = th.zeros(n, n, device=device)
        upper_triangle = th.mul(th.ones(n, n).triu(diagonal=1), (th.rand(n, n) < sparsity).int().triu(diagonal=1))
        test_data = upper_triangle + upper_triangle.transpose(-1, -2)
        np.save(f'./data/maxcut/N{n}Sparsity{sparsity}.npy', test_data.cpu().numpy())
    return test_data


class Opt_net(nn.Module):
    def __init__(self, N, hidden_layers):
        super(Opt_net, self).__init__()
        self.N = N
        self.hidden_layers = hidden_layers
        self.lstm = nn.LSTM(self.N, self.hidden_layers, 1, batch_first=True)
        self.output = nn.Linear(hidden_layers, self.N)

    def forward(self, configuration, hidden_state, cell_state):
        x, (h, c) = self.lstm(configuration, (hidden_state, cell_state))
        return self.output(x).sigmoid(), h, c

# for maxcut problem, gset txt to npy
def maxcut_gset2npy(id: int):
    file1 = open(f"./data/maxcut/gset_{id}.txt", 'r')
    Lines = file1.readlines()

    count = 0
    for line in Lines:
        count += 1
        s = line.split()
        if count == 1:
            N = int(s[0])
            edge = int(s[1])
            adjacency = th.zeros(N, N)
        else:
            i = int(s[0])
            j = int(s[1])
            w = int(s[2])
            adjacency[i - 1, j - 1] = w
            adjacency[j - 1, i - 1] = w
    sparsity = edge / (N * N)
    np.save(f"./data/maxcut/gset_{id}.npy", adjacency)



