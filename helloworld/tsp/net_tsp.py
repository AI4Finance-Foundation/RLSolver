import torch
import numpy as np
import torch.nn as nn

class Policy_Net_TSP(nn.Module):
    def __init__(self, mid_dim=1024, N=4, encode_dim=512):
        super(Policy_Net_TSP, self).__init__()
        self.encode_dim = encode_dim
        self.N = N
        self.input_shape = 2
        self.state_dim = (self.input_shape, N, N)
        self.action_dim = N * N
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.sigmoid = nn.Sigmoid()
        self.net = nn.Sequential(
        BiConvNet(mid_dim, self.state_dim, mid_dim * 4), nn.ReLU(),
        nn.Linear(mid_dim * 4, mid_dim * 2), nn.ReLU(),
        nn.Linear(mid_dim * 2, mid_dim * 1),
        DenseNet(mid_dim * 1), nn.ReLU(),
        nn.Linear(mid_dim * 4, mid_dim * 2), nn.Hardswish(),
        nn.Linear(mid_dim * 2, self.action_dim),
        )

    def forward(self, state):
        mat_H, mat_W = state
        vec_H = mat_H.reshape(-1, self.N * self.N)
        vec_W = mat_W.reshape(-1, self.N * self.N)
        net_input = torch.cat((vec_H, vec_W), 1).reshape(-1, self.input_shape, self.N * self.N)
        net_input = net_input.reshape(-1, self.input_shape, self.N, self.N)
        vec_W_new = (self.sigmoid(self.net(net_input)) - 0.5) * 2
        vec_W_new = vec_W_new / torch.norm(vec_W_new, dim=1, keepdim=True)
        mat_W_new =  vec_W_new.reshape(-1, self.N, self.N)
        mat_W = mat_W.softmax(dim=1)
        mat_W = mat_W.softmax(dim=2)
        return mat_W_new
    
class DenseNet(nn.Module):
    def __init__(self, lay_dim):
        super().__init__()
        self.dense1 = nn.Sequential(nn.Linear(lay_dim * 1, lay_dim * 1), nn.Hardswish())
        self.dense2 = nn.Sequential(nn.Linear(lay_dim * 2, lay_dim * 2), nn.Hardswish())
        self.inp_dim = lay_dim
        self.out_dim = lay_dim * 4

    def forward(self, x1):
        x2 = torch.cat((x1, self.dense1(x1)), dim=1)
        x3 = torch.cat((x2, self.dense2(x2)), dim=1)
        return x3

class BiConvNet(nn.Module):
    def __init__(self, mid_dim, inp_dim, out_dim):
        super().__init__()
        i_c_dim, i_h_dim, i_w_dim = inp_dim 
        self.cnn_h = nn.Sequential(
            nn.Conv2d(i_c_dim * 1, mid_dim * 2, (1, i_w_dim), bias=True), nn.LeakyReLU(inplace=True),
            nn.Conv2d(mid_dim * 2, mid_dim * 1, (1, 1), bias=True), nn.ReLU(inplace=True),)
        self.linear_h = nn.Linear(i_h_dim * mid_dim, out_dim)
        self.cnn_w = nn.Sequential(
            nn.Conv2d(i_c_dim * 1, mid_dim * 2, (i_h_dim, 1), bias=True), nn.LeakyReLU(inplace=True),
            nn.Conv2d(mid_dim * 2, mid_dim * 1, (1, 1), bias=True), nn.ReLU(inplace=True),)
        self.linear_w = nn.Linear(i_w_dim * mid_dim, out_dim)

    def forward(self, state):
        ch = self.cnn_h(state)
        xh = self.linear_h(ch.reshape(ch.shape[0], -1))
        cw = self.cnn_w(state)
        xw = self.linear_w(cw.reshape(cw.shape[0], -1))
        return xw + xh
