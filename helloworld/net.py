import torch as th
import torch 
import numpy as np
import torch.nn as nn

class Net_MIMO(nn.Module):
    def __init__(self, mid_dim=256, K=4, N=4, total_power=10, encode_dim=512, gnn_loop=4):
        super(Net_MIMO, self).__init__()
        self.encode_dim = encode_dim
        self.total_power = total_power
        self.K = K
        self.N = N
        self.state_dim = (6,K,N)
        self.action_dim = 2 * K * N
        self.loop = gnn_loop
        self.theta_0 = nn.Linear(self.K * 2, self.encode_dim)
        self.if_gnn = False
        self.device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
        self.sigmoid = nn.Sigmoid()
        self.net = nn.Sequential(
            BiConvNet(mid_dim, self.state_dim, mid_dim * 4), nn.ReLU(),
            nn.Linear(mid_dim * 4, mid_dim * 2), nn.ReLU(),
            nn.Linear(mid_dim * 2, mid_dim * 1),
            DenseNet(mid_dim * 1), nn.ReLU(),
            nn.Linear(mid_dim * 4, mid_dim * 2), nn.Hardswish(),
            nn.Linear(mid_dim * 2, self.action_dim),
        )
        if self.if_gnn:
            self.gnn_weight = nn.ModuleList([ nn.Linear(self.K * 2, self.encode_dim), 
                                        nn.Linear(8, self.encode_dim), 
                                        nn.Linear(self.encode_dim, self.encode_dim), 
                                        nn.Linear(self.encode_dim, self.encode_dim),
                                        nn.Linear(1, self.encode_dim),
                                        nn.Linear(2 * self.encode_dim, self.N * 2),
                                        nn.Linear(self.encode_dim, self.encode_dim), 
                                        nn.Linear(self.encode_dim, self.encode_dim), 
                                        nn.Linear(self.encode_dim, self.encode_dim),
                                        nn.Linear(self.encode_dim * 4, self.encode_dim)])
            self.mid = nn.ReLU()

    def forward(self, channel, state, configure):
        state = th.cat((state.real, state.imag), dim=-1)
        channel = channel.type(th.float32).reshape(channel.shape[0], 6, 16)
        if self.if_gnn:
            mu = [self.gnn_weight[0](th.cat((configure.real, configure.imag), dim=-1)) for _ in range(self.loop + 1)]
            configure = th.cat((configure.real, configure.imag), dim=-1)
            for j in range(self.loop):
                x4 = self.mid(self.gnn_weight[4](state.reshape(-1, 1)).reshape(state.shape[0], state.shape[1], state.shape[2], -1))
                for i in range(state.shape[1]):
                    x1 = self.gnn_weight[1](configure[:, i])
                    x2 = self.gnn_weight[2](mu[j].sum(dim = 1) - mu[j][:, i, :])
                    x4_ = x4[:, i, i] / ((x4.sum(dim = 2)[:, i] - x4[:, i, i] ) + 0.0001)
                    x8 = self.gnn_weight[8](mu[j][:, i])
                    x3 = self.gnn_weight[3](x4_)
                    t = th.cat((x1, x2, x3, x8), dim=1)
                    mu[j + 1][:, i] = self.mid(self.gnn_weight[9](t))
            w = th.zeros(state.shape[0],self.K, self.N * 2 ).to(self.device)
            for i in range(state.shape[1]):
                x6 = self.gnn_weight[6](mu[-1].sum(dim=1))
                x7 = self.gnn_weight[7](mu[-1][:, i])
                t = th.cat((x6, x7), dim=1)
                t = self.mid(t)
                w[:, i] = self.gnn_weight[5](t)
            w = w.reshape(state.shape[0], state.shape[1], 2, -1)
            w = w[:,:,  0] + 1.j * w[:,:, 1]
            w = w / th.norm(w, dim=1, keepdim=True)
            w = w.reshape(-1, self.K*self.N)
            w = th.cat((w.real, w.imag), dim=1)
            w = w.reshape(-1, 2, self.K*self.N)
            channel = th.cat((channel, w), dim=1)
        channel = channel.reshape(channel.shape[0], 6 + self.if_gnn * 2, self.K, self.N)
        t = (self.sigmoid(self.net(channel)) - 0.5) * 2
        return t / torch.norm(t, dim=1, keepdim=True) * np.sqrt(self.total_power)

    def calc_mmse(self, channel):
        lambda_ = th.ones(self.K).repeat((channel.shape[0], 1)).to(self.device) * self.total_power / self.K
        p = th.ones(self.K).repeat((channel.shape[0], 1)).to(self.device) * np.sqrt(self.total_power / self.K)
        effective_channel = channel.conj().transpose(1,2).to(th.cfloat).to(self.device)
        channel = channel.to(self.device)
        eye_N = (th.zeros(lambda_.shape[0], self.N) + 1).to(self.device)
        eye_N = th.diag_embed(eye_N)
        lambda_ = th.diag_embed(lambda_)
        channel = th.bmm(lambda_.to(th.cfloat), channel.type(th.cfloat))
        denominator = th.inverse(eye_N + th.bmm(effective_channel,channel))
        wslnr_max = th.zeros((lambda_.shape[0], self.N, self.K), dtype=th.cfloat).to(self.device)
        wslnr_max = th.bmm(denominator, effective_channel)
        wslnr_max = wslnr_max.transpose(1,2)
        wslnr_max = wslnr_max / wslnr_max.norm(dim=2, keepdim=True)
        p = th.diag_embed(p)
        W = th.bmm(p.to(th.cfloat), wslnr_max)
        return W

    def calc_sum_rate(self,channel, precoder):
        HTF = channel
        HTFGW = torch.bmm(HTF.to(th.cfloat), precoder.to(th.cfloat).transpose(-1, -2))
        S = torch.abs(torch.diagonal(HTFGW, dim1=-2, dim2=-1))**2
        I = torch.sum(torch.abs(HTFGW)**2, dim=-1) - torch.abs(torch.diagonal(HTFGW, dim1=-2, dim2=-1))**2
        N = 1
        SINR = S/(I+N)
        return torch.log2(1+SINR).sum(dim=-1).unsqueeze(-1)
    
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