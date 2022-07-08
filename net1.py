import torch as th
import torch 
import numpy as np
import torch.nn as nn
class Net(nn.Module):
    def __init__(self, init):
        super(Net, self).__init__()
        # self.step_1 = th.tensor(init_1, dtype=th.float32, requires_grad=True)
        # self.step_2 = th.tensor(init_2, dtype=th.float32, requires_grad=True)
        # self.step_3 = th.tensor(init_3, dtype=th.float32, requires_grad=True)
        # self.step_4 = th.tensor(init_4, dtype=th.float32, requires_grad=True)
        self.register_parameter(name='step', param=th.nn.Parameter( th.tensor(init, dtype=th.float32)))
        
    def forward(self, ):
        return self.step.unsqueeze(0)
def weights_init_uniform(m):
        classname = m.__class__.__name__
        # for every Linear layer in a model..
        if classname.find('Linear') != -1:
            # apply a uniform distribution to the weights and a bias=0
            m.weight.data.uniform_(0.0, 1.0)
            m.bias.data.fill_(0)

def layer_norm(layer, std=1.0, bias_const=1e-6):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
class MMSE_Net(nn.Module):
    def __init__(self, mid_dim=256):
        super(MMSE_Net, self).__init__()
        #self.net1 = nn.Sequential(nn.Linear(32, 512), nn.ReLU())#, nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, 256), nn.ReLU(), nn.Linear(256,32))  
        #self.net2 = nn.Sequential(nn.Linear(32 + 512, 512), nn.ReLU())
        #self.net3 = nn.Sequential(nn.Linear(32 +  512 + 512, 32), nn.ReLU())
        state_dim = (2,4,4)
        action_dim = 32
        self.action_dim2 = action_dim // 2
        s_c_dim, s_h_dim, s_w_dim = state_dim  # inp_for_cnn.shape == (N, C, H, W)

        self.net1 = nn.Sequential(
            BiConvNet(mid_dim, state_dim, mid_dim * 4), nn.ReLU(),
            nn.Linear(mid_dim * 4, mid_dim * 1),
            DenseNet(mid_dim),
        )

        net2_inp_dim = (+ mid_dim * 4
                        + s_c_dim * s_h_dim * s_w_dim)
        self.net2 = nn.Sequential(
            nn.Linear(net2_inp_dim, mid_dim * 4), nn.ReLU(),
            nn.Linear(mid_dim * 4, mid_dim * 2), nn.ReLU(),
            nn.Linear(mid_dim * 2, action_dim)
        )
        layer_norm(self.net2[-1], std=0.1)  # output layer for action

    def forward(self, state):
        state = state.type(th.float32).reshape(state.shape[0], 2, 16)
        state = state.reshape(state.shape[0], 2, 4, 4)

        state_hw = self.net1(state)
        state_flatten = state.reshape((state.shape[0], -1))
        action = self.net2(torch.cat((state_hw, state_flatten), dim=1))
        '''power norm'''
        return action / torch.norm(action, dim=1, keepdim=True)
    def calc_wsr(self, user_weights, channel, precoder, noise_power, selected_users):
        result = 0
        nr_of_users = 4

        for user_index in range(nr_of_users):
          if user_index in selected_users:
            user_sinr = self.compute_sinr_th(channel, precoder, noise_power, user_index, selected_users)
            result = result + user_weights[user_index]*th.log2(1 + user_sinr)
        return result
    def compute_sinr_th(self, channel, precoder, noise_power, user_id, selected_users):
        #print(precoder.shape)
        nr_of_users = np.size(channel,0)
        #print(type(precoder))
        numerator = (th.absolute(th.bmm(th.conj(th.as_tensor(channel[:,user_id,:], dtype=th.complex64).view(channel.shape[0], 1, 4)),th.as_tensor(precoder[:, user_id,:].view(channel.shape[0], 4, 1), dtype=th.complex64)).view(channel.shape[0], 1)))**2
        #print(type(precoder))
        inter_user_interference = 0
        for user_index in range(nr_of_users):
            if user_index != user_id and user_index in selected_users:
                inter_user_interference = inter_user_interference + (th.absolute(th.bmm(th.conj(th.as_tensor(channel[:,user_id,:], dtype=th.complex64).view(channel.shape[0], 1, 4)),th.as_tensor(precoder[:,user_index,:], dtype=th.complex64).view(channel.shape[0], 4, 1))).view(channel.shape[0], 1))**2
        denominator = noise_power + inter_user_interference
        #print(result.shape)
        result = numerator/denominator
        #print(result.shape)

        return result



class DenseNet(nn.Module):  # plan to hyper-param: layer_number
    def __init__(self, lay_dim):
        super().__init__()
        self.dense1 = nn.Sequential(nn.Linear(lay_dim * 1, lay_dim * 1), nn.Hardswish())
        self.dense2 = nn.Sequential(nn.Linear(lay_dim * 2, lay_dim * 2), nn.Hardswish())
        self.inp_dim = lay_dim
        self.out_dim = lay_dim * 4

    def forward(self, x1):  # x1.shape==(-1, lay_dim*1)
        x2 = torch.cat((x1, self.dense1(x1)), dim=1)
        x3 = torch.cat((x2, self.dense2(x2)), dim=1)
        return x3  # x2.shape==(-1, lay_dim*4)
class BiConvNet(nn.Module):
    def __init__(self, mid_dim, inp_dim, out_dim):
        super().__init__()
        i_c_dim, i_h_dim, i_w_dim = inp_dim  # inp_for_cnn.shape == (N, C, H, W)

        self.cnn_h = nn.Sequential(
            nn.Conv2d(i_c_dim * 1, mid_dim * 4, (1, i_w_dim), bias=True), nn.LeakyReLU(inplace=True),
            nn.Conv2d(mid_dim * 4, mid_dim * 1, (1, 1), bias=True), nn.ReLU(inplace=True),
            NnReshape(-1),  # shape=(-1, i_h_dim * mid_dim)
            nn.Linear(i_h_dim * mid_dim, out_dim),
        )
        self.cnn_w = nn.Sequential(
            nn.Conv2d(i_c_dim * 1, mid_dim * 4, (i_h_dim, 1), bias=True), nn.LeakyReLU(inplace=True),
            nn.Conv2d(mid_dim * 4, mid_dim * 1, (1, 1), bias=True), nn.ReLU(inplace=True),
            NnReshape(-1),  # shape=(-1, i_w_dim * mid_dim)
            nn.Linear(i_w_dim * mid_dim, out_dim),
        )

    def forward(self, state):
        xh = self.cnn_h(state)
        xw = self.cnn_w(state)
        return xw + xh

class NnReshape(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.args = args

    def forward(self, x):
        return x.view((x.size(0),) + self.args)



class ActorBiConv(nn.Module):
    def __init__(self, mid_dim, state_dim, action_dim):
        super().__init__()
        self.action_dim2 = action_dim // 2

        self.net = nn.Sequential(
            BiConvNet(mid_dim, state_dim, mid_dim * 4),
            nn.Linear(mid_dim * 4, mid_dim * 2), nn.ReLU(),
            nn.Linear(mid_dim * 2, mid_dim * 1),
            DenseNet(mid_dim * 1),
            nn.Linear(mid_dim * 4, mid_dim * 2), nn.Hardswish(),
            nn.Linear(mid_dim * 2, action_dim),
        )

