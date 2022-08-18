import torch as th
import torch 
import numpy as np
import torch.nn as nn
class Net(nn.Module):
    def __init__(self, init):
        super(Net, self).__init__()
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
            
class MMSE_Net(nn.Module):
    def __init__(self, mid_dim=256):
        super(MMSE_Net, self).__init__()
        state_dim = (8,4,4)
        action_dim = 32
        self.encode_dim = 512
        self.total_power = 10
        self.input_dim = 1
        self.K = 4
        self.N = 4
        self.loop = 4
        self.theta_0 = nn.Linear(8, self.encode_dim)
        self.loop_ = 1
        self.theta_1 = nn.ModuleList([nn.Linear(8, self.encode_dim) for _ in range(self.loop_)])
        self.theta_2 = nn.ModuleList([nn.Linear(self.encode_dim, self.encode_dim) for _ in range(self.loop_)])
        self.theta_3 = nn.ModuleList([nn.Linear(self.encode_dim, self.encode_dim) for _ in range(self.loop_)])
        self.theta_4 = nn.ModuleList([nn.Linear(1, self.encode_dim) for _ in range(self.loop_)])
        self.theta_5 = nn.Linear(2 * self.encode_dim, self.N * 2)
        self.theta_6 = nn.Linear(self.encode_dim, self.encode_dim)
        self.theta_7 = nn.Linear(self.encode_dim, self.encode_dim)
        self.theta_8  = nn.ModuleList([nn.Linear(self.encode_dim, self.encode_dim) for _ in range(self.loop_)])
        self.theta_9 = nn.Linear(self.encode_dim * 4, self.encode_dim)
        self.mid = nn.ReLU()
        self.device = th.device("cuda:0")
        self.net = nn.Sequential(
            BiConvNet(mid_dim, state_dim, mid_dim * 4), nn.ReLU(),
            nn.Linear(mid_dim * 4, mid_dim * 2), nn.ReLU(),
            nn.Linear(mid_dim * 2, mid_dim * 1),
            DenseNet(mid_dim * 1), nn.ReLU(),
            nn.Linear(mid_dim * 4, mid_dim * 2), nn.Hardswish(),
            nn.Linear(mid_dim * 2, action_dim),
        ) 
        self.s = nn.Sigmoid()
    def forward(self, channel, state, configure):
        
        state = th.cat((state.real, state.imag), dim=-1)

        mu = [self.theta_0(th.cat((configure.real, configure.imag), dim=-1)) for _ in range(self.loop + 1)]
        configure = th.cat((configure.real, configure.imag), dim=-1)

        for j in range(self.loop):
            j = 0
            x4 = self.theta_4[j](state.reshape(-1, 1))
            x4 = x4.reshape(state.shape[0], state.shape[1], state.shape[2], -1)
            x4 = self.mid(x4)
            
            for i in range(state.shape[1]):
                x1 = self.theta_1[j](configure[:, i])
                x2 = self.theta_2[j](mu[j].sum(dim = 1) - mu[j][:, i, :])
                x4_ = x4[:, i, i] / ((x4.sum(dim = 2)[:, i] - x4[:, i, i] ) + 0.0001)
                x8 = self.theta_8[j](mu[j][:, i])
                x3 = self.theta_3[j](x4_)
                t = th.cat((x1, x2, x3, x8), dim=1)
                mu[j + 1][:, i] = self.mid(self.theta_9(t))
            
        w = th.zeros(state.shape[0],self.K, self.N * 2 ).to(th.device("cuda:0"))
        for i in range(state.shape[1]):
            x6 = self.theta_6(mu[-1].sum(dim=1))
            x7 = self.theta_7(mu[-1][:, i])
            t = th.cat((x6, x7), dim=1)
            t = self.mid(t)
            w[:, i] = self.theta_5(t)
        
        w = w.reshape(state.shape[0], state.shape[1], 2, -1)
        w = w[:,:,  0] + 1.j * w[:,:, 1]
        w = w / th.norm(w, dim=1, keepdim=True)
        w = w.reshape(-1, self.K*self.N)
        w = th.cat((w.real, w.imag), dim=1)
        w = w.reshape(-1, 2, self.K*self.N)
        
        channel = channel.type(th.float32).reshape(channel.shape[0], 6, 16)
        channel = th.cat((channel, w), dim=1)
        channel = channel.reshape(channel.shape[0], 8, 4, 4)
        t = (self.s(self.net(channel)) - 0.5) * 2
        return t / torch.norm(t, dim=1, keepdim=True)* np.sqrt(10)

    def calc_mmse(self, channel):
        K = self.K
        N = self.N

        lambda_ = th.ones(self.K).repeat((channel.shape[0], 1)).to(self.device) * self.total_power / self.K
        p = th.ones(self.K).repeat((channel.shape[0], 1)).to(self.device) * np.sqrt(self.total_power / self.K)
        
        effective_channel = channel.conj().transpose(1,2).to(th.cfloat).to(self.device)
        channel = channel.to(self.device)

        eye_N = (th.zeros(lambda_.shape[0], self.N) + 1).to(self.device)
        eye_N = th.diag_embed(eye_N)
        lambda_ = th.diag_embed(lambda_)

        channel = th.bmm(lambda_.to(th.cfloat), channel.type(th.cfloat))

        denominator = th.inverse(eye_N + th.bmm(effective_channel,channel))

        wslnr_max = th.zeros((lambda_.shape[0], N, K), dtype=th.cfloat).to(self.device)
        wslnr_max = th.bmm(denominator, effective_channel)
        wslnr_max = wslnr_max.transpose(1,2)
        wslnr_max = wslnr_max / wslnr_max.norm(dim=2, keepdim=True)

        p = th.diag_embed(p)

        W = th.bmm(p.to(th.cfloat), wslnr_max)
        return W


    def calc_wsr(self, user_weights, channel, precoder, noise_power, selected_users):
        result = 0
        nr_of_users = self.K
        #print(th.matmul(channel[0].to(th.cfloat), precoder[0].T.to(th.cfloat)))
        for user_index in range(nr_of_users):
            if user_index in selected_users:
                user_sinr = self.compute_sinr_th(channel, precoder, noise_power, user_index, selected_users)
                result = result + user_weights[user_index]*th.log2(1 + user_sinr)
        return result

    def compute_sinr_th(self, channel, precoder, noise_power, user_id, selected_users):
        #print(precoder.shape)
        nr_of_users = self.K
        numerator = (th.absolute(th.bmm(th.as_tensor(channel[:,user_id,:], dtype=th.complex64).view(channel.shape[0], 1, self.N),th.as_tensor(precoder[:, user_id,:].view(channel.shape[0], self.N, 1), dtype=th.complex64)).view(channel.shape[0], 1)))**2
        #print(type(precoder))

        inter_user_interference = 0

        for user_index in range(nr_of_users):
            if user_index != user_id and user_index in selected_users:
                inter_user_interference = inter_user_interference + (th.absolute(th.bmm(th.as_tensor(channel[:,user_id,:], dtype=th.complex64).view(channel.shape[0], 1, self.N),th.as_tensor(precoder[:,user_index,:], dtype=th.complex64).view(channel.shape[0], self.N, 1))).view(channel.shape[0], 1))**2
        denominator = noise_power + inter_user_interference
        result = numerator/denominator
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
            nn.Conv2d(i_c_dim * 1, mid_dim * 2, (1, i_w_dim), bias=True), nn.LeakyReLU(inplace=True),
            nn.Conv2d(mid_dim * 2, mid_dim * 1, (1, 1), bias=True), nn.ReLU(inplace=True),
            NnReshape(-1),  # shape=(-1, i_h_dim * mid_dim)
            nn.Linear(i_h_dim * mid_dim, out_dim),
        )
        self.cnn_w = nn.Sequential(
            nn.Conv2d(i_c_dim * 1, mid_dim * 2, (i_h_dim, 1), bias=True), nn.LeakyReLU(inplace=True),
            nn.Conv2d(mid_dim * 2, mid_dim * 1, (1, 1), bias=True), nn.ReLU(inplace=True),
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