import numpy as np
import numpy.random as rd
import torch
import torch.nn as nn

'''[ElegantRL.2022.05.05](github.com/AI4Fiance-Foundation/ElegantRL)'''

Tensor = torch.Tensor

'''DQN (Q network)'''


def layer_norm(layer, std=1.0, bias_const=1e-6):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias,1)
    torch.nn.init.constant_(layer.weight,1)
class QNet(nn.Module):  # `nn.Module` is a PyTorch module for neural network
    """
    Critic class for **Q-network**.

    :param mid_dim[int]: the middle dimension of networks
    :param num_layer[int]: the number of MLP network layer
    :param state_dim[int]: the dimension of state (the number of state vector)
    :param action_dim[int]: the dimension of action (the number of discrete action)
    """

    def __init__(self, mid_dim: int, num_layer: int, state_dim: int, action_dim: int):
        super().__init__()
        self.net = build_mlp(mid_dim, num_layer, input_dim=state_dim, output_dim=action_dim)

        self.explore_rate = None  # float ∈ [0, 1]
        self.action_dim = action_dim

    def forward(self, state: Tensor) -> Tensor:
        """
        The forward function for **Dueling Q-network**.

        :param state: [tensor] the input state. state.shape == (batch_size, state_dim)
        :return: Q values for multiple actions [tensor]. q_values.shape == (batch_size, action_dim)
        """
        q_values = self.net(state)
        return th.softmax(q_values, dim=1)

    def get_action(self, state: Tensor) -> Tensor:  # return [int], which is the index of discrete action
        """
        return the action for exploration with the epsilon-greedy.

        :param state: [tensor] the input state. state.shape == (batch_size, state_dim)
        :return: action [tensor.int]. action.shape == (batch_size, 1)
        """
        #if True:
        a = self.net(state).argmax(dim=1, keepdim=True)
        action = torch.zeros(self.action_dim)
        action[a[0].item()] = 1
        if self.explore_rate < rd.rand():
        
            #action = self.net(state).argmax(dim=1, keepdim=True)
            return action
        else:
            index = torch.randint(self.action_dim, size=(state.shape[0], 1))
            action[index[0].item()] = 1
            if self.explore_rate > rd.rand():
                index = torch.randint(self.action_dim, size=(state.shape[0], 1))
                action[index[0].item()] = 1
            return action

import torch as th
class QNet_mimo(nn.Module):  # `nn.Module` is a PyTorch module for neural network
    """
    Critic class for **Q-network**.

    :param mid_dim[int]: the middle dimension of networks
    :param num_layer[int]: the number of MLP network layer
    :param state_dim[int]: the dimension of state (the number of state vector)
    :param action_dim[int]: the dimension of action (the number of discrete action)
    """

    def __init__(self, mid_dim: int, num_layer: int, state_dim: int, action_dim: int):
        super().__init__()
        
        self.explore_rate = None  # float ∈ [0, 1]
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.action_dim = action_dim
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
        layer_norm(self.net2[-1], std=0.1)  # output layer fo

    def forward(self, state: Tensor) -> Tensor:
        """
        The forward function for **Dueling Q-network**.

        :param state: [tensor] the input state. state.shape == (batch_size, state_dim)
        :return: Q values for multiple actions [tensor]. q_values.shape == (batch_size, action_dim)
        """
        state = state.type(th.float32).reshape(state.shape[0], 2, 40)
        state = state.reshape(state.shape[0], 2, 8, 5)

        state_hw = self.net1(state)
        state_flatten = state.reshape((state.shape[0], -1))
        action = self.net2(torch.cat((state_hw, state_flatten), dim=1))
        q_values = action #/ torch.norm(action, dim=1, keepdim=True)
        #print(q_values)
        #assert 0
        return q_values

    def get_action(self, state: Tensor, I) -> Tensor:  # return [int], which is the index of discrete action
        """
        return the action for exploration with the epsilon-greedy.

        :param state: [tensor] the input state. state.shape == (batch_size, state_dim)
        :return: action [tensor.int]. action.shape == (batch_size, 1)
        """
        #if True:
        a = self.forward(state)
        #print(a, I.flatten())
        a = a - a.abs().max() * 100 * th.as_tensor(( I.flatten())).to(th.device("cuda:0"))
        #print(a)

        a = a.argmax(dim=1, keepdim=True)
        #print(a)
        #assert 0
        action = torch.zeros(self.action_dim)
        #action[a[0].item()] = 1
        if self.explore_rate < rd.rand():
            action[a[0].item()] = 1
            #action = self.net(state).argmax(dim=1, keepdim=True)
            return action
        else:
            #print(np.where(I.flatten() == 0))
            #print(np.random.choice(np.where(I.flatten() == 0)[0], 1))
            action[np.random.choice(np.where(I.flatten() == 0)[0], 1)] = 1
            return action

    def get_action_r(self, state, I):
        a = th.rand(1,self.action_dim).to(th.device("cuda:0"))
        
        a = a -  a.abs().max() * 100 *  th.as_tensor((I.flatten())).to(th.device("cuda:0"))
        a = a.argmax(dim=1, keepdim=True)
        action = torch.zeros(self.action_dim)
        action[a[0].item()] = 1
        return action


class QNet_mimo_graph(nn.Module):
    def __init__(self, mid_dim: int, num_layer: int, state_dim: int, action_dim, p, K, T=4):
        self.theta_1 = nn.Linear(1, p, bias=False)
        self.theta_2 = nn.Linear(p, p, bias = False)
        self.theta_3 = nn.Linear(p, p, bias = False)
        self.theta_4 = nn.Linear(1, p, bias = False)
        self.K = K
        self.p = p
        self.theta_5 = nn.Linear(2 * p,1, bias = False)
        self.theta_6 = nn.Linear(p, p, bias = False)
        self.theta_7 = nn.Linear(p, p, bias = False)
        self.T = T
        self.device = th.device("cuda:0")
    def forward(self, x, w):
        
        mu = th.zeros((x.shape[0], self.K, self.p)).to(self.device)
        for t in range(self.T):
            input_theta_2 = mu.sum(dim=1)
            input_theta_3 = th.zeros((x.shape[0], self.p)).to(self.device)
            input_theta_3_ = []
            for k in range(self.K):
                t = th.nn.ReLU(self.theta_4(w[:, k]))
                input_theta_3 += t
                input_theta_3_.append(t)
            for k in range(self.K):
                mu[:, k] = nn.ReLU(self.theta_1(x[:, k]) + self.theta_2(input_theta_2 - mu[:, k]) 
                                   + self.theta_3(input_theta_3 - input_theta_3_[k]))
        q_values = th.zeros((x.shape[0], self.K, 1)).to(self.device)
        for k in range(self.K):
            q_values[:, k] = self.theta_5(th.concatenate(nn.ReLU(self.theta_6(mu.sum(dim=1))),self.theta_7(mu[:,k]) ))
        
        return q_values

    def get_action(self, x, w) -> Tensor:  # return [int], which is the index of discrete action
        """
        return the action for exploration with the epsilon-greedy.

        :param state: [tensor] the input state. state.shape == (batch_size, state_dim)
        :return: action [tensor.int]. action.shape == (batch_size, 1)
        """
        #if True:
        a = self.forward(x, w)
        #print(a, I.flatten())
        a = a - a.abs().max() * 100 * th.as_tensor(( I.flatten())).to(th.device("cuda:0"))
        #print(a)

        a = a.argmax(dim=1, keepdim=True)
        #print(a)
        #assert 0
        action = torch.zeros(self.action_dim)
        #action[a[0].item()] = 1
        if self.explore_rate < rd.rand():
            action[a[0].item()] = 1
            #action = self.net(state).argmax(dim=1, keepdim=True)
            return action
        else:
            #print(np.where(I.flatten() == 0))
            #print(np.random.choice(np.where(I.flatten() == 0)[0], 1))
            action[np.random.choice(np.where(I.flatten() == 0)[0], 1)] = 1
            return action

    def get_action_r(self, state, I):
        a = th.rand(1,self.action_dim).to(th.device("cuda:0"))
        
        a = a -  a.abs().max() * 100 *  th.as_tensor((I.flatten())).to(th.device("cuda:0"))
        a = a.argmax(dim=1, keepdim=True)
        action = torch.zeros(self.action_dim)
        action[a[0].item()] = 1
        return action

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


class QNetDuel(nn.Module):  # Dueling DQN
    """
    Critic class for **Dueling Q-network**.

    :param mid_dim[int]: the middle dimension of networks
    :param num_layer[int]: the number of MLP network layer
    :param state_dim[int]: the dimension of state (the number of state vector)
    :param action_dim[int]: the dimension of action (the number of discrete action)
    """

    def __init__(self, mid_dim: int, num_layer: int, state_dim: int, action_dim: int):
        super().__init__()
        self.net_state = build_mlp(mid_dim, num_layer - 1, input_dim=state_dim, output_dim=mid_dim)
        self.net_adv = nn.Sequential(nn.Linear(mid_dim, mid_dim), nn.Hardswish(),
                                     nn.Linear(mid_dim, 1))  # advantage function value
        self.net_val = nn.Sequential(nn.Linear(mid_dim, mid_dim), nn.Hardswish(),
                                     nn.Linear(mid_dim, action_dim))  # Q value
        self.explore_rate = None  # float ∈ [0, 1)
        self.action_dim = action_dim

    def forward(self, state: Tensor) -> Tensor:
        """
        The forward function for **Dueling Q-network**.

        :param state: [tensor] the input state.
        :return: the output tensor.
        """
        s_tmp = self.net_state(state)  # encoded state
        q_val = self.net_val(s_tmp)
        q_adv = self.net_adv(s_tmp)
        return q_val - q_val.mean(dim=1, keepdim=True) + q_adv  # dueling Q value

    def get_action(self, state: Tensor) -> Tensor:  # return [int], which is the index of discrete action
        if self.explore_rate < rd.rand():
            s_tmp = self.net_state(state)
            q_val = self.net_val(s_tmp)
            action = q_val.argmax(dim=1, keepdim=True)
        else:
            action = torch.randint(self.action_dim, size=(state.shape[0], 1))
        return action


class QNetTwin(nn.Module):  # Double DQN
    def __init__(self, mid_dim: int, num_layer: int, state_dim: int, action_dim: int):
        super().__init__()
        self.net_state = build_mlp(mid_dim, num_layer - 1, input_dim=state_dim, output_dim=mid_dim)
        self.net_q1 = nn.Sequential(nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                    nn.Linear(mid_dim, action_dim))  # q1 value
        self.net_q2 = nn.Sequential(nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                    nn.Linear(mid_dim, action_dim))  # q2 value
        self.explore_rate = None  # float ∈ [0, 1)
        self.action_dim = action_dim
        self.soft_max = nn.Softmax(dim=1)

    def forward(self, state: Tensor) -> Tensor:
        tmp = self.net_state(state)
        return self.net_q1(tmp)  # one group of Q values

    def get_q1_q2(self, state: Tensor) -> (Tensor, Tensor):
        tmp = self.net_state(state)
        return self.net_q1(tmp), self.net_q2(tmp)  # two groups of Q values

    def get_action(self, state: Tensor) -> Tensor:  # return [int], which is the index of discrete action
        s = self.net_state(state)
        q = self.net_q1(s)
        if self.explore_rate < rd.rand():
            action = q.argmax(dim=1, keepdim=True)
        else:
            a_prob = self.soft_max(q)
            action = torch.multinomial(a_prob, num_samples=1)
        return action


class QNetTwinDuel(nn.Module):  # D3QN: Dueling Double DQN
    def __init__(self, mid_dim: int, num_layer: int, state_dim: int, action_dim: int):
        super().__init__()
        self.net_state = build_mlp(mid_dim, num_layer - 1, input_dim=state_dim, output_dim=mid_dim)
        self.net_val1 = nn.Sequential(nn.Linear(mid_dim, mid_dim), nn.Hardswish(),
                                      nn.Linear(mid_dim, action_dim))  # q1 value
        self.net_val2 = nn.Sequential(nn.Linear(mid_dim, mid_dim), nn.Hardswish(),
                                      nn.Linear(mid_dim, action_dim))  # q2 value
        self.net_adv1 = nn.Sequential(nn.Linear(mid_dim, mid_dim), nn.Hardswish(),
                                      nn.Linear(mid_dim, 1))  # advantage function value 1
        self.net_adv2 = nn.Sequential(nn.Linear(mid_dim, mid_dim), nn.Hardswish(),
                                      nn.Linear(mid_dim, 1))  # advantage function value 1
        self.explore_rate = None  # float ∈ [0, 1)
        self.action_dim = action_dim
        self.soft_max = nn.Softmax(dim=1)

    def forward(self, state: Tensor) -> Tensor:
        t_tmp = self.net_state(state)
        q_val = self.net_val1(t_tmp)
        q_adv = self.net_adv1(t_tmp)
        return q_val - q_val.mean(dim=1, keepdim=True) + q_adv  # one dueling Q value

    def get_q1_q2(self, state: Tensor) -> (Tensor, Tensor):
        s_tmp = self.net_state(state)

        q_val1 = self.net_val1(s_tmp)
        q_adv1 = self.net_adv1(s_tmp)
        q_duel1 = q_val1 - q_val1.mean(dim=1, keepdim=True) + q_adv1

        q_val2 = self.net_val2(s_tmp)
        q_adv2 = self.net_adv2(s_tmp)
        q_duel2 = q_val2 - q_val2.mean(dim=1, keepdim=True) + q_adv2
        return q_duel1, q_duel2  # two dueling Q values

    def get_action(self, state: Tensor) -> Tensor:
        s = self.net_state(state)
        q = self.net_val1(s)
        if self.explore_rate < rd.rand():
            action = q.argmax(dim=1, keepdim=True)
        else:
            a_prob = self.soft_max(q)
            action = torch.multinomial(a_prob, num_samples=1)
        return action


'''Actor (policy network)'''


class Actor(nn.Module):
    def __init__(self, mid_dim: int, num_layer: int, state_dim: int, action_dim: int):
        super().__init__()
        self.net = build_mlp(mid_dim, num_layer, input_dim=state_dim, output_dim=action_dim)
        self.explore_noise_std = 0.1  # standard deviation of exploration action noise
        self.log_sqrt_2pi = np.log(np.sqrt(2 * np.pi))

    def forward(self, state: Tensor) -> Tensor:
        return self.net(state).tanh()  # action

    def get_action(self, state: Tensor) -> Tensor:  # for exploration
        action = self.net(state).tanh()
        noise = (torch.randn_like(action) * self.explore_noise_std).clamp(-0.5, 0.5)
        return (action + noise).clamp(-1.0, 1.0)

    def get_action_noise(self, state: Tensor, action_std: float) -> Tensor:
        action = self.net(state).tanh()
        noise = (torch.randn_like(action) * action_std).clamp(-0.5, 0.5)
        return (action + noise).clamp(-1.0, 1.0)

    def get_logprob(self, state: Tensor, action: Tensor) -> Tensor:
        action_avg = self.net(state)
        action_std = torch.ones_like(action_avg) * self.explore_noise_std
        action_std_log = action_std.log()

        delta = ((action_avg - action) / action_std).pow(2).__mul__(0.5)
        logprob = -(action_std_log + self.log_sqrt_2pi + delta)  # new_logprob
        return logprob

    def get_logprob_fixed(self, state: Tensor, action: Tensor) -> Tensor:
        action_avg = self.net(state)  # NOTICE! `action_avg` is a tensor without .tanh()
        action_std = self.explore_noise_std
        action_std_log = np.log(action_std)  # assert isinstance(action_std, float)

        action_tanh = action_avg.tanh()  # action.tanh()

        logprob = action_std_log + self.log_sqrt_2pi + (action_tanh - action).pow(2).__mul__(0.5)
        logprob += (-action_tanh.pow(2) + 1.000001).log()  # fix logprob using the derivative of action.tanh()
        return logprob


class ActorSAC(nn.Module):
    def __init__(self, mid_dim: int, num_layer: int, state_dim: int, action_dim: int):
        super().__init__()
        self.net_state = build_mlp(mid_dim, num_layer - 1, input_dim=state_dim, output_dim=mid_dim)
        self.net_action_avg = nn.Sequential(nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                            nn.Linear(mid_dim, action_dim))  # the average of action
        self.net_action_std = nn.Sequential(nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                            nn.Linear(mid_dim, action_dim))  # the log_std of action
        self.log_sqrt_2pi = np.log(np.sqrt(2 * np.pi))

    def forward(self, state: Tensor) -> Tensor:
        tmp = self.net_state(state)
        return self.net_action_avg(tmp).tanh()  # action

    def get_action(self, state: Tensor) -> Tensor:
        t_tmp = self.net_state(state)
        action_avg = self.net_action_avg(t_tmp)  # NOTICE! it is action_avg without .tanh()
        action_std = self.net_action_std(t_tmp).clamp(-20, 2).exp()
        return torch.normal(action_avg, action_std).tanh()  # re-parameterize

    def get_action_logprob(self, state: Tensor) -> (Tensor, Tensor):
        t_tmp = self.net_state(state)
        action_avg = self.net_action_avg(t_tmp)  # NOTICE! it needs action_avg.tanh()
        action_std_log = self.net_action_std(t_tmp).clamp(-20, 2)
        action_std = action_std_log.exp()

        noise = torch.randn_like(action_avg, requires_grad=True)
        a_tan = (action_avg + action_std * noise).tanh()  # action.tanh()

        logprob = action_std_log + self.log_sqrt_2pi + noise.pow(2).__mul__(0.5)  # noise.pow(2) * 0.5
        logprob += (-a_tan.pow(2) + 1.000001).log()  # fix logprob using the derivative of action.tanh()
        return a_tan, logprob.sum(1, keepdim=True)


class ActorFixSAC(nn.Module):
    def __init__(self, mid_dim: int, num_layer: int, state_dim: int, action_dim: int):
        super().__init__()
        self.net_state = build_mlp(mid_dim, num_layer - 1, input_dim=state_dim, output_dim=mid_dim)
        self.net_action_avg = nn.Sequential(nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                            nn.Linear(mid_dim, action_dim))  # the average of action
        self.net_action_std = nn.Sequential(nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                            nn.Linear(mid_dim, action_dim))  # the log_std of action
        self.log_sqrt_2pi = np.log(np.sqrt(2 * np.pi))
        self.soft_plus = nn.Softplus()

    def forward(self, state: Tensor) -> Tensor:
        tmp = self.net_state(state)
        return self.net_action_avg(tmp).tanh()  # action

    def get_action(self, state: Tensor) -> Tensor:
        t_tmp = self.net_state(state)
        action_avg = self.net_action_avg(t_tmp)  # NOTICE! it is action_avg without .tanh()
        action_std = self.net_action_std(t_tmp).clamp(-20, 2).exp()
        return torch.normal(action_avg, action_std).tanh()  # re-parameterize

    def get_action_log_std(self, state: Tensor) -> Tensor:
        t_tmp = self.net_state(state)
        a_log_std = self.net_action_std(t_tmp).clamp(-20, 2).exp()
        return a_log_std

    def get_logprob(self, state: Tensor, action: Tensor) -> Tensor:
        t_tmp = self.net_state(state)
        action_avg = self.net_action_avg(t_tmp)  # NOTICE! `action_avg` is a tensor without .tanh()
        action_std_log = self.net_action_std(t_tmp).clamp(-20, 2)
        action_std = action_std_log.exp()

        '''add noise to a_noise in stochastic policy'''
        a_noise = action_avg + action_std * torch.randn_like(action_avg, requires_grad=True)

        logprob = action_std_log + self.log_sqrt_2pi + (a_noise - action).pow(2).__mul__(0.5)  # noise.pow(2) * 0.5
        logprob += (np.log(2.) - a_noise - self.soft_plus(-2. * a_noise)) * 2.  # better than below
        return logprob

    def get_action_logprob(self, state: Tensor) -> (Tensor, Tensor):
        t_tmp = self.net_state(state)
        action_avg = self.net_action_avg(t_tmp)  # NOTICE! it needs action_avg.tanh()
        action_std_log = self.net_action_std(t_tmp).clamp(-20, 2)
        action_std = action_std_log.exp()

        '''add noise to a_noise in stochastic policy'''
        noise = torch.randn_like(action_avg, requires_grad=True)
        a_noise = action_avg + action_std * noise
        # Can only use above code instead of below, because the tensor need gradients here.
        # a_noise = torch.normal(action_avg, action_std, requires_grad=True)

        '''compute logprob according to mean and std of a_noise (stochastic policy)'''
        # self.sqrt_2pi_log = np.log(np.sqrt(2 * np.pi))
        logprob = action_std_log + self.log_sqrt_2pi + noise.pow(2).__mul__(0.5)  # noise.pow(2) * 0.5
        """same as below:
        from torch.distributions.normal import Normal
        logprob = Normal(action_avg, action_std).logprob(a_noise)
        # same as below:
        a_delta = (action_avg - a_noise).pow(2) /(2*action_std.pow(2))
        logprob = -a_delta - action_std.log() - np.log(np.sqrt(2 * np.pi))
        """

        '''fix logprob of action.tanh'''
        logprob += (np.log(2.) - a_noise - self.soft_plus(-2. * a_noise)) * 2.  # better than below
        """same as below:
        epsilon = 1e-6
        a_noise_tanh = a_noise.tanh()
        logprob = logprob - (1 - a_noise_tanh.pow(2) + epsilon).log()

        Thanks for:
        https://github.com/denisyarats/pytorch_sac/blob/81c5b536d3a1c5616b2531e446450df412a064fb/agent/actor.py#L37
        ↑ MIT License， Thanks for https://www.zhihu.com/people/Z_WXCY 2ez4U
        They use action formula that is more numerically stable, see details in the following link
        https://pytorch.org/docs/stable/_modules/torch/distributions/transforms.html#TanhTransform
        https://github.com/tensorflow/probability/commit/ef6bb176e0ebd1cf6e25c6b5cecdd2428c22963f
        """
        return a_noise.tanh(), logprob.sum(1, keepdim=True)


class ActorPPO(nn.Module):
    def __init__(self, mid_dim: int, num_layer: int, state_dim: int, action_dim: int):
        super().__init__()
        self.action_dim = action_dim
        self.net = build_mlp(mid_dim, num_layer, input_dim=state_dim, output_dim=action_dim * 2)

        # the logarithm (log) of standard deviation (std) of action, it is a trainable parameter
        self.action_std_log = nn.Parameter(torch.zeros((1, action_dim * 2)) - 0.5, requires_grad=True)
        self.log_sqrt_2pi = np.log(np.sqrt(2 * np.pi))
        self.softmax = nn.Softmax(dim = 2)

    def forward(self, state: Tensor) -> Tensor:
        action_avg = self.net(state)  # action.tanh()
        action_avg = action_avg.reshape((1, self.action_dim, 2)) * 100
        action_avg = self.softmax(action_avg)
        action_avg = action_avg.argmax(dim = 2)
        return action_avg

    def get_action(self, state: Tensor) -> (Tensor, Tensor):
        action_avg = self.net(state)
       

        action_std = self.action_std_log.exp()
        noise = torch.randn_like(action_avg)
        action1 = action_avg + noise * action_std

        action = action1.reshape((1, self.action_dim, 2)) * 100 
        action = self.softmax(action) 
        action = action.argmax(dim = 2)   
        return action, noise, action1

    def get_logprob(self, state: Tensor, action: Tensor) -> Tensor:
        action_avg = self.net(state)
        action_std = self.action_std_log.exp()

        delta = ((action_avg - action) / action_std).pow(2).__mul__(0.5)
        logprob = -(self.action_std_log + self.log_sqrt_2pi + delta)  # new_logprob
        return logprob

    def get_logprob_entropy(self, state: Tensor, action: Tensor) -> (Tensor, Tensor):
        action_avg = self.net(state)
        action_std = self.action_std_log.exp()


        delta = ((action_avg - action) / action_std).pow(2) * 0.5
        logprob = -(self.action_std_log + self.log_sqrt_2pi + delta).sum(1)  # new_logprob

        dist_entropy = (logprob.exp() * logprob).mean()  # policy entropy
        return logprob, dist_entropy

    def get_old_logprob(self, _action: Tensor, noise: Tensor) -> Tensor:  # noise = action - a_noise
        delta = noise.pow(2).__mul__(0.5)
        return -(self.action_std_log + self.log_sqrt_2pi + delta).sum(1)  # old_logprob

    @staticmethod
    def convert_action_for_env(action: Tensor) -> Tensor:
        return action


class ActorDiscretePPO(nn.Module):
    def __init__(self, mid_dim: int, num_layer: int, state_dim: int, action_dim: int):
        super().__init__()
        self.net = build_mlp(mid_dim, num_layer, input_dim=state_dim, output_dim=action_dim)
        self.action_dim = action_dim
        self.soft_max = nn.Softmax(dim=-1)
        self.Categorical = torch.distributions.Categorical

    def forward(self, state: Tensor) -> Tensor:
        return self.net(state)  # action_prob without softmax

    def get_action(self, state: Tensor) -> (Tensor, Tensor):
        a_prob = self.soft_max(self.net(state))
        # action = Categorical(a_prob).sample()
        samples_2d = torch.multinomial(a_prob, num_samples=1, replacement=True)
        action = samples_2d.reshape(state.size(0))
        return action, a_prob

    def get_logprob_entropy(self, state, a_int):
        # assert a_int.shape == (batch_size, 1)
        a_prob = self.soft_max(self.net(state))
        dist = self.Categorical(a_prob)
        return dist.log_prob(a_int.squeeze(1)), dist.entropy().mean()

    def get_old_logprob(self, a_int, a_prob):
        # assert a_int.shape == (batch_size, 1)
        dist = self.Categorical(a_prob)
        return dist.log_prob(a_int.squeeze(1))

    @staticmethod
    def convert_action_for_env(action: Tensor) -> Tensor:
        return action.int()


'''Critic (value network)'''


class Critic(nn.Module):
    def __init__(self, mid_dim: int, num_layer: int, state_dim: int, action_dim: int):
        super().__init__()
        self.net = build_mlp(mid_dim, num_layer, input_dim=state_dim + action_dim, output_dim=1)

    def forward(self, state: Tensor, action: Tensor) -> Tensor:
        return self.net(torch.cat((state, action), dim=1))  # q value


class CriticPPO(nn.Module):
    def __init__(self, mid_dim: int, num_layer: int, state_dim: int, _action_dim: int):
        super().__init__()
        self.net = build_mlp(mid_dim, num_layer, input_dim=state_dim, output_dim=1)

    def forward(self, state: Tensor) -> Tensor:
        return self.net(state)  # advantage value


class CriticTwin(nn.Module):  # shared parameter
    def __init__(self, mid_dim: int, num_layer: int, state_dim: int, action_dim: int):
        super().__init__()
        self.net_sa = build_mlp(mid_dim, num_layer - 1, input_dim=state_dim + action_dim, output_dim=mid_dim)

        self.net_q1 = nn.Sequential(nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                    nn.Linear(mid_dim, 1))  # q1 value
        self.net_q2 = nn.Sequential(nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                    nn.Linear(mid_dim, 1))  # q2 value

    def forward(self, state: Tensor, action: Tensor) -> Tensor:
        return torch.add(*self.get_q1_q2(state, action)) / 2.  # mean Q value

    def get_q_min(self, state: Tensor, action: Tensor) -> Tensor:
        return torch.min(*self.get_q1_q2(state, action))  # min Q value

    def get_q1_q2(self, state: Tensor, action: Tensor) -> (Tensor, Tensor):
        tmp = self.net_sa(torch.cat((state, action), dim=1))
        return self.net_q1(tmp), self.net_q2(tmp)  # two Q values


class CriticREDq(nn.Module):  # modified REDQ (Randomized Ensemble Double Q-learning)
    def __init__(self, mid_dim: int, num_layer: int, state_dim: int, action_dim: int):
        super().__init__()
        self.critic_num = 8
        self.critic_list = list()
        for critic_id in range(self.critic_num):
            child_cri_net = Critic(mid_dim, num_layer, state_dim, action_dim).net
            setattr(self, f'critic{critic_id:02}', child_cri_net)
            self.critic_list.append(child_cri_net)

    def forward(self, state: Tensor, action: Tensor) -> Tensor:
        return self.get_q_values(state, action).mean(dim=1, keepdim=True)  # mean Q value

    def get_q_min(self, state: Tensor, action: Tensor) -> Tensor:
        tensor_qs = self.get_q_values(state, action)
        q_min = torch.min(tensor_qs, dim=1, keepdim=True)[0]  # min Q value
        q_sum = tensor_qs.sum(dim=1, keepdim=True)  # mean Q value
        return (q_min * (self.critic_num * 0.5) + q_sum) / (self.critic_num * 1.5)  # better than min

    def get_q_values(self, state: Tensor, action: Tensor) -> Tensor:
        tensor_sa = torch.cat((state, action), dim=1)
        tensor_qs = [cri_net(tensor_sa) for cri_net in self.critic_list]
        tensor_qs = torch.cat(tensor_qs, dim=1)
        return tensor_qs  # multiple Q values


def build_mlp(mid_dim: int, num_layer: int, input_dim: int, output_dim: int):  # MLP (MultiLayer Perceptron)
    assert num_layer >= 1
    net_list = list()
    if num_layer == 1:
        net_list.extend([nn.Linear(input_dim, output_dim), ])
    else:  # elif num_layer >= 2:
        net_list.extend([nn.Linear(input_dim, mid_dim), nn.ReLU()])
        for _ in range(num_layer - 2):
            net_list.extend([nn.Linear(mid_dim, mid_dim), nn.ReLU()])
        net_list.extend([nn.Linear(mid_dim, output_dim), ])
    return nn.Sequential(*net_list)
