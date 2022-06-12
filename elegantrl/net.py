import torch
import torch.nn as nn
from torch import Tensor
import numpy as np
import numpy.random as rd


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


class ActorPPO(nn.Module):
    def __init__(self, mid_dim: int, num_layer: int, state_dim: int, action_dim: int):
        super().__init__()
        # self.net = build_mlp(mid_dim, num_layer, input_dim=state_dim, output_dim=action_dim)
        # check https://arxiv.org/pdf/2204.07137.pdf A.2.2
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, action_dim)
        )

        # the logarithm (log) of standard deviation (std) of action, it is a trainable parameter
        self.action_std_log = nn.Parameter(torch.zeros((1, action_dim)) - 0.5, requires_grad=True)
        self.log_sqrt_2pi = np.log(np.sqrt(2 * np.pi))

    def forward(self, state: Tensor) -> Tensor:
        return self.net(state).tanh()  # action.tanh()

    def get_action(self, state: Tensor) -> (Tensor, Tensor):
        action_avg = self.net(state)
        action_std = self.action_std_log.exp()

        noise = torch.randn_like(action_avg)
        action = action_avg + noise * action_std
        return action, noise

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
        return action.tanh()


class CriticPPO(nn.Module):
    def __init__(self, mid_dim: int, num_layer: int, state_dim: int, _action_dim: int):
        super().__init__()
        # self.net = build_mlp(mid_dim, num_layer, input_dim=state_dim, output_dim=1)
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, state: Tensor) -> Tensor:
        return self.net(state)  # advantage value


class ActorSAC(nn.Module):
    def __init__(self, mid_dim: int, num_layer: int, state_dim: int, action_dim: int):
        super().__init__()
        #self.net_state = build_mlp(mid_dim, num_layer - 1, input_dim=state_dim, output_dim=mid_dim)
        
        self.net_state = nn.Sequential(nn.Linear(state_dim, 128), nn.ELU(), nn.LayerNorm(128),
                                       nn.Linear(128, 64), nn.ELU(), nn.LayerNorm(64), nn.Linear(64, 32), nn.ELU(), nn.LayerNorm(32))
        #self.net_action_avg = nn.Sequential(nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                            # nn.Linear(mid_dim, action_dim))  # the average of action
        self.net_action_avg = nn.Sequential(nn.Linear(32, action_dim))  # the average of action
        #self.net_action_std = nn.Sequential(nn.Linear(mid_dim, mid_dim), nn.ReLU(),
        #                                    nn.Linear(mid_dim, action_dim))  # the log_std of action
        self.net_action_std = nn.Sequential(nn.Linear(32, action_dim))
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


class CriticTwin(nn.Module):  # shared parameter
    def __init__(self, mid_dim: int, num_layer: int, state_dim: int, action_dim: int):
        super().__init__()
        #self.net_sa = build_mlp(mid_dim, num_layer - 1, input_dim=state_dim + action_dim, output_dim=mid_dim)
        self.net_q1 = nn.Sequential(nn.Linear(state_dim + action_dim, 64), nn.ELU(),nn.LayerNorm(64),
                                    nn.Linear(64, 64), nn.ELU(),nn.LayerNorm(64), nn.Linear(64, 1))  # q1 value
        self.net_q2 = nn.Sequential(nn.Linear(state_dim + action_dim, 64), nn.ELU(),nn.LayerNorm(64),
                                    nn.Linear(64, 64), nn.ELU(),nn.LayerNorm(64), nn.Linear(64, 1))  # q2 value

    def forward(self, state: Tensor, action: Tensor) -> Tensor:
        return torch.add(*self.get_q1_q2(state, action)) / 2.  # mean Q value

    def get_q_min(self, state: Tensor, action: Tensor) -> Tensor:
        return torch.min(*self.get_q1_q2(state, action))  # min Q value

    def get_q1_q2(self, state: Tensor, action: Tensor) -> (Tensor, Tensor):
        tmp = torch.cat((state, action), dim=1)
        return self.net_q1(tmp), self.net_q2(tmp)  # two Q values
