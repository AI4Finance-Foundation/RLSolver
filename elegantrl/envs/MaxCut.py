import numpy as np
import torch.nn as nn
import torch

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

class Policy(nn.Module):
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


def get_adjacency_matrix(size=10):
    '''generate a binary symmetric matrix'''
    mat = np.random.randint(0, 2, (size, size))
    mat ^= mat.T
    return mat

def run(seed=1,):
    np.random.seed(seed)
    for i in range(10):
        mat = get_adjacency_matrix(2)
        print(mat)
if __name__ =='main':
    
    run()