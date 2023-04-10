import torch as th
import torch.nn as nn
from copy import deepcopy
import numpy as np
from torch import Tensor

class MaxcutEnv():
    def __init__(self, N = 20, num_env=128, device=th.device("cuda:0"), episode_length=6):
        self.N = N
        self.num_env = num_env
        self.device = device
        self.episode_length = episode_length
        self.get_cut_value_tensor = th.vmap(self.get_cut_value, in_dims=(0, 0))
        self.adjacency_matrix = None

    def load_graph(self, file_name):
        self.adjacency_matrix = th.as_tensor(np.load(file_name), device=self.device)

    def reset(self):
        self.configuration = th.rand(self.num_env, self.N).to(self.device)
        self.num_steps = 0
        return self.configuration

    # def step(self, configuration):
    #     self.configuration = configuration   # num_env x N x 1
    #     self.reward = self.get_cut_value_tensor(self.configuration)
    #     self.num_steps +=1
    #     self.done = True if self.num_steps >= self.episode_length else False
    #     return  self.configuration.detach(), self.reward, self.done

    def generate_adjacency_symmetric_matrix(self, sparsity): # sparsity for binary
        upper_triangle = th.mul(th.rand(self.N, self.N).triu(diagonal=1), (th.rand(self.N, self.N) < sparsity).int().triu(diagonal=1))
        adjacency_matrix = upper_triangle + upper_triangle.transpose(-1, -2)
        return adjacency_matrix # num_env x self.N x self.N

    def get_cut_value(self, mu1, mu2):
        return th.mul(th.matmul(mu1.reshape(self.N, 1), \
                                (1 - mu2.reshape(-1, self.N, 1)).transpose(-1, -2)), \
                      self.adjacency_matrix)\
                   .flatten().sum(dim=-1) \
               + ((mu1-mu2)**2).sum()