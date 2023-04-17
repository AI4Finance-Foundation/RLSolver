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

    # make sure that mu1 and mu2 are different tensors. If they are the same, use get_cut_value_one_tensor
    def get_cut_value(self, mu1, mu2):
        # return th.mul(th.matmul(mu1.reshape(self.N, 1), \
        #                         (1 - mu2.reshape(-1, self.N, 1)).transpose(-1, -2)), \
        #               self.adjacency_matrix)\
        #            .flatten().sum(dim=-1) \
        #        + ((mu1-mu2)**2).sum()

        # mu1 = mu1.reshape(self.N, 1)
        # mu1_1 = 1 - mu1
        # mu2 = mu2.reshape(-1, self.N, 1)
        # mu2_t = mu2.transpose(-1, -2)
        # mu2_1_t = (1 - mu2).transpose(-1, -2)
        # mu12_ = th.mul(th.matmul(mu1, mu2_1_t), self.adjacency_matrix)
        # mu1_2 = th.mul(th.matmul(mu1_1, mu2_t), self.adjacency_matrix)
        # mu12 = th.min(th.ones_like(mu12_), mu12_ + mu1_2)
        # cut12 = mu12.flatten().sum(dim=-1)
        # cut1 = self.get_cut_value_one_tensor(mu1)
        # cut2 = self.get_cut_value_one_tensor(mu2)
        # cut = cut1 + cut2 + cut12
        # return cut

        return self.get_cut_value_one_tensor(mu1) \
               + self.get_cut_value_one_tensor(mu2) \
               + th.mul(th.matmul(mu1.reshape(-1, self.N, 1), (1 - mu2.reshape(-1, self.N, 1)).transpose(-1, -2)),
                        self.adjacency_matrix) \
               + th.mul(th.matmul(1 - mu1.reshape(-1, self.N, 1), mu2.reshape(-1, self.N, 1).transpose(-1, -2)),
                        self.adjacency_matrix)

    def get_cut_value_one_tensor(self, mu1):
        mu1 = mu1.reshape(-1, self.N, 1)
        mu2 = mu1.reshape(-1, self.N, 1)
        mu2_1_t = (1 - mu2).transpose(-1, -2)
        mu12 = th.mul(th.matmul(mu1, mu2_1_t), self.adjacency_matrix)
        cut = mu12.flatten().sum(dim=-1) / self.num_env
        return cut