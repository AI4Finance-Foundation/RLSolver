import torch as th
import torch.nn as nn
from copy import deepcopy
import numpy as np
from torch import Tensor

# from rlsolver.rlsolver_learn2opt.np_complete_problems.envs._base_env import _BaseEnv

# from _base_env import _BaseEnv
class MaxcutEnv():
    def __init__(self, num_nodes=20, num_envs=128, device=th.device("cuda:0"), episode_length=6):
        self.num_nodes = num_nodes
        self.num_envs = num_envs
        self.device = device
        self.episode_length = episode_length
        self.x = th.rand(self.num_envs, self.num_nodes).to(self.device)
        self.best_x = None
        self.calc_obj_for_two_graphs_vmap = th.vmap(self.reward, in_dims=(0, 0))
        self.adjacency_matrix = None

    def load_graph(self, file_name: str):
        self.adjacency_matrix = th.as_tensor(np.load(file_name), device=self.device)

    def reset(self):
        self.x = th.rand(self.num_envs, self.num_nodes).to(self.device)
        self.num_steps = 0
        return self.x

    # make sure that mu1 and mu2 are different tensors. If they are the same, use calc_obj_for_one_graph
    def reward(self, mu1: Tensor, mu2: Tensor):
        cut1 = self.obj(mu1)
        cut2 = self.obj(mu2)
        cut12 = th.mul(th.matmul(mu1.reshape(-1, self.num_nodes, 1), (1 - mu2.reshape(-1, self.num_nodes, 1)).transpose(-1, -2)),
                        self.adjacency_matrix) \
               + th.mul(th.matmul(1 - mu1.reshape(-1, self.num_nodes, 1), mu2.reshape(-1, self.num_nodes, 1).transpose(-1, -2)),
                        self.adjacency_matrix)
        cut = cut1 + cut2 + cut12
        return cut

    def obj(self, mu: Tensor):
        # mu1 = mu1.reshape(-1, self.N, 1)
        # mu2 = mu1.reshape(-1, self.N, 1)
        # mu2_1_t = (1 - mu2).transpose(-1, -2)
        # mu12 = th.mul(th.matmul(mu1, mu2_1_t), self.adjacency_matrix)
        # cut = mu12.flatten().sum(dim=-1) / self.num_env
        # return cut

        return th.mul(th.matmul(mu.reshape(-1, self.num_nodes, 1), (1 - mu.reshape(-1, self.num_nodes, 1)).transpose(-1, -2)), self.adjacency_matrix).flatten().sum(dim=-1) / self.num_envs
