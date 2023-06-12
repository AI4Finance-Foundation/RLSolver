import torch as th
import torch.nn as nn
from copy import deepcopy
import numpy as np
from torch import Tensor

class _BaseEnv():
    def __init__(self, num_nodes: int = 20, num_envs: int=128, device: th.device =th.device("cuda:0"), episode_length: int=6):
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

    def reset(self, add_noise_for_best_x=False, sample_ratio_envs=0.2, sample_ratio_nodes=0.2):
        if add_noise_for_best_x and self.best_x is not None:
            e = max(1, int(sample_ratio_envs * self.num_envs))
            n = max(1, int(sample_ratio_nodes * self.num_nodes))
            indices_envs = th.randint(0, self.num_envs, e)  # indices of selected envs/rows
            indices_nodes = th.randint(0, self.num_nodes, n)
            # noise = th.randn(n, self.num_nodes).to(self.device)
            noise = th.rand(self.num_envs, self.num_nodes).to(self.device)
            mask = th.zeros(self.num_envs, self.num_nodes, dtype=bool).to(self.device)
            mask[indices_envs, indices_nodes.unsqueeze(1)] = True
            noise = th.mul(noise, mask).to(self.device)

            mask2 = th.zeros(self.num_envs, self.num_nodes, dtype=bool).to(self.device)
            mask2[indices_envs, :] = True
            add_noise_for_best_x = th.mul(self.best_x.repeat(self.num_envs, 1), mask2).to(self.device)

            mask3 = th.ones(self.num_envs, self.num_nodes, dtype=bool).to(self.device)
            mask3[indices_envs, :] = False
            x = th.mul(th.rand(self.num_envs, self.num_nodes), mask3).to(self.device)

            self.x = x + add_noise_for_best_x + noise
            self.x[0, :] = self.best_x  # the first row is best_x, no noise
        else:
            self.x = th.rand(self.num_envs, self.num_nodes).to(self.device)
        self.num_steps = 0
        return self.x



    def generate_symmetric_adjacency_matrix(self, sparsity: float): # sparsity for binary
        upper_triangle = th.mul(th.rand(self.num_nodes, self.num_nodes).triu(diagonal=1), (th.rand(self.num_nodes, self.num_nodes) < sparsity).int().triu(diagonal=1))
        adjacency_matrix = upper_triangle + upper_triangle.transpose(-1, -2)
        return adjacency_matrix # num_env x self.N x self.N

    # make sure that mu1 and mu2 are different tensors. If they are the same, use get_cut_value_one_tensor
    def reward(self, mu1: Tensor, mu2: Tensor):
        pass

    def obj(self, mu: Tensor):
        pass