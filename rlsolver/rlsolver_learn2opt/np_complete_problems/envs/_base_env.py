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
        self.calc_obj_for_two_graphs_vmap = th.vmap(self.calc_obj_for_two_graphs, in_dims=(0, 0))
        self.adjacency_matrix = None

    def load_graph(self, file_name: str):
        self.adjacency_matrix = th.as_tensor(np.load(file_name), device=self.device)

    def reset(self, best_x_noise=False, best_x_ratio=0.2):
        self.x = th.rand(self.num_envs, self.num_nodes).to(self.device)
        if best_x_noise and self.best_x is not None:
            n = max(1, int(th.floor(best_x_ratio * self.num_envs)))
            self.x[0: n, :] = self.best_x + th.randn(n, self.num_nodes)
        self.num_steps = 0
        return self.x



    def generate_symmetric_adjacency_matrix(self, sparsity: float): # sparsity for binary
        upper_triangle = th.mul(th.rand(self.num_nodes, self.num_nodes).triu(diagonal=1), (th.rand(self.num_nodes, self.num_nodes) < sparsity).int().triu(diagonal=1))
        adjacency_matrix = upper_triangle + upper_triangle.transpose(-1, -2)
        return adjacency_matrix # num_env x self.N x self.N

    # make sure that mu1 and mu2 are different tensors. If they are the same, use get_cut_value_one_tensor
    def calc_obj_for_two_graphs(self, mu1: Tensor, mu2: Tensor):
        pass

    def calc_obj_for_one_graph(self, mu: Tensor):
        pass