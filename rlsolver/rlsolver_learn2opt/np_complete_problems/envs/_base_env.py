import torch as th
import torch.nn as nn
from copy import deepcopy
import numpy as np
from torch import Tensor

class _BaseEnv():
    def __init__(self, num_nodes = 20, num_envs=128, device=th.device("cuda:0"), episode_length=6):
        self.num_nodes = num_nodes
        self.num_envs = num_envs
        self.device = device
        self.episode_length = episode_length
        self.calc_obj_for_two_graphs_vmap = th.vmap(self.calc_obj_for_two_graphs, in_dims=(0, 0))
        self.adjacency_matrix = None

    def load_graph(self, file_name):
        self.adjacency_matrix = th.as_tensor(np.load(file_name), device=self.device)

    def reset(self):
        self.configuration = th.rand(self.num_envs, self.num_nodes).to(self.device)
        self.num_steps = 0
        return self.configuration



    def generate_adjacency_symmetric_matrix(self, sparsity): # sparsity for binary
        upper_triangle = th.mul(th.rand(self.num_nodes, self.num_nodes).triu(diagonal=1), (th.rand(self.num_nodes, self.num_nodes) < sparsity).int().triu(diagonal=1))
        adjacency_matrix = upper_triangle + upper_triangle.transpose(-1, -2)
        return adjacency_matrix # num_env x self.N x self.N

    # make sure that mu1 and mu2 are different tensors. If they are the same, use get_cut_value_one_tensor
    def calc_obj_for_two_graphs(self, mu1, mu2):
        pass

    def calc_obj_for_one_graph(self, mu1):
        pass