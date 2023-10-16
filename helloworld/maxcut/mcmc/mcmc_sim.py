import torch as th
import torch.nn as nn
from copy import deepcopy
import numpy as np
from torch import Tensor
# from functorch import vmap
# from rlsolver.rlsolver_learn2opt.np_complete_problems.env._base_env import _BaseEnv
from utils import read_txt_as_networkx_graph
import networkx as nx
class MCMCSim():
    def __init__(self, filename: str, num_samples=128, device=th.device("cuda:0"), episode_length=6):
        self.graph = read_txt_as_networkx_graph(filename)
        self.num_nodes = self.graph.number_of_nodes()
        self.num_edges = self.graph.number_of_edges()
        self.num_samples = num_samples
        self.device = device
        self.x = th.rand(self.num_samples, self.num_nodes).to(self.device)
        adj = nx.to_numpy_array(self.graph)
        self.adjacency_matrix = Tensor(adj)
        self.adjacency_matrix = self.adjacency_matrix.to(self.device)
        self.episode_length = episode_length
        self.best_x = None
        self.calc_obj_for_two_graphs_vmap = th.vmap(self.obj2, in_dims=(0, 0))


    def init(self, add_noise_for_best_x=False, sample_ratio_envs=0.6, sample_ratio_nodes=0.7):
        if add_noise_for_best_x and self.best_x is not None:
            e = max(1, int(sample_ratio_envs * self.num_samples))
            n = max(1, int(sample_ratio_nodes * self.num_nodes))
            indices_envs = th.randint(0, self.num_samples, (e, 1))  # indices of selected env/rows
            indices_nodes = th.randint(0, self.num_nodes, (n, 1))
            # noise = th.randn(n, self.num_nodes).to(self.device)
            noise = th.rand(self.num_samples, self.num_nodes).to(self.device)
            mask = th.zeros(self.num_samples, self.num_nodes, dtype=bool).to(self.device)
            mask[indices_envs, indices_nodes.unsqueeze(1)] = True
            noise = th.mul(noise, mask).to(self.device)

            mask2 = th.zeros(self.num_samples, self.num_nodes, dtype=bool).to(self.device)
            mask2[indices_envs, :] = True
            add_noise_for_best_x = th.mul(self.best_x.repeat(self.num_samples, 1), mask2).to(self.device)

            mask3 = th.ones(self.num_samples, self.num_nodes, dtype=bool)
            mask3[indices_envs, :] = False
            x = th.mul(th.rand(self.num_samples, self.num_nodes), mask3).to(self.device)

            self.x = x + add_noise_for_best_x + noise
            self.x[0, :] = self.best_x  # the first row is best_x, no noise
        else:
            self.x = th.rand(self.num_samples, self.num_nodes).to(self.device)
        self.num_steps = 0
        return self.x



    # make sure that mu1 and mu2 are different tensors. If they are the same, use obj function
    # calc obj for two graphs
    def obj2(self, mu1: Tensor, mu2: Tensor):
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

        return self.obj(mu1) \
               + self.obj(mu2) \
               + th.mul(th.matmul(mu1.reshape(-1, self.num_nodes, 1), (1 - mu2.reshape(-1, self.num_nodes, 1)).transpose(-1, -2)),
                        self.adjacency_matrix) \
               + th.mul(th.matmul(1 - mu1.reshape(-1, self.num_nodes, 1), mu2.reshape(-1, self.num_nodes, 1).transpose(-1, -2)),
                        self.adjacency_matrix)


    # calc obj for one graph
    def obj(self, mu: Tensor):
        # mu1 = mu1.reshape(-1, self.N, 1)
        # mu2 = mu1.reshape(-1, self.N, 1)
        # mu2_1_t = (1 - mu2).transpose(-1, -2)
        # mu12 = th.mul(th.matmul(mu1, mu2_1_t), self.adjacency_matrix)
        # cut = mu12.flatten().sum(dim=-1) / self.num_env
        # return cut

        return th.mul(th.matmul(mu.reshape(-1, self.num_nodes, 1), (1 - mu.reshape(-1, self.num_nodes, 1)).transpose(-1, -2)), self.adjacency_matrix).flatten().sum(dim=-1) / self.num_samples
