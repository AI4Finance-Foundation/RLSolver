import torch as th
import torch.nn as nn
from copy import deepcopy
import numpy as np
from torch import Tensor
# from envs._base_env import _BaseEnv
from rlsolver.rlsolver_learn2opt.np_complete_problems.envs._base_env import _BaseEnv
# from _base_env import _BaseEnv
class TspEnv(_BaseEnv):
    def __init__(self, num_nodes=20, num_envs=128, device=th.device("cuda:0"), episode_length=6):
        self.num_nodes = num_nodes
        self.num_envs = num_envs
        self.device = device
        self.episode_length = episode_length
        self.calc_obj_for_two_graphs_vmap = th.vmap(self.calc_obj_for_two_graphs, in_dims=(0, 0))
        self.adjacency_matrix = None


    # make sure that mu1 and mu2 are different tensors. If they are the same, use get_cut_value_one_tensor
    def calc_obj_for_two_graphs(self, mu1: Tensor, mu2: Tensor):
        return self.calc_obj_for_one_graph(mu1) \
               + self.calc_obj_for_one_graph(mu2)

    # if select_nonedge is true, the function is selecting the cases which are not edges, and use adjacency_matrix otherwise.
    def calc_path_dist(self, mu: Tensor, select_nonedge: bool):
        mu_roll = mu.roll(shifts=self.num_nodes - 1, dims=1)  # roll
        if select_nonedge:
            indices = (self.adjacency_matrix == 0).nonzero(as_tuple=True)
        else:
            indices = (self.adjacency_matrix != 0).nonzero(as_tuple=True)
        indices_i = indices[0]  # indices for rows in mu
        indices_j = indices[1]  # indices for rows in mu_roll
        indices_k = (indices_i != indices_j).nonzero(as_tuple=True)  # remove the indices like (0, 0), (1, 1), ...
        indices_i_ = indices_i[indices_k]
        indices_j_ = indices_j[indices_k]
        if select_nonedge:
            matrix = th.mul(mu[indices_i_], mu_roll[indices_j_])
        else:
            adj = self.adjacency_matrix[indices_i_, :][:, indices_j_]
            adj2 = self.adjacency_matrix[indices_i_, indices_j_]
            mi = mu[indices_i_]
            mj = mu_roll[indices_j_]
            m = th.mul(mu[indices_i_], mu_roll[indices_j_])
            matrix = th.mul(adj, m)
        dist = matrix.sum()
        return dist

    # weight is the weight of final part compared with the previous parts. default: 1.0
    def calc_obj_for_one_graph(self, mu: Tensor, weight: float = 1.0):
        part1 = (1 - mu.sum(dim=0))
        part2 = (1 - mu.sum(dim=1))
        # mu_roll = mu.roll(shifts=self.num_nodes - 1, dims=1)  # roll
        # indices = (self.adjacency_matrix == 0).nonzero(as_tuple=True)
        # indices_i = indices[0]  # indices for rows
        # indices_j = indices[1]  # indices for cols
        # indices_k = (indices_i != indices_j).nonzero(as_tuple=True)  # remove the indices like (0, 0), (1, 1), ...
        # indices_i_ = indices_i[indices_k]
        # indices_j_ = indices_j[indices_k]
        # part3 = th.mul(mu[indices_i_], mu_roll[indices_j_])
        part3 = self.calc_path_dist(mu, True)
        part4 = self.calc_path_dist(mu, False)
        return ((part1 ** 2).sum() + (part2 ** 2).sum() + part3 + weight * part4) / self.num_envs

if __name__ == '__main__':
    env = TspEnv()
    env.num_envs = 1
    env.num_nodes = 4
    env.adjacency_matrix = th.Tensor([[0, 1, 5, 2],
                                      [1, 0, 3, 1000],
                                      [5, 3, 0, 1],
                                      [2, 1000, 1, 0]])
    mu = th.Tensor([[1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]])
    obj = env.calc_obj_for_one_graph(mu)
    print()