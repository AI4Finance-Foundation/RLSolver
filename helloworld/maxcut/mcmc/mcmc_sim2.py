import sys
import time
import torch as th
import torch.nn as nn
import os
from typing import Union, List
import numpy as np
import torch as th
import torch.nn as nn
from torch import Tensor
from torch.distributions import Bernoulli  # BinaryDist
from tqdm import tqdm
import matplotlib

class MCMCSim2:
    def __init__(self, graph_name: str = 'gset_14', gpu_id: int = -1):
        self.grapj_name = graph_name
        device = th.device(f'cuda:{gpu_id}' if th.cuda.is_available() and gpu_id >= 0 else 'cpu')
        int_type = th.int32
        self.device = device
        self.int_type = int_type
        self.read_txt_as_adjacency_matrix(f'data/{graph_name}.txt')

    def step(self, probs: th.Tensor):
        assert probs.shape[-1] == self.num_nodes
        num_samples = probs.shape[0]
        sum_dts = []
        for env_i in range(num_samples):
            env_probs = probs[env_i]
            map_node_to_probs = []
            for node_id in range(self.num_nodes):
                adjacency_nodes = th.where(self.adjacency_matrix[node_id] != -1)[
                    0]  # find all adjacency nodes given the node_id
                adjacency_probs = env_probs[adjacency_nodes]  # get the probability of adjacency nodes of the node_id
                map_node_to_probs.append(adjacency_probs)

            sum_dt = []
            for _p0, _p1 in zip(env_probs, map_node_to_probs):  # iterate all nodes, node_i
                # the prob of node_i in set A and its adjacent node in set B: `_p0 * (1-_p1)` (1)
                # the prob of node_i in set B and its adjacent node in set A: `_p1 * (1-_p0)` (2)
                # dt = (1)+ (2) = _p0 * (1-_p1) + _p1 * (1-_p0)
                dt = _p0 + _p1 - 2 * _p0 * _p1
                sum_dt.append(dt.sum(dim=0))
            sum_dt = th.stack(sum_dt).sum(dim=-1)  # the objective of the current environment
            sum_dts.append(sum_dt)
        sum_dts = th.hstack(sum_dts)
        return -sum_dts

    def get_rand_probs(self, num_samples: int):
        # generate random probability for each node, mainly for initilization
        return th.rand((num_samples, self.num_nodes), dtype=th.float32, device=self.device)

    @staticmethod
    def make_decision(prob: th.Tensor, thresh=0.5):
        # make decision of each node whether it is in set A by the probability given the threshold
        return prob > thresh

    def get_score(self, decisions: th.Tensor):
        # get the score of the decision
        num_samples = decisions.shape[0]
        env_ids = th.arange(num_samples, dtype=self.int_type, device=self.device)
        # unsqueeze(1) is to make the shape of env_ids to (num_samples, 1), so that it can be broadcasted to (num_samples, num_edges)
        p0 = decisions[env_ids.unsqueeze(1), self.map_edge_to_n0_n1_dt[:, 0].repeat(num_samples, 1)]
        p1 = decisions[env_ids.unsqueeze(1), self.map_edge_to_n0_n1_dt[:, 1].repeat(num_samples, 1)]
        return (p0 ^ p1).sum(1)

    def read_txt_as_adjacency_matrix(self, filename: str) -> np.array:
        with open(filename, 'r') as file:
            lines = file.readlines()
            lines = [[int(i1) for i1 in i0.split()] for i0 in lines]
        num_nodes, num_edges = lines[0]
        self.num_nodes = num_nodes
        self.num_edges = num_edges
        map_edge_to_n0_n1_dt = [(n0 - 1, n1 - 1, dt) for n0, n1, dt in lines[
                                                                       1:]]  # re-index the node from 0, and save the edge by a 3-tuple (left-node, right-node, distance)
        self.map_edge_to_n0_n1_dt = th.tensor(map_edge_to_n0_n1_dt, dtype=th.int,
                                              device=self.device)  # transfer to tensor
        adjacency_matrix = th.empty((num_nodes, num_nodes), dtype=th.float32, device=self.device)
        adjacency_matrix[:] = -1  # initialize the adjacency matrix
        for n0, n1, dt in self.map_edge_to_n0_n1_dt:
            adjacency_matrix[n0, n1] = dt
        assert num_nodes == adjacency_matrix.shape[0] == adjacency_matrix.shape[1]
        assert num_edges == (adjacency_matrix != -1).sum()
        self.adjacency_matrix = adjacency_matrix

    def write_result(self, result: Union[Tensor, List, np.array], filename: str = 'result/result.txt'):
        # assert len(result.shape) == 1
        # N = result.shape[0]
        num_nodes = len(result)
        directory = filename.split('/')[0]
        if not os.path.exists(directory):
            os.mkdir(directory)
        with open(filename, 'w', encoding="UTF-8") as file:
            for node in range(num_nodes):
                file.write(f'{node + 1} {int(result[node] + 1)}\n')
