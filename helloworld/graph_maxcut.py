import os
import sys
import time
import math
import json
import numpy as np
import torch as th
import torch.nn as nn
from torch import Tensor
from torch.distributions import Bernoulli  # BinaryDist
from tqdm import tqdm
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from typing import Union, List

class GraphMaxCutEnv:
    def __init__(self, graph_name: str = 'gset_14', gpu_id: int = -1):
        self.grapj_name = graph_name
        device = th.device(f'cuda:{gpu_id}' if th.cuda.is_available() and gpu_id >= 0 else 'cpu')
        int_type = th.int32
        self.device = device
        self.int_type = int_type
        self.read_txt_as_adjacency_matrix(f'data/{graph_name}.txt')

        
    def step(self, probs:th.Tensor):
        assert probs.shape[-1] == self.num_nodes
        num_envs = probs.shape[0]
        sum_dts = []
        for env_i in range(num_envs): 
            env_probs = probs[env_i]
            map_node_to_probs = []
            for node_id in range(self.num_nodes): 
                adjacency_nodes = th.where(self.adjacency_matrix[node_id] != -1)[0]  # find all adjacency nodes given the node_id
                adjacency_probs = env_probs[adjacency_nodes]  # get the probability of adjacency nodes of the node_id
                map_node_to_probs.append(adjacency_probs)

            sum_dt = []
            for _p0, _p1 in zip(env_probs, map_node_to_probs): # iterate all nodes, node_i
                # the prob of node_i in set A and its adjacent node in set B: `_p0 * (1-_p1)` (1) 
                # the prob of node_i in set B and its adjacent node in set A: `_p1 * (1-_p0)` (2)
                # dt = (1)+ (2) = _p0 * (1-_p1) + _p1 * (1-_p0)
                dt = _p0 + _p1 - 2 * _p0 * _p1
                sum_dt.append(dt.sum(dim=0))
            sum_dt = th.stack(sum_dt).sum(dim=-1)  # the objective of the current environment
            sum_dts.append(sum_dt)
        sum_dts = th.hstack(sum_dts)  
        return -sum_dts
    
    def get_rand_probs(self, num_envs: int):
        # generate random probability for each node, mainly for initilization
        return th.rand((num_envs, self.num_nodes), dtype=th.float32, device=self.device)
    
    
    @staticmethod
    def make_decision(prob:th.Tensor, thresh=0.5):
        # make decision of each node whether it is in set A by the probability given the threshold
        return prob > thresh
    
    def get_score(self,decisions:th.Tensor):
        # get the score of the decision
        num_envs = decisions.shape[0]
        env_ids = th.arange(num_envs, dtype=self.int_type, device=self.device)
        # unsqueeze(1) is to make the shape of env_ids to (num_envs, 1), so that it can be broadcasted to (num_envs, num_edges)
        p0 = decisions[env_ids.unsqueeze(1), self.map_edge_to_n0_n1_dt[:, 0].repeat(num_envs, 1)]
        p1 = decisions[env_ids.unsqueeze(1), self.map_edge_to_n0_n1_dt[:, 1].repeat(num_envs, 1)]
        return (p0 ^ p1).sum(1)
    
    
    def read_txt_as_adjacency_matrix(self, filename: str) -> np.array:
        with open(filename, 'r') as file:
            lines = file.readlines()
            lines = [[int(i1) for i1 in i0.split()] for i0 in lines]
        num_nodes, num_edges = lines[0]
        self.num_nodes = num_nodes
        self.num_edges = num_edges
        map_edge_to_n0_n1_dt = [(n0 - 1, n1 - 1, dt) for n0, n1, dt in lines[1:]]  # re-index the node from 0, and save the edge by a 3-tuple (left-node, right-node, distance)
        self.map_edge_to_n0_n1_dt = th.tensor(map_edge_to_n0_n1_dt, dtype=th.int, device=self.device) # transfer to tensor
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
    
    
    
    
def search_by_grad():
    gpu_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    num_envs = 2 ** 6
    graph_name = 'g70'

    '''hyper-parameters'''
    lr = 1e-3
    opt_num = 2 ** 12
    eval_gap = 2 

    '''init task'''
    env = GraphMaxCutEnv(graph_name=graph_name, gpu_id=gpu_id)
    probs = env.get_rand_probs(num_envs=num_envs)
    probs.requires_grad_(True)

    '''loop'''
    best_score = -th.inf
    current_score = -th.inf
    decision = None
    with tqdm(total=opt_num) as p_bar:
        for i in range(1, opt_num + 1):
            p_bar.update(1)
            p_bar.set_description("current max score:{}, history max score:{}".format(current_score, best_score))
            obj = env.step(probs).mean()
            obj.backward()

            grads = probs.grad.data
            probs.data.add_(-lr * grads).clip_(0, 1)

            if i % eval_gap == 0:
                decision = env.make_decision(probs)
                scores = env.get_score(decision)
                current_score = scores.max().item()
                max_score, max_id = th.max(scores, dim=0)
                if max_score > best_score:
                    best_score = max_score
                    tqdm.write(f"\rbest_score {best_score}")

    print()
    print(f"best_score {best_score}")
    env.write_result(decision[0].cpu().numpy(), f'result/{graph_name}.txt')
    
    
    
def search_by_adam():
    gpu_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    num_envs = 2 ** 4
    graph_name = 'G14'
    # graph_name = 'g70'

    '''hyper-parameters'''
    lr = 1e-3
    opt_num = 2 ** 12
    eval_gap = 2 

    '''init task'''
    env = GraphMaxCutEnv(graph_name=graph_name, gpu_id=gpu_id)
    probs = env.get_rand_probs(num_envs=num_envs)
    probs.requires_grad_(True)

    '''init opti'''
    optim = th.optim.Adam([probs, ], lr=lr)

    
    
    '''loop'''
    best_score = -th.inf
    current_score = -th.inf
    with tqdm(total=opt_num) as p_bar:
        for i in range(1, opt_num + 1):
            p_bar.update(1)
            p_bar.set_description("current max score:{}, history max score:{}".format(current_score, best_score))
            obj = env.step(probs).mean()
            optim.zero_grad()
            obj.backward()
            optim.step()

            probs.data.clip_(0, 1)


            if i % eval_gap == 0:
                decision = env.make_decision(probs)
                scores = env.get_score(decision)
                current_score = scores.max().item()
                max_score, max_id = th.max(scores, dim=0)
                if max_score > best_score:
                    best_score = max_score
                    tqdm.write(f"\rbest_score {best_score}")
    

    print()
    print(f"best_score {best_score}")
    env.write_result(decision[0].cpu().numpy(), f'result/{graph_name}.txt')





class OptimizerLSTM(nn.Module):
    def __init__(self, inp_dim, mid_dim, out_dim, num_layers):
        super().__init__()
        self.inp_dim = inp_dim
        self.mid_dim = mid_dim
        self.out_dim = out_dim
        self.num_layers = num_layers

        self.rnn0 = nn.LSTM(inp_dim, mid_dim, num_layers=num_layers)
        self.mlp0 = nn.Linear(mid_dim, out_dim)
        self.rnn1 = nn.LSTM(1, 8, num_layers=num_layers)
        self.mlp1 = nn.Linear(8, 1)

    def forward(self, inp, hid0=None, hid1=None):
        tmp0, hid0 = self.rnn0(inp, hid0)
        out0 = self.mlp0(tmp0)

        d0, d1, d2 = inp.shape
        inp1 = inp.reshape(d0, d1 * d2, 1)
        tmp1, hid1 = self.rnn1(inp1, hid1)
        out1 = self.mlp1(tmp1).reshape(d0, d1, d2)

        out = out0 + out1
        return out, hid0, hid1
    

def search_by_optimizer_optimizee():
    gpu_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    num_envs = 2 ** 6
    graph_name = 'G14'
    # graph_name = 'g70'

    '''hyper-parameters'''
    lr = 1e-3
    mid_dim = 2 ** 8
    num_layers = 1
    seq_len = 2 ** 5
    reset_gap = 2 ** 6

    opt_num = int(2 ** 16 / num_envs)
    eval_gap = 2 ** 1

    '''init task'''
    env = GraphMaxCutEnv(graph_name=graph_name, gpu_id=gpu_id)
    dim = env.num_nodes
    probs = env.get_rand_probs(num_envs=num_envs)
    probs.requires_grad_(True)
    obj = None
    hidden0 = None
    hidden1 = None

    '''init opti'''
    device = th.device(f'cuda:{gpu_id}' if th.cuda.is_available() and gpu_id >= 0 else 'cpu')
    opt_opti = OptimizerLSTM(inp_dim=dim, mid_dim=mid_dim, out_dim=dim, num_layers=num_layers).to(device)
    opt_base = th.optim.Adam(opt_opti.parameters(), lr=lr)

    '''loop'''
    best_score = -th.inf
    current_score = -th.inf
    with tqdm(total=opt_num) as p_bar:
        for i in range(1, opt_num + 1):
            p_bar.update(1)
            p_bar.set_description("current max score:{}, history max score:{}".format(current_score, best_score))
            if i % reset_gap == 0:
                probs = env.get_rand_probs(num_envs=num_envs)
                probs.requires_grad_(True)
                obj = None
                hidden0 = None
                hidden1 = None

            prob_ = probs.clone()
            updates = []

            for j in range(seq_len):
                obj = env.step(probs).mean()
                obj.backward()

                grad_s = probs.grad.data
                update, hidden0, hidden1 = opt_opti(grad_s.unsqueeze(0), hidden0, hidden1)
                update = (update.squeeze_(0) - grad_s) * lr
                updates.append(update)
                probs.data.add_(update).clip_(0, 1)
            hidden0 = [h.detach() for h in hidden0]
            hidden1 = [h.detach() for h in hidden1]

            updates = th.stack(updates, dim=0)
            prob_ = (prob_ + updates.mean(0)).clip(0, 1)
            obj_ = env.step(prob_).mean()

            opt_base.zero_grad()
            obj_.backward()
            opt_base.step()

            probs.data[:] = prob_

            if i % eval_gap == 0:
                decision = env.make_decision(probs)
                scores = env.get_score(decision)
                current_score = scores.max().item()
                max_score, max_id = th.max(scores, dim=0)
                if max_score > best_score:
                    best_score = max_score
                
    

    print()
    print(f"best_score {best_score}")
    env.write_result(decision[0].cpu().numpy(), f'result/{graph_name}.txt')




class AutoRegressiveModel(nn.Module):
    def __init__(self, inp_dim, mid_dim, out_dim, num_layers):
        super().__init__()
        self.rnn1 = nn.LSTM(inp_dim, mid_dim, num_layers=num_layers)
        self.mlp1 = nn.Sequential(nn.Linear(mid_dim, out_dim), nn.Sigmoid())

    def forward(self, inp, hid=None):
        tmp, hid = self.rnn1(inp, hid)
        out = self.mlp1(tmp)
        return out, hid
    
    def sample(self, sequence_dim, sequence_num, return_prob_entropy=True, device='cpu'):
        hidden = None
        sample = th.zeros((sequence_num, 1), dtype=th.float32, device=device)
        node_prob = th.zeros((sequence_num, 1), dtype=th.float32, device=device)

        samples = []
        logprobs = []
        entropies = []

        samples.append(sample)
        for _ in range(sequence_dim - 1):
            obs = th.hstack((sample, node_prob))
            node_prob, hidden = self.forward(obs, hidden)
            dist = Bernoulli(node_prob.squeeze(0))
            sample = dist.sample()

            samples.append(sample)
            if return_prob_entropy:
                logprobs.append(dist.log_prob(sample))
                entropies.append(dist.entropy())

        samples = th.stack(samples).squeeze(2)
        samples = samples.permute(1, 0).to(th.int)

        if return_prob_entropy:
            logprobs = th.stack(logprobs).squeeze(2).sum(0)
            logprobs = logprobs - logprobs.mean()

            entropies = th.stack(entropies).squeeze(2).mean(0)
        return samples, logprobs, entropies

           


def search_by_auto_regressive_model():
    gpu_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    num_envs = 2 ** 8
    # graph_name, num_limit = 'G14', sys.maxsize
    graph_name, num_limit = 'G14', 28

    '''hyper-parameters'''
    lr = 1e-3
    mid_dim = 2 ** 8
    num_layers = 1
    opt_num = int(2 ** 24 / num_envs)
    eval_gap = 2 ** 4

    alpha_period = 2 ** 10
    alpha_weight = 1.0

    '''init task'''
    env = GraphMaxCutEnv(graph_name=graph_name, gpu_id=gpu_id)
    dim = env.num_nodes

    '''init opti'''
    device = th.device(f'cuda:{gpu_id}' if th.cuda.is_available() and gpu_id >= 0 else 'cpu')
    model = AutoRegressiveModel(inp_dim=2, mid_dim=mid_dim, out_dim=1, num_layers=num_layers).to(device)
    opt_base = th.optim.Adam(model.parameters(), lr=lr)

    '''loop'''
    best_score = -th.inf
    current_score = -th.inf
    with tqdm(total=opt_num) as p_bar:
        for i in range(1, opt_num + 1):
            p_bar.update(1)
            p_bar.set_description("current max score:{}, history max score:{}".format(current_score, best_score))
   
            alpha = (math.cos(i * math.pi / alpha_period) + 1) / 2
            samples, logprobs, entropies = model.sample(dim, num_envs, True, device=device)
            scores = env.get_score(samples).detach().to(th.float32)
            scores = (scores - scores.min()) / (scores.std() + 1e-4)

            obj_probs = logprobs.exp()
            obj = -((obj_probs / obj_probs.mean()) * scores + (alpha * alpha_weight) * entropies).mean()

            opt_base.zero_grad()
            obj.backward()
            opt_base.step()

            if i % eval_gap == 0:
                _samples, _, _ =  model.sample(dim, num_envs, False, device=device)
                _scores = env.get_score(_samples)

                samples = th.vstack((samples, _samples))
                scores = th.hstack((scores, _scores))

            max_score, max_id = th.max(scores, dim=0)
            current_score = max_score.item()
            if max_score > best_score:
                best_score = max_score
    print()
    print(f"best_score {best_score}")
    env.write_result(decision[0].cpu().numpy(), f'result/{graph_name}.txt')





if __name__ == '__main__':
    search_by_grad()
    # search_by_adam()
    # search_by_optimizer_optimizee()
    # search_by_auto_regressive_model()
