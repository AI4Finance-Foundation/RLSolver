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
from mcmc_sim.mcmc_sim2 import MCMCSim2

def search_by_grad():
    gpu_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    num_samples = 2 ** 6
    graph_name = 'g70'

    '''hyper-parameters'''
    lr = 1e-3
    opt_num = 2 ** 12
    eval_gap = 2

    '''init task'''
    mcmc_sim = MCMCSim2(graph_name=graph_name, gpu_id=gpu_id)
    probs = mcmc_sim.get_rand_probs(num_samples=num_samples)
    probs.requires_grad_(True)

    '''loop'''
    best_score = -th.inf
    current_score = -th.inf
    decision = None
    with tqdm(total=opt_num) as p_bar:
        for i in range(1, opt_num + 1):
            p_bar.update(1)
            p_bar.set_description("current max score:{}, history max score:{}".format(current_score, best_score))
            obj = mcmc_sim.step(probs).mean()
            obj.backward()

            grads = probs.grad.data
            probs.data.add_(-lr * grads).clip_(0, 1)

            if i % eval_gap == 0:
                decision = mcmc_sim.make_decision(probs)
                scores = mcmc_sim.get_score(decision)
                current_score = scores.max().item()
                max_score, max_id = th.max(scores, dim=0)
                if max_score > best_score:
                    best_score = max_score
                    tqdm.write(f"\rbest_score {best_score}")

    print()
    print(f"best_score {best_score}")
    mcmc_sim.write_result(decision[0].cpu().numpy(), f'result/{graph_name}.txt')


def search_by_adam():
    gpu_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    num_samples = 2 ** 4
    graph_name = 'G14'
    # graph_name = 'g70'

    '''hyper-parameters'''
    lr = 1e-3
    opt_num = 2 ** 12
    eval_gap = 2

    '''init task'''
    mcmc_sim = MCMCSim2(graph_name=graph_name, gpu_id=gpu_id)
    probs = mcmc_sim.get_rand_probs(num_samples=num_samples)
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
            obj = mcmc_sim.step(probs).mean()
            optim.zero_grad()
            obj.backward()
            optim.step()

            probs.data.clip_(0, 1)

            if i % eval_gap == 0:
                decision = mcmc_sim.make_decision(probs)
                scores = mcmc_sim.get_score(decision)
                current_score = scores.max().item()
                max_score, max_id = th.max(scores, dim=0)
                if max_score > best_score:
                    best_score = max_score
                    tqdm.write(f"\rbest_score {best_score}")

    print()
    print(f"best_score {best_score}")
    mcmc_sim.write_result(decision[0].cpu().numpy(), f'result/{graph_name}.txt')


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
    num_samples = 2 ** 6
    graph_name = 'G14'
    # graph_name = 'g70'

    '''hyper-parameters'''
    lr = 1e-3
    mid_dim = 2 ** 8
    num_layers = 1
    seq_len = 2 ** 5
    reset_gap = 2 ** 6

    opt_num = int(2 ** 16 / num_samples)
    eval_gap = 2 ** 1

    '''init task'''
    mcmc_sim = MCMCSim2(graph_name=graph_name, gpu_id=gpu_id)
    dim = mcmc_sim.num_nodes
    probs = mcmc_sim.get_rand_probs(num_samples=num_samples)
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
                probs = mcmc_sim.get_rand_probs(num_samples=num_samples)
                probs.requires_grad_(True)
                obj = None
                hidden0 = None
                hidden1 = None

            prob_ = probs.clone()
            updates = []

            for j in range(seq_len):
                obj = mcmc_sim.step(probs).mean()
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
            obj_ = mcmc_sim.step(prob_).mean()

            opt_base.zero_grad()
            obj_.backward()
            opt_base.step()

            probs.data[:] = prob_

            if i % eval_gap == 0:
                decision = mcmc_sim.make_decision(probs)
                scores = mcmc_sim.get_score(decision)
                current_score = scores.max().item()
                max_score, max_id = th.max(scores, dim=0)
                if max_score > best_score:
                    best_score = max_score

    print()
    print(f"best_score {best_score}")
    mcmc_sim.write_result(decision[0].cpu().numpy(), f'result/{graph_name}.txt')


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
    num_samples = 2 ** 8
    # graph_name, num_limit = 'G14', sys.maxsize
    graph_name, num_limit = 'G14', 28

    '''hyper-parameters'''
    lr = 1e-3
    mid_dim = 2 ** 8
    num_layers = 1
    opt_num = int(2 ** 24 / num_samples)
    eval_gap = 2 ** 4

    alpha_period = 2 ** 10
    alpha_weight = 1.0

    '''init task'''
    mcmc_sim = MCMCSim2(graph_name=graph_name, gpu_id=gpu_id)
    dim = mcmc_sim.num_nodes

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
            samples, logprobs, entropies = model.sample(dim, num_samples, True, device=device)
            scores = mcmc_sim.get_score(samples).detach().to(th.float32)
            scores = (scores - scores.min()) / (scores.std() + 1e-4)

            obj_probs = logprobs.exp()
            obj = -((obj_probs / obj_probs.mean()) * scores + (alpha * alpha_weight) * entropies).mean()

            opt_base.zero_grad()
            obj.backward()
            opt_base.step()

            if i % eval_gap == 0:
                _samples, _, _ = model.sample(dim, num_samples, False, device=device)
                _scores = mcmc_sim.get_score(_samples)

                samples = th.vstack((samples, _samples))
                scores = th.hstack((scores, _scores))

            max_score, max_id = th.max(scores, dim=0)
            current_score = max_score.item()
            if max_score > best_score:
                best_score = max_score
    print()
    print(f"best_score {best_score}")
    mcmc_sim.write_result(decision[0].cpu().numpy(), f'result/{graph_name}.txt')


if __name__ == '__main__':
    search_by_grad()
    # search_by_adam()
    # search_by_optimizer_optimizee()
    # search_by_auto_regressive_model()
