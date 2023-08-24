import os
import sys
import time
import json
import numpy as np
import torch as th
import torch.nn as nn
from torch.distributions import Binomial
from graph_max_cut_simulator import load_graph, build_adjacency_matrix, build_adjacency_index
from graph_max_cut_simulator import GraphMaxCutSimulator, EncoderBase64

TEN = th.Tensor


class OptimizerNN(nn.Module):
    def __init__(self, inp_dim, mid_dim, out_dim, num_nodes):
        super().__init__()
        self.inp_dim = inp_dim
        self.mid_dim = mid_dim
        self.out_dim = out_dim
        self.num_nodes = num_nodes

        self.mat_dim = mat_dim = int(num_nodes ** 0.5)
        self.rnn_dim = rnn_dim = inp_dim * 4
        self.inp_rnn = nn.LSTMCell(inp_dim, rnn_dim)

        self.mat_enc = nn.Sequential(nn.Linear(num_nodes, mid_dim), nn.ReLU(),
                                     nn.Linear(mid_dim, mat_dim))

        self.inp_enc = nn.Sequential(nn.Linear(inp_dim * 4 + mat_dim, mid_dim), nn.ReLU(),
                                     nn.Linear(mid_dim, mid_dim))

        self.tmp_enc = nn.Sequential(nn.Linear(mid_dim + mid_dim, mid_dim), nn.ReLU(),
                                     nn.Linear(mid_dim, out_dim), nn.Sigmoid())

    def forward(self, inp, hid, mat, idx):
        size = inp.shape[0]
        device = inp.device
        inp = inp.reshape(size * self.num_nodes, self.inp_dim)
        hid, cell = self.inp_rnn(inp, hid)
        inp = hid.reshape(size, self.num_nodes, -1)
        # inp.shape == (size, num_nodes, inp_dim*4)
        # mat.shape == (num_nodes, num_nodes)
        # isinstance(idx, list) and isinstance(idx[0], TEN)

        mat = self.mat_enc(mat)  # (num_nodes, mid_dim)

        tmp0 = th.cat((inp, mat.repeat(size, 1, 1)), dim=2)
        tmp1 = self.inp_enc(tmp0)  # (size, num_nodes, inp_dim)

        env_i = th.arange(size, device=device)
        tmp2 = th.stack([tmp1[env_i[:, None], ids[None, :]].sum(dim=1) for ids in idx], dim=1)

        tmp3 = th.cat((tmp1, tmp2), dim=2)
        prob = self.tmp_enc(tmp3)[:, :, 0]  # (size, num_nodes)
        return prob, (hid, cell)


def run():
    sim_ids = 48
    inp_dim = 3  # sample, prob, score
    mid_dim = 64
    out_dim = 1
    seq_len = 64
    num_epoch = 32
    lr = 1e-3
    graph_name = 'powerlaw_32_ID01'
    gpu_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    int_type = th.int32

    print_gap = 4

    device = th.device(f'cuda:{gpu_id}' if th.cuda.is_available() and gpu_id >= 0 else 'cpu')

    '''init task'''
    graph, num_nodes, num_edges = load_graph(graph_name=graph_name)

    mat = build_adjacency_matrix(graph, num_nodes, if_bidirectional=True).to(device)
    idx = build_adjacency_index(graph, num_nodes, if_bidirectional=True)[0]
    idx = [t.to(int_type).to(device) for t in idx]

    sim = GraphMaxCutSimulator(graph_name='powerlaw_32', gpu_id=gpu_id, graph_tuple=(graph, num_nodes, num_edges))
    enc = EncoderBase64(num_nodes=sim.num_nodes)

    '''init opti'''
    opt_opti = OptimizerNN(inp_dim=inp_dim, mid_dim=mid_dim, out_dim=out_dim, num_nodes=num_nodes)
    opt_base = th.optim.Adam(opt_opti.parameters(), lr=lr)

    prob = th.rand(sim_ids, num_nodes, device=device)
    dist = Binomial(probs=prob, total_count=1)
    sample_best = dist.sample(th.Size((1,)))[0].to(int_type)
    score_best = -th.inf
    start_time = time.time()

    for j in range(num_epoch):
        hid = (th.zeros(sim_ids * num_nodes, opt_opti.rnn_dim),
               th.zeros(sim_ids * num_nodes, opt_opti.rnn_dim),)
        prob = th.rand(sim_ids, num_nodes, device=device)
        dist = Binomial(probs=prob, total_count=1)
        sample = dist.sample(th.Size((1,)))[0].to(int_type)
        score = sim.get_scores(sample)

        samples = th.zeros((seq_len, sim_ids, num_nodes), dtype=th.float32, device=device)
        scores = th.zeros((seq_len, sim_ids), dtype=th.float32, device=device)
        logprobs = th.zeros((seq_len, sim_ids), dtype=th.float32, device=device)

        for i in range(seq_len):
            inp = th.stack((prob, sample, score[:, None].repeat(1, num_nodes)), dim=2)
            prob, hid = opt_opti(inp, hid, mat, idx)

            dist = Binomial(probs=prob, total_count=1)
            sample = dist.sample(th.Size((1,)))[0].to(int_type)
            score = sim.get_scores(sample)

            samples[i] = sample
            scores[i] = score
            logprobs[i] = dist.log_prob(sample).sum(dim=1)

        logprobs = logprobs.sum(dim=0)
        obj_probs = ((logprobs - logprobs.mean())/logprobs.std()).exp()

        scores_max = scores.max(dim=0)[0]

        signals = th.ones_like(scores_max, dtype=th.float32)
        signals[scores_max != scores_max.max()] = -1

        obj_opti = -(obj_probs * signals).mean()

        opt_base.zero_grad()
        obj_opti.backward()
        opt_base.step()

        score_max = scores.max()
        if score_max > score_best:
            score_best = score_max
            sample_best = samples[scores == score_best][0]

        if j % print_gap == 0:
            used_time = time.time() - start_time
            print(f"|{used_time:9.0f}  {j:6}  {obj_opti.item():9.3f}  "
                  f"score {scores.max().item():6.0f}  {score_best:6.0f}  ")

            if j % (print_gap * 256) == 0:
                print(f"best_score {score_best}  best_sln_x \n{enc.bool_to_str(sample_best)}")


run()
