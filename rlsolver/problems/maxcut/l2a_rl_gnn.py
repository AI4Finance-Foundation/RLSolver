import os
import sys
import time
import json
import numpy as np
import torch as th
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.distributions import Categorical
from simulator import load_graph, build_adjacency_matrix, build_adjacency_index
from simulator import GraphMaxCutSimulator, EncoderBase64

try:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

TEN = th.Tensor
INT = th.IntTensor


class PolicyGNN(nn.Module):
    def __init__(self, inp_dim, mid_dim, out_dim, num_nodes):
        super().__init__()
        self.inp_dim = inp_dim
        self.mid_dim = mid_dim
        self.out_dim = out_dim
        self.num_nodes = num_nodes

        self.mat_dim = mat_dim = int(num_nodes ** 0.5)

        self.mat_enc = nn.Sequential(nn.Linear(num_nodes, mid_dim), nn.ReLU(),
                                     nn.Linear(mid_dim, mat_dim))

        self.inp_enc = nn.Sequential(nn.Linear(inp_dim + mat_dim, mid_dim), nn.ReLU(),
                                     nn.Linear(mid_dim, mid_dim))

        self.tmp_enc = nn.Sequential(nn.Linear(mid_dim + mid_dim, mid_dim), nn.ReLU(),
                                     nn.Linear(mid_dim, out_dim), nn.Tanh())

    def forward(self, inp, mat, idx):
        size = inp.shape[0]
        device = inp.device
        mat = self.mat_enc(mat)  # (num_nodes, mid_dim)

        tmp0 = th.cat((inp, mat.repeat(size, 1, 1)), dim=2)
        tmp1 = self.inp_enc(tmp0)  # (size, num_nodes, inp_dim)

        env_i = th.arange(size, device=device)
        tmp2 = th.stack([tmp1[env_i[:, None], ids[None, :]].sum(dim=1) for ids in idx], dim=1)

        tmp3 = th.cat((tmp1, tmp2), dim=2)
        tmp4 = self.tmp_enc(tmp3)[:, :, 0]  # (size, num_nodes)
        return tmp4


def run():
    num_sim = 1024
    inp_dim = 2  # x, score
    mid_dim = 64
    out_dim = 1
    seq_len = 8
    reset_gap = 256
    num_epoch = 64
    lr = 2e-4
    clip_grad_norm = 1.0
    num_nodes = 300
    graph_name = f'powerlaw_{num_nodes}_ID01'
    gpu_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    save_dir = f"./{graph_name}_{gpu_id}"
    int_type = th.int32

    print_gap = 8

    device = th.device(f'cuda:{gpu_id}' if th.cuda.is_available() and gpu_id >= 0 else 'cpu')
    mpl.use('Agg') if plt else None  # Generating matplotlib graphs without a running X server [duplicate]
    os.makedirs(save_dir, exist_ok=True)

    '''init task'''
    graph, num_nodes, num_edges = load_graph(graph_name=graph_name)

    mat = build_adjacency_matrix(graph, num_nodes, if_bidirectional=True).to(device)
    idx = build_adjacency_index(graph, num_nodes, if_bidirectional=True)[0]
    idx = [t.to(int_type).to(device) for t in idx]

    sim = GraphMaxCutSimulator(graph_name=graph_name, gpu_id=gpu_id, graph_tuple=(graph, num_nodes, num_edges))
    enc = EncoderBase64(num_nodes=sim.num_nodes)

    '''init opti'''
    opt_opti = PolicyGNN(inp_dim=inp_dim, mid_dim=mid_dim, out_dim=out_dim, num_nodes=num_nodes).to(device)
    opt_base = th.optim.Adam(opt_opti.parameters(), lr=lr, maximize=True)

    x_best = th.randint(0, 2, size=(num_nodes,), device=device)
    v_best = sim.get_scores(x_best[None, :])[0]
    recorder = []

    start_time = time.time()

    softmax = nn.Softmax(dim=1)
    temperature = 1e2
    sim_ids = th.arange(num_sim, device=device)
    for j in range(num_epoch):
        x = th.randint(0, 2, size=(num_sim, num_nodes,), device=device)
        v = sim.get_scores(x)

        for k in range(reset_gap):
            v_good = v.max()
            x_good = x[v == v_good][0]
            inp = th.stack((x_good, v_good[None].repeat(num_nodes)), dim=1)[None, :, :]
            out = opt_opti(inp, mat, idx)

            dist = Categorical(probs=softmax(out / temperature))
            sample = dist.sample(th.Size((num_sim,)))[:, 0].to(int_type)

            x = x_good.repeat(num_sim, 1)
            x[sim_ids, sample] = 1 - x[sim_ids, sample]

            xs = th.zeros((seq_len, num_sim, num_nodes), dtype=th.float64, device=device)
            vs = th.zeros((seq_len, num_sim), dtype=th.float64, device=device)
            logprobs = th.zeros((seq_len, num_sim), dtype=th.float64, device=device)
            for i in range(seq_len):
                inp = th.stack((x, v[:, None].repeat(1, num_nodes)), dim=2)
                out = opt_opti(inp, mat, idx)
                dist = Categorical(probs=softmax(out))
                sample = dist.sample(th.Size((1,)))[0].to(int_type)

                x[sim_ids, sample] = 1 - x[sim_ids, sample]
                v = sim.get_scores(x)

                xs[i] = x
                vs[i] = v
                logprobs[i] = dist.log_prob(sample)

            vs_max = vs.max(dim=0)[0]
            mask = th.eq(vs_max, vs_max.max())

            obj_logprob = logprobs[:, mask].mean(dim=0).clip(-10, -0.1).exp().mean()
            obj_opti = obj_logprob  # + obj_entropy

            opt_base.zero_grad()
            obj_opti.backward()
            clip_grad_norm_(parameters=opt_base.param_groups[0]["params"], max_norm=clip_grad_norm)
            opt_base.step()

            '''evaluate'''
            count = j * reset_gap + k

            v_max = vs.max()
            if v_max > v_best:
                v_best = v_max
                x_best = xs[vs == v_best][0]
                print(f"v_best {v_best}  x_best \n{enc.bool_to_str(x_best)}")

                recorder.append((count, v_best))
                recorder_ary = th.tensor(recorder)
                if plt:
                    plt.plot(recorder_ary[:, 0], recorder_ary[:, 1])
                    plt.scatter(recorder_ary[:, 0], recorder_ary[:, 1])
                    plt.grid()
                    plt.title(f"best_obj_value {v_best}")
                    plt.savefig(f"{save_dir}/recorder.jpg")
                    plt.close('all')
                else:
                    np.savetxt(f"{save_dir}/recorder.txt", recorder_ary.data.detach().cpu().numpy())

            if count % print_gap == 0:
                used_time = time.time() - start_time
                print(f"| {used_time:9.0f}  {j:6}  {k:6}  "
                      # f"| {obj_logprob.item():9.4f}  {obj_entropy.item():9.4f}  "
                      f"| {obj_logprob.item():9.4f}  "
                      f"| obj_value {v_max:6.0f}  {v_best:6.0f}  ")

                # if count % (print_gap * reset_gap) == 0:
                #     print(f"best_score {v_best}  best_x \n{enc.bool_to_str(x_best)}")


if __name__ == '__main__':
    run()
