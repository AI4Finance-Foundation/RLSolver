import sys
import time
import math
import torch as th
import torch.nn as nn
from torch.distributions import Bernoulli  # BinaryDist

from env import GraphMaxCutEnv

TEN = th.Tensor

"""optimizer"""


class Optimizer(nn.Module):
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


def evaluate_and_print(probs, env, start_time, best_score, i, obj):
    sln_xs = env.node_prob_to_bool(probs)
    scores = env.get_scores(sln_xs)
    used_time = time.time() - start_time
    print(f"|{used_time:9.0f}  {i:6}  {obj.item():9.3f}  {scores.max().item():9.0f}")

    max_score, max_id = th.max(scores, dim=0)
    if max_score > best_score:
        best_score = max_score
        best_sln_x = sln_xs[max_id]
        print(f"best_score {best_score}  best_sln_x \n{env.node_bool_to_int(best_sln_x)}")


def run_optimizer_by_grad():
    gpu_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    num_envs = 2 ** 6
    graph_name = 'syn_20_42'

    '''hyper-parameters'''
    lr = 1e-3
    opt_num = 2 ** 12
    eval_gap = 2 ** 6

    '''init task'''
    env = GraphMaxCutEnv(graph_name=graph_name, gpu_id=gpu_id)
    probs = env.get_rand_probs(num_envs=num_envs)
    probs.requires_grad_(True)

    '''loop'''
    best_score = -th.inf
    start_time = time.time()
    for i in range(1, opt_num + 1):
        obj = env.get_objectives(probs).mean()
        obj.backward()

        grads = probs.grad.data
        probs.data.add_(-lr * grads).clip_(0, 1)
        evaluate_and_print(probs, env, start_time, best_score, i, obj) if i % eval_gap == 0 else None


def run_optimizer_by_adam():
    gpu_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    num_envs = 2 ** 4
    graph_name = 'syn_20_42'

    '''hyper-parameters'''
    lr = 1e-3
    opt_num = 2 ** 12
    eval_gap = 2 ** 6

    '''init task'''
    env = GraphMaxCutEnv(graph_name=graph_name, gpu_id=gpu_id)
    probs = env.get_rand_probs(num_envs=num_envs)
    probs.requires_grad_(True)

    '''init opti'''
    opt_base = th.optim.Adam([probs, ], lr=lr)

    '''loop'''
    best_score = -th.inf
    start_time = time.time()
    for i in range(1, opt_num + 1):
        obj = env.get_objectives(probs).mean()
        opt_base.zero_grad()
        obj.backward()
        opt_base.step()

        probs.data.clip_(0, 1)
        evaluate_and_print(probs, env, start_time, best_score, i, obj) if i % eval_gap == 0 else None


def run_optimizer_by_opti():
    gpu_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    num_envs = 2 ** 6
    graph_name = 'syn_20_42'

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
    opt_opti = Optimizer(inp_dim=dim, mid_dim=mid_dim, out_dim=dim, num_layers=num_layers).to(device)
    opt_base = th.optim.Adam(opt_opti.parameters(), lr=lr)

    '''loop'''
    best_score = -th.inf
    start_time = time.time()
    for i in range(1, opt_num + 1):
        if i % reset_gap == 0:
            probs = env.get_rand_probs(num_envs=num_envs)
            probs.requires_grad_(True)
            obj = None
            hidden0 = None
            hidden1 = None

        prob_ = probs.clone()
        updates = []

        for j in range(seq_len):
            obj = env.get_objectives(probs).mean()
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
        obj_ = env.get_objectives(prob_).mean()

        opt_base.zero_grad()
        obj_.backward()
        opt_base.step()

        probs.data[:] = prob_
        evaluate_and_print(probs, env, start_time, best_score, i, obj) if i % eval_gap == 0 else None


"""generator"""


class Generator(nn.Module):
    def __init__(self, inp_dim, mid_dim, out_dim, num_layers):
        super().__init__()
        self.rnn1 = nn.LSTM(inp_dim, mid_dim, num_layers=num_layers)
        self.mlp1 = nn.Sequential(nn.Linear(mid_dim, out_dim), nn.Sigmoid())

    def forward(self, inp, hid=None):
        tmp, hid = self.rnn1(inp, hid)
        out = self.mlp1(tmp)
        return out, hid


def get_solution_xs(device, gene: Generator, dim: int, num_envs: int, if_train=True) -> (TEN, TEN, TEN):
    hidden = None
    sample = th.zeros((num_envs, 1), dtype=th.float32, device=device)
    node_prob = th.zeros((num_envs, 1), dtype=th.float32, device=device)

    samples = []
    logprobs = []
    entropies = []

    samples.append(sample)
    for _ in range(dim - 1):
        obs = th.hstack((sample, node_prob))
        node_prob, hidden = gene(obs, hidden)
        dist = Bernoulli(node_prob.squeeze(0))
        sample = dist.sample()

        samples.append(sample)
        if if_train:
            logprobs.append(dist.log_prob(sample))
            entropies.append(dist.entropy())

    samples = th.stack(samples).squeeze(2)
    sln_xs = samples.permute(1, 0).to(th.int)

    if if_train:
        logprobs = th.stack(logprobs).squeeze(2).sum(0)
        logprobs = logprobs - logprobs.mean()

        entropies = th.stack(entropies).squeeze(2).mean(0)
    return sln_xs, logprobs, entropies


def run_generator():
    gpu_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    num_envs = 2 ** 8
    graph_name = 'syn_20_42'

    '''hyper-parameters'''
    lr = 1e-3
    mid_dim = 2 ** 8
    num_layers = 1
    num_opt = int(2 ** 24 / num_envs)
    eval_gap = 2 ** 4
    print_gap = 2 ** 8

    alpha_period = 2 ** 10
    alpha_weight = 1.0

    '''init task'''
    env = GraphMaxCutEnv(graph_name=graph_name, gpu_id=gpu_id)
    dim = env.num_nodes

    '''init opti'''
    device = th.device(f'cuda:{gpu_id}' if th.cuda.is_available() and gpu_id >= 0 else 'cpu')
    opt_gene = Generator(inp_dim=2, mid_dim=mid_dim, out_dim=1, num_layers=num_layers).to(device)
    opt_base = th.optim.Adam(opt_gene.parameters(), lr=lr)

    '''loop'''
    best_score = -th.inf
    start_time = time.time()
    for i in range(num_opt):
        alpha = (math.cos(i * math.pi / alpha_period) + 1) / 2
        sln_xs, logprobs, entropies = get_solution_xs(device, opt_gene, dim, num_envs, if_train=True)
        scores = env.get_scores(samples=sln_xs).detach().to(th.float32)
        scores = (scores - scores.min()) / (scores.std() + 1e-4)

        obj_probs = logprobs.exp()
        obj = -((obj_probs / obj_probs.mean()) * scores + (alpha * alpha_weight) * entropies).mean()

        opt_base.zero_grad()
        obj.backward()
        opt_base.step()

        if i % eval_gap == 0:
            _sln_xs = get_solution_xs(device, opt_gene, dim, num_envs, if_train=False)[0]
            _scores = env.get_scores(_sln_xs)

            sln_xs = th.vstack((sln_xs, _sln_xs))
            scores = th.hstack((scores, _scores))

        max_score, max_id = th.max(scores, dim=0)
        if max_score > best_score:
            best_score = max_score
            best_sln_x = sln_xs[max_id]
            print(f"best_score {best_score}  best_sln_x {env.node_bool_to_int(best_sln_x)}")

        if i % print_gap == 0:
            used_time = time.time() - start_time
            print(f"|{used_time:9.0f}  {i:6}  {obj.item():9.3f}  "
                  f"score {scores.max().item():6.0f}  {best_score:6.0f}  "
                  f"entropy {entropies.mean().item():6.3f}  alpha {alpha:5.3f}")


if __name__ == '__main__':
    run_generator()
