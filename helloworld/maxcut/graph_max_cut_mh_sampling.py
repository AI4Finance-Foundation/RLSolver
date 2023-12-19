import os
import sys
import time

import torch as th
import torch.nn as nn
import tqdm
from torch.nn.utils import clip_grad_norm_

from graph_max_cut_simulator import SimulatorGraphMaxCut
from graph_max_cut_local_search import SolverLocalSearch, show_gpu_memory, update_xs_by_vs, EncoderBase64
from evaluator import Evaluator


class BnMLP(nn.Module):
    def __init__(self, dims, activation=None):
        super(BnMLP, self).__init__()

        assert len(dims) >= 3
        mlp_list = [nn.Linear(dims[0], dims[1]), ]
        for i in range(1, len(dims) - 1):
            dim_i = dims[i]
            dim_j = dims[i + 1]
            mlp_list.extend([nn.GELU(), nn.LayerNorm(dim_i), nn.Linear(dim_i, dim_j)])

        if activation is not None:
            mlp_list.append(activation)

        self.mlp = nn.Sequential(*mlp_list)

        if activation is not None:
            layer_init_with_orthogonal(self.mlp[-2], std=0.1)
        else:
            layer_init_with_orthogonal(self.mlp[-1], std=0.1)

    def forward(self, inp):
        return self.mlp(inp)


def layer_init_with_orthogonal(layer, std=1.0, bias_const=1e-6):
    th.nn.init.orthogonal_(layer.weight, std)
    th.nn.init.constant_(layer.bias, bias_const)


class PolicyMLP(nn.Module):
    def __init__(self, inp_dim, mid_dim, out_dim):
        super().__init__()
        self.net1 = BnMLP(dims=(inp_dim, mid_dim, mid_dim, out_dim), activation=nn.Sigmoid())

    def forward(self, xs):
        return self.net1(xs)

    def auto_regression(self, xs, ids_ary):
        num_sims, num_iter = ids_ary.shape
        sim_ids = th.arange(num_sims, device=xs.device)
        xs = xs.detach().clone()

        for i in range(num_iter):
            ids = ids_ary[:, i]

            # ps = self.forward(xs.detach().clone())[sim_ids, ids]
            ps = self.forward(xs.clone())[sim_ids, ids]  # todo
            xs[sim_ids, ids] = ps
        return xs


class PolicyGNN(nn.Module):
    def __init__(self, inp_dim, mid_dim, out_dim, adj_matrix):
        super().__init__()
        self.adj_inp = BnMLP(dims=(inp_dim, mid_dim, out_dim), activation=nn.GELU())
        self.adj_dec = BnMLP(dims=(out_dim, out_dim, out_dim), activation=nn.GELU())
        self.adj_wgt = BnMLP(dims=(out_dim, mid_dim, inp_dim), activation=nn.Softmax(dim=1))

        self.adj_query = BnMLP(dims=(out_dim * 2, out_dim, out_dim), activation=nn.Tanh())

        self.net_out = BnMLP(dims=(inp_dim + out_dim, mid_dim, 1), activation=nn.Sigmoid())

        self.num_nodes = inp_dim
        self.adj_matrix = adj_matrix  # shape==(num_nodes, num_nodes)

    def auto_regression(self, xs, ids_ary):
        num_sims, num_iter = ids_ary.shape
        assert num_sims == 1

        adj_inp = self.adj_inp(self.adj_matrix)
        adj_wgt = self.adj_wgt(adj_inp)

        adj_tmp = th.concat(tensors=(adj_inp[:, None, :].repeat(1, self.num_nodes, 1),
                                     adj_inp[None, :, :].repeat(self.num_nodes, 1, 1)), dim=2)
        adj_out = (self.adj_query(adj_tmp) * adj_wgt[:, :, None]).sum(dim=1)

        xs = xs.detach().clone()
        i = 0
        for idx in ids_ary[i]:
            inp = xs.detach().clone()
            adj = adj_out[idx, None, :]
            xs[i, idx] = self.net_out(th.concat((inp, adj), dim=1))[i]
        return xs


def metropolis_hastings_sampling(prob, start_xs, num_iters):  # mcmc sampling with transition kernel and accept ratio
    xs = start_xs.clone()
    num_sims, num_nodes = xs.shape
    device = xs.device

    sim_ids = th.arange(num_sims, device=device)

    count = 0
    for _ in range(num_iters * 8):
        if count >= num_sims * num_iters:
            break

        ids = th.randint(low=0, high=num_nodes, size=(num_sims,), device=device)
        chosen_p = prob[ids]
        chosen_xs = xs[sim_ids, ids]
        chosen_ps = th.where(chosen_xs, chosen_p, 1 - chosen_p)

        accept_rates = (1 - chosen_ps) / chosen_ps
        accept_masks = th.rand(num_sims, device=device).lt(accept_rates)
        xs[sim_ids, ids] = th.where(accept_masks, th.logical_not(chosen_xs), chosen_xs)

        count += accept_masks.sum()
    return xs


def run_in_single_graph():
    gpu_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    device = th.device(f'cuda:{gpu_id}' if th.cuda.is_available() and gpu_id >= 0 else 'cpu')

    sim_name = 'gset_14'

    weight_decay = 0  # 4e-5
    learning_rate = 1e-3

    reset_gap = 32
    show_gap = 4
    mid_dim = 2 ** 8
    num_sims = 2 ** 6
    num_repeat = 2 ** 7
    if os.name == 'nt':  # windowsOS (new type)
        mid_dim = 2 ** 6
        num_sims = 2 ** 2
        num_repeat = 2 ** 3
    save_path = f'net_{sim_name}_{gpu_id}.pth'

    '''simulator'''
    sim = SimulatorGraphMaxCut(sim_name=sim_name, device=device)
    num_nodes = sim.num_nodes

    '''addition'''
    solver = SolverLocalSearch(simulator=sim, num_nodes=num_nodes)

    xs = sim.generate_xs_randomly(num_sims=num_sims)
    solver.reset(xs)
    for _ in tqdm.trange(8, ascii=True):
        solver.random_search(num_iters=8)
    temp_xs = solver.good_xs
    temp_vs = solver.good_vs
    evaluator = Evaluator(save_dir=f"{sim_name}_{gpu_id}", num_nodes=num_nodes, x=temp_xs[0], v=temp_vs[0].item())

    '''model'''
    net = PolicyMLP(inp_dim=num_nodes, mid_dim=mid_dim, out_dim=num_nodes).to(device)
    # net = PolicyGNN(inp_dim=num_nodes, mid_dim=mid_dim, out_dim=mid_dim, adj_matrix=sim.adjacency_matrix).to(device)
    net_params = list(net.parameters())
    optimizer = th.optim.Adam(net_params, lr=learning_rate, maximize=True) if weight_decay \
        else th.optim.AdamW(net_params, lr=learning_rate, maximize=True, weight_decay=weight_decay)

    '''loop'''
    soft_max = nn.Softmax(dim=0)
    sim_ids = th.arange(num_sims, device=device)
    for i in range(256):
        ids_ary = th.randperm(num_nodes, device=device)[None, :]
        probs0 = th.rand(size=(1, num_nodes), device=device) * 0.02 + 0.49
        probs1 = net.auto_regression(probs0, ids_ary=ids_ary).clip(1e-9, 1 - 1e-9)

        start_xs = temp_xs.repeat(num_repeat, 1)
        xs = metropolis_hastings_sampling(prob=probs1[0], start_xs=start_xs, num_iters=int(num_nodes * 0.2))
        vs = solver.reset(xs)
        for _ in range(4):
            xs, vs, num_update = solver.random_search(num_iters=8)

        advantages = vs.detach().float()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        logprobs = th.log(th.where(xs, probs1, 1 - probs1)).sum(dim=1)
        obj_values = (soft_max(logprobs) * advantages).sum()

        objective = obj_values  # + obj_entropy
        optimizer.zero_grad()
        objective.backward()
        clip_grad_norm_(net_params, 3)
        optimizer.step()

        '''update temp_xs'''
        xs = xs.reshape(num_repeat, num_sims, num_nodes)
        vs = vs.reshape(num_repeat, num_sims)
        temp_i = vs.argmax(dim=0)
        temp_xs = xs[temp_i, sim_ids]
        temp_vs = vs[temp_i, sim_ids]

        '''update good_x'''
        good_i = temp_vs.argmax()
        good_x = temp_xs[good_i]
        good_v = temp_vs[good_i]
        if_show_x = evaluator.record2(i=i, v=good_v, x=good_x)
        # if_show_x = if_show_x and (good_v >= 3050)

        if (i + 1) % show_gap == 0 or if_show_x:
            entropy = -th.mean(probs1 * th.log2(probs1), dim=1)
            obj_entropy = entropy.mean()

            show_str = (f"| obj {obj_values:9.3f}  entropy {obj_entropy:9.3f} "
                        f"| cut_value {temp_vs.float().mean().long():6} < {temp_vs.max():6}")
            evaluator.logging_print(x=good_x, v=good_v, show_str=show_str, if_show_x=if_show_x)

        if (i + 1) % reset_gap == 0:
            print(f"| reset {show_gpu_memory(device=device)}")
            temp_xs[0, :] = evaluator.best_x
            temp_xs[1:] = sim.generate_xs_randomly(num_sims=num_sims - 1)

            th.save(net.state_dict(), save_path)


def run_in_graph_distribution():
    gpu_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    device = th.device(f'cuda:{gpu_id}' if th.cuda.is_available() and gpu_id >= 0 else 'cpu')

    num_nodes = 500
    sim_name = f'powerlaw_{num_nodes}'

    weight_decay = 0  # 4e-5
    learning_rate = 1e-3

    reset_gap = 32
    show_gap = 4
    mid_dim = 2 ** 8
    num_sims = 2 ** 6
    num_repeat = 2 ** 7
    if os.name == 'nt':  # windowsOS (new type)
        mid_dim = 2 ** 6
        num_sims = 2 ** 2
        num_repeat = 2 ** 3
    save_path = f'net_{sim_name}_{gpu_id}.pth'

    '''simulator'''
    sim = SimulatorGraphMaxCut(sim_name=sim_name, device=device)
    num_nodes = sim.num_nodes

    '''addition'''
    solver = SolverLocalSearch(simulator=sim, num_nodes=num_nodes)

    xs = sim.generate_xs_randomly(num_sims=num_sims)
    solver.reset(xs)
    for _ in tqdm.trange(8, ascii=True):
        solver.random_search(num_iters=8)
    temp_xs = solver.good_xs
    temp_vs = solver.good_vs
    evaluator = Evaluator(save_dir=f"{sim_name}_{gpu_id}", num_nodes=num_nodes, x=temp_xs[0], v=temp_vs[0].item())

    '''model'''
    net = PolicyGNN(inp_dim=num_nodes, mid_dim=mid_dim, out_dim=mid_dim, adj_matrix=sim.adjacency_matrix).to(device)
    net_params = list(net.parameters())
    optimizer = th.optim.Adam(net_params, lr=learning_rate, maximize=True) if weight_decay \
        else th.optim.AdamW(net_params, lr=learning_rate, maximize=True, weight_decay=weight_decay)

    '''loop'''
    soft_max = nn.Softmax(dim=0)
    sim_ids = th.arange(num_sims, device=device)
    for i in range(256):
        ids_ary = th.randperm(num_nodes, device=device)[None, :]
        probs0 = th.rand(size=(1, num_nodes), device=device) * 0.02 + 0.49
        probs1 = net.auto_regression(probs0, ids_ary=ids_ary).clip(1e-9, 1 - 1e-9)

        start_xs = temp_xs.repeat(num_repeat, 1)
        xs = metropolis_hastings_sampling(prob=probs1[0], start_xs=start_xs, num_iters=int(num_nodes * 0.2))
        vs = solver.reset(xs)
        for _ in range(4):
            xs, vs, num_update = solver.random_search(num_iters=8)

        advantages = vs.detach().float()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        logprobs = th.log(th.where(xs, probs1, 1 - probs1)).sum(dim=1)
        obj_values = (soft_max(logprobs) * advantages).sum()

        objective = obj_values  # + obj_entropy
        optimizer.zero_grad()
        objective.backward()
        clip_grad_norm_(net_params, 3)
        optimizer.step()

        '''update temp_xs'''
        xs = xs.reshape(num_repeat, num_sims, num_nodes)
        vs = vs.reshape(num_repeat, num_sims)
        temp_i = vs.argmax(dim=0)
        temp_xs = xs[temp_i, sim_ids]
        temp_vs = vs[temp_i, sim_ids]

        '''update good_x'''
        good_i = temp_vs.argmax()
        good_x = temp_xs[good_i]
        good_v = temp_vs[good_i]
        if_show_x = evaluator.record2(i=i, v=good_v, x=good_x)
        # if_show_x = if_show_x and (good_v >= 3050)

        if (i + 1) % show_gap == 0 or if_show_x:
            entropy = -th.mean(probs1 * th.log2(probs1), dim=1)
            obj_entropy = entropy.mean()

            show_str = (f"| obj {obj_values:9.3f}  entropy {obj_entropy:9.3f} "
                        f"| cut_value {temp_vs.float().mean().long():6} < {temp_vs.max():6}")
            evaluator.logging_print(x=good_x, v=good_v, show_str=show_str, if_show_x=if_show_x)

        if (i + 1) % reset_gap == 0:
            print(f"| reset {show_gpu_memory(device=device)}")
            sim = SimulatorGraphMaxCut(sim_name=sim_name, device=device)
            temp_xs[:] = sim.generate_xs_randomly(num_sims=num_sims - 1)
            net.adj_matrix = sim.adjacency_matrix

            th.save(net.state_dict(), save_path)


if __name__ == '__main__':
    run_in_single_graph()
    # run_in_graph_distribution()
