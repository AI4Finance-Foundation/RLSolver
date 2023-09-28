import os
import sys
import torch as th
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.distributions import Categorical

from graph_max_cut_simulator import load_graph, SimulatorGraphMaxCut
from graph_max_cut_evaluator import Evaluator, EncoderBase64

TEN = th.Tensor


class SolverRandomLocalSearch:
    def __init__(self, simulator: SimulatorGraphMaxCut, num_nodes: int, spin_thresh: float):
        self.simulator = simulator
        self.num_nodes = num_nodes
        self.spin_thresh = spin_thresh

        self.num_sims = 0
        self.good_xs = th.tensor([])  # solution x
        self.good_vs = th.tensor([])  # objective value

    def reset(self, xs: TEN):
        self.good_xs = xs
        self.good_vs = self.simulator.calculate_obj_values(xs=xs)
        self.num_sims = xs.shape[0]

    def reset_search(self, num_sims):
        xs = th.empty((num_sims, self.num_nodes), dtype=th.bool, device=self.simulator.device)
        for sim_id in range(num_sims):
            _xs = self.simulator.generate_xs_randomly(num_sims=num_sims)
            _vs = self.simulator.calculate_obj_values(_xs)
            xs[sim_id] = _xs[_vs.argmax()]
        return xs

    def random_search(self, num_iters):
        device = self.good_xs.device

        xss = self.good_xs.unsqueeze(0).repeat(num_iters, 1, 1)
        vss = th.empty((num_iters, self.num_sims), dtype=th.long, device=device)

        spin_masks = th.rand_like(xss, dtype=th.float32).lt(self.spin_thresh)
        spin_masks[:, :, 0] = False
        xss[spin_masks] = th.logical_not(xss[spin_masks])

        xss[0] = self.good_xs
        vss[0] = self.good_vs
        for i in range(1, num_iters):
            xs = xss[i]
            vs = self.simulator.calculate_obj_values(xs)
            vss[i] = vs

        good_is = vss.argmax(dim=0)
        sim_id = th.arange(self.num_sims, device=device)
        self.good_xs = xss[good_is, sim_id]
        self.good_vs = vss[good_is, sim_id]

        return self.good_xs, self.good_vs


def check_x():
    num_nodes, graph_id = 128, 1
    graph_name = f'powerlaw_{num_nodes}_ID{graph_id}'
    # x_str = "32tbLf8nvhrD7VqvobBftE"  # 363
    x_str = "0vCQYstA7KhmvWB6DQ6M8H"  # 363

    graph = load_graph(graph_name=graph_name)

    simulator = SimulatorGraphMaxCut(graph=graph)
    encoder_base64 = EncoderBase64(num_nodes=num_nodes)

    x = encoder_base64.str_to_bool(x_str)
    v = simulator.calculate_obj_values(xs=x.unsqueeze(0))[0].item()
    print(f"best_obj_value {v:8}  {x_str}")


class PolicyGNN(nn.Module):
    def __init__(self, inp_dim, mid_dim, out_dim, num_nodes):
        super().__init__()
        self.inp_dim = inp_dim
        self.mid_dim = mid_dim
        self.out_dim = out_dim
        self.num_nodes = num_nodes

        self.mat_dim = mat_dim = int(num_nodes ** 0.25)

        self.mat_enc = nn.Sequential(nn.Linear(num_nodes, mid_dim), nn.ReLU(),
                                     nn.Linear(mid_dim, mat_dim))

        self.inp_enc = nn.Sequential(nn.Linear(inp_dim + mat_dim, mid_dim), nn.ReLU(),
                                     nn.Linear(mid_dim, mid_dim))

        self.tmp_enc = nn.Sequential(nn.Linear(mid_dim + mid_dim, mid_dim), nn.ReLU(),
                                     nn.Linear(mid_dim, out_dim), nn.Softmax(dim=1))

    def forward(self, inp, mat, ids_list):
        size = inp.shape[0]
        device = inp.device
        mat = self.mat_enc(mat)  # (num_nodes, mid_dim)

        tmp0 = th.cat((inp, mat.repeat(size, 1, 1)), dim=2)
        tmp1 = self.inp_enc(tmp0)  # (size, num_nodes, inp_dim)

        env_i = th.arange(size, device=device)
        tmp2 = th.stack([tmp1[env_i[:, None], ids[None, :]].sum(dim=1) for ids in ids_list], dim=1)

        tmp3 = th.cat((tmp1, tmp2), dim=2)
        tmp4 = self.tmp_enc(tmp3)[:, :, 0]  # (size, num_nodes)
        return tmp4


def search_and_evaluate_random_search():
    gpu_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0

    graph_name, num_nodes = 'gset_14', 800
    num_sims = 2 ** 9
    num_reset = 2 ** 6

    num_iter0 = 2 ** 1
    num_iter1 = 2 ** 8
    spin_thresh = 8 / num_nodes

    num_skip = 2 ** 2  # 2 ** 4
    gap_print = 2 ** 1

    # graph_name, num_nodes = 'gset_70', 10000
    # num_sims = 2 ** 8

    if os.name == 'nt':
        num_sims = 2 ** 6
        num_iter0 = 2 ** 2
        num_iter1 = 2 ** 4
        num_reset = 2 ** 1

    device = th.device(f'cuda:{gpu_id}' if th.cuda.is_available() and gpu_id >= 0 else 'cpu')
    graph = load_graph(graph_name=graph_name)

    simulator = SimulatorGraphMaxCut(graph=graph, device=device, if_bidirectional=False)
    solver = SolverRandomLocalSearch(simulator=simulator, num_nodes=num_nodes, spin_thresh=spin_thresh)

    temp_xs = simulator.generate_xs_randomly(num_sims=1)
    temp_vs = simulator.calculate_obj_values(xs=temp_xs)
    evaluator = Evaluator(save_dir=f"{graph_name}_{gpu_id}", num_nodes=num_nodes, x=temp_xs[0], v=temp_vs[0].item())

    print("start searching")
    sim_ids = th.arange(num_sims, device=device)
    for j2 in range(1, num_reset + 1):
        prev_xs = simulator.generate_xs_randomly(num_sims)
        prev_vs = simulator.calculate_obj_values(prev_xs)

        for j1 in range(1, num_iter1 + 1):
            prev_i = prev_vs.argmax()
            prev_xs[:] = prev_xs[prev_i]
            prev_vs[:] = prev_vs[prev_i]

            '''update xs via probability, obtain logprobs for VPG'''
            xs = prev_xs.clone()
            output_tensor = th.ones_like(xs) * (1 / num_nodes)
            for _ in range(num_iter0):
                dist = Categorical(probs=output_tensor)
                sample = dist.sample(th.Size((num_sims,)))[:, 0]
                xs[sim_ids, sample] = th.logical_not(xs[sim_ids, sample])

            '''update xs via max local search'''
            solver.reset(xs)

            good_vs_list = []
            for j0 in range(2 ** 5):
                solver.random_search(2 ** 8)
                good_vs_list.append(solver.good_vs.clone())

            prev_xs = solver.good_xs
            prev_vs = solver.good_vs
            th.set_grad_enabled(True)

            if j1 > num_skip and j1 % gap_print == 0:
                good_i = solver.good_vs.argmax()
                i = j2 * num_iter1 + j1
                x = solver.good_xs[good_i]
                v = solver.good_vs[good_i].item()

                evaluator.record2(i=i, v=v, x=x)
                evaluator.logging_print(v=v)
        evaluator.plot_record()


def search_and_evaluate_reinforce():
    gpu_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0

    graph_name, num_nodes = 'gset_14', 800
    num_sims = 2 ** 9
    num_reset = 2 ** 6

    num_iter0 = 2 ** 1
    num_iter1 = 2 ** 8
    spin_thresh = 8 / num_nodes

    inp_dim = 2
    mid_dim = 32
    out_dim = 1
    learning_rate = 1e-3
    clip_grad_norm = 3.0
    if_bidirectional = False  # GPU 4567
    # if_bidirectional = True  # GPU 0123
    temperature = (gpu_id + 1) * 4

    num_skip = 2 ** 2  # 2 ** 4
    gap_plot = 2 ** 1

    if os.name == 'nt':
        num_sims = 2 ** 6
        num_iter = 2 ** 2
        num_update = 2 ** 4
        num_reset = 2 ** 1
        record_gap = 2 ** 0

    device = th.device(f'cuda:{gpu_id}' if th.cuda.is_available() and gpu_id >= 0 else 'cpu')
    graph = load_graph(graph_name=graph_name)

    simulator = SimulatorGraphMaxCut(graph=graph, device=device, if_bidirectional=if_bidirectional)
    matrix_tensor = simulator.adjacency_matrix
    indies_tensor = simulator.adjacency_indies

    solver = SolverRandomLocalSearch(simulator=simulator, num_nodes=num_nodes, spin_thresh=spin_thresh)
    opt_opti = PolicyGNN(inp_dim=inp_dim, mid_dim=mid_dim, out_dim=out_dim, num_nodes=num_nodes).to(device)
    opt_base = th.optim.Adam(opt_opti.parameters(), lr=learning_rate, maximize=True)  # todo

    temp_xs = simulator.generate_xs_randomly(num_sims=1)
    temp_vs = simulator.calculate_obj_values(xs=temp_xs)
    evaluator = Evaluator(save_dir=f"{graph_name}_{gpu_id}", num_nodes=num_nodes, x=temp_xs[0], v=temp_vs[0].item())

    sim_ids = th.arange(num_sims, device=device)
    output_tensor = None

    print("start searching")
    for j2 in range(1, num_reset + 1):
        prev_xs = simulator.generate_xs_randomly(num_sims)
        prev_vs = simulator.calculate_obj_values(prev_xs)

        for j1 in range(1, num_iter1 + 1):
            prev_i = prev_vs.argmax()
            prev_xs[:] = prev_xs[prev_i]
            prev_vs[:] = prev_vs[prev_i]

            '''update xs via probability, obtain logprobs for VPG'''
            xs = prev_xs.clone()
            logprobs = []
            for _ in range(num_iter0):
                input_tensor = build_input_tensor(xs, simulator, num_sims, num_nodes, device)
                output_tensor = opt_opti(input_tensor, matrix_tensor, indies_tensor) + temperature / num_nodes
                dist = Categorical(probs=output_tensor / output_tensor.sum(dim=1, keepdim=True))
                sample = dist.sample(th.Size((num_sims,)))[:, 0]
                xs[sim_ids, sample] = th.logical_not(xs[sim_ids, sample])

                logprob = dist.log_prob(sample)

                logprobs.append(logprob)
            logprobs = th.stack(logprobs, dim=0).mean(dim=0)

            '''update xs via max local search, obtain good_vs for VPG'''
            th.set_grad_enabled(False)
            for _ in range(num_iter0):
                input_tensor = build_input_tensor(xs, simulator, num_sims, num_nodes, device)
                output_tensor = opt_opti(input_tensor, matrix_tensor, indies_tensor)
                sample = output_tensor.argmax(dim=1)
                xs[sim_ids, sample] = th.logical_not(xs[sim_ids, sample])

            solver.reset(xs)

            good_vs_list = []
            for j0 in range(2 ** 5):
                solver.random_search(2 ** 8)
                good_vs_list.append(solver.good_vs.clone())
            scaled_vs0 = th.stack(good_vs_list, dim=0).float().mean(dim=0)
            # scaled_vs0 = solver.good_vs.float() # todo

            scaled_vs = (scaled_vs0 - scaled_vs0.mean()) / (scaled_vs0.std() + 1e-4)
            prev_xs = solver.good_xs
            prev_vs = solver.good_vs
            th.set_grad_enabled(True)

            '''update objective of optimizer'''
            obj_logprobs = ((logprobs - logprobs.mean().detach()) / (logprobs.std().detach() + 1e-6)).exp()
            obj_opti = (obj_logprobs * scaled_vs).mean()

            opt_base.zero_grad()
            obj_opti.backward()
            clip_grad_norm_(parameters=opt_base.param_groups[0]["params"], max_norm=clip_grad_norm)
            opt_base.step()

            if j1 > num_skip and j1 % gap_plot == 0:
                i = j2 * num_iter1 + j1
                v = solver.good_vs.max().item()
                evaluator.record1(i=i, v=v)
                out = output_tensor.detach().cpu().numpy()
                print(f"| i {j2:6} {j1:6}  "
                      f"output  {out.min():8.1e} {out.mean():8.1e}  {out.std():8.1e}  {out.max():8.1e}  |  "
                      f"avg_value {prev_vs.float().mean().item():8.2f}  "
                      f"max_value {prev_vs.max().item():8.2f}")

        i = j2 * num_iter1 + num_iter1
        good_i = solver.good_vs.argmax()
        x = solver.good_xs[good_i]
        v = solver.good_vs[good_i].item()
        evaluator.record2(i=i, v=v, x=x)
        evaluator.plot_record()
        evaluator.logging_print(v=v)


def build_input_tensor(xs, simulator, num_sims, num_nodes, device):
    input_tensor = th.empty((num_sims, num_nodes, 2), dtype=th.float32, device=device)
    input_tensor[:, :, 0] = xs
    _vs = simulator.calculate_obj_values(xs, if_sum=False)
    for i in range(num_sims):
        input_tensor[i, :, 1] = th.bincount(simulator.n0_ids[i], weights=_vs[i], minlength=num_nodes)
    return input_tensor


if __name__ == '__main__':
    search_and_evaluate_random_search()
    search_and_evaluate_reinforce()
