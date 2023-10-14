import os
import sys
import torch as th
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.distributions import Categorical

from simulator2 import load_graph, SimulatorGraphMaxCut
from evaluator import Evaluator, EncoderBase64

TEN = th.Tensor


class SolverLocalSearch:
    def __init__(self, simulator: SimulatorGraphMaxCut, num_nodes: int):
        self.simulator = simulator
        self.num_nodes = num_nodes

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

    def random_search(self, num_iters: int = 8, num_spin: int = 8, noise_std: float = 0.3):
        th.set_grad_enabled(False)
        sim = self.simulator
        kth = self.num_nodes - num_spin

        prev_xs = self.good_xs.clone()
        prev_vs_raw = sim.calculate_obj_values_for_loop(prev_xs, if_sum=False)
        prev_vs = prev_vs_raw.sum(dim=1)

        thresh = None
        for _ in range(num_iters):
            ws = sim.n0_num_n1 - (4 if sim.if_bidirectional else 2) * prev_vs_raw
            ws_std = ws.max(dim=0, keepdim=True)[0] - ws.min(dim=0, keepdim=True)[0]

            spin_rand = ws + th.randn_like(ws, dtype=th.float32) * (ws_std.float() * noise_std)
            thresh = th.kthvalue(spin_rand, k=kth, dim=1)[0][:, None] if thresh is None else thresh
            spin_mask = spin_rand.gt(thresh)

            xs = prev_xs.clone()
            xs[spin_mask] = th.logical_not(xs[spin_mask])

            vs_raw = sim.calculate_obj_values_for_loop(xs, if_sum=False)
            vs = vs_raw.sum(dim=1)

            good_is = vs.gt(self.good_vs)
            prev_xs[good_is] = xs[good_is]
            prev_vs[good_is] = vs[good_is]
            prev_vs_raw[good_is] = vs_raw[good_is]

        self.good_xs = prev_xs
        self.good_vs = prev_vs_raw.sum(dim=1)
        th.set_grad_enabled(True)
        return self.good_xs, self.good_vs


def search_and_evaluate_local_search():
    gpu_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0

    graph_name, num_nodes = 'gset_14', 800
    num_reset = 2 ** 0
    num_iter1 = 2 ** 6
    num_iter0 = 2 ** 2
    num_sims = 2 ** 12

    num_skip = 2 ** 0
    gap_print = 2 ** 0

    if os.name == 'nt':
        num_sims = 2 ** 6
        num_reset = 2 ** 1
        num_iter1 = 2 ** 4
        num_iter0 = 2 ** 2

    device = th.device(f'cuda:{gpu_id}' if th.cuda.is_available() and gpu_id >= 0 else 'cpu')
    graph = load_graph(graph_name=graph_name)

    sim = SimulatorGraphMaxCut(graph=graph, device=device, if_bidirectional=True)
    solver = SolverLocalSearch(simulator=sim, num_nodes=num_nodes)

    temp_xs = sim.generate_xs_randomly(num_sims=1)
    temp_vs = sim.calculate_obj_values(xs=temp_xs)
    evaluator = Evaluator(save_dir=f"{graph_name}_{gpu_id}", num_nodes=num_nodes, x=temp_xs[0], v=temp_vs[0].item())

    print("start searching")
    sim_ids = th.arange(num_sims, device=device)
    for j2 in range(1, num_reset + 1):
        prev_xs = sim.generate_xs_randomly(num_sims)
        prev_vs = sim.calculate_obj_values(prev_xs)

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
            solver.random_search(num_iters=2 ** 6, num_spin=8, noise_std=0.2)

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
        print('\n' * 3)


if __name__ == '__main__':
    search_and_evaluate_local_search()
