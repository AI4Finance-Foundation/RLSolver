import os

import torch
import sys
from torch_geometric.data import Data
from graph_max_cut_evaluator import EncoderBase64
from graph_max_cut_simulator import SimulatorGraphMaxCut, load_graph

"""
pip install torch_geometric
"""

GPU_ID = int(sys.argv[1]) if len(sys.argv) > 1 else 0


def metro_sampling(probs, start_status, max_transfer_time, device=None):
    # Metropolis-Hastings sampling
    if device is None:
        device = torch.device(f'cuda:{GPU_ID}' if torch.cuda.is_available() else 'cpu')

    num_node = len(probs)
    num_chain = start_status.shape[1]
    index_col = torch.tensor(list(range(num_chain)), device=device)

    samples = start_status.bool().to(device)
    probs = probs.detach().to(device)

    count = 0
    for t in range(max_transfer_time * 5):
        if count >= num_chain * max_transfer_time:
            break

        index_row = torch.randint(low=0, high=num_node, size=[num_chain], device=device)
        chosen_probs_base = probs[index_row]
        chosen_value = samples[index_row, index_col]
        chosen_probs = torch.where(chosen_value, chosen_probs_base, 1 - chosen_probs_base)
        accept_rate = (1 - chosen_probs) / chosen_probs

        is_accept = torch.rand(num_chain, device=device).lt(accept_rate)
        samples[index_row, index_col] = torch.where(is_accept, ~chosen_value, chosen_value)

        count += is_accept.sum()
    return samples.float().to(device)


def sampler_func(data, simulator, xs_sample,
                 num_ls, total_mcmc_num, repeat_times,
                 device=torch.device(f'cuda:{GPU_ID}' if torch.cuda.is_available() else 'cpu')):
    k = 1 / 4

    num_nodes = data.num_nodes
    num_edges = data.num_edges
    n0s_tensor = data.edge_index[0]
    n1s_tensor = data.edge_index[1]

    xs_loc_sample = xs_sample.clone()
    xs_loc_sample = xs_loc_sample * 2 - 0.5  # map (0, 1) to (-0.5, 1.5)

    # local search
    for cnt in range(num_ls):
        for node_index in range(0, num_nodes):
            node0_id = data.sorted_degree_nodes[node_index]
            node1_ids = data.neighbors[node0_id]

            node_rand_v = xs_loc_sample[node1_ids].sum(dim=0) + torch.rand(total_mcmc_num * repeat_times, device=device) * k
            xs_loc_sample[node0_id] = node_rand_v.lt((data.weighted_degree[node0_id] + k) / 2).long()

    nlr_probs = 2 * xs_loc_sample[n0s_tensor.type(torch.long)][:] - 1
    nlc_probs = 2 * xs_loc_sample[n1s_tensor.type(torch.long)][:] - 1
    expected_cut = (nlr_probs * nlc_probs).sum(dim=0)

    expected_cut_reshape = expected_cut.reshape((-1, total_mcmc_num))
    index = torch.argmin(expected_cut_reshape, dim=0)
    index = torch.arange(total_mcmc_num, device=device) + index * total_mcmc_num
    max_cut = expected_cut[index]
    temp_max = (num_edges - max_cut) / 2  # todo

    value = (expected_cut - torch.mean(expected_cut)).to(device)
    return temp_max, xs_loc_sample[:, index], value


class Simpler(torch.nn.Module):
    def __init__(self, output_num):
        super().__init__()
        self.lin = torch.nn.Linear(1, output_num)
        self.sigmoid = torch.nn.Sigmoid()

    def reset_parameters(self):
        self.lin.reset_parameters()

    def forward(self, device=torch.device(f'cuda:{GPU_ID}' if torch.cuda.is_available() else 'cpu')):
        x = torch.ones(1).to(device)
        x = self.lin(x)
        x = self.sigmoid(x)

        x = (x - 0.5) * 0.6 + 0.5
        return x

    def __repr__(self):
        return self.__class__.__name__


def maxcut_dataloader(path, device=torch.device(f'cuda:{GPU_ID}' if torch.cuda.is_available() else 'cpu')):
    with open(path) as f:
        fline = f.readline()
        fline = fline.split()
        num_nodes, num_edges = int(fline[0]), int(fline[1])
        edge_index = torch.LongTensor(2, num_edges)
        cnt = 0
        while True:
            lines = f.readlines(num_edges * 2)
            if not lines:
                break
            for line in lines:
                line = line.rstrip('\n').split()
                edge_index[0][cnt] = int(line[0]) - 1
                edge_index[1][cnt] = int(line[1]) - 1
                cnt += 1

        data = Data(num_nodes=num_nodes, edge_index=edge_index.to(device))
        data = append_neighbors(data)

        data.single_degree = []
        data.weighted_degree = []
        tensor_abs_weighted_degree = []
        for i0 in range(data.num_nodes):
            data.single_degree.append(len(data.neighbors[i0]))
            data.weighted_degree.append(
                float(torch.sum(data.neighbor_edges[i0])))
            tensor_abs_weighted_degree.append(
                float(torch.sum(torch.abs(data.neighbor_edges[i0]))))
        tensor_abs_weighted_degree = torch.tensor(tensor_abs_weighted_degree)
        data.sorted_degree_nodes = torch.argsort(
            tensor_abs_weighted_degree, descending=True)

        edge_degree = []
        add = torch.zeros(3, num_edges).to(device)
        for i0 in range(num_edges):
            edge_degree.append(
                tensor_abs_weighted_degree[edge_index[0][i0]] + tensor_abs_weighted_degree[edge_index[1][i0]])
            node_r = edge_index[0][i0]
            node_c = edge_index[1][i0]
            add[0][i0] = 1 - data.weighted_degree[node_r] / 2 - 0.05
            add[1][i0] = 1 - data.weighted_degree[node_c] / 2 - 0.05
            add[2][i0] = 1 + 0.05

        for i0 in range(num_nodes):
            data.neighbor_edges[i0] = data.neighbor_edges[i0].unsqueeze(0)
        data.add = add
        edge_degree = torch.tensor(edge_degree)
        data.sorted_degree_edges = torch.argsort(
            edge_degree, descending=True)
        return data, num_nodes


def append_neighbors(data, device=torch.device(f'cuda:{GPU_ID}' if torch.cuda.is_available() else 'cpu')):
    data.neighbors = []
    data.neighbor_edges = []
    num_nodes = data.num_nodes
    for i in range(num_nodes):
        data.neighbors.append([])
        data.neighbor_edges.append([])
    edge_number = data.edge_index.shape[1]

    edge_weight = 1
    for index in range(0, edge_number):
        row = data.edge_index[0][index]
        col = data.edge_index[1][index]

        data.neighbors[row].append(col.item())
        data.neighbor_edges[row].append(edge_weight)
        data.neighbors[col].append(row.item())
        data.neighbor_edges[col].append(edge_weight)

    data.n0 = []
    data.n1 = []
    data.n0_edges = []
    data.n1_edges = []
    for index in range(0, edge_number):
        row = data.edge_index[0][index]
        col = data.edge_index[1][index]
        data.n0.append(data.neighbors[row].copy())
        data.n1.append(data.neighbors[col].copy())
        data.n0_edges.append(data.neighbor_edges[row].copy())
        data.n1_edges.append(data.neighbor_edges[col].copy())
        i = 0
        for i in range(len(data.n0[index])):
            if data.n0[index][i] == col:
                break
        data.n0[index].pop(i)
        data.n0_edges[index].pop(i)
        for i in range(len(data.n1[index])):
            if data.n1[index][i] == row:
                break
        data.n1[index].pop(i)
        data.n1_edges[index].pop(i)

        data.n0[index] = torch.LongTensor(data.n0[index]).to(device)
        data.n1[index] = torch.LongTensor(data.n1[index]).to(device)
        data.n0_edges[index] = torch.tensor(
            data.n0_edges[index]).unsqueeze(0).to(device)
        data.n1_edges[index] = torch.tensor(
            data.n1_edges[index]).unsqueeze(0).to(device)

    for i in range(num_nodes):
        data.neighbors[i] = torch.LongTensor(data.neighbors[i]).to(device)
        data.neighbor_edges[i] = torch.tensor(
            data.neighbor_edges[i]).to(device)

    return data


def get_return(probs, samples, value):
    log_prob = (samples * probs + (1 - samples) * (1 - probs)).log()
    objective = log_prob.sum(dim=1) * value.t().detach()
    return objective.mean()


def run():
    max_epoch_num = 2 ** 13
    sample_epoch_num = 8
    repeat_times = 128

    num_ls = 8
    reset_epoch_num = 128
    total_mcmc_num = 512
    path = 'data/gset_14.txt'
    # path = 'data/gset_15.txt'
    # path = 'data/gset_49.txt'
    # path = 'data/gset_50.txt'

    # num_ls = 6
    # reset_epoch_num = 192
    # total_mcmc_num = 224
    # path = 'data/gset_22.txt'

    num_ls = 8
    reset_epoch_num = 128
    total_mcmc_num = 256
    path = 'data/gset_55.txt'

    # num_ls = 6
    # reset_epoch_num = 256
    # total_mcmc_num = 192
    # path = 'data/gset_70.txt'

    show_gap = 2 ** 4

    if os.name == 'nt':
        max_epoch_num = 2 ** 4
        repeat_times = 32
        reset_epoch_num = 32
        total_mcmc_num = 64
        show_gap = 2 ** 0

    '''init'''
    data, num_nodes = maxcut_dataloader(path)
    device = torch.device(f'cuda:{GPU_ID}' if torch.cuda.is_available() else 'cpu')

    change_times = int(num_nodes / 10)  # transition times for metropolis sampling

    net = Simpler(num_nodes)
    net.to(device).reset_parameters()
    optimizer = torch.optim.Adam(net.parameters(), lr=2e-2)

    '''addition'''
    from graph_max_cut_simulator import load_graph, SimulatorGraphMaxCut
    from graph_max_cut_random_search import SolverRandomLocalSearch
    graph_name = path.split('/')[-1][:-4]
    graph = load_graph(graph_name=graph_name)
    simulator = SimulatorGraphMaxCut(graph=graph, device=device)
    solver = SolverRandomLocalSearch(simulator=simulator, num_nodes=num_nodes, spin_thresh=8 / num_nodes)

    xs = simulator.generate_xs_randomly(num_sims=total_mcmc_num)
    solver.reset(xs)
    solver.random_search(num_iters=repeat_times)
    now_max_info = solver.good_xs.t()
    now_max_res = solver.good_vs

    '''loop'''
    net.train()
    xs_prob = (torch.zeros(num_nodes) + 0.5).to(device)
    xs_bool = now_max_info.repeat(1, repeat_times)

    for epoch in range(1, max_epoch_num + 1):
        net.to(device).reset_parameters()

        for j1 in range(reset_epoch_num // sample_epoch_num):
            xs_sample = metro_sampling(xs_prob, xs_bool.clone(), change_times)

            temp_max, temp_max_info, value = sampler_func(
                data, simulator, xs_sample, num_ls, total_mcmc_num, repeat_times)
            # update now_max
            for i0 in range(total_mcmc_num):
                if temp_max[i0] > now_max_res[i0]:
                    now_max_res[i0] = temp_max[i0]
                    now_max_info[:, i0] = temp_max_info[:, i0]

            # update if min is too small
            now_max = max(now_max_res).item()
            now_max_index = torch.argmax(now_max_res)
            now_min_index = torch.argmin(now_max_res)
            now_max_res[now_min_index] = now_max
            now_max_info[:, now_min_index] = now_max_info[:, now_max_index]
            temp_max_info[:, now_min_index] = now_max_info[:, now_max_index]

            # select best samples
            xs_bool = temp_max_info.clone()
            xs_bool = xs_bool.repeat(1, repeat_times)
            # construct the start point for next iteration
            start_samples = xs_sample.t()
            print(f"value     {max(now_max_res).item():9.2f}")

            for _ in range(sample_epoch_num):
                xs_prob = net()
                ret_loss_ls = get_return(xs_prob, start_samples, value)

                optimizer.zero_grad()
                ret_loss_ls.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1)
                optimizer.step()

            if j1 % show_gap == 0:
                total_max = now_max_res
                best_sort = torch.argsort(now_max_res, descending=True)
                total_best_info = torch.squeeze(now_max_info[:, best_sort[0]])

                objective_value = max(total_max)
                solution = total_best_info

                encoder = EncoderBase64(num_nodes=num_nodes)
                x_str = encoder.bool_to_str(x_bool=solution)
                print(f"epoch {epoch:6}  objective value  {objective_value.item():8.2f}  solution {x_str}")


if __name__ == '__main__':
    run()