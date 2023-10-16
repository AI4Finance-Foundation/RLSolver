import torch
import time
import os
import numpy as np
import scipy.io as scio
from torch_geometric.data import Data
from torch.distributions.bernoulli import Bernoulli

"""
pip install torch_geometric
"""

'''mcpg_solver'''


# from model import simple
# from sampling import sampler_select, sample_initializer
# from mcpg_solver import mcpg_solver


def sample_initializer(problem_type, probs, config,
                       device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), data=None):
    if problem_type in ["r_cheegercut", "n_cheegercut"]:
        samples = torch.zeros(config['total_mcmc_num'], data.num_nodes)
        index = data.sorted_degree_nodes[- config['total_mcmc_num']:]
        for i in range(config['total_mcmc_num']):
            samples[i][index[i]] = 1
        samples = samples.repeat(config['repeat_times'], 1)
        return samples.t()
    m = Bernoulli(probs)
    samples = m.sample([config['total_mcmc_num'] * config['repeat_times']])
    samples = samples.detach().to(device)
    return samples.t()


def sampler_select(problem_type):
    if problem_type == "maxcut":
        return mcpg_sampling_maxcut
    elif problem_type == "maxcut_edge":
        return mcpg_sampling_maxcut_edge
    elif problem_type == "mimo":
        return mcpg_sampling_mimo
    elif problem_type == "qubo":
        return mcpg_sampling_qubo
    elif problem_type == "r_cheegercut":
        return mcpg_sampling_rcheegercut
    elif problem_type == "n_cheegercut":
        return mcpg_sampling_ncheegercut
    else:
        raise (Exception("Unrecognized problem type {}".format(problem_type)))


def metro_sampling(probs, start_status, max_transfer_time,
                   device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    num_node = len(probs)
    num_chain = start_status.shape[1]
    index_col = torch.tensor(list(range(num_chain))).to(device)

    probs = probs.detach().to(device)
    samples = start_status.bool().to(device)

    count = 0
    for t in range(max_transfer_time * 5):
        if count >= num_chain * max_transfer_time:
            break
        index_row = torch.randint(low=0, high=num_node, size=[
            num_chain], device=device)
        chosen_probs_base = probs[index_row]
        chosen_value = samples[index_row, index_col]
        chosen_probs = torch.where(
            chosen_value, chosen_probs_base, 1 - chosen_probs_base)
        accept_rate = (1 - chosen_probs) / chosen_probs
        r = torch.rand(num_chain, device=device)
        is_accept = (r < accept_rate)
        samples[index_row, index_col] = torch.where(
            is_accept, ~chosen_value, chosen_value)

        count += is_accept.sum()

    return samples.float().to(device)


def mcpg_sampling_maxcut(data,
                         start_result, probs,
                         num_ls, change_times, total_mcmc_num,
                         device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    probs = probs.to(torch.device("cpu"))
    num_nodes = data.num_nodes
    edges = data.edge_index
    nlr_graph = edges[0]
    nlc_graph = edges[1]
    edge_weight = data.edge_attr
    edge_weight_sum = data.edge_weight_sum
    graph_probs = start_result.clone()
    # get probs
    graph_probs = metro_sampling(
        probs, graph_probs, change_times)
    start = graph_probs.clone()

    temp = graph_probs[data.sorted_degree_nodes[0]].clone()
    graph_probs += temp
    graph_probs = graph_probs % 2

    graph_probs = (graph_probs - 0.5) * 2 + 0.5

    # local search
    expected_cut = torch.zeros(graph_probs.size(dim=1))
    cnt = 0
    while True:
        cnt += 1
        for node_index in range(0, num_nodes):
            node = data.sorted_degree_nodes[node_index]
            neighbor_index = data.neighbors[node]
            neighbor_edge_weight = data.neighbor_edges[node]
            node_temp_v = torch.mm(
                neighbor_edge_weight, graph_probs[neighbor_index])
            node_temp_v = torch.squeeze(node_temp_v)
            node_temp_v += torch.rand(node_temp_v.shape[0],
                                      device=torch.device(device)) / 4
            graph_probs[node] = (node_temp_v <
                                 data.weighted_degree[node] / 2 + 0.125).int()
        if cnt >= num_ls:
            break

    # maxcut

    expected_cut[:] = ((2 * graph_probs[nlr_graph.type(torch.long)][:] - 1) * (
            2 * graph_probs[nlc_graph.type(torch.long)][:] - 1) * edge_weight).sum(dim=0)

    expected_cut_reshape = torch.reshape(expected_cut, (-1, total_mcmc_num))
    index = torch.argmin(expected_cut_reshape, dim=0)
    for i0 in range(total_mcmc_num):
        index[i0] = i0 + index[i0] * total_mcmc_num
    max_cut = expected_cut[index]
    return ((edge_weight_sum - max_cut) / 2), graph_probs[:, index], start, (
            expected_cut - torch.mean(expected_cut)).to(device)


def mcpg_sampling_maxcut_edge(data,
                              start_result, probs,
                              num_ls, change_times, total_mcmc_num,
                              device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    probs = probs.to(torch.device("cpu"))
    num_edges = data.num_edges
    edges = data.edge_index
    nlr_graph = edges[0]
    nlc_graph = edges[1]
    edge_weight = data.edge_attr
    edge_weight_sum = data.edge_weight_sum
    graph_probs = start_result.clone()
    # get probs
    graph_probs = metro_sampling(
        probs, graph_probs, change_times, device)
    start = graph_probs.clone()

    temp = graph_probs[data.sorted_degree_nodes[0]].clone()
    graph_probs += temp
    graph_probs = graph_probs % 2

    graph_probs = (graph_probs - 0.5) * 2 + 0.5

    # local search
    temp = torch.zeros(4, graph_probs.size(dim=1)).to(device)
    expected_cut = torch.zeros(graph_probs.size(dim=1))
    cnt = 0
    while True:
        cnt += 1
        for i in range(num_edges):
            index = data.sorted_degree_edges[i]
            node_r = nlr_graph[index]
            node_c = nlc_graph[index]
            edges_r = data.n0_edges[index]
            edges_c = data.n1_edges[index]
            add_0 = data.add[0][index]
            add_1 = data.add[1][index]
            add_2 = data.add[2][index]

            temp_r_v = torch.mm(edges_r, graph_probs[data.n0[index]])
            temp_c_v = torch.mm(edges_c, graph_probs[data.n1[index]])

            temp[1] = temp_r_v + torch.rand(graph_probs.size(dim=1), device=torch.device('cuda:0')) * 0.1 + add_0
            temp[2] = temp_c_v + torch.rand(graph_probs.size(dim=1), device=torch.device('cuda:0')) * 0.1 + add_1
            temp[0] = temp[1] + temp[2] + torch.rand(graph_probs.size(dim=1),
                                                     device=torch.device('cuda:0')) * 0.1 - add_2

            max_index = torch.argmax(temp, dim=0)
            graph_probs[node_r] = torch.floor(max_index / 2)
            graph_probs[node_c] = max_index % 2

        if cnt >= num_ls:
            break

    # maxcut
    expected_cut[:] = ((2 * graph_probs[nlr_graph.type(torch.long)][:] - 1) * (
            2 * graph_probs[nlc_graph.type(torch.long)][:] - 1) * edge_weight).sum(dim=0)

    expected_cut_reshape = torch.reshape(expected_cut, (-1, total_mcmc_num))
    index = torch.argmin(expected_cut_reshape, dim=0)
    for i0 in range(total_mcmc_num):
        index[i0] = i0 + index[i0] * total_mcmc_num
    max_cut = expected_cut[index]

    return ((edge_weight_sum - max_cut) / 2), graph_probs[:, index], start, (
            expected_cut - torch.mean(expected_cut)).to(device)


def mcpg_sampling_rcheegercut(data,
                              start_result, probs,
                              num_ls, change_times, total_mcmc_num,
                              device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    probs = probs.to(torch.device("cpu"))
    nvar = data.num_nodes
    edges = data.edge_index
    nlr_graph, nlc_graph = edges
    graph_probs = start_result.clone()
    # get probs
    raw_samples = metro_sampling(
        probs, graph_probs, change_times)
    samples = raw_samples.clone()

    res_cut = ((2 * samples[nlr_graph.type(torch.long)][:] - 1) * (
            2 * samples[nlc_graph.type(torch.long)][:] - 1)).sum(dim=0)
    res_cut[:] = (data.edge_weight_sum - res_cut) / 2
    res_node = samples.sum(dim=0)
    cheeger_cut = res_cut / torch.min(res_node, nvar - res_node)

    for cnt in range(num_ls):
        for node_index in range(nvar):
            node = data.sorted_degree_nodes[node_index]
            neighbor_index = data.neighbors[node]

            change_cut_size = torch.sum(samples[neighbor_index], dim=0)
            new_res_cut = res_cut - \
                          (2 * samples[node] - 1) * \
                          (data.weighted_degree[node] - 2 * change_cut_size)
            new_res_node = res_node - (2 * samples[node] - 1)
            new_cheeger_cut = new_res_cut / \
                              torch.min(new_res_node, nvar - new_res_node)
            new_min_node = torch.min(new_res_node, nvar - new_res_node)
            cond = torch.logical_or(
                (cheeger_cut < new_cheeger_cut), (new_min_node < 0.0000001))
            samples[node] = torch.where(cond, samples[node], 1 - samples[node])
            res_cut = torch.where(cond, res_cut, new_res_cut)
            res_node = torch.where(cond, res_node, new_res_node)
            cheeger_cut = torch.where(cond, cheeger_cut, new_cheeger_cut)
    # maxcut
    cheeger_cut_reshape = torch.reshape(cheeger_cut, (-1, total_mcmc_num))
    index = torch.argmin(cheeger_cut_reshape, dim=0)
    for i0 in range(total_mcmc_num):
        index[i0] = i0 + index[i0] * total_mcmc_num
    min_cheeger_cut = cheeger_cut[index]

    return -min_cheeger_cut, samples[:, index], raw_samples, (cheeger_cut - torch.mean(cheeger_cut)).to(device)


def mcpg_sampling_ncheegercut(data,
                              start_result, probs,
                              num_ls, change_times, total_mcmc_num,
                              device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    probs = probs.to(torch.device("cpu"))
    nvar = data.num_nodes
    edges = data.edge_index
    nlr_graph, nlc_graph = edges
    graph_probs = start_result.clone()
    # get probs
    raw_samples = metro_sampling(
        probs, graph_probs, change_times)
    samples = raw_samples.clone()

    res_cut = ((2 * samples[nlr_graph.type(torch.long)][:] - 1) * (
            2 * samples[nlc_graph.type(torch.long)][:] - 1)).sum(dim=0)
    res_cut[:] = (data.edge_weight_sum - res_cut) / 2
    res_node = samples.sum(dim=0)
    cheeger_cut = res_cut * (1 / res_node + 1 / (nvar - res_node))

    for cnt in range(num_ls):
        for node_index in range(nvar):
            node = data.sorted_degree_nodes[node_index]
            neighbor_index = data.neighbors[node]

            change_cut_size = torch.sum(samples[neighbor_index], dim=0)
            new_res_cut = res_cut - \
                          (2 * samples[node] - 1) * \
                          (data.weighted_degree[node] - 2 * change_cut_size)
            new_res_node = res_node - (2 * samples[node] - 1)
            new_cheeger_cut = new_res_cut * \
                              (1 / new_res_node + 1 / (nvar - new_res_node))
            new_min_node = torch.min(new_res_node, nvar - new_res_node)
            cond = torch.logical_or(
                (cheeger_cut < new_cheeger_cut), (new_min_node < 0.0000001))
            samples[node] = torch.where(cond, samples[node], 1 - samples[node])
            res_cut = torch.where(cond, res_cut, new_res_cut)
            res_node = torch.where(cond, res_node, new_res_node)
            cheeger_cut = torch.where(cond, cheeger_cut, new_cheeger_cut)
    # maxcut
    cheeger_cut_reshape = torch.reshape(cheeger_cut, (-1, total_mcmc_num))
    index = torch.argmin(cheeger_cut_reshape, dim=0)
    for i0 in range(total_mcmc_num):
        index[i0] = i0 + index[i0] * total_mcmc_num
    min_cheeger_cut = cheeger_cut[index]

    return -min_cheeger_cut, samples[:, index], raw_samples, (cheeger_cut - torch.mean(cheeger_cut)).to(device)


def mcpg_sampling_mimo(data,
                       start_result, probs,
                       num_ls, change_times, total_mcmc_num,
                       device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    Sigma = data[0]
    Diag = data[1]
    num_n = data[0].shape[0]
    info = start_result.clone()
    # get probs
    info = metro_sampling(
        probs, info, change_times, device)
    start = info.clone()

    info = (info - 0.5) * 4  # convert to 2, -2

    # local search
    cnt = 0
    while True:
        for node in range(0, num_n):
            if cnt >= num_ls * num_n:
                break
            cnt += 1
            neighbor_weight = Sigma[node].unsqueeze(0)
            node_temp_v = torch.matmul(
                neighbor_weight, info)
            node_temp_v = torch.squeeze(node_temp_v)
            temp = (node_temp_v < - Diag[node] / 2).int()
            info[node] = temp * 2 - 1
        if cnt >= num_ls * num_n:
            break

    # compute value
    expected = (info * torch.matmul(Sigma, info)).sum(dim=0)
    expected += torch.matmul(Diag.unsqueeze(0), info).sum(dim=0)
    expected_cut_reshape = torch.reshape(expected, (-1, total_mcmc_num))
    index = torch.argmin(expected_cut_reshape, dim=0)
    for i0 in range(total_mcmc_num):
        index[i0] = i0 + index[i0] * total_mcmc_num
    min_res = expected[index]
    info = (info + 1) / 2
    return -(min_res + data[3]), info[:, index], start, (expected - torch.mean(expected)).to(device)


def mcpg_sampling_qubo(data, start_result, probs, num_ls, change_times, total_mcmc_num,
                       device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    Q = data['Q']
    nvar = data['nvar']
    raw_samples = start_result.clone()
    # get probs
    raw_samples = metro_sampling(
        probs, raw_samples, change_times, device)
    samples = raw_samples.clone()
    samples = samples * 2 - 1
    # local search
    for cnt in range(num_ls):
        for index in range(nvar):
            samples[index] = 0
            res = torch.matmul(Q[index], samples)
            ind = (res > 0)
            samples[index] = 2 * ind - 1
    # compute value
    res_sample = torch.matmul(Q, samples)
    res_sample = torch.sum(torch.mul(samples, res_sample), dim=0)
    res_sample_reshape = torch.reshape(res_sample, (-1, total_mcmc_num))
    index = torch.argmax(res_sample_reshape, dim=0)
    index = torch.tensor(list(range(total_mcmc_num)),
                         device=device) + index * total_mcmc_num
    max_res = res_sample[index]
    samples = (samples + 1) / 2
    return max_res, samples[:, index], raw_samples, -(res_sample - torch.mean(res_sample.float())).to(device)


class simple(torch.nn.Module):
    def __init__(self, output_num):
        super(simple, self).__init__()
        self.lin = torch.nn.Linear(1, output_num)
        self.sigmoid = torch.nn.Sigmoid()

    def reset_parameters(self):
        self.lin.reset_parameters()

    def forward(self, alpha=0.1, start_samples=None, value=None,
                device=torch.device("cuda")):
        x = torch.ones(1).to(device)
        x = self.lin(x)
        x = self.sigmoid(x)

        x = (x - 0.5) * 0.6 + 0.5
        probs = x
        probs = probs.squeeze()
        retdict = {}
        reg = probs * torch.log(probs) + (1 - probs) * torch.log(1 - probs)
        reg = torch.mean(reg)
        if start_samples == None:
            retdict["output"] = [probs.squeeze(-1), "hist"]  # output
            retdict["reg"] = [reg, "sequence"]
            retdict["loss"] = [alpha * reg, "sequence"]
            return retdict

        res_samples = value.t().detach()

        start_samples_idx = start_samples * \
                            probs + (1 - start_samples) * (1 - probs)
        log_start_samples_idx = torch.log(start_samples_idx)
        log_start_samples = log_start_samples_idx.sum(dim=1)
        loss_ls = torch.mean(log_start_samples * res_samples)
        loss = loss_ls + alpha * reg

        retdict["output"] = [probs.squeeze(-1), "hist"]  # output
        retdict["reg"] = [reg, "sequence"]
        retdict["loss"] = [loss, "sequence"]
        return retdict

    def __repr__(self):
        return self.__class__.__name__


def mcpg_solver(nvar, config, data, verbose=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    sampler = sampler_select(config["problem_type"])

    change_times = int(nvar / 10)  # transition times for metropolis sampling

    net = simple(nvar)
    net.to(device).reset_parameters()
    optimizer = torch.optim.Adam(net.parameters(), lr=config['lr_init'])

    start_samples = None
    for epoch in range(config['max_epoch_num']):

        if epoch % config['reset_epoch_num'] == 0:
            net.to(device).reset_parameters()
            regular = config['regular_init']

        net.train()
        if epoch <= 0:
            retdict = net(regular, None, None)
        else:
            retdict = net(regular, start_samples, value)

        retdict["loss"][0].backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1)
        optimizer.step()

        # get start samples
        if epoch == 0:
            probs = (torch.zeros(nvar) + 0.5).to(device)
            tensor_probs = sample_initializer(
                config["problem_type"], probs, config, data=data)
            temp_max, temp_max_info, temp_start_samples, value = sampler(
                data, tensor_probs, probs, config['num_ls'], 0, config['total_mcmc_num'])
            now_max_res = temp_max
            now_max_info = temp_max_info
            tensor_probs = temp_max_info.clone()
            tensor_probs = tensor_probs.repeat(1, config['repeat_times'])
            start_samples = temp_start_samples.t().to(device)

        # get samples
        if epoch % config['sample_epoch_num'] == 0 and epoch > 0:
            probs = retdict["output"][0]
            probs = probs.detach()
            temp_max, temp_max_info, start_samples_temp, value = sampler(
                data, tensor_probs, probs, config['num_ls'], change_times, config['total_mcmc_num'])
            # update now_max
            for i0 in range(config['total_mcmc_num']):
                if temp_max[i0] > now_max_res[i0]:
                    now_max_res[i0] = temp_max[i0]
                    now_max_info[:, i0] = temp_max_info[:, i0]

            # update if min is too small
            now_max = max(now_max_res).item()
            now_max_index = torch.argmax(now_max_res)
            now_min = min(now_max_res).item()
            now_min_index = torch.argmin(now_max_res)
            now_max_res[now_min_index] = now_max
            now_max_info[:, now_min_index] = now_max_info[:, now_max_index]
            temp_max_info[:, now_min_index] = now_max_info[:, now_max_index]

            # select best samples
            tensor_probs = temp_max_info.clone()
            tensor_probs = tensor_probs.repeat(1, config['repeat_times'])
            # construct the start point for next iteration
            start_samples = start_samples_temp.t()
            if verbose:
                if config["problem_type"] == "maxsat" and len(data.pdata) == 7:
                    res = max(now_max_res).item()
                    if res > data.pdata[5] * data.pdata[6]:
                        res -= data.pdata[5] * data.pdata[6]
                        print("o {:.3f}".format(res))
                elif "obj_type" in config and config["obj_type"] == "neg":
                    print("o {:.3f}".format((-max(now_max_res).item())))
                else:
                    print("o {:.3f}".format((max(now_max_res).item())))
        del (retdict)

    total_max = now_max_res
    best_sort = torch.argsort(now_max_res, descending=True)
    total_best_info = torch.squeeze(now_max_info[:, best_sort[0]])

    return max(total_max).item(), total_best_info, now_max_res, now_max_info


'''dataloader.py'''


# from dataloader import dataloader_select


def dataloader_select(problem_type):
    if problem_type in ["maxcut", "maxcut_edge", "r_cheegercut", "n_cheegercut"]:
        return maxcut_dataloader
    elif problem_type == "maxsat":
        return maxsat_dataloader
    elif problem_type == "qubo":
        return qubo_dataloader
    else:
        raise (Exception("Unrecognized problem type {}".format(problem_type)))


def maxcut_dataloader(path, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    with open(path) as f:
        fline = f.readline()
        fline = fline.split()
        num_nodes, num_edges = int(fline[0]), int(fline[1])
        edge_index = torch.LongTensor(2, num_edges)
        edge_attr = torch.Tensor(num_edges, 1)
        cnt = 0
        while True:
            lines = f.readlines(num_edges * 2)
            if not lines:
                break
            for line in lines:
                line = line.rstrip('\n').split()
                edge_index[0][cnt] = int(line[0]) - 1
                edge_index[1][cnt] = int(line[1]) - 1
                edge_attr[cnt][0] = float(line[2])
                cnt += 1
        data_maxcut = Data(num_nodes=num_nodes,
                           edge_index=edge_index, edge_attr=edge_attr)
        data_maxcut = data_maxcut.to(device)
        data_maxcut.edge_weight_sum = float(torch.sum(data_maxcut.edge_attr))

        data_maxcut = append_neighbors(data_maxcut)

        data_maxcut.single_degree = []
        data_maxcut.weighted_degree = []
        tensor_abs_weighted_degree = []
        for i0 in range(data_maxcut.num_nodes):
            data_maxcut.single_degree.append(len(data_maxcut.neighbors[i0]))
            data_maxcut.weighted_degree.append(
                float(torch.sum(data_maxcut.neighbor_edges[i0])))
            tensor_abs_weighted_degree.append(
                float(torch.sum(torch.abs(data_maxcut.neighbor_edges[i0]))))
        tensor_abs_weighted_degree = torch.tensor(tensor_abs_weighted_degree)
        data_maxcut.sorted_degree_nodes = torch.argsort(
            tensor_abs_weighted_degree, descending=True)

        edge_degree = []
        add = torch.zeros(3, num_edges).to(device)
        for i0 in range(num_edges):
            edge_degree.append(abs(edge_attr[i0].item()) * (
                    tensor_abs_weighted_degree[edge_index[0][i0]] + tensor_abs_weighted_degree[edge_index[1][i0]]))
            node_r = edge_index[0][i0]
            node_c = edge_index[1][i0]
            add[0][i0] = - data_maxcut.weighted_degree[node_r] / \
                         2 + data_maxcut.edge_attr[i0] - 0.05
            add[1][i0] = - data_maxcut.weighted_degree[node_c] / \
                         2 + data_maxcut.edge_attr[i0] - 0.05
            add[2][i0] = data_maxcut.edge_attr[i0] + 0.05

        for i0 in range(num_nodes):
            data_maxcut.neighbor_edges[i0] = data_maxcut.neighbor_edges[i0].unsqueeze(
                0)
        data_maxcut.add = add
        edge_degree = torch.tensor(edge_degree)
        data_maxcut.sorted_degree_edges = torch.argsort(
            edge_degree, descending=True)

        return data_maxcut, num_nodes


def append_neighbors(data, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    data.neighbors = []
    data.neighbor_edges = []
    num_nodes = data.num_nodes
    for i in range(num_nodes):
        data.neighbors.append([])
        data.neighbor_edges.append([])
    edge_number = data.edge_index.shape[1]

    for index in range(0, edge_number):
        row = data.edge_index[0][index]
        col = data.edge_index[1][index]
        edge_weight = data.edge_attr[index][0].item()

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


class Data_MaxSAT(object):
    def __init__(self, pdata=None, ndata=None):
        self.pdata = pdata
        self.ndata = ndata


def maxsat_dataloader(path, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    ext = os.path.splitext(path)[-1]
    if ext == ".cnf":
        ptype = 'n'
    elif ext == ".wcnf":
        ptype = 'p'
    else:
        raise (Exception("Unrecognized file type {}".format(path)))

    with open(path) as f:
        lines = f.readlines()
        variable_index = []
        clause_index = []
        neg_index = []
        clause_cnt = 0
        nhard = 0
        nvi = []
        nci = []
        nneg = []
        tempvi = []
        tempneg = []
        vp = []
        vn = []
        for line in lines:
            line = line.split()
            if len(line) == 0:
                continue
            elif line[0] == "c":
                continue
            elif line[0] == "p":
                if ptype == 'p':
                    weight = int(line[4])
                nvar, nclause = int(line[2]), int(line[3])
                for i0 in range(nvar):
                    nvi.append([])
                    nci.append([])
                    nneg.append([])
                vp = [0] * nvar
                vn = [0] * nvar
                continue
            tempvi = []
            tempneg = []
            if ptype == 'p':
                clause_weight_i = int(line[0])
                if clause_weight_i == weight:
                    nhard += 1
                for ety in line[1:-1]:
                    ety = int(ety)
                    variable_index.append(abs(ety) - 1)
                    tempvi.append(abs(ety) - 1)
                    clause_index.append(clause_cnt)
                    neg_index.append(int(ety / abs(ety)) * clause_weight_i)
                    tempneg.append(int(ety / abs(ety)) * clause_weight_i)
                    if ety > 0:
                        vp[abs(ety) - 1] += 1
                    else:
                        vn[abs(ety) - 1] += 1
            else:
                for ety in line:
                    if ety == '0':
                        continue
                    ety = int(ety)
                    variable_index.append(abs(ety) - 1)
                    tempvi.append(abs(ety) - 1)
                    clause_index.append(clause_cnt)
                    neg_index.append(int(ety / abs(ety)))
                    tempneg.append(int(ety / abs(ety)))
                    if ety > 0:
                        vp[abs(ety) - 1] += 1
                    else:
                        vn[abs(ety) - 1] += 1
            for i0 in range(len(tempvi)):
                node = tempvi[i0]
                nvi[node] += tempvi
                nneg[node] += tempneg
                temp = len(nci[node])
                if temp > 0:
                    temp = nci[node][temp - 1] + 1
                nci[node] += [temp] * len(tempvi)
            clause_cnt += 1
    degree = []
    for i0 in range(nvar):
        nvi[i0] = torch.LongTensor(nvi[i0]).to(device)
        nci[i0] = torch.LongTensor(nci[i0]).to(device)
        nneg[i0] = torch.tensor(nneg[i0]).to(device)
        degree.append(vp[i0] + vn[i0])
    degree = torch.FloatTensor(degree).to(device)
    sorted = torch.argsort(degree, descending=True).to('cpu')
    neg_index = torch.tensor(neg_index).to(device)
    ci_cuda = torch.tensor(clause_index).to(device)

    ndata = [nvi, nci, nneg, sorted, degree]
    ndata = sort_node(ndata)

    pdata = [nvar, nclause, variable_index, ci_cuda, neg_index]
    if ptype == 'p':
        pdata = [nvar, nclause, variable_index,
                 ci_cuda, neg_index, weight, nhard]
    return Data_MaxSAT(pdata=pdata, ndata=ndata), pdata[0]


def sort_node(ndata):
    degree = ndata[4]
    device = degree.device
    temp = degree + (torch.rand(degree.shape[0], device=device) - 0.5) / 2
    sorted = torch.argsort(temp, descending=True).to('cpu')
    ndata[3] = sorted
    return ndata


def qubo_dataloader(path, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    Q = np.load(path)
    Q = torch.tensor(Q).float().to(device)
    data = {'Q': Q, 'nvar': Q.shape[0]}
    return data, Q.shape[0]


def read_data_mimo(K, N, SNR, X_num, r_seed, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    path = "../data/mimo/4QAM{}_{}/4QAM{}H{}.mat".format(N, K, K, int(r_seed // X_num + 1))
    data = scio.loadmat(path)
    H = data["save_H"]
    path = "../data/mimo/4QAM{}_{}/4QAM{}X{}.mat".format(N, K, K, int(r_seed // X_num + 1))
    data = scio.loadmat(path)
    X = data["save_X"][r_seed % X_num]
    path = "../data/mimo/4QAM{}_{}/4QAM{}v{}.mat".format(N, K, K, int(r_seed // X_num + 1))
    data = scio.loadmat(path)
    v = data["save_v"][r_seed % X_num]
    v = np.sqrt(2 * K * 10 ** (-SNR / 10)) * v

    Y = H.dot(X) + v
    noise = np.linalg.norm(v)

    Sigma = H.T.dot(H)
    Diag = -2 * Y.T.dot(H)
    sca = Y.T.dot(Y)
    for i in range(Sigma.shape[0]):
        sca += Sigma[i][i]
        Sigma[i][i] = 0

    # to cuda
    Sigma = torch.tensor(Sigma).to(device)
    Diag = torch.tensor(Diag).to(device)
    X = torch.tensor(X).to(device)
    sca = torch.tensor(sca).to(device)

    data = [Sigma, Diag, X, sca, noise]
    return data


'''mcpg.py'''


def run():
    # gpu_id = 1
    # device = th.device(f'cuda:{gpu_id}' if th.cuda.is_available() and gpu_id >= 0 else 'cpu')

    config = {
        'problem_type': 'maxcut', 'lr_init': 0.1, 'regular_init': 0, 'sample_epoch_num': 8,
        'max_epoch_num': 80, 'reset_epoch_num': 80, 'total_mcmc_num': 400, 'repeat_times': 120, 'num_ls': 5
    }

    path = '../data/gset/gset_14.txt'
    start_time = time.perf_counter()
    dataloader = dataloader_select(config["problem_type"])
    data, nvar = dataloader(path)

    dataloader_t = time.perf_counter()
    res, solutions, _, _ = mcpg_solver(nvar, config, data, verbose=True)
    mcpg_t = time.perf_counter()

    if config["problem_type"] == "maxsat" and len(data.pdata) == 7:
        if res > data.pdata[5] * data.pdata[6]:
            res -= data.pdata[5] * data.pdata[6]
            print("SATISFIED")
            print("SATISFIED SOFT CLAUSES:", res)
            print("UNSATISFIED SOFT CLAUSES:", data.pdata[1] - data.pdata[-1] - res)
        else:
            res = res // data.pdata[5] - data.pdata[6]
            print("UNSATISFIED")

    elif "obj_type" in config and config["obj_type"] == "neg":
        print("OUTPUT: {:.2f}".format(-res))
    else:
        print("OUTPUT: {:.2f}".format(res))

    print("DATA LOADING TIME: {:.2f}".format(dataloader_t - start_time))
    print("MCPG RUNNING TIME: {:.2f}".format(mcpg_t - dataloader_t))


if __name__ == '__main__':
    run()
