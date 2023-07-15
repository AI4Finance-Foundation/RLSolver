import copy

import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
import operator
from itertools import islice
import os
from collections import OrderedDict
import collections.abc as container_abcs
import functools
import torch as th
import torch.nn as nn
import numpy as np
from typing import List, Union
import networkx as nx
import matplotlib.pyplot as plt

from torch import Tensor


# read graph file, e.g., gset_14.txt, as networkx.Graph
# The nodes in file start from 1, but the nodes start from 0 in our codes.
def read_txt_as_networkx_graph(filename: str, plot_fig: bool = True) -> nx.Graph():
    with open(filename, 'r') as file:
        lines = file.readlines()
        lines = [[int(i1) for i1 in i0.split()] for i0 in lines]
    num_nodes, num_edges = lines[0]
    g = nx.Graph()
    nodes = list(range(num_nodes))
    g.add_nodes_from(nodes)
    for item in lines[1:]:
        g.add_edge(item[0] - 1, item[1] - 1, weight=item[2])
    if plot_fig:
        nx.draw_networkx(g)
        fig_filename = filename.split('.')[0] + '.png'
        plt.savefig(fig_filename)
        plt.show()
    return g


def obj_maxcut(result: Union[Tensor, List[int], np.array], graph: nx.Graph):
    num_nodes = len(result)
    cut = 0
    adj_matrix = nx.adjacency_matrix(graph)
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if result[i] != result[j]:
                cut += adj_matrix[(i, j)]
    # print('obj: ', cut)
    return cut


# write a tensor/list/np.array (dim: 1) to a txt file.
# The nodes start from 0, and the label of classified set is 0 or 1 in our codes, but the nodes written to file start from 1, and the label is 1 or 2
def write_result(result: Union[Tensor, List, np.array], filename: str = 'result/result.txt'):
    # assert len(result.shape) == 1
    # N = result.shape[0]
    num_nodes = len(result)
    directory = filename.split('/')[0]
    if not os.path.exists(directory):
        os.mkdir(directory)
    with open(filename, 'w', encoding="UTF-8") as file:
        for node in range(num_nodes):
            file.write(f'{node + 1} {int(result[node] + 1)}\n')


# genete a graph, and output a symmetric_adjacency_matrix and networkx_graph. The graph will be written to a file.
# weight_low (inclusive) and weight_high (exclusive) are the low and high int values for weight, and should be int.
# If writing the graph to file, the node starts from 1, not 0. The first node index < the second node index. Only the non-zero weight will be written.
# If writing the graph, the file name will be revised, e.g., syn.txt will be revised to syn_n_m.txt, where n is num_nodes, and m is num_edges.
def generate_write_symmetric_adjacency_matrix_and_networkx_graph(num_nodes: int,
                                                                 num_edges: int,
                                                                 filename: str = 'data/syn.txt',
                                                                 weight_low=0,
                                                                 weight_high=2) -> (List[List[int]], nx.Graph):
    if weight_low == 0:
        weight_low += 1
    adjacency_matrix = []
    # generate adjacency_matrix where each row has num_edges_per_row edges
    num_edges_per_row = int(np.ceil(2 * num_edges / num_nodes))
    for i in range(num_nodes):
        indices = []
        while True:
            all_indices = list(range(0, num_nodes))
            np.random.shuffle(all_indices)
            indices = all_indices[: num_edges_per_row]
            if i not in indices:
                break
        row = [0] * num_nodes
        weights = np.random.randint(weight_low, weight_high, size=num_edges_per_row)
        for k in range(len(indices)):
            row[indices[k]] = weights[k]
        adjacency_matrix.append(row)
    # the num of edges of the generated adjacency_matrix may not be the specified, so we revise it.
    indices1 = []  # num of non-zero weights for i < j
    indices2 = []  # num of non-zero weights for i > j
    for i in range(num_nodes):
        for j in range(num_nodes):
            if adjacency_matrix[i][j] != 0:
                if i < j:
                    indices1.append((i, j))
                else:
                    indices2.append((i, j))
    # if |indices1| > |indices2|, we get the new adjacency_matrix by swapping symmetric elements
    # based on adjacency_matrix so that |indices1| < |indices2|
    if len(indices1) > len(indices2):
        indices1 = []
        indices2 = []
        new_adjacency_matrix = copy.deepcopy(adjacency_matrix)
        for i in range(num_nodes):
            for j in range(num_nodes):
                new_adjacency_matrix[i][j] = adjacency_matrix[j][i]
                if new_adjacency_matrix[i][j] != 0:
                    if i < j:
                        indices1.append((i, j))
                    else:
                        indices2.append((i, j))
        adjacency_matrix = new_adjacency_matrix
    # We first set some elements of indices2 0 so that |indices2| = num_edges,
    # then, fill the adjacency_matrix so that the symmetric elements along diagonal are the same
    if len(indices1) <= len(indices2):
        num_set_0 = len(indices2) - num_edges
        if num_set_0 < 0:
            raise ValueError("wrong num_set_0")
        while True:
            all_ind_set_0 = list(range(0, len(indices2)))
            np.random.shuffle(all_ind_set_0)
            ind_set_0 = all_ind_set_0[: num_set_0]
            indices2_set_0 = [indices2[k] for k in ind_set_0]
            new_indices2 = set([indices2[k] for k in range(len(indices2)) if k not in ind_set_0])
            my_list = list(range(num_nodes))
            my_set: set = set()
            satisfy = True
            # check if all nodes exist in new_indices2. If yes, the condition is satisfied, and iterate again otherwise.
            for i, j in new_indices2:
                my_set.add(i)
                my_set.add(j)
            for item in my_list:
                if item not in my_set:
                    satisfy = False
                    break
            if satisfy:
                break
        for (i, j) in indices2_set_0:
            adjacency_matrix[i][j] = 0
        if len(new_indices2) != num_edges:
            raise ValueError("wrong new_indices2")
        # fill elements of adjacency_matrix based on new_indices2
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                if (j, i) in new_indices2:
                    adjacency_matrix[i][j] = adjacency_matrix[j][i]
                else:
                    adjacency_matrix[i][j] = 0
    # create a networkx graph
    g = nx.Graph()
    nodes = list(range(num_nodes))
    g.add_nodes_from(nodes)
    num_edges = len(new_indices2)
    # create a new filename, and write the graph to the file.
    new_filename = filename.split('.')[0] + '_' + str(num_nodes) + '_' + str(num_edges) + '.txt'
    with open(new_filename, 'w', encoding="UTF-8") as file:
        file.write(f'{num_nodes} {num_edges} \n')
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                weight = int(adjacency_matrix[i][j])
                g.add_edge(i, j, weight=weight)
                if weight != 0:
                    file.write(f'{i + 1} {j + 1} {weight}\n')
    return adjacency_matrix, g

def calc_file_name(front: str, id2: int, val: int, end: str):
    return front + "_" + str(id2) + "_" + str(val) + end + "pkl"


def detach_var(v, device):
    var = Variable(v.data, requires_grad=True).to(device)
    var.retain_grad()
    return var


def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition('.')
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)


def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)

    return functools.reduce(_getattr, [obj] + attr.split('.'))


class Opt_net(nn.Module):
    def __init__(self, N, hidden_layers):
        super(Opt_net, self).__init__()
        self.N = N
        self.hidden_layers = hidden_layers
        self.lstm = nn.LSTM(self.N, self.hidden_layers, 1, batch_first=True)
        self.output = nn.Linear(hidden_layers, self.N)

    def forward(self, configuration, hidden_state, cell_state):
        x, (h, c) = self.lstm(configuration, (hidden_state, cell_state))
        return self.output(x).sigmoid(), h, c


def plot_figs(scoress: List[List[int]], num_steps: int, labels: List[str]):
    num = len(scoress)
    x = list(range(num_steps))
    dic = {'0': 'ro', '1': 'gs', '2': 'b^', '3': 'c>', '4': 'm<', '5': 'yp'}
    for i in range(num):
        plt(x, scoress[i], dic[str(i)], labels[i])
    plt.legend(labels, loc=0)
    plt.show()

def plot_fig(scores: List[int], label: str):
    x = list(range(len(scores)))
    dic = {'0': 'ro-', '1': 'gs', '2': 'b^', '3': 'c>', '4': 'm<', '5': 'yp'}
    plt.plot(x, scores, dic['0'])
    plt.legend([label], loc=0)
    plt.savefig('result/' + label + '.png')
    plt.show()

if __name__ == '__main__':
    graph1 = read_txt_as_networkx_graph('data/gset_14.txt')
    graph2 = read_txt_as_networkx_graph('data/syn_5_5.txt')

    # result = Tensor([0, 1, 0, 1, 0, 1, 1])
    # write_result(result)
    # result = [0, 1, 0, 1, 0, 1, 1]
    # write_result(result)
    result = [1, 0, 1, 0, 1]
    write_result(result)
    adj_matrix, graph3 = generate_write_symmetric_adjacency_matrix_and_networkx_graph(6, 8)
    graph4 = read_txt_as_networkx_graph('data/syn_6_8.txt')
    obj_maxcut(result, graph4)

    # generate synthetic data
    # num_nodes_edges = [(20, 50), (30, 110), (50, 190), (100, 460), (200, 1004), (400, 1109), (800, 2078), (1000, 4368), (2000, 9386), (3000, 11695), (4000, 25654), (5000, 50543), (10000, 100457)]
    num_nodes_edges = [(3000, 25695), (4000, 38654), (5000, 50543),  (6000, 73251), (7000, 79325), (8000, 83647), (9000, 96324), (10000, 100457), (13000, 18634), (16000, 19687), (20000, 26358)]
    # num_nodes_edges = [(100, 460)]
    num_datasets = 1
    for num_nodes, num_edges in num_nodes_edges:
        for n in range(num_datasets):
            generate_write_symmetric_adjacency_matrix_and_networkx_graph(num_nodes, num_edges + n)
    print()
