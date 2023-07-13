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
def read_as_networkx_graph(filename: str) -> nx.Graph():
    with open(filename, 'r') as file:
        lines = file.readlines()
        lines = [[int(i1) for i1 in i0.split()] for i0 in lines]
    num_nodes, num_edges = lines[0]
    g = nx.Graph()
    nodes = list(range(num_nodes))
    g.add_nodes_from(nodes)
    for item in lines[1:]:
        g.add_edge(item[0] - 1, item[1] - 1, weight=item[2])
    # nx.draw(g, with_labels=False)
    # plt.savefig('result/graph.png')
    # plt.show()
    return g


def obj_maxcut(result: Union[Tensor, List[int], np.array], graph: nx.Graph):
    num_nodes = len(result)
    cut = 0
    adj_matrix = nx.adj_matrix(graph)
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if result[i] != result[j]:
                cut += adj_matrix[(i, j)]
    # print('obj: ', cut)
    return cut


# write a tensor/list/np.array (dim: 1) to a txt file.
def write_result(result: Union[Tensor, List, np.array], filename: str = 'result/result.txt'):
    # assert len(result.shape) == 1
    # N = result.shape[0]
    N = len(result)
    directory = filename.split('/')[0]
    if not os.path.exists(directory):
        os.mkdir(directory)
    with open(filename, 'w', encoding="UTF-8") as file:
        for i in range(N):
            file.write(f'{i} {int(result[i])}\n')



# weight_low (inclusive) and weight_high (exclusive) are the low and high int values for weight, and should be int.
# If writing the graph to file, the node starts from 1, not 0. The first node index < the second node index. Only the non-zero weight will be written.
# If writing the graph, the name of file will be revised, e.g., graph.txt will be revised to graph_n_m.txt, where n is num_nodes, and m is num_edges.
def generate_write_symmetric_adjacency_matrix_and_networkx_graph(num_nodes: int,
                                                                 density: float,
                                                                 filename: str = 'data/syn.txt',
                                                                 weight_low=0,
                                                                 weight_high=2):
    upper_triangle = torch.triu((th.rand(num_nodes, num_nodes) < density).int(), diagonal=1)
    upper_triangle2 = th.mul(th.randint(weight_low, weight_high, (num_nodes, num_nodes)), upper_triangle)
    adjacency_matrix = upper_triangle2 + upper_triangle2.transpose(-1, -2)
    g = nx.Graph()
    nodes = list(range(num_nodes))
    g.add_nodes_from(nodes)
    num_edges = int(th.count_nonzero(adjacency_matrix) / 2)
    new_filename = filename.split('.')[0] + '_' + str(num_nodes) + '_' + str(num_edges) + '.txt'
    with open(new_filename, 'w', encoding="UTF-8") as file:
        file.write(f'{num_nodes} {num_edges} \n')
        for j in range(num_nodes):
            for i in range(0, j):
                weight = int(adjacency_matrix[i, j])
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
    graph1 = read_as_networkx_graph('data/gset_14.txt')
    graph2 = read_as_networkx_graph('data/syn_5_5.txt')
    # result = Tensor([0, 1, 0, 1, 0, 1, 1])
    # write_result(result)
    # result = [0, 1, 0, 1, 0, 1, 1]
    # write_result(result)
    result = [1, 0, 1, 0, 1]
    write_result(result)
    adj_matrix, graph = generate_write_symmetric_adjacency_matrix_and_networkx_graph(30, 0.5)
    obj_maxcut(result, graph2)
    print()
