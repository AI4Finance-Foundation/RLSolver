import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import copy
from torch.autograd import Variable
import os
import functools
import time
import torch.nn as nn
import numpy as np
from typing import List, Union, Tuple
import networkx as nx
import pandas as pd
import torch as th
from torch import Tensor
from os import system
from config import *
import math
from enum import Enum
import tqdm
import re
from config import *
try:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

TEN = th.Tensor
INT = th.IntTensor
TEN = th.Tensor
GraphList = List[Tuple[int, int, int]]
IndexList = List[List[int]]
from config import GSET_DIR
DataDir = GSET_DIR

class MyGraph:
    def __init__(self):
        num_nodes = 0
        num_edges = 0
        graph = List[Tuple[int, int, int]]

def plot_nxgraph(g: nx.Graph()):
    import matplotlib.pyplot as plt
    nx.draw_networkx(g)
    fig_filename = '.result/fig.png'
    plt.savefig(fig_filename)
    plt.show()

# read graph file, e.g., gset_14.txt, as networkx.Graph
# The nodes in file start from 1, but the nodes start from 0 in our codes.
def read_nxgraph(filename: str) -> nx.Graph():
    graph = nx.Graph()
    with open(filename, 'r') as file:
        # lines = []
        line = file.readline()
        is_first_line = True
        while line is not None and line != '':
            if '//' not in line:
                if is_first_line:
                    strings = line.split(" ")
                    num_nodes = int(strings[0])
                    num_edges = int(strings[1])
                    nodes = list(range(num_nodes))
                    graph.add_nodes_from(nodes)
                    is_first_line = False
                else:
                    node1, node2, weight = line.split(" ")
                    graph.add_edge(int(node1) - 1, int(node2) - 1, weight=weight)
            line = file.readline()
    return graph

#
def transfer_nxgraph_to_adjacencymatrix(graph: nx.Graph):
    return nx.to_numpy_array(graph)

# the returned weightmatrix has the following format： node1 node2 weight
# For example: 1 2 3 // the weight of node1 and node2 is 3
def transfer_nxgraph_to_weightmatrix(graph: nx.Graph):
    # edges = nx.edges(graph)
    res = np.array([])
    edges = graph.edges()
    for u, v in edges:
        u = int(u)
        v = int(v)
        # weight = graph[u][v]["weight"]
        weight = float(graph.get_edge_data(u, v)["weight"])
        vec = np.array([u, v, weight])
        if len(res) == 0:
            res = vec
        else:
            res = np.vstack((res, vec))
    return res

# weightmatrix: format of each vector: node1 node2 weight
# num_nodes: num of nodes
def transfer_weightmatrix_to_nxgraph(weightmatrix: List[List[int]], num_nodes: int) -> nx.Graph():
    graph = nx.Graph()
    nodes = list(range(num_nodes))
    graph.add_nodes_from(nodes)
    for i, j, weight in weightmatrix:
        graph.add_edge(i, j, weight=weight)
    return graph

# max total cuts
def obj_maxcut(result: Union[Tensor, List[int], np.array], graph: nx.Graph):
    num_nodes = len(result)
    obj = 0
    adj_matrix = transfer_nxgraph_to_adjacencymatrix(graph)
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if result[i] != result[j]:
                obj += adj_matrix[(i, j)]
    return obj

# min total cuts
def obj_graph_partitioning(solution: Union[Tensor, List[int], np.array], graph: nx.Graph):
    num_nodes = len(solution)
    obj = 0
    adj_matrix = transfer_nxgraph_to_adjacencymatrix(graph)
    sum1 = 0
    for i in range(num_nodes):
        if solution[i] == 0:
            sum1 += 1
        for j in range(num_nodes):
            if i != j and solution[i] != solution[j]:
                obj -= adj_matrix[(i, j)]
    if sum1 != num_nodes / 2:
        return -INF
    return obj

def cover_all_edges(solution: List[int], graph: nx.Graph):
    if graph.number_of_nodes() == 0:
        return False
    cover_all = True
    for node1, node2 in graph.edges:
        if solution[node1] == 0 and solution[node2] == 0:
            cover_all = False
            break
    return cover_all

def obj_minimum_vertex_cover(solution: Union[Tensor, List[int], np.array], graph: nx.Graph, need_check_cover_all_edges=True):
    num_nodes = len(solution)
    obj = 0
    for i in range(num_nodes):
        if solution[i] == 1:
            obj -= 1
    if need_check_cover_all_edges:
        if not cover_all_edges(solution, graph):
                return -INF
    return obj

# make sure solution[i] = 0 or 1
def obj_maximum_independent_set(solution: Union[Tensor, List[int], np.array], graph: nx.Graph):
    sol = set(solution)
    # if len(solution) > 0:
    #     assert len(sol) == 2
    max_elem = max(sol)
    min_elem = min(sol)
    if max_elem == min_elem:
        max_elem += 1
    obj = 0
    edges = list(graph.edges)
    num_nodes = int(graph.number_of_nodes())
    for i, j in edges:
        if solution[i] == max_elem and solution[j] == max_elem:
            return -INF
    for i in range(num_nodes):
        if solution[i] == max_elem:
            obj += 1
    return obj

# write a tensor/list/np.array (dim: 1) to a txt file.
# The nodes start from 0, and the label of classified set is 0 or 1 in our codes, but the nodes written to file start from 1, and the label is 1 or 2
def write_result(result: Union[Tensor, List, np.array],
                 filename: str = './result/result.txt',
                 obj: Union[int, float] = None,
                 running_duration: Union[int, float] = None):
    # assert len(result.shape) == 1
    # N = result.shape[0]
    num_nodes = len(result)
    directory = filename.split('/')[0]
    if not os.path.exists(directory):
        os.mkdir(directory)
    with open(filename, 'w', encoding="UTF-8") as file:
        if obj is not None:
            file.write(f'// obj: {obj}\n')
        if running_duration is not None:
            file.write(f'// running_duration: {running_duration}\n')
        for node in range(num_nodes):
            file.write(f'{node + 1} {int(result[node] + 1)}\n')


# genete a graph, and output a symmetric_adjacency_matrix and networkx_graph. The graph will be written to a file.
# weight_low (inclusive) and weight_high (exclusive) are the low and high int values for weight, and should be int.
# If writing the graph to file, the node starts from 1, not 0. The first node index < the second node index. Only the non-zero weight will be written.
# If writing the graph, the file name will be revised, e.g., syn.txt will be revised to syn_n_m.txt, where n is num_nodes, and m is num_edges.
def generate_write_adjacencymatrix_and_nxgraph(num_nodes: int,
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
    graph = nx.Graph()
    nodes = list(range(num_nodes))
    graph.add_nodes_from(nodes)
    num_edges = len(new_indices2)
    # create a new filename, and write the graph to the file.
    new_filename = filename.split('.')[0] + '_' + str(num_nodes) + '_' + str(num_edges) + '.txt'
    with open(new_filename, 'w', encoding="UTF-8") as file:
        file.write(f'{num_nodes} {num_edges} \n')
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                weight = int(adjacency_matrix[i][j])
                graph.add_edge(i, j, weight=weight)
                if weight != 0:
                    file.write(f'{i + 1} {j + 1} {weight}\n')
    return adjacency_matrix, graph

def generate_write_distribution(num_nodess: List[int], num_graphs: int, graph_type: GraphDistriType, dir: str):
    for num_nodes in num_nodess:
        for i in range(num_graphs):
            weightmatrix, num_nodes, num_edges = generate_graph(num_nodes, graph_type)
            graph = transfer_weightmatrix_to_nxgraph(weightmatrix, num_nodes)
            filename = dir + '/' + graph_type.value + '_' + str(num_nodes) + '_ID' + str(i) + '.txt'
            write_nxgraph(graph, filename)

def write_nxgraph(g: nx.Graph(), filename: str):
    num_nodes = nx.number_of_nodes(g)
    num_edges = nx.number_of_edges(g)
    adjacency_matrix = nx.to_numpy_array(g)
    with open(filename, 'w', encoding="UTF-8") as file:
        file.write(f'{num_nodes} {num_edges} \n')
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                weight = int(adjacency_matrix[i][j])
                g.add_edge(i, j, weight=weight)
                if weight != 0:
                    file.write(f'{i + 1} {j + 1} {weight}\n')

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


def plot_fig(scores: List[int], label: str):
    import matplotlib.pyplot as plt
    plt.figure()
    x = list(range(len(scores)))
    dic = {'0': 'ro-', '1': 'gs', '2': 'b^', '3': 'c>', '4': 'm<', '5': 'yp'}
    plt.plot(x, scores, dic['0'])
    plt.legend([label], loc=0)
    plt.savefig('../result/' + label + '.png')
    plt.show()

def plot_fig_over_durations(objs: List[int], durations: List[int], label: str):
    import matplotlib.pyplot as plt
    plt.figure()
    x = durations
    dic = {'0': 'ro-', '1': 'gs', '2': 'b^', '3': 'c>', '4': 'm<', '5': 'yp'}
    # plt.ylim(0, max(objs))
    plt.plot(x, objs, dic['0'])
    plt.legend([label], loc=0)
    plt.savefig('./result/' + label + '.png')
    plt.show()

# return: num_nodes, ID, running_duration:, obj,
def read_result_comments(filename: str):
    num_nodes, ID, running_duration, obj = None, None, None, None
    ID = int(filename.split('ID')[1].split('_')[0])
    with open(filename, 'r') as file:
        # lines = []
        line = file.readline()
        while line is not None and line != '':
            if '//' in line:
                if 'num_nodes:' in line:
                    num_nodes = float(line.split('num_nodes:')[1])
                    break
                if 'running_duration:' in line:
                    running_duration = obtain_first_number(line)
                if 'obj:' in line:
                    obj = float(line.split('obj:')[1])
            line = file.readline()
    return int(num_nodes), ID, running_duration, obj

def read_result_comments_multifiles(dir: str, prefixes: str, running_durations: List[int]):
    res = {}
    num_nodess = set()
    # for prefix in prefixes:
    files = calc_txt_files_with_prefix(dir, prefixes)
    num_ids = NUM_IDS
    for i in range(len(files)):
        file = files[i]
        num_nodes, ID, running_duration, obj = read_result_comments(file)
        if running_duration not in running_durations:
            continue
        index = running_durations.index(running_duration)
        num_nodess.add(num_nodes)
        if str(num_nodes) not in res.keys():
            res[str(num_nodes)] = [[None] * len(running_durations) for _ in range(num_ids)]
        res[str(num_nodes)][ID][index] = obj
            # res[str(num_nodes)] = {**res[str(num_nodes)], **tmp_dict}
    for num_nodes_str in res.keys():
        for ID in range(num_ids):
            last_nonNone = None
            for i in range(len(running_durations)):
                if res[num_nodes_str][ID][i] is not None:
                    last_nonNone = res[num_nodes_str][ID][i]
                if res[num_nodes_str][ID][i] is None and last_nonNone is not None:
                    res[num_nodes_str][ID][i] = last_nonNone

    num_nodess = list(num_nodess)
    num_nodess.sort()
    for num_nodes in num_nodess:
        objs = []
        for i in range(len(running_durations)):
            sum_obj = 0
            for ID in range(num_ids):
                if res[str(num_nodes)][ID][i] is not None:
                    sum_obj += res[str(num_nodes)][ID][i]
            obj = sum_obj / num_ids
            objs.append(obj)
        label = f"num_nodes={num_nodes}"
        print(f"objs: {objs}, running_duration: {running_durations}, label: {label}")
        plot_fig_over_durations(objs, running_durations, label)


def calc_txt_files_with_prefix(directory: str, prefix: str):
    res = []
    files = os.listdir(directory)
    for file in files:
        if prefix in file and '.txt' in file:
            res.append(directory + '/' + file)
    return res

def calc_files_with_prefix_suffix(directory: str, prefix: str, suffix: str, extension: str = '.txt'):
    res = []
    files = os.listdir(directory)
    new_suffix = '_' + suffix + extension
    for file in files:
        if prefix in file and new_suffix in file:
            res.append(directory + '/' + file)
    return res

# if the file name is '../data/syn_10_27.txt', the return is '../result/syn_10_27.txt'
# if the file name is '../result/syn_10_27.txt', the return is '../result/syn_10_27.txt'
# if the file name is '../data/syn/syn_10_27.txt', the return is '../result/syn_10_27.txt'
def calc_result_file_name(file: str, add_tail: str= ''):
    new_file = copy.deepcopy(file)
    if 'data' in new_file:
        new_file = new_file.replace('data', 'result')
    # if file[0: 2] == '..':
    #     new_file = new_file.split('.txt')[0]
    #     new_file = new_file.split('/')[0] + '/' + new_file.split('/')[1] + '/' + new_file.split('/')[-1]
    # else:
    #     new_file = new_file.split('.')[0]
    #     new_file = new_file.split('/')[0] + '/' + new_file.split('/')[-1]
    new_file = new_file.split('result')[0] + 'result/' + new_file.split('/')[-1]
    if add_tail is not None:
        new_file = new_file.replace('.txt', '') + add_tail + '.txt'
    return new_file

# For example, syn_10_21_3600.txt, the prefix is 'syn_10_', time_limit is 3600 (seconds).
# The gap and running_duration are also be calculated.
def calc_avg_std_of_obj(directory: str, prefix: str, time_limit: int):
    init_time_limit = copy.deepcopy(time_limit)
    objs = []
    gaps = []
    obj_bounds = []
    running_durations = []
    suffix = str(time_limit)
    files = calc_files_with_prefix_suffix(directory, prefix, suffix)
    for i in range(len(files)):
        with open(files[i], 'r') as file:
            line = file.readline()
            assert 'obj' in line
            obj = float(line.split('obj:')[1].split('\n')[0])
            objs.append(obj)

            line2 = file.readline()
            running_duration_ = line2.split('running_duration:')
            running_duration = float(running_duration_[1]) if len(running_duration_) >= 2 else None
            running_durations.append(running_duration)

            line3 = file.readline()
            gap_ = line3.split('gap:')
            gap = float(gap_[1]) if len(gap_) >= 2 else None
            gaps.append(gap)

            line4 = file.readline()
            obj_bound_ = line4.split('obj_bound:')
            obj_bound = float(obj_bound_[1]) if len(obj_bound_) >= 2 else None
            obj_bounds.append(obj_bound)
    if len(objs) == 0:
        return
    avg_obj = np.average(objs)
    std_obj = np.std(objs)
    avg_running_duration = np.average(running_durations)
    avg_gap = np.average(gaps) if None not in gaps else None
    avg_obj_bound = np.average(obj_bounds) if None not in obj_bounds else None
    print(f'{directory} prefix {prefix}, suffix {suffix}: avg_obj {avg_obj}, std_obj {std_obj}, avg_running_duration {avg_running_duration}, avg_gap {avg_gap}, avg_obj_bound {avg_obj_bound}')
    if time_limit != init_time_limit:
        print()
    return {(prefix, time_limit): (avg_obj, std_obj, avg_running_duration, avg_gap, avg_obj_bound)}

def calc_avg_std_of_objs(directory: str, prefixes: List[str], time_limits: List[int]):
    res = []
    for i in range(len(prefixes)):
        for k in range(len(time_limits)):
            avg_std = calc_avg_std_of_obj(directory, prefixes[i], int(time_limits[k]))
            res.append(avg_std)
    return res

# transfer flot to binary. For example, 1e-7 -> 0, 1 + 1e-8 -> 1
def transfer_float_to_binary(value: float) -> int:
    if abs(value) < 1e-4:
        value = 0
    elif abs(value - 1) < 1e-4:
        value = 1
    else:
        raise ValueError('wrong value')
    return value

def fetch_node(line: str):
    if 'x[' in line:
        node = int(line.split('x[')[1].split(']')[0])
    else:
        node = None
    return node

# e.g., s = "// time_limit: ('TIME_LIMIT', <class 'float'>, 36.0, 0.0, inf, inf)",
# then returns 36
def obtain_first_number(s: str):
    res = ''
    pass_first_digit = False
    for i in range(len(s)):
        if s[i].isdigit() or s[i] == '.':
            res += s[i]
            pass_first_digit = True
        elif pass_first_digit:
            break
    value = int(float(res))
    return value


def load_graph_from_txt(txt_path: str = './data/gset_14.txt'):
    with open(txt_path, 'r') as file:
        lines = file.readlines()
        lines = [[int(i1) for i1 in i0.split()] for i0 in lines]
    num_nodes, num_edges = lines[0]
    graph = [(n0 - 1, n1 - 1, dt) for n0, n1, dt in lines[1:]]  # node_id “从1开始”改为“从0开始”
    return graph, num_nodes, num_edges

def get_adjacency_matrix(graph, num_nodes):
    adjacency_matrix = np.empty((num_nodes, num_nodes))
    adjacency_matrix[:] = -1  # 选用-1而非0表示表示两个node之间没有edge相连，避免两个节点的距离为0时出现冲突
    for n0, n1, dt in graph:
        adjacency_matrix[n0, n1] = dt
    return adjacency_matrix

def load_graph(graph_name: str):
    data_dir = DATA_DIR
    graph_types = GRAPH_DISTRI_TYPES
    if os.path.exists(f"{data_dir}/{graph_name}.txt"):
        txt_path = f"{data_dir}/{graph_name}.txt"
        graph, num_nodes, num_edges = load_graph_from_txt(txt_path=txt_path)
    elif graph_name.split('_')[0] in graph_types:
        g_type, num_nodes = graph_name.split('_')
        num_nodes = int(num_nodes)
        graph, num_nodes, num_edges = generate_graph(num_nodes=num_nodes, g_type=g_type)
    else:
        raise ValueError(f"graph_name {graph_name}")
    return graph, num_nodes, num_edges

def load_graph_auto(graph_name: str):
    import random
    graph_types = GRAPH_DISTRI_TYPES
    if os.path.exists(f"{DataDir}/{graph_name}.txt"):
        txt_path = f"{DataDir}/{graph_name}.txt"
        graph = load_graph_from_txt(txt_path=txt_path)
    elif graph_name.split('_')[0] in graph_types and len(graph_name.split('_')) == 3:
        graph_type, num_nodes, valid_i = graph_name.split('_')
        num_nodes = int(num_nodes)
        valid_i = int(valid_i[len('ID'):])
        random.seed(valid_i)
        graph = generate_graph(num_nodes=num_nodes, graph_type=graph_type)
        random.seed()
    elif graph_name.split('_')[0] in graph_types and len(graph_name.split('_')) == 2:
        graph_type, num_nodes = graph_name.split('_')
        num_nodes = int(num_nodes)
        graph = generate_graph(num_nodes=num_nodes, graph_type=graph_type)
    else:
        raise ValueError(f"DataDir {DataDir} | graph_name {graph_name}")
    return graph

def save_graph_info_to_txt(txt_path, graph, num_nodes, num_edges):
    formatted_content = f"{num_nodes} {num_edges}\n"
    for node0, node1, distance in graph:
        row = [node0 + 1, node1 + 1, distance]  # node+1 is a bad design
        formatted_content += " ".join(str(item) for item in row) + "\n"
    with open(txt_path, "w") as file:
        file.write(formatted_content)


def generate_graph(num_nodes: int, g_type: str):
    graph_types = GRAPH_DISTRI_TYPES
    assert g_type in graph_types

    if g_type == GraphDistriType.erdos_renyi:
        g = nx.erdos_renyi_graph(n=num_nodes, p=0.15)
    elif g_type == GraphDistriType.powerlaw:
        g = nx.powerlaw_cluster_graph(n=num_nodes, m=4, p=0.05)
    elif g_type == GraphDistriType.barabasi_albert:
        g = nx.barabasi_albert_graph(n=num_nodes, m=4)
    else:
        raise ValueError(f"g_type {g_type} should in {graph_types}")

    graph = []
    for node0, node1 in g.edges:
        distance = 1
        graph.append((node0, node1, distance))
    num_nodes = num_nodes
    num_edges = len(graph)
    return graph, num_nodes, num_edges


def generate_graph_for_validation():
    import random
    num_nodes_list = [20, 50, 100, 200, 300]
    g_type = GRAPH_DISTRI_TYPE
    num_valid = 6
    seed_num = 0
    data_dir = DATA_DIR
    os.makedirs(data_dir, exist_ok=True)

    '''generate'''
    for num_nodes in num_nodes_list:
        random.seed(seed_num)  # must write in the for loop
        for i in range(num_valid):
            txt_path = f"{data_dir}/graph_{g_type}_{num_nodes}_ID{i:03}.txt"

            graph, num_nodes, num_edges = generate_graph(num_nodes=num_nodes, g_type=g_type)
            save_graph_info_to_txt(txt_path, graph, num_nodes, num_edges)

    '''load'''
    for num_nodes in num_nodes_list:
        for i in range(num_valid):
            txt_path = f"{data_dir}/graph_{g_type}_{num_nodes}_ID{i:03}.txt"

            graph, num_nodes, num_edges = load_graph_from_txt(txt_path)
            adjacency_matrix = build_adjacency_matrix(graph, num_nodes)
            print(adjacency_matrix.shape)


def build_adjacency_matrix(graph, num_nodes):
    adjacency_matrix = np.empty((num_nodes, num_nodes))
    adjacency_matrix[:] = -1  # 选用-1而非0表示表示两个node之间没有edge相连，避免两个节点的距离为0时出现冲突
    for n0, n1, dt in graph:
        adjacency_matrix[n0, n1] = dt
    return adjacency_matrix

def build_adjacency_matrix_auto(graph: GraphList, if_bidirectional: bool = False):
    """例如，无向图里：
    - 节点0连接了节点1
    - 节点0连接了节点2
    - 节点2连接了节点3

    用邻接阶矩阵Ary的上三角表示这个无向图：
      0 1 2 3
    0 F T T F
    1 _ F F F
    2 _ _ F T
    3 _ _ _ F

    其中：
    - Ary[0,1]=True
    - Ary[0,2]=True
    - Ary[2,3]=True
    - 其余为False
    """
    not_connection = -1  # 选用-1去表示表示两个node之间没有edge相连，不选用0是为了避免两个节点的距离为0时出现冲突
    print(f"graph before enter: {graph}")
    num_nodes = obtain_num_nodes_auto(graph=graph)

    adjacency_matrix = th.zeros((num_nodes, num_nodes), dtype=th.float32)
    adjacency_matrix[:] = not_connection
    for n0, n1, distance in graph:
        adjacency_matrix[n0, n1] = distance
        if if_bidirectional:
            adjacency_matrix[n1, n0] = distance
    return adjacency_matrix

def build_adjacency_indies_auto(graph: MyGraph, if_bidirectional: bool = False) -> (IndexList, IndexList):
    """
    用二维列表list2d表示这个图：
    [
        [1, 2],
        [],
        [3],
        [],
    ]
    其中：
    - list2d[0] = [1, 2]
    - list2d[2] = [3]

    对于稀疏的矩阵，可以直接记录每条边两端节点的序号，用shape=(2,N)的二维列表 表示这个图：
    0, 1
    0, 2
    2, 3
    如果条边的长度为1，那么表示为shape=(2,N)的二维列表，并在第一行，写上 4个节点，3条边的信息，帮助重建这个图，然后保存在txt里：
    4, 3
    0, 1, 1
    0, 2, 1
    2, 3, 1
    """
    num_nodes = obtain_num_nodes_auto(graph=graph)

    n0_to_n1s = [[] for _ in range(num_nodes)]  # 将 node0_id 映射到 node1_id
    n0_to_dts = [[] for _ in range(num_nodes)]  # 将 mode0_id 映射到 node1_id 与 node0_id 的距离
    for n0, n1, distance in graph:
        n0_to_n1s[n0].append(n1)
        n0_to_dts[n0].append(distance)
        if if_bidirectional:
            n0_to_n1s[n1].append(n0)
            n0_to_dts[n1].append(distance)
    n0_to_n1s = [th.tensor(node1s) for node1s in n0_to_n1s]
    n0_to_dts = [th.tensor(node1s) for node1s in n0_to_dts]
    assert num_nodes == len(n0_to_n1s)
    assert num_nodes == len(n0_to_dts)

    '''sort'''
    for i, node1s in enumerate(n0_to_n1s):
        sort_ids = th.argsort(node1s)
        n0_to_n1s[i] = n0_to_n1s[i][sort_ids]
        n0_to_dts[i] = n0_to_dts[i][sort_ids]
    return n0_to_n1s, n0_to_dts

def obtain_num_nodes_auto(graph: GraphList) -> int:
    return max([max(n0, n1) for n0, n1, distance in graph]) + 1


def convert_matrix_to_vector(matrix):
    vector = [row[i + 1:] for i, row in enumerate(matrix)]
    return th.hstack(vector)

def write_result2(obj, running_duration, num_nodes, alg_name, filename: str):
    add_tail = '_' + str(int(running_duration)) if 'data' in filename else None
    new_filename = calc_result_file_name(filename, add_tail)
    with open(new_filename, 'w', encoding="UTF-8") as new_file:
        prefix = '// '
        new_file.write(f"{prefix}obj: {obj}\n")
        new_file.write(f"{prefix}running_duration: {running_duration}\n")
        new_file.write(f"// num_nodes: {num_nodes}\n")
        new_file.write(f"{prefix}alg_name: {alg_name}\n")


def run_greedy_over_multiple_files(alg, alg_name, num_steps, set_init_0: bool, directory_data: str, prefixes: List[str])-> List[List[float]]:
    if PROBLEM == Problem.graph_partitioning:
        assert set_init_0 is False
    if PROBLEM in [Problem.maxcut, Problem.minimum_vertex_cover, Problem.maximum_independent_set]:
        assert set_init_0 is True
    scoress = []
    for prefix in prefixes:
        files = calc_txt_files_with_prefix(directory_data, prefix)
        files.sort()
        for i in range(len(files)):
            start_time = time.time()
            filename = files[i]
            print(f'The {i}-th file: {filename}')
            graph = read_nxgraph(filename)
            if set_init_0:
                init_solution = [0] * graph.number_of_nodes()
            else:
                init_solution = [0] * int(graph.number_of_nodes() / 2) + [1] * int(graph.number_of_nodes() / 2)
            score, solution, scores = alg(init_solution, num_steps, graph)
            scoress.append(scores)
            print(f"score, scores: {score}, {scores}")
            running_duration = time.time() - start_time
            num_nodes = int(graph.number_of_nodes())
            write_result2(score, running_duration, num_nodes, alg_name, filename)
    return scoress

def run_sdp_over_multiple_files(alg, alg_name, directory_data: str, prefixes: List[str])-> List[List[float]]:
    scores = []
    for prefix in prefixes:
        files = calc_txt_files_with_prefix(directory_data, prefix)
        files.sort()
        for i in range(len(files)):
            start_time = time.time()
            filename = files[i]
            print(f'The {i}-th file: {filename}')
            score, solution = alg(filename)
            scores.append(score)
            print(f"score: {score}")
            running_duration = time.time() - start_time
            graph = read_nxgraph(filename)
            num_nodes = int(graph.number_of_nodes())
            write_result2(score, running_duration, num_nodes, alg_name, filename)
    return scores

def run_simulated_annealing_over_multiple_files(alg, alg_name, init_temperature, num_steps, directory_data: str, prefixes: List[str])-> List[List[float]]:
    scoress = []
    for prefix in prefixes:
        files = calc_txt_files_with_prefix(directory_data, prefix)
        files.sort()
        for i in range(len(files)):
            start_time = time.time()
            filename = files[i]
            print(f'The {i}-th file: {filename}')
            graph = read_nxgraph(filename)
            score, solution, scores = alg(init_temperature, num_steps, graph)
            scoress.append(scores)
            print(f"score, scores: {score}, {scores}")
            running_duration = time.time() - start_time
            num_nodes = int(graph.number_of_nodes())
            write_result2(score, running_duration, num_nodes, alg_name, filename)
    return scoress


if __name__ == '__main__':
    s = "// time_limit: ('TIME_LIMIT', <class 'float'>, 36.0, 0.0, inf, inf)"
    val = obtain_first_number(s)

    read_txt = True
    if read_txt:
        graph1 = read_nxgraph('data/gset/gset_14.txt')
        graph2 = read_nxgraph('data/syn_5_5.txt')

    # result = Tensor([0, 1, 0, 1, 0, 1, 1])
    # write_result(result)
    # result = [0, 1, 0, 1, 0, 1, 1]
    # write_result(result)
    write_result_ = False
    if write_result_:
        result = [1, 0, 1, 0, 1]
        write_result(result)

    generate_read = False
    if generate_read:
        adj_matrix, graph3 = generate_write_adjacencymatrix_and_nxgraph(6, 8)
        graph4 = read_nxgraph('data/syn_6_8.txt')
        obj_maxcut(result, graph4)

    # generate synthetic data
    generate_data = False
    if generate_data:
        # num_nodes_edges = [(20, 50), (30, 110), (50, 190), (100, 460), (200, 1004), (400, 1109), (800, 2078), (1000, 4368), (2000, 9386), (3000, 11695), (4000, 25654), (5000, 50543), (10000, 100457)]
        num_nodes_edges = [(3000, 25695), (4000, 38654), (5000, 50543),  (6000, 73251), (7000, 79325), (8000, 83647), (9000, 96324), (10000, 100457), (13000, 18634), (16000, 19687), (20000, 26358)]
        # num_nodes_edges = [(100, 460)]
        num_datasets = 1
        for num_nodes, num_edges in num_nodes_edges:
            for n in range(num_datasets):
                generate_write_adjacencymatrix_and_nxgraph(num_nodes, num_edges + n)
        print()


    # directory = 'result'
    # prefix = 'syn_10_'
    # time_limit = 3600
    # avg_std = calc_avg_std_of_obj(directory, prefix, time_limit)

    if_calc_avg_std = False
    if if_calc_avg_std:
        directory_result = 'result'
        # prefixes = ['syn_10_', 'syn_50_', 'syn_100_', 'syn_300_', 'syn_500_', 'syn_700_', 'syn_900_', 'syn_1000_', 'syn_3000_', 'syn_5000_', 'syn_7000_', 'syn_9000_', 'syn_10000_']
        prefixes = ['syn_10_', 'syn_50_', 'syn_100_']
        time_limits = GUROBI_TIME_LIMITS
        avgs_stds = calc_avg_std_of_objs(directory_result, prefixes, time_limits)

    # filename = 'result/syn_10_21_1800.sta'
    # new_filename = 'result/syn_10_21_1800.txt'
    # transfer_write_solver_result(filename, new_filename)

    # from_extension = '.sov'
    # to_extension = '.txt'
    # transfer_write_solver_results(directory_result, prefixes, time_limits, from_extension, to_extension)

    if_plot = True
    if(if_plot):
        dir = 'result/syn_PL_gurobi'
        prefixes = 'powerlaw_1400_'
        running_durations = RUNNING_DURATIONS
        read_result_comments_multifiles(dir, prefixes, running_durations)

    if_generate_distribution = False
    if if_generate_distribution:
        num_nodess = [20, 40] + list(range(100, 3001, 100))
        # num_nodess = list(range(2100, 3001, 100))
        # num_nodess = [1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000]
        # num_nodess = [20]
        num_graphs = 30
        graph_type = GraphDistriType.barabasi_albert
        dir = 'data/syn_BA'
        generate_write_distribution(num_nodess, num_graphs, graph_type, dir)


    print()
