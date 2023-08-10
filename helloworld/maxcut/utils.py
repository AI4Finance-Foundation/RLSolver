import copy
from torch.autograd import Variable
import os
import functools
import torch.nn as nn
import numpy as np
from typing import List, Union
import networkx as nx


from torch import Tensor


# read graph file, e.g., gset_14.txt, as networkx.Graph
# The nodes in file start from 1, but the nodes start from 0 in our codes.
def read_txt_as_networkx_graph(filename: str, plot_fig: bool = False) -> nx.Graph():
    with open(filename, 'r') as file:
        lines = []
        line = file.readline()  # 读取第一行
        while line is not None and line != '':
            if '//' not in line:
                lines.append(line)
            line = file.readline()  # 读取下一行
        # lines = file.readlines()
        lines = [[int(i1) for i1 in i0.split()] for i0 in lines]
    num_nodes, num_edges = lines[0]
    g = nx.Graph()
    nodes = list(range(num_nodes))
    g.add_nodes_from(nodes)
    for item in lines[1:]:
        g.add_edge(item[0] - 1, item[1] - 1, weight=item[2])
    if plot_fig:
        import matplotlib.pyplot as plt
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
def write_result(result: Union[Tensor, List, np.array], filename: str = 'result/result.txt', obj: Union[int, float] = None, running_duration: Union[int, float] = None):
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

def write_networkx_graph(g: nx.Graph(), new_filename: str):
    num_nodes = nx.number_of_nodes(g)
    num_edges = nx.number_of_edges(g)
    adjacency_matrix = nx.to_numpy_array(g)
    with open(new_filename, 'w', encoding="UTF-8") as file:
        file.write(f'{num_nodes} {num_edges} \n')
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                weight = int(adjacency_matrix[i][j])
                g.add_edge(i, j, weight=weight)
                if weight != 0:
                    file.write(f'{i + 1} {j + 1} {weight}\n')

def calc_networkx_graph(node_node_weight: List[List[int]], num_nodes: int) -> nx.Graph():
    g = nx.Graph()
    nodes = list(range(num_nodes))
    g.add_nodes_from(nodes)
    for i, j, weight in node_node_weight:
        g.add_edge(i, j, weight=weight)
    return g


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


# def plot_figs(scoress: List[List[int]], num_steps: int, labels: List[str]):
#     num = len(scoress)
#     x = list(range(num_steps))
#     dic = {'0': 'ro', '1': 'gs', '2': 'b^', '3': 'c>', '4': 'm<', '5': 'yp'}
#     for i in range(num):
#         plt(x, scoress[i], dic[str(i)], labels[i])
#     plt.legend(labels, loc=0)
#     plt.show()

def plot_fig(scores: List[int], label: str):
    import matplotlib.pyplot as plt
    plt.figure()
    x = list(range(len(scores)))
    dic = {'0': 'ro-', '1': 'gs', '2': 'b^', '3': 'c>', '4': 'm<', '5': 'yp'}
    plt.plot(x, scores, dic['0'])
    plt.legend([label], loc=0)
    plt.savefig('result/' + label + '.png')
    plt.show()

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

# if the file name is 'data/syn_10_27.txt', the return is 'result/syn_10_27'
def calc_result_file_name(file: str):
    new_file = copy.deepcopy(file)
    if 'data' in new_file:
        new_file = new_file.replace('data', 'result')
    new_file = new_file.split('.')[0]
    new_file = new_file.split('/')[0] + '/' + new_file.split('/')[-1]
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

    avg_obj = np.average(objs)
    std_obj = np.std(objs)
    avg_running_duration = np.average(running_durations)
    avg_gap = np.average(gaps)
    avg_obj_bound = np.average(obj_bounds)
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
def float_to_binary(value: float) -> int:
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

# transfer result file,
# e.g.,
# x[0]: 1.0
# x[1]: 0.0
# x[2]: 1.0
# to
# 1 2
# 2 1
# 3 2
def transfer_write_solver_result(filename: str, new_filename: str):
    # assert '.txt' in filename
    nodes = []
    values = []
    with open(filename, 'r') as file:
        find_x = False
        while True:
            line = file.readline()
            if 'x[' in line:
                find_x = True
                node = int(line.split('x[')[1].split(']')[0])
                value = float(line.split(':')[1].split('\n')[0])
                value = float_to_binary(value)
                nodes.append(node)
                values.append(value)
            if find_x and 'x[' not in line:
                break
    with open(new_filename, 'w', encoding="UTF-8") as file:
        for i in range(len(nodes)):
            file.write(f'{nodes[i] + 1} {values[i] + 1}\n')

# For example, syn_10_21_3600.sov, the prefix is 'syn_10_', time_limit is 3600 (seconds).
# extension is '.txt' or '.sta'
def transfer_write_solver_results(directory: str, prefixes: List[str], time_limits: List[int], from_extension: str, to_extension: str):
    for i in range(len(prefixes)):
        for k in range(len(time_limits)):
            suffix = str(int(time_limits[k]))
            files = calc_files_with_prefix_suffix(directory, prefixes[i], suffix, from_extension)
            for filename in files:
                new_filename = filename.split('.')[0] + to_extension
                transfer_write_solver_result(filename, new_filename)

# e.g., rename 'txt' files in directory to 'sta'
def rename_files(directory: str, orig: str, dest: str):
    files = os.listdir(directory)
    for file in files:
        filename = directory + '/' + file
        if orig in filename:
            new_filename = filename.replace(orig, dest)
            os.rename(filename, new_filename)


if __name__ == '__main__':
    read_txt = True
    if read_txt:
        graph1 = read_txt_as_networkx_graph('data/gset/gset_14.txt')
        graph2 = read_txt_as_networkx_graph('data/syn_5_5.txt')

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
        adj_matrix, graph3 = generate_write_symmetric_adjacency_matrix_and_networkx_graph(6, 8)
        graph4 = read_txt_as_networkx_graph('data/syn_6_8.txt')
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
                generate_write_symmetric_adjacency_matrix_and_networkx_graph(num_nodes, num_edges + n)
        print()


    # directory = 'result'
    # prefix = 'syn_10_'
    # time_limit = 3600
    # avg_std = calc_avg_std_of_obj(directory, prefix, time_limit)

    directory_result = 'result'
    prefixes = ['syn_10_', 'syn_50_', 'syn_100_', 'syn_300_', 'syn_500_', 'syn_700_', 'syn_900_', 'syn_1000_', 'syn_3000_', 'syn_5000_', 'syn_7000_', 'syn_9000_', 'syn_10000_']
    # prefixes = ['syn_10_', 'syn_50_', 'syn_100_']
    time_limits = [0.5 * 3600]
    avgs_stds = calc_avg_std_of_objs(directory_result, prefixes, time_limits)

    filename = 'result/syn_10_21_1800.sta'
    new_filename = 'result/syn_10_21_1800.txt'
    transfer_write_solver_result(filename, new_filename)

    from_extension = '.sov'
    to_extension = '.txt'
    transfer_write_solver_results(directory_result, prefixes, time_limits, from_extension, to_extension)

    print()