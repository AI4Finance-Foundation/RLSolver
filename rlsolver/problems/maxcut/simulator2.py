import os
import sys
import time
import torch as th
import networkx as nx
from typing import List, Tuple

'''graph'''

TEN = th.Tensor
GraphList = List[Tuple[int, int, int]]
IndexList = List[List[int]]


def load_graph_from_txt(txt_path: str = 'G14.txt') -> GraphList:
    with open(txt_path, 'r') as file:
        lines = file.readlines()
        lines = [[int(i1) for i1 in i0.split()] for i0 in lines]
    num_nodes, num_edges = lines[0]
    graph = [(n0 - 1, n1 - 1, dt) for n0, n1, dt in lines[1:]]  # 将node_id 由“从1开始”改为“从0开始”

    assert num_nodes == obtain_num_nodes(graph=graph)
    assert num_edges == len(graph)
    return graph


def generate_graph(graph_type: str, num_nodes: int) -> GraphList:
    graph_types = ['erdos_renyi', 'powerlaw', 'barabasi_albert']
    assert graph_type in graph_types

    if graph_type == 'erdos_renyi':
        g = nx.erdos_renyi_graph(n=num_nodes, p=0.15)
    elif graph_type == 'powerlaw':
        g = nx.powerlaw_cluster_graph(n=num_nodes, m=4, p=0.05)
    elif graph_type == 'barabasi_albert':
        g = nx.barabasi_albert_graph(n=num_nodes, m=4)
    else:
        raise ValueError(f"g_type {graph_type} should in {graph_types}")

    distance = 1
    graph = [(node0, node1, distance) for node0, node1 in g.edges]
    return graph


def load_graph(graph_name: str):
    import random
    data_dir = './data'
    graph_types = ['erdos_renyi', 'powerlaw', 'barabasi_albert']

    if os.path.exists(f"{data_dir}/{graph_name}.txt"):
        txt_path = f"{data_dir}/{graph_name}.txt"
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
        raise ValueError(f"graph_name {graph_name}")
    return graph


def obtain_num_nodes(graph: GraphList) -> int:
    return max([max(n0, n1) for n0, n1, distance in graph]) + 1


def build_adjacency_matrix(graph: GraphList, if_bidirectional: bool = False):
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
    num_nodes = obtain_num_nodes(graph=graph)

    adjacency_matrix = th.zeros((num_nodes, num_nodes), dtype=th.float32)
    adjacency_matrix[:] = not_connection
    for n0, n1, distance in graph:
        adjacency_matrix[n0, n1] = distance
        if if_bidirectional:
            adjacency_matrix[n1, n0] = distance
    return adjacency_matrix


def build_adjacency_indies(graph: GraphList, if_bidirectional: bool = False) -> (IndexList, IndexList):
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
    num_nodes = obtain_num_nodes(graph=graph)

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


def get_gpu_info_str(device) -> str:
    if not th.cuda.is_available():
        return 'th.cuda.is_available() == False'

    total_memory = th.cuda.get_device_properties(device).total_memory / (1024 ** 3)  # GB
    max_allocated = th.cuda.max_memory_allocated(device) / (1024 ** 3)  # GB
    memory_allocated = th.cuda.memory_allocated(device) / (1024 ** 3)  # GB

    return (f"RAM(GB) {memory_allocated:.2f} < {max_allocated:.2f} < {total_memory:.2f}  "
            f"Rate {(max_allocated / total_memory):5.2f}")


'''simulator'''


class SimulatorGraphMaxCut:
    def __init__(self, graph: GraphList, device=th.device('cpu'), if_bidirectional: bool = False):
        self.device = device
        self.int_type = int_type = th.long
        self.if_bidirectional = if_bidirectional

        '''建立邻接矩阵'''
        self.adjacency_matrix = build_adjacency_matrix(graph=graph, if_bidirectional=if_bidirectional).to(device)

        '''建立邻接索引'''
        n0_to_n1s, n0_to_dts = build_adjacency_indies(graph=graph, if_bidirectional=if_bidirectional)
        n0_to_n1s = [t.to(int_type).to(device) for t in n0_to_n1s]
        self.num_nodes = obtain_num_nodes(graph)
        self.num_edges = len(graph)
        self.adjacency_indies = n0_to_n1s

        '''基于邻接索引，建立基于边edge的索引张量：(n0_ids, n1_ids)是所有边(第0个, 第1个)端点的索引'''
        n0_to_n0s = [(th.zeros_like(n1s) + i) for i, n1s in enumerate(n0_to_n1s)]
        self.n0_ids = th.hstack(n0_to_n0s)[None, :]
        self.n1_ids = th.hstack(n0_to_n1s)[None, :]
        len_sim_ids = self.num_edges * (2 if if_bidirectional else 1)
        self.sim_ids = th.zeros(len_sim_ids, dtype=int_type, device=device)[None, :]
        self.n0_num_n1 = th.tensor([n1s.shape[0] for n1s in n0_to_n1s], device=device)[None, :]

    def calculate_obj_values(self, xs: TEN, if_sum: bool = True) -> TEN:
        num_sims = xs.shape[0]
        if num_sims != self.sim_ids.shape[0]:
            self.n0_ids = self.n0_ids[0].repeat(num_sims, 1)
            self.n1_ids = self.n1_ids[0].repeat(num_sims, 1)
            self.sim_ids = self.sim_ids[0:1] + th.arange(num_sims, dtype=self.int_type, device=self.device)[:, None]

        values = xs[self.sim_ids, self.n0_ids] ^ xs[self.sim_ids, self.n1_ids]
        if if_sum:
            values = values.sum(1)
        if self.if_bidirectional:
            values = values.float() / 2
        return values

    def calculate_obj_values_for_loop(self, xs: TEN, if_sum: bool = True) -> TEN:  # 有更高的并行度，但计算耗时增加一倍。
        num_sims, num_nodes = xs.shape
        values = th.zeros((num_sims, num_nodes), dtype=self.int_type, device=self.device)
        for node0 in range(num_nodes):
            node1s = self.adjacency_indies[node0]
            if node1s.shape[0] > 0:
                values[:, node0] = (xs[:, node0, None] ^ xs[:, node1s]).sum(dim=1)

        if if_sum:
            values = values.sum(dim=1)
        if self.if_bidirectional:
            values = values.float() / 2
        return values

    def generate_xs_randomly(self, num_sims):
        xs = th.randint(0, 2, size=(num_sims, self.num_nodes), dtype=th.bool, device=self.device)
        xs[:, 0] = 0
        return xs


def check_simulator():
    gpu_id = -1
    num_sims = 16
    num_nodes = 24
    graph_name = f'powerlaw_{num_nodes}'

    graph = load_graph(graph_name=graph_name)
    device = th.device(f'cuda:{gpu_id}' if th.cuda.is_available() and gpu_id >= 0 else 'cpu')
    simulator = SimulatorGraphMaxCut(graph=graph, device=device)

    for i in range(8):
        xs = simulator.generate_xs_randomly(num_sims=num_sims)
        obj = simulator.calculate_obj_values(xs=xs)
        print(f"| {i}  max_obj_value {obj.max().item()}")
    pass


def find_best_num_sims():
    gpu_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0

    calculate_obj_func = 'calculate_obj_values'
    graph_name = 'gset_14'
    num_sims = 2 ** 16
    num_iter = 2 ** 6
    # calculate_obj_func = 'calculate_obj_values_for_loop'
    # graph_name = 'gset_14'
    # num_sims = 2 ** 13
    # num_iter = 2 ** 9

    if os.name == 'nt':
        graph_name = 'powerlaw_64'
        num_sims = 2 ** 4
        num_iter = 2 ** 3

    graph = load_graph(graph_name=graph_name)
    device = th.device(f'cuda:{gpu_id}' if th.cuda.is_available() and gpu_id >= 0 else 'cpu')
    simulator = SimulatorGraphMaxCut(graph=graph, device=device, if_bidirectional=False)

    print('find the best num_sims')
    from math import ceil
    for j in (1, 1, 1, 1.5, 2, 3, 4, 6, 8, 12, 16, 24, 32):
        _num_sims = int(num_sims * j)
        _num_iter = ceil(num_iter * num_sims / _num_sims)

        timer = time.time()
        for i in range(_num_iter):
            xs = simulator.generate_xs_randomly(num_sims=_num_sims)
            vs = getattr(simulator, calculate_obj_func)(xs=xs)
            assert isinstance(vs, TEN)
            # print(f"| {i}  max_obj_value {vs.max().item()}")
        print(f"_num_iter {_num_iter:8}  "
              f"_num_sims {_num_sims:8}  "
              f"UsedTime {time.time() - timer:9.3f}  "
              f"GPU {get_gpu_info_str(device)}")
    """
'''calculate_obj_values'''
find the best num_sims
_num_iter      512  _num_sims     8192  UsedTime     3.189  GPU RAM(GB) 1.73 < 2.52 < 10.75  Rate  0.23
_num_iter      512  _num_sims     8192  UsedTime     4.141  GPU RAM(GB) 1.73 < 2.52 < 10.75  Rate  0.23
_num_iter      512  _num_sims     8192  UsedTime     4.140  GPU RAM(GB) 1.73 < 2.52 < 10.75  Rate  0.23
_num_iter      342  _num_sims    12288  UsedTime     3.632  GPU RAM(GB) 2.59 < 3.77 < 10.75  Rate  0.35
_num_iter      256  _num_sims    16384  UsedTime     3.624  GPU RAM(GB) 3.45 < 5.03 < 10.75  Rate  0.47
_num_iter      171  _num_sims    24576  UsedTime     3.247  GPU RAM(GB) 5.18 < 7.54 < 10.75  Rate  0.70

'''calculate_obj_values_for_loop (lower effective, lower GPU RAM, higher parallel)'''
find the best num_sims
_num_iter       64  _num_sims    65536  UsedTime     7.018  GPU RAM(GB) 0.05 < 0.52 < 10.75  Rate  0.05
_num_iter       64  _num_sims    65536  UsedTime     6.965  GPU RAM(GB) 0.05 < 0.52 < 10.75  Rate  0.05
_num_iter       64  _num_sims    65536  UsedTime     6.962  GPU RAM(GB) 0.05 < 0.52 < 10.75  Rate  0.05
_num_iter       43  _num_sims    98304  UsedTime     6.887  GPU RAM(GB) 0.08 < 0.77 < 10.75  Rate  0.07
_num_iter       32  _num_sims   131072  UsedTime     6.815  GPU RAM(GB) 0.10 < 1.03 < 10.75  Rate  0.10
_num_iter       22  _num_sims   196608  UsedTime     6.957  GPU RAM(GB) 0.15 < 1.54 < 10.75  Rate  0.14
_num_iter       16  _num_sims   262144  UsedTime     6.681  GPU RAM(GB) 0.20 < 2.06 < 10.75  Rate  0.19
_num_iter       11  _num_sims   393216  UsedTime     6.836  GPU RAM(GB) 0.30 < 3.08 < 10.75  Rate  0.29
_num_iter        8  _num_sims   524288  UsedTime     6.594  GPU RAM(GB) 0.40 < 4.11 < 10.75  Rate  0.38
_num_iter        6  _num_sims   786432  UsedTime     7.597  GPU RAM(GB) 0.59 < 6.16 < 10.75  Rate  0.57
_num_iter        4  _num_sims  1048576  UsedTime     6.716  GPU RAM(GB) 0.79 < 8.21 < 10.75  Rate  0.76
    """


if __name__ == '__main__':
    check_simulator()
