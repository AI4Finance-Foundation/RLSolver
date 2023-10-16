import os
import sys
import time
import math
import json
import networkx as nx
import numpy as np
import torch as th
import torch.nn as nn
from tqdm import tqdm
from utils import write_networkx_graph
from utils import calc_networkx_graph
try:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

"""变量名缩写约定
edge: graph 里的edge
每条edge 由端点node0 指向 端点node1 。在无向图里，端点可以随意选取固定顺序
n0: index of 左端点node0 
n1: index of 右端点node1
dt: distance 这条edge 的长度，就是node0 和 node1 的距离

最大割问题 将node 分为两个集合，集合set0 和 集合set1
p0: 端点node0 属于set0 的概率, (1-p0): 端点node0 属于set1 的概率
p1: 端点node1 属于set0 的概率, (1-p1): 端点node1 属于set1 的概率

prob: probability 按node0 的顺序，标记每个节点的概率。是GraphMaxCut 问题的解的概率形式
sln_x: solution_x 按node0 的顺序，标记每个节点所属的集合。 是GraphMaxCut 问题的解的二进制表示
"""

TEN = th.Tensor
INT = th.IntTensor

'''graph'''


def load_graph_from_txt(txt_path: str = './data/gset_14.txt'):
    with open(txt_path, 'r') as file:
        lines = file.readlines()
        lines = [[int(i1) for i1 in i0.split()] for i0 in lines]
    num_nodes, num_edges = lines[0]
    graph = [(n0 - 1, n1 - 1, dt) for n0, n1, dt in lines[1:]]  # node_id “从1开始”改为“从0开始”
    return graph, num_nodes, num_edges


def generate_graph(num_nodes: int, g_type: str):
    graph_types = ['erdos_renyi', 'powerlaw', 'barabasi_albert']
    assert g_type in graph_types

    if g_type == 'erdos_renyi':
        g = nx.erdos_renyi_graph(n=num_nodes, p=0.15)
    elif g_type == 'powerlaw':
        g = nx.powerlaw_cluster_graph(n=num_nodes, m=4, p=0.05)
    elif g_type == 'barabasi_albert':
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


def load_graph(graph_name: str):
    data_dir = './data'
    graph_types = ['erdos_renyi', 'powerlaw', 'barabasi_albert']

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


class EncoderBase64:
    def __init__(self, num_nodes: int):
        self.num_nodes = num_nodes

        self.base_digits = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz_$"
        self.base_num = len(self.base_digits)

    def bool_to_str(self, x_bool: TEN) -> str:
        x_int = int(''.join([('1' if i else '0') for i in x_bool.tolist()]), 2)

        '''bin_int_to_str'''
        base_num = len(self.base_digits)
        x_str = ""
        while True:
            remainder = x_int % base_num
            x_str = self.base_digits[remainder] + x_str
            x_int //= base_num
            if x_int == 0:
                break

        x_str = '\n'.join([x_str[i:i + 120] for i in range(0, len(x_str), 120)])
        return x_str.zfill(math.ceil(self.num_nodes // 6 + 1))

    def str_to_bool(self, x_str: str) -> TEN:
        x_b64 = x_str.replace('\n', '')

        '''b64_str_to_int'''
        x_int = 0
        base_len = len(x_b64)
        for i in range(base_len):
            digit = self.base_digits.index(x_b64[i])
            power = base_len - 1 - i
            x_int += digit * (self.base_num ** power)

        return self.int_to_bool(x_int)

    def int_to_bool(self, x_int: int) -> TEN:
        x_bin: str = bin(x_int)[2:]
        x_bool = th.zeros(self.num_nodes, dtype=th.int8)
        x_bool[-len(x_bin):] = th.tensor([int(i) for i in x_bin], dtype=th.int8)
        return x_bool


'''simulator'''


def build_adjacency_matrix(graph, num_nodes):
    adjacency_matrix = th.empty((num_nodes, num_nodes), dtype=th.float32)
    adjacency_matrix[:] = -1  # 选用-1而非0表示表示两个node之间没有edge相连，避免两个节点的距离为0时出现冲突
    for n0, n1, dt in graph:
        adjacency_matrix[n0, n1] = dt
    return adjacency_matrix


def convert_matrix_to_vector(matrix):
    vector = [row[i + 1:] for i, row in enumerate(matrix)]
    return th.hstack(vector)


def check_adjacency_matrix_vector():
    num_nodes = 32
    graph_types = ['erdos_renyi', 'powerlaw', 'barabasi_albert']

    for g_type in graph_types:
        print(f"g_type {g_type}")
        for i in range(8):
            graph, num_nodes, num_edges = generate_graph(num_nodes=num_nodes, g_type=g_type)
            print(i, num_nodes, num_edges, graph)

            adjacency_matrix = build_adjacency_matrix(graph, num_nodes)  # 邻接矩阵
            adjacency_vector = convert_matrix_to_vector(adjacency_matrix)  # 邻接矩阵的上三角拍平为矢量，传入神经网络
            print(adjacency_vector)


class GraphMaxCutSimulator:  # Markov Chain Monte Carlo Simulator
    def __init__(self, graph_name: str = 'powerlaw_64', gpu_id: int = -1, graph_tuple=None):
        device = th.device(f'cuda:{gpu_id}' if th.cuda.is_available() and gpu_id >= 0 else 'cpu')
        int_type = th.int32
        self.device = device
        self.int_type = int_type

        if graph_tuple:
            graph, num_nodes, num_edges = graph_tuple
        else:
            graph, num_nodes, num_edges = load_graph(graph_name=graph_name)

        # 建立邻接矩阵，不预先保存索引的邻接矩阵不适合GPU并行
        '''
        例如，无向图里：
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
        '''
        adjacency_matrix = th.empty((num_nodes, num_nodes), dtype=th.float32, device=device)
        adjacency_matrix[:] = -1  # 选用-1而非0表示表示两个node之间没有edge相连，避免两个节点的距离为0时出现冲突
        for n0, n1, dt in graph:
            adjacency_matrix[n0, n1] = dt
        assert num_nodes == adjacency_matrix.shape[0] == adjacency_matrix.shape[1]
        assert num_edges == (adjacency_matrix != -1).sum()
        self.adjacency_matrix = adjacency_matrix
        self.adjacency_vector = convert_matrix_to_vector(adjacency_matrix)

        # 建立二维列表n0_to_n1s 表示这个图，
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
        n0_to_n1s = [[] for _ in range(num_nodes)]  # 将 node0_id 映射到 node1_id
        n0_to_dts = [[] for _ in range(num_nodes)]  # 将 mode0_id 映射到 node1_id 与 node0_id 的距离
        for n0, n1, dist in graph:
            n0_to_n1s[n0].append(n1)
            n0_to_dts[n0].append(dist)
        n0_to_n1s = [th.tensor(node1s, dtype=int_type, device=device) for node1s in n0_to_n1s]
        n0_to_dts = [th.tensor(node1s, dtype=int_type, device=device) for node1s in n0_to_dts]
        assert num_nodes == len(n0_to_n1s)
        assert num_nodes == len(n0_to_dts)
        assert num_edges == sum([len(n0_to_n1) for n0_to_n1 in n0_to_n1s])
        assert num_edges == sum([len(n0_to_dt) for n0_to_dt in n0_to_dts])
        self.num_nodes = len(n0_to_n1s)
        self.num_edges = sum([len(n0_to_n1) for n0_to_n1 in n0_to_n1s])

        # 根据二维列表n0_to_n1s 建立基于edge 的node0 node1 的索引，用于高效计算
        """
        在K个子环境里，需要对N个点进行索引去计算计算GraphMaxCut距离：
        - 建立邻接矩阵的方法，计算GraphMaxCut距离时，需要索引K*N次
        - 下面这种方法直接保存并行索引信息，仅需要索引1次

        为了用GPU加速计算，可以用两个固定长度的张量记录端点序号，再用两个固定长度的张量记录端点信息。去表示这个图：
        我们直接将每条edge两端的端点称为：左端点node0 和 右端点node1 （在无向图里，左右端点可以随意命名）
        node0_id   [0, 0, 2]  # 依次保存三条边的node0，用于索引
        node0_prob [p, p, p]  # 依次根据索引得到node0 的概率，用于计算
        node1_id   [1, 2, 3]  # 依次保存三条边的node1，用于索引
        node1_prob [p, p, p]  # 依次根据索引得到node1 的概率，用于计算

        env_id     [0, 1, 2, ..., num_envs]  # 保存了并行维度的索引信息
        """
        n0_ids = []
        n1_ids = []
        for i, n1s in enumerate(n0_to_n1s):
            n0_ids.extend([i, ] * n1s.shape[0])
            n1_ids.extend(n1s)
        self.n0_ids = th.tensor(n0_ids, dtype=int_type, device=device).unsqueeze(0)
        self.n1_ids = th.tensor(n1_ids, dtype=int_type, device=device).unsqueeze(0)
        self.env_is = th.zeros(self.num_edges, dtype=int_type, device=device).unsqueeze(0)

    def get_objectives_using_for_loop(self, probs: TEN) -> TEN:  # 使用for循环重复查找索引，不适合GPU并行
        assert probs.shape[-1] == self.num_nodes
        num_envs = probs.shape[0]

        sum_dts = []
        for env_i in range(num_envs):  # 逐个访问子环境
            p0 = probs[env_i]

            n0_to_p1 = []
            for n0 in range(self.num_nodes):  # 逐个访问节点
                n1s = th.where(self.adjacency_matrix[n0] != -1)[0]  # 根据邻接矩阵，找出与node0 相连的多个节点的索引
                p1 = p0[n1s]  # 根据索引找出node1 属于集合的概率
                n0_to_p1.append(p1)

            sum_dt = []
            for _p0, _p1 in zip(p0, n0_to_p1):
                # `_p0 * (1-_p1)` node_0 属于这个集合 且 node1 属于那个集合的概率
                # `_p1 * (1-_p0)` node_1 属于这个集合 且 node0 属于那个集合的概率
                # dt = _p0 * (1-_p1) + _p1 * (1-_p0)  # 等价于以下一行代码，相加计算出了这条边两端的节点分别属于两个集合的概率
                dt = _p0 + _p1 - 2 * _p0 * _p1
                # 此计算只能算出的局部梯度，与全局梯度有差别，未考虑无向图里节点间的复杂关系，需要能跳出局部最优的求解器
                sum_dt.append(dt.sum(dim=0))
            sum_dt = th.stack(sum_dt).sum(dim=-1)  # 求和得到这个子环境的 objective
            sum_dts.append(sum_dt)
        sum_dts = th.hstack(sum_dts)  # 堆叠结果，得到 num_envs 个子环境的 objective
        return -sum_dts

    def get_objectives(self, probs: TEN):
        p0s, p1s = self.get_p0s_p1s(probs)
        return -(p0s + p1s - 2 * p0s * p1s).sum(1)

    def get_scores(self, probs: INT) -> INT:
        p0s, p1s = self.get_p0s_p1s(probs)
        return (p0s ^ p1s).sum(1)

    def get_p0s_p1s(self, probs: TEN) -> (TEN, TEN):
        num_envs = probs.shape[0]
        if num_envs != self.env_is.shape[0]:
            self.n0_ids = self.n0_ids[0].repeat(num_envs, 1)
            self.n1_ids = self.n1_ids[0].repeat(num_envs, 1)
            self.env_is = self.env_is[0:1] + th.arange(num_envs, device=self.device).unsqueeze(1)

        p0s = probs[self.env_is, self.n0_ids]
        p1s = probs[self.env_is, self.n1_ids]
        return p0s, p1s

    def get_rand_probs(self, num_envs: int) -> TEN:
        return th.rand((num_envs, self.num_nodes), dtype=th.float32, device=self.device)

    @staticmethod
    def prob_to_bool(p0s, thresh=0.5):
        return p0s > thresh


def draw_adjacency_matrix():
    graph_name = 'powerlaw_48'
    env = GraphMaxCutSimulator(graph_name=graph_name)
    ary = (env.adjacency_matrix != -1).to(th.int).data.cpu().numpy()

    d0 = d1 = ary.shape[0]
    if plt:
        plt.imshow(1 - ary[:, ::-1], cmap='hot', interpolation='nearest', extent=[0, d1, 0, d0])
        plt.gca().set_xticks(np.arange(0, d1, 1))
        plt.gca().set_yticks(np.arange(0, d0, 1))
        plt.grid(True, color='grey', linewidth=1)
        plt.title('black denotes connect')
        plt.show()


def check_simulator_encoder():
    th.manual_seed(0)
    num_envs = 6
    graph_name = 'powerlaw_64'

    sim = GraphMaxCutSimulator(graph_name=graph_name)
    enc = EncoderBase64(num_nodes=sim.num_nodes)

    probs = sim.get_rand_probs(num_envs=num_envs)
    print(sim.get_objectives(probs))

    best_score = -th.inf
    best_str_x = ''
    for _ in range(8):
        probs = sim.get_rand_probs(num_envs=num_envs)
        sln_xs = sim.prob_to_bool(probs)
        scores = sim.get_scores(sln_xs)

        max_score, max_id = th.max(scores, dim=0)
        if max_score > best_score:
            best_score = max_score
            best_sln_x = sln_xs[max_id]
            best_str_x = enc.bool_to_str(best_sln_x)
            print(f"best_score {best_score}  best_sln_x {best_str_x}")

    best_sln_x = enc.str_to_bool(best_str_x)
    best_score = sim.get_scores(best_sln_x.unsqueeze(0)).squeeze(0)
    print(f"NumNodes {sim.num_nodes}  NumEdges {sim.num_edges}")
    print(f"score {best_score}  sln_x \n{enc.bool_to_str(best_sln_x)}")


def check_convert_sln_x():
    gpu_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    num_envs = 1
    graph_name = 'powerlaw_64'

    sim = GraphMaxCutSimulator(graph_name=graph_name, gpu_id=gpu_id)
    enc = EncoderBase64(num_nodes=sim.num_nodes)

    x_prob = sim.get_rand_probs(num_envs=num_envs)[0]
    x_bool = sim.prob_to_bool(x_prob)

    x_str = enc.bool_to_str(x_bool)
    print(x_str)
    x_bool = enc.str_to_bool(x_str)

    assert all(x_bool == sim.prob_to_bool(x_prob))


def exhaustion_search():
    gpu_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    graph_name = 'powerlaw_64'
    num_envs = 2 ** 14
    th.manual_seed(0)

    th.set_grad_enabled(False)

    sim = GraphMaxCutSimulator(graph_name=graph_name, gpu_id=gpu_id)
    enc = EncoderBase64(num_nodes=sim.num_nodes)

    best_score = -th.inf
    best_sln_x = None
    dim = sim.num_nodes

    num_iter = 2 ** (dim - 1)
    print(f"NumNodes {sim.num_nodes}  NumEdges {sim.num_edges}  Total search num_iter {num_iter}")
    time.sleep(0.1)
    _num_envs = min(num_iter, num_envs)
    i_iter = tqdm(range(0, num_iter, _num_envs), ascii=True)
    all_score = np.empty(num_iter, dtype=np.int16)
    for i in i_iter:
        sln_xs = [enc.int_to_bool(i + j) for j in range(_num_envs)]
        sln_xs = th.stack(sln_xs).to(sim.device)
        scores = sim.get_scores(sln_xs)

        max_score, max_id = th.max(scores, dim=0)
        if max_score > best_score:
            best_score = max_score
            best_sln_x = sln_xs[max_id]
            i_iter.set_description(
                f"best_score {best_score:6.0f}  best_sln_x {enc.bool_to_str(best_sln_x)}")
            print()

        all_score[i:i + _num_envs] = scores.data.cpu().numpy()
    i_iter.close()

    num_best = np.count_nonzero(all_score == best_score.item())
    print(f"NumNodes {sim.num_nodes}  NumEdges {sim.num_edges}  NumSearch {2 ** dim}  "
          f"best score {best_score:6.0f}  sln_x {enc.bool_to_str(best_sln_x)}  count 2*{num_best}")


"""find solution_x using optimizer"""


class OptimizerLSTM(nn.Module):
    def __init__(self, inp_dim, mid_dim, out_dim, num_layers):
        super().__init__()
        self.inp_dim = inp_dim
        self.mid_dim = mid_dim
        self.out_dim = out_dim
        self.num_layers = num_layers

        self.rnn0 = nn.LSTM(inp_dim, mid_dim, num_layers=num_layers)
        self.mlp0 = nn.Linear(mid_dim, out_dim)
        self.rnn1 = nn.LSTM(1, 8, num_layers=num_layers)
        self.mlp1 = nn.Linear(8, 1)

    def forward(self, inp, hid0=None, hid1=None):
        tmp0, hid0 = self.rnn0(inp, hid0)
        out0 = self.mlp0(tmp0)

        d0, d1, d2 = inp.shape
        inp1 = inp.reshape(d0, d1 * d2, 1)
        tmp1, hid1 = self.rnn1(inp1, hid1)
        out1 = self.mlp1(tmp1).reshape(d0, d1, d2)

        out = out0 + out1
        return out, hid0, hid1


class OptimizerLSTM2(nn.Module):
    def __init__(self, inp_dim, mid_dim, out_dim, num_layers):
        super().__init__()
        inp_dim0 = inp_dim[0]
        inp_dim1 = inp_dim[1]

        self.rnn0 = nn.LSTM(inp_dim0, mid_dim, num_layers=num_layers)
        self.enc0 = nn.Sequential(nn.Linear(inp_dim1, mid_dim), nn.ReLU(),
                                  nn.Linear(mid_dim, mid_dim))  # todo graph dist
        self.mlp0 = nn.Sequential(nn.Linear(mid_dim + mid_dim, mid_dim), nn.ReLU(),
                                  nn.Linear(mid_dim, out_dim))

        self.rnn1 = nn.LSTM(1, 8, num_layers=num_layers)
        self.mlp1 = nn.Linear(8, 1)

    def forward(self, inp, graph, hid0=None, hid1=None):
        d0, d1, d2 = inp.shape
        inp1 = inp.reshape(d0, d1 * d2, 1)
        tmp1, hid1 = self.rnn1(inp1, hid1)
        out1 = self.mlp1(tmp1).reshape(d0, d1, d2)

        tmp0, hid0 = self.rnn0(inp, hid0)

        graph = self.enc0(graph) * th.ones((d0, d1, 1), device=tmp0.device)
        tmp0 = th.concat((tmp0, graph), dim=2)  # todo graph dist
        out0 = self.mlp0(tmp0)

        out = out0 + out1
        return out, hid0, hid1


class Evaluator:
    def __init__(self, sim, enc):
        self.start_time = time.time()
        self.best_score = -th.inf
        self.best_sln_x = 'none'
        self.get_scores = sim.get_scores
        self.bool_to_str = enc.bool_to_str

    def evaluate_and_print(self, sln_xs, i, obj):
        scores = self.get_scores(sln_xs)
        used_time = time.time() - self.start_time

        max_score, max_id = th.max(scores, dim=0)
        if max_score > self.best_score:
            self.best_score = max_score
            self.best_sln_x = sln_xs[max_id]
            print(f"best_score {self.best_score}  best_sln_x \n{self.bool_to_str(self.best_sln_x)}")

        print(f"|{used_time:9.0f}  {i:6}  {obj.item():9.3f}  "
              f"max_score {max_score.item():9.0f}  "
              f"best_score {self.best_score:9.0f}")


class Agent:  # Demo
    def __init__(self, graph_name='powerlaw_64', gpu_id: int = 0, json_path: str = '', graph_tuple=None):
        pass

        """数据超参数（路径，存储，预处理）"""
        self.json_path = f'./GraphMaxCut_{graph_name}.json'  # 存放超参数的json文件。将会保存所有类似为 str int float 的参数
        self.graph_name = graph_name
        self.gpu_id = gpu_id

        '''GPU memory'''
        self.num_envs = 2 ** 6
        self.mid_dim = 2 ** 6
        self.num_layers = 2
        self.seq_len = 2 ** 6
        self.reset_gap = 2 ** 6
        self.learning_rate = 1e-3

        '''train and evaluate'''
        self.num_opti = 2 ** 16
        self.eval_gap = 2 ** 2
        self.save_dir = f"./result_{graph_name}_{gpu_id}"

        if graph_tuple is None:
            if os.path.exists(json_path):
                self.load_from_json(json_path=json_path)
                self.save_as_json(json_path=json_path)
            vars_str = str(vars(self)).replace(", '", ", \n'")
            print(f"| Config\n{vars_str}")

        """agent"""
        '''init simulator'''
        self.sim = GraphMaxCutSimulator(graph_name=graph_name, gpu_id=gpu_id, graph_tuple=graph_tuple)
        self.enc = EncoderBase64(num_nodes=self.sim.num_nodes)
        self.num_nodes = self.sim.num_nodes
        self.num_edges = self.sim.num_edges

        '''init optimizer'''
        self.device = th.device(f'cuda:{gpu_id}' if th.cuda.is_available() and gpu_id >= 0 else 'cpu')
        self.opt_opti = OptimizerLSTM(
            inp_dim=self.num_nodes,
            mid_dim=self.mid_dim,
            out_dim=self.num_nodes,
            num_layers=self.num_layers
        ).to(self.device)
        self.opt_base = th.optim.Adam(self.opt_opti.parameters(), lr=self.learning_rate)

        '''init evaluator'''
        self.evaluator = Evaluator(sim=self.sim, enc=self.enc)

    def iter_reset(self):
        probs = self.sim.get_rand_probs(num_envs=self.num_envs)
        probs.requires_grad_(True)
        hidden0 = None
        hidden1 = None
        return probs, hidden0, hidden1

    def iter_loop(self, probs, hidden0, hidden1):
        prob_ = probs.clone()
        updates = []

        for j in range(self.seq_len):
            obj = self.sim.get_objectives(probs).mean()
            obj.backward()

            grad_s = probs.grad.data
            update, hidden0, hidden1 = self.opt_opti(grad_s.unsqueeze(0), hidden0, hidden1)
            update = (update.squeeze_(0) - grad_s) * self.learning_rate
            updates.append(update)
            probs.data.add_(update).clip_(0, 1)
        hidden0 = [h.detach() for h in hidden0]
        hidden1 = [h.detach() for h in hidden1]

        updates = th.stack(updates, dim=0)
        prob_ = (prob_ + updates.mean(0)).clip(0, 1)
        return prob_, hidden0, hidden1

    def search(self, j):
        probs = self.sim.get_rand_probs(num_envs=self.num_envs)
        probs.requires_grad_(True)
        hidden0 = None
        hidden1 = None

        total_i = j * self.reset_gap
        for i in range(self.reset_gap):
            prob_, hidden0, hidden1 = self.iter_loop(probs=probs, hidden0=hidden0, hidden1=hidden1)

            obj_ = self.sim.get_objectives(prob_).mean()
            self.opt_base.zero_grad()
            obj_.backward()
            self.opt_base.step()

            probs.data[:] = prob_

            if i % self.eval_gap:
                sln_xs = self.sim.prob_to_bool(probs)
                self.evaluator.evaluate_and_print(sln_xs=sln_xs, i=total_i + i, obj=obj_)

        best_sln_x = self.evaluator.best_sln_x
        best_score = self.evaluator.best_score
        return best_sln_x, best_score

    def load_from_json(self, json_path: str):
        with open(json_path, "r") as file:
            json_dict = json.load(file)
        for key, value in json_dict.items():
            setattr(self, key, value)

    def save_as_json(self, json_path: str):
        json_dict = dict([(key, value) for key, value in vars(self).items()
                          if type(value) in {str, int, float}])
        print(json_dict)
        with open(json_path, "w") as file:
            json.dump(json_dict, file, indent=4)


class AgentDist(Agent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.opt_opti = OptimizerLSTM2(
            inp_dim=(self.num_nodes, self.sim.adjacency_vector.shape[0]),
            mid_dim=self.mid_dim,
            out_dim=self.num_nodes,
            num_layers=self.num_layers
        ).to(self.device)
        self.opt_base = th.optim.Adam(self.opt_opti.parameters(), lr=self.learning_rate)

    def iter_loop(self, probs, hidden0, hidden1):
        prob_ = probs.clone()
        updates = []
        graph_tensor = self.sim.adjacency_vector

        for j in range(self.seq_len):
            obj = self.sim.get_objectives(probs).mean()
            obj.backward()

            grad_s = probs.grad.data
            update, hidden0, hidden1 = self.opt_opti(
                grad_s.unsqueeze(0),
                graph_tensor.unsqueeze(0),
                hidden0,
                hidden1,
            )  # todo graph dist
            update = (update.squeeze_(0) - grad_s) * self.learning_rate
            updates.append(update)
            probs.data.add_(update).clip_(0, 1)
        hidden0 = [h.detach() for h in hidden0]
        hidden1 = [h.detach() for h in hidden1]

        updates = th.stack(updates, dim=0)
        prob_ = (prob_ + updates.mean(0)).clip(0, 1)
        return prob_, hidden0, hidden1

    def search_for_valid(self, j):
        self.opt_base = None

        probs = self.sim.get_rand_probs(num_envs=self.num_envs)
        probs.requires_grad_(True)
        hidden0 = None
        hidden1 = None

        total_i = j * self.reset_gap
        for i in range(self.reset_gap):
            prob_, hidden0, hidden1 = self.iter_loop(probs=probs, hidden0=hidden0, hidden1=hidden1)
            probs.data[:] = prob_

            if i % self.eval_gap:
                sln_xs = self.sim.prob_to_bool(probs)
                self.evaluator.evaluate_and_print(sln_xs=sln_xs, i=total_i + i, obj=th.tensor(0))

        best_sln_x = self.evaluator.best_sln_x
        best_score = self.evaluator.best_score
        return best_sln_x, best_score


def run_v32_find_xs_using_opti():
    gpu_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    num_nodes = 48
    graph_name = f"powerlaw_{num_nodes}"

    # todo graph dist
    num_valid, seed_num = 100, 0
    np.random.seed(seed_num)
    th.manual_seed(seed_num)
    g_type, num_nodes = graph_name.split('_')
    num_nodes = int(num_nodes)
    graphs = []
    agents = []
    directory = 'data/syndistri/'
    for i in range(num_valid):
        graph, num_nodes, num_edges = generate_graph(num_nodes=num_nodes, g_type=g_type)
        filename = directory + "syndistri_" + str(num_nodes) + '_' + str(num_edges) + '.txt'
        g = calc_networkx_graph(graph, num_nodes)
        write_networkx_graph(g, filename)
        agent = AgentDist(graph_name=graph_name, gpu_id=gpu_id, json_path='auto_build',
                          graph_tuple=(graph, num_nodes, num_edges))

        graphs.append(graph)
        agents.append(agent)
    valid_opt_opti = agents[0].opt_opti
    for i in range(1, num_valid):
        agent = agents[i]
        agent.opt_opti = valid_opt_opti
        agent.opt_base = None

    agent = AgentDist(graph_name=graph_name, gpu_id=gpu_id, json_path='auto_build')
    valid_opt_opti.load_state_dict(agent.opt_opti.state_dict())

    best_sln_xs: list = [0, ] * num_valid
    best_scores: list = [0, ] * num_valid
    for j in range(agent.num_opti):
        agent.sim.__init__(graph_name=graph_name, gpu_id=gpu_id)
        _best_sln_x, _best_score = agent.search(j=j)

        valid_opt_opti.load_state_dict(agent.opt_opti.state_dict())
        for j_valid in range(num_valid):
            _agent = agents[j_valid]
            best_sln_x, best_score = _agent.search_for_valid(j=j)
            best_sln_xs[j_valid] = best_sln_x
            best_scores[j_valid] = best_score
        print(f"| best_scores.avg {sum(best_scores) / len(best_scores)}")

    print(f"\nbest_sln_x {best_sln_xs}"
          f"\nbest_score {best_scores}")


if __name__ == '__main__':
    run_v32_find_xs_using_opti()
