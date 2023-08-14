import os
import sys
import time
import math
import json
import networkx as nx
import numpy as np
import torch as th
import torch.nn as nn
from torch.distributions import Bernoulli  # BinaryDist
from tqdm import tqdm
from utils import generate_graph
from utils import load_graph
from utils import load_graph_from_txt
from mcmc_sim.mcmc_sim2 import MCMCSim2
from mcmc_sim.mcmc_sim2 import convert_matrix_to_vector

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




def draw_adjacency_matrix():
    graph_name = 'syn_20_42'
    env = MCMCSim2(graph_name=graph_name)
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

    sim = MCMCSim2(graph_name=graph_name)
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

    sim = MCMCSim2(graph_name=graph_name, gpu_id=gpu_id)
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

    sim = MCMCSim2(graph_name=graph_name, gpu_id=gpu_id)
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


class UniqueBuffer:  # for GraphMaxCut
    def __init__(self, max_size: int, num_nodes: int, gpu_id: int = 0):
        self.max_size = max_size
        self.device = th.device(f"cuda:{gpu_id}" if (th.cuda.is_available() and (gpu_id >= 0)) else "cpu")

        self.sln_xs = th.empty((max_size, num_nodes), dtype=th.bool, device=self.device)
        self.scores = th.empty(max_size, dtype=th.int64, device=self.device)
        self.min_score = -th.inf
        self.min_index = None

        enc = EncoderBase64(num_nodes=num_nodes)
        self.bool_to_str = enc.bool_to_str

    def update(self, sln_x, score):
        if (score < self.min_score) or th.any(th.all(sln_x.unsqueeze(0) == self.sln_xs, dim=1)):
            return False

        self.sln_xs[self.min_index] = sln_x
        self.scores[self.min_index] = score
        self.min_score, self.min_index = th.min(self.scores, dim=0)
        return True

    def save_or_load_history(self, cwd: str, if_save: bool):

        item_paths = (
            (self.sln_xs, f"{cwd}/buffer_sln_xs.npz"),
            (self.scores, f"{cwd}/buffer_scores.npz"),
        )

        if if_save:
            # print(f"| buffer.save_or_load_history(): Save {cwd}")
            for item, path in item_paths:
                np.savez_compressed(path, arr=item.data.cpu().numpy())

        elif all([os.path.isfile(path) for item, path in item_paths]):
            for item, path in item_paths:
                buf_item = np.load(path)['arr']
                print(f"| buffer.save_or_load_history(): Load {path}    {buf_item.shape}")

                max_size = buf_item.shape[0]
                max_size = min(self.max_size, max_size)
                item[:max_size] = th.tensor(buf_item[-max_size:], dtype=item.dtype, device=item.device)
            self.min_score, self.min_index = th.min(self.scores, dim=0)

    def init_with_random(self, sim):
        probs = sim.get_rand_probs(num_envs=self.max_size)
        sln_xs = sim.prob_to_bool(probs)
        scores = sim.get_scores(sln_xs)
        self.sln_xs[:] = sln_xs
        self.scores[:] = scores
        self.min_score, self.min_index = th.min(scores, dim=0)

    def print_sln_x_str(self):
        print(f"{'score':>8}  {'sln_x':8}")

        sort_ids = th.argsort(self.scores)
        for i in sort_ids:
            sln_x = self.sln_xs[i]
            score = self.scores[i]

            sln_x_str = self.bool_to_str(sln_x)
            enter_str = '\n' if len(sln_x_str) > 60 else ''
            print(f"score {score:8}  {enter_str}{sln_x_str}")

        ys = self.scores.sort().values.data.cpu().numpy()
        xs = th.arange(ys.shape[0]).data.cpu().numpy()
        if plt:
            plt.scatter(xs, ys)
            plt.title(f"max score {ys.max().item()}  top {ys.shape[0]}")
            plt.grid()
            plt.show()
        else:
            print(f"max score {ys.max().item()}  top {ys.shape[0]}")


def check_unique_buffer():
    gpu_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    graph_name = 'powerlaw_64'

    sim = MCMCSim2(graph_name=graph_name, gpu_id=gpu_id)
    enc = EncoderBase64(num_nodes=sim.num_nodes)
    dim = sim.num_nodes

    max_size = 4
    buffer = UniqueBuffer(max_size=max_size, num_nodes=dim, gpu_id=gpu_id)
    buffer.init_with_random(sim) if (buffer.min_score is None) else None

    x_prob = sim.get_rand_probs(num_envs=1).unsqueeze(0)
    x_bool = sim.prob_to_bool(x_prob)
    x_str = enc.bool_to_str(x_bool)
    sln_x = enc.str_to_bool(x_str)
    score = sim.get_scores(sln_x.unsqueeze(0)).squeeze(0)
    buffer.update(sln_x, score)
    print(buffer.scores)

    max_size = 2 ** 7
    buffer = UniqueBuffer(max_size=max_size, num_nodes=dim, gpu_id=gpu_id)
    buffer.init_with_random(sim) if (buffer.min_score is None) else None

    result_paths = [f"result_{graph_name}_{i}" for i in (5, 6, 7)]
    for result_path in result_paths:
        if os.path.exists(result_path):
            sln_xs = th.tensor(np.load(f"{result_path}/buffer_sln_xs.npz")['arr'], device=buffer.device)
            scores = th.tensor(np.load(f"{result_path}/buffer_scores.npz")['arr'], device=buffer.device)
            for i in range(sln_xs.shape[0]):
                buffer.update(sln_xs[i], scores[i])
        else:
            print(result_path)

    buffer.print_sln_x_str()
    # buffer.save_or_load_history(cwd=result_paths[0], if_save=True)


class Config:  # Demo
    def __init__(self, json_path: str = '', graph_name: str = 'gset_14', gpu_id: int = 0):
        # 数据超参数（路径，存储，预处理）
        self.json_path = './GraphMaxCut.json'  # 存放超参数的json文件。将会保存 class Config 里的所有变量 `vars(Config())`
        self.graph_name = graph_name
        self.gpu_id = gpu_id

        self.save_dir = f"./result_{graph_name}_{gpu_id}"

        '''GPU memory'''
        self.num_envs = 2 ** 6
        self.mid_dim = 2 ** 6
        self.num_layers = 2
        self.seq_len = 2 ** 6
        self.reset_gap = 2 ** 6

        '''exploit and explore'''
        self.learning_rate = 1e-3
        self.buf_size = 2 ** 8
        self.alpha_rate = 8
        self.diff_ratio = 0.1
        self.explore_weight = 1.0

        '''train and evaluate'''
        self.num_opti = 2 ** 16
        self.eval_gap = 2 ** 2

        if os.path.exists(json_path):
            self.load_from_json(json_path=json_path)
            self.save_as_json(json_path=json_path)
        vars_str = str(vars(self)).replace(", '", ", \n'")
        print(f"| Config\n{vars_str}")

    def load_from_json(self, json_path: str):
        with open(json_path, "r") as file:
            json_dict = json.load(file)
        for key, value in json_dict.items():
            setattr(self, key, value)

    def save_as_json(self, json_path: str):
        json_dict = vars(self)
        with open(json_path, "w") as file:
            json.dump(json_dict, file, indent=4)


def run_v1_find_sln_x_using_grad():
    gpu_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    num_envs = 2 ** 6
    graph_name = 'G14'
    # graph_name = 'g70'

    '''hyper-parameters'''
    lr = 1e-3
    opt_num = 2 ** 12
    eval_gap = 2 ** 6

    '''init task'''
    sim = MCMCSim2(graph_name=graph_name, gpu_id=gpu_id)
    enc = EncoderBase64(num_nodes=sim.num_nodes)

    probs = sim.get_rand_probs(num_envs=num_envs)
    probs.requires_grad_(True)

    '''loop'''
    best_score = -th.inf
    best_sln_x = probs
    start_time = time.time()
    for i in range(1, opt_num + 1):
        obj = sim.get_objectives(probs).mean()
        obj.backward()

        grads = probs.grad.data
        probs.data.add_(-lr * grads).clip_(0, 1)

        if i % eval_gap == 0:
            sln_xs = sim.prob_to_bool(probs)
            scores = sim.get_scores(sln_xs)
            used_time = time.time() - start_time
            print(f"|{used_time:9.0f}  {i:6}  {obj.item():9.3f}  {scores.max().item():9.0f}")

            max_score, max_id = th.max(scores, dim=0)
            if max_score > best_score:
                best_score = max_score
                best_sln_x = sln_xs[max_id]
                print(f"best_score {best_score}  best_sln_x \n{enc.bool_to_str(best_sln_x)}")

    print()
    print(f"best_score {best_score}  best_sln_x \n{enc.bool_to_str(best_sln_x)}")


def run_v2_find_sln_x_using_adam():
    gpu_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    num_envs = 2 ** 4
    graph_name = 'G14'
    # graph_name = 'g70'

    '''hyper-parameters'''
    lr = 1e-3
    opt_num = 2 ** 12
    eval_gap = 2 ** 6

    '''init task'''
    sim = MCMCSim2(graph_name=graph_name, gpu_id=gpu_id)
    enc = EncoderBase64(num_nodes=sim.num_nodes)

    probs = sim.get_rand_probs(num_envs=num_envs)
    probs.requires_grad_(True)

    '''init opti'''
    opt_base = th.optim.Adam([probs, ], lr=lr)

    '''loop'''
    best_score = -th.inf
    best_sln_x = sim.prob_to_bool(probs[0])
    start_time = time.time()
    for i in range(1, opt_num + 1):
        obj = sim.get_objectives(probs).mean()
        opt_base.zero_grad()
        obj.backward()
        opt_base.step()

        probs.data.clip_(0, 1)

        if i % eval_gap == 0:
            sln_xs = sim.prob_to_bool(probs)
            scores = sim.get_scores(sln_xs)
            used_time = time.time() - start_time
            print(f"|{used_time:9.0f}  {i:6}  {obj.item():9.3f}  {scores.max().item():9.0f}")

            max_score, max_id = th.max(scores, dim=0)
            if max_score > best_score:
                best_score = max_score
                best_sln_x = sln_xs[max_id]
                print(f"best_score {best_score}  best_sln_x \n{enc.bool_to_str(best_sln_x)}")

    print()
    print(f"best_score {best_score}  best_sln_x \n{enc.bool_to_str(best_sln_x)}")


def run_v3_find_sln_x_using_opti():
    gpu_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    num_envs = 2 ** 6
    graph_name = 'G14'
    # graph_name = 'g70'

    '''hyper-parameters'''
    lr = 1e-3
    mid_dim = 2 ** 8
    num_layers = 1
    seq_len = 2 ** 5
    reset_gap = 2 ** 6

    opt_num = int(2 ** 16 / num_envs)
    eval_gap = 2 ** 1

    '''init task'''
    sim = MCMCSim2(graph_name=graph_name, gpu_id=gpu_id)
    enc = EncoderBase64(num_nodes=sim.num_nodes)

    dim = sim.num_nodes
    probs = sim.get_rand_probs(num_envs=num_envs)
    probs.requires_grad_(True)
    obj = None
    hidden0 = None
    hidden1 = None

    '''init opti'''
    device = th.device(f'cuda:{gpu_id}' if th.cuda.is_available() and gpu_id >= 0 else 'cpu')
    opt_opti = OptimizerLSTM(inp_dim=dim, mid_dim=mid_dim, out_dim=dim, num_layers=num_layers).to(device)
    opt_base = th.optim.Adam(opt_opti.parameters(), lr=lr)

    '''loop'''
    best_score = -th.inf
    best_sln_x = sim.prob_to_bool(probs[0])
    start_time = time.time()
    for i in range(1, opt_num + 1):
        if i % reset_gap == 0:
            probs = sim.get_rand_probs(num_envs=num_envs)
            probs.requires_grad_(True)
            obj = None
            hidden0 = None
            hidden1 = None

        prob_ = probs.clone()
        updates = []

        for j in range(seq_len):
            obj = sim.get_objectives(probs).mean()
            obj.backward()

            grad_s = probs.grad.data
            update, hidden0, hidden1 = opt_opti(grad_s.unsqueeze(0), hidden0, hidden1)
            update = (update.squeeze_(0) - grad_s) * lr
            updates.append(update)
            probs.data.add_(update).clip_(0, 1)
        hidden0 = [h.detach() for h in hidden0]
        hidden1 = [h.detach() for h in hidden1]

        updates = th.stack(updates, dim=0)
        prob_ = (prob_ + updates.mean(0)).clip(0, 1)
        obj_ = sim.get_objectives(prob_).mean()

        opt_base.zero_grad()
        obj_.backward()
        opt_base.step()

        probs.data[:] = prob_

        if i % eval_gap == 0:
            sln_xs = sim.prob_to_bool(probs)
            scores = sim.get_scores(sln_xs)
            used_time = time.time() - start_time
            print(f"|{used_time:9.0f}  {i:6}  {obj.item():9.3f}  "
                  f"max_score {scores.max().item():9.0f}  "
                  f"best_score {best_score}")

            max_score, max_id = th.max(scores, dim=0)
            if max_score > best_score:
                best_score = max_score
                best_sln_x = sln_xs[max_id]

        if i % eval_gap * 256:
            print(f"best_score {best_score}  best_sln_x \n{enc.bool_to_str(best_sln_x)}")

    print()
    print(f"best_score {best_score}  best_sln_x \n{enc.bool_to_str(best_sln_x)}")


def run_v4_find_sln_x_using_opti_and_buffer():
    gpu_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    args = Config(json_path='', graph_name='gset/gset_55', gpu_id=gpu_id)

    graph_name = args.graph_name
    save_dir = args.save_dir

    num_envs = args.num_envs
    mid_dim = args.mid_dim
    num_layers = args.num_layers
    seq_len = args.seq_len
    reset_gap = args.reset_gap

    lr = args.learning_rate
    buf_size = args.buf_size
    alpha_rate = args.alpha_rate
    diff_ratio = args.diff_ratio
    explore_weight = args.explore_weight

    num_opti = args.num_opti
    eval_gap = args.eval_gap

    '''init task'''
    sim = MCMCSim2(graph_name=graph_name, gpu_id=gpu_id)
    enc = EncoderBase64(num_nodes=sim.num_nodes)

    dim = sim.num_nodes
    probs = sim.get_rand_probs(num_envs=num_envs)
    probs.requires_grad_(True)
    sln_xs = sim.prob_to_bool(probs)
    scores = sim.get_scores(sln_xs)
    max_score, max_id = th.max(scores, dim=0)

    obj = None
    hidden0 = [th.zeros((num_layers, num_envs, mid_dim), dtype=th.float32, device=sim.device) for _ in range(2)]
    hidden1 = [th.zeros((num_layers, num_envs * dim, 8), dtype=th.float32, device=sim.device) for _ in range(2)]

    '''init opti'''
    device = th.device(f'cuda:{gpu_id}' if th.cuda.is_available() and gpu_id >= 0 else 'cpu')
    opt_opti = OptimizerLSTM(inp_dim=dim, mid_dim=mid_dim, out_dim=dim, num_layers=num_layers).to(device)
    opt_base = th.optim.Adam(opt_opti.parameters(), lr=lr, amsgrad=False)
    # opt_base = th.optim.SGD(opt_opti.parameters(), lr=lr, momentum=0.99)

    '''inti buffer'''
    os.makedirs(save_dir, exist_ok=True)
    buffer = UniqueBuffer(max_size=buf_size, num_nodes=dim, gpu_id=gpu_id)
    buffer.save_or_load_history(cwd=save_dir, if_save=False)
    buffer.init_with_random(sim) if (buffer.min_score is None) else None

    '''init recorder'''
    best_score = -th.inf
    recorder = []

    '''loop'''
    start_time = time.time()
    pbar = tqdm(total=reset_gap * alpha_rate, ascii=True)
    #mpl.use('Agg')  # Generating matplotlib graphs without a running X server [duplicate]
    for i in range(1, num_opti + 1):
        if i % reset_gap == 0:
            buffer.update(sln_x=sln_xs[max_id], score=max_score)
            probs = sim.get_rand_probs(num_envs=num_envs)
            probs.requires_grad_(True)
            obj = None
            hidden0 = None
            hidden1 = None

        prob_ = probs.clone()
        updates = []

        for j in range(seq_len):
            obj = sim.get_objectives(probs).mean()
            obj.backward()

            grad_s = probs.grad.data
            update, hidden0, hidden1 = opt_opti(grad_s.unsqueeze(0), hidden0, hidden1)
            update = (update.squeeze_(0) - grad_s) * lr
            updates.append(update)
            probs.data.add_(update).clip_(0, 1)
        hidden0 = [h.detach() for h in hidden0]
        hidden1 = [h.detach() for h in hidden1]

        updates = th.stack(updates, dim=0)
        prob_ = (prob_ + updates.mean(0)).clip(0, 1)
        obj_exploit = sim.get_objectives(prob_).mean()

        if i % (reset_gap * alpha_rate) == 0:
            pbar = tqdm(total=reset_gap * alpha_rate, ascii=True)
        if i % (reset_gap * alpha_rate) > (reset_gap * (alpha_rate - 1)):
            buf_prob = buffer.sln_xs.float()
            obj_explore = (th.abs(prob_.unsqueeze(1) - buf_prob.unsqueeze(0)) - 0.5).clamp_min(0).sum(dim=(1, 2))
            obj_explore = obj_explore.clamp_max(dim / 2 * diff_ratio).mean() * explore_weight
        else:
            obj_explore = 0

        obj_ = obj_exploit + obj_explore
        opt_base.zero_grad()
        obj_.backward()
        opt_base.step()

        probs.data[:] = prob_

        sln_xs = sim.prob_to_bool(probs)
        scores = sim.get_scores(sln_xs)
        used_time = time.time() - start_time
        max_score, max_id = th.max(scores, dim=0)

        if max_score > best_score:
            best_score = max_score
            best_sln_x = sln_xs[max_id]
            best_sln_x_str = enc.bool_to_str(best_sln_x)
            enter_str = '\n' if len(best_sln_x_str) > 60 else ''
            print(f"\nbest_score {best_score}  best_sln_x {enter_str}{best_sln_x_str}")

            recorder.append((i, best_score))
            recorder_ary = th.tensor(recorder)
            th.save(recorder_ary, f"{save_dir}/recorder.pth")
            buffer.save_or_load_history(cwd=save_dir, if_save=True)

            if plt:
                plt.plot(recorder_ary[:, 0], recorder_ary[:, 1])
                plt.scatter(recorder_ary[:, 0], recorder_ary[:, 1])
                plt.grid()
                plt.title(f"best_score {best_score}")
                plt.savefig(f"{save_dir}/recorder.jpg")
                plt.close('all')
            else:
                np.savetxt(f"{save_dir}/recorder.txt", recorder_ary.data.detach().cpu().numpy())

        if i % eval_gap == 0:
            pbar.set_description(
                f"|{used_time:6.0f}  {i:6}  {obj.item():9.3f}  score {scores.max().item():6.0f}  {best_score}")
            pbar.update(eval_gap)
    pbar.close()


"""find solution_x using auto regression"""


class NetLSTM(nn.Module):
    def __init__(self, inp_dim, mid_dim, out_dim, num_layers):
        super().__init__()
        self.rnn1 = nn.LSTM(inp_dim, mid_dim, num_layers=num_layers)
        self.mlp1 = nn.Sequential(nn.Linear(mid_dim, out_dim), nn.Sigmoid())

    def forward(self, inp, hid=None):
        tmp, hid = self.rnn1(inp, hid)
        out = self.mlp1(tmp)
        return out, hid


def sample_sln_x_using_auto_regression(num_envs, device, dim, opt_opti, if_train=True):
    hidden = None
    sample = th.zeros((num_envs, 1), dtype=th.float32, device=device)
    node_prob = th.zeros((num_envs, 1), dtype=th.float32, device=device)

    samples = []
    logprobs = []
    entropies = []

    samples.append(sample)
    for _ in range(dim - 1):
        obs = th.hstack((sample, node_prob))
        node_prob, hidden = opt_opti(obs, hidden)
        dist = Bernoulli(node_prob.squeeze(0))
        sample = dist.sample()

        samples.append(sample)
        if if_train:
            logprobs.append(dist.log_prob(sample))
            entropies.append(dist.entropy())

    samples = th.stack(samples).squeeze(2)
    sln_xs = samples.permute(1, 0).to(th.int)

    if if_train:
        logprobs = th.stack(logprobs).squeeze(2).sum(0)
        logprobs = logprobs - logprobs.mean()

        entropies = th.stack(entropies).squeeze(2).mean(0)
    return sln_xs, logprobs, entropies


def run_v1_find_sln_x_using_auto_regression():
    gpu_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    num_envs = 2 ** 8
    # graph_name, num_limit = 'G14', sys.maxsize
    graph_name, num_limit = 'G14', 28

    '''hyper-parameters'''
    lr = 1e-3
    mid_dim = 2 ** 8
    num_layers = 1
    num_opt = int(2 ** 24 / num_envs)
    eval_gap = 2 ** 4
    print_gap = 2 ** 8

    alpha_period = 2 ** 10
    alpha_weight = 1.0

    '''init task'''
    sim = MCMCSim2(graph_name=graph_name, gpu_id=gpu_id)
    enc = EncoderBase64(num_nodes=sim.num_nodes)
    dim = sim.num_nodes

    '''init opti'''
    device = th.device(f'cuda:{gpu_id}' if th.cuda.is_available() and gpu_id >= 0 else 'cpu')
    opt_opti = NetLSTM(inp_dim=2, mid_dim=mid_dim, out_dim=1, num_layers=num_layers).to(device)
    opt_base = th.optim.Adam(opt_opti.parameters(), lr=lr)

    '''loop'''
    best_score = -th.inf
    best_sln_x = sim.get_rand_probs(num_envs=1)[0]
    start_time = time.time()
    for i in range(num_opt):
        alpha = (math.cos(i * math.pi / alpha_period) + 1) / 2
        sln_xs, logprobs, entropies = sample_sln_x_using_auto_regression(num_envs, device, dim, opt_opti, if_train=True)
        scores = sim.get_scores(probs=sln_xs).detach().to(th.float32)
        scores = (scores - scores.min()) / (scores.std() + 1e-4)

        obj_probs = logprobs.exp()
        obj = -((obj_probs / obj_probs.mean()) * scores + (alpha * alpha_weight) * entropies).mean()

        opt_base.zero_grad()
        obj.backward()
        opt_base.step()

        if i % eval_gap == 0:
            _sln_xs, _, _ = sample_sln_x_using_auto_regression(num_envs, device, dim, opt_opti, if_train=False)
            _scores = sim.get_scores(_sln_xs)

            sln_xs = th.vstack((sln_xs, _sln_xs))
            scores = th.hstack((scores, _scores))

        max_score, max_id = th.max(scores, dim=0)
        if max_score > best_score:
            best_score = max_score
            best_sln_x = sln_xs[max_id]
            print(f"best_score {best_score}  best_sln_x \n{enc.bool_to_str(best_sln_x)}")

        if i % print_gap == 0:
            used_time = time.time() - start_time
            print(f"|{used_time:9.0f}  {i:6}  {obj.item():9.3f}  "
                  f"score {scores.max().item():6.0f}  {best_score:6.0f}  "
                  f"entropy {entropies.mean().item():6.3f}  alpha {alpha:5.3f}")

            if i % (print_gap * 256) == 0:
                print(f"best_score {best_score}  best_sln_x \n{enc.bool_to_str(best_sln_x)}")


if __name__ == '__main__':
    # run_v1_find_sln_x_using_grad()
    # run_v2_find_sln_x_using_adam()
    # run_v3_find_sln_x_using_opti()
    run_v4_find_sln_x_using_opti_and_buffer()
    # run_v1_find_sln_x_using_auto_regression()
