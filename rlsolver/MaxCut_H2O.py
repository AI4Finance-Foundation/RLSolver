import os
import sys
import time
import math
import torch as th
import torch.nn as nn
from torch.distributions import Bernoulli  # BinaryDist

"""
backup in 2023-07-03 08:30:33
H2O_MaxCut_0015_stable_g14.py
"""
TEN = th.Tensor

ThetaG14L10 = "2U"  # best_score 21  count 2*2
ThetaG14L12 = "09v"  # best_score 27  count 2*6
ThetaG14L14 = "0nd"  # best_score 34  count 2*2
ThetaG14L16 = "2US"  # best_score 40  count 2*9
ThetaG14L18 = "09vo"  # best_score 47  count 2*23
ThetaG14L20 = "1QQb"  # best_score 54  count 2*1
ThetaG14L22 = "1E_E"  # best_score 61  count 2*1
ThetaG14L24 = "09xeR"  # best_score 68  count 2*4
ThetaG14L26 = "17dBj"  # best_score 76  count 2*1
ThetaG14L28 = "4SwUU"  # best_score 83  count 2*4
ThetaG14L30 = "0Hpfvw"  # best_score 91  count 2*1

ThetaG14L128_Gene = "1Fd$$$810612J1KLO9JAuN"  # best_score 409 by generate
ThetaG14L128_Opti = "2bnSUT7dTem7GZoHXLU6zH"  # best_score 462 by optimize

ThetaG14 = """
2dbChJAXfdo2GRp49ecgPjQwfRSIJqcfANlGZMwwAOZzjFXMXFcLYmRi27fT49J38CH8NUHFf8nLFzuUQh_LltgK6ofnt0P2NEwwUdURMPtFC8ZhlzftdQJj
MQ1aEyvV6RqIv8
"""  # 3029
ThetaG70 = """
Ca$26VjCodrX19jP9y9ReEIV$393n6siIgCwy7xvS_yJg5M07JuJQnKsmKyHC2Ji3zQFtdKEy4DwGwdPiYeeM_DxYHGlOpmwSeyY0TcR31jX9HUXYgNZrWpP
Im3I2dEuQGS49jDE0kMDKkrLZmy1tnFxUWZWqy_vJj3v0uB$c9cpK$9EKv9NMjvCgH9myDgHQl2DyQ9nGNliQUEAizCbe7IXnGSvCHq3uT1zdN4y$1ot5ma9
7JEkvTk_WlRH6GXEZZm$Mvzpi0SqbxVn0fSAKyDHQ21hEOkpe3ygUR7qSlIMM4zlSz992JlwLh9JNKTBSAAhn9fG4RCT86SPUGe36qxMZip_tjcCtF8f3nv2
l9SokkqtRFVeg7hJlys7D8nhfzRuDU8Ab_h1laQ1D_uPDWNaegSx1UIJQhqdWb121i5XwBr7TM$dawUEGDTJGkEmNmsagkkrlbom6YGnDQ6mKP$wn4nhPpoT
vmdd1W8Rdb9ijEZcNrx3dqdxu3_X50LS1Ub9ARdIKpqAmn2aV33eRA6rjRzGR5y$Mvx78$DIIjIu8UsDtJzpLZEIcvPt$Q6hXGCxcO5$eUv7v9dAZS2RDFFT
NqnttGE9ZEXTPo4$9T5Yt5Xcqk8NzsSZq020YDEy9YxBY3fJzRGYvfr2gZQd$yh6G$FLAJMcmHwDE9B6K$SETREgXgdf44Nw782u0j4cTsmmYBfwhOvVun2d
bE1WpeEpb6ZFXYRs7oL6PJJn2YtmT0_T1R3UMZ1Co1zFX$LhaTSUDQR2rC9ke5c1wAVaYn1xJPOv3$VnKba2rAioDfrD3XlzeWM$FbXH5Fwa027cJGRD7$O$
qBJp1IKQ5xlejSDQlZFuSgXuR3vY7Hx8QPfNOuQizr1RtkiVn20ZIAuAco5RBMyTjnSQB7oddim$HaAobRy1KdgjfqYVo9x11r_gKFkJnQas1Lz1Rq31jpf6
OnBYKAYQ6ugEx3yyU3tu7NIA9O0OpcIIOe$JeikBSr4NQerCgkfliA0YXuFQYMojF$BiGwfY9tYz0CCXeESqYxsmM76b_jjR3vvUc1N2t6qRTqgQ0IcZtFz7
gA9zdDUsVdH2aJFm4E3Dy83lZFGp43K3w5V9gWIZpKcZ9g8RkWzDxHv7kaSzQJPL2LzRqj_9OdozoDUBQHoHWQbLlfGy0EetIbrxN$1GsStTyHuCCD4RXy2c
MmJaDfrFf4NZdxH4kWe9aXQ3TJ5m7dIJ6KlnmDuHPecz7yyFTQGP_y6IsxhhqCGcFUp9BCGEgv4jOFJ7yO8$$9pSQ66$Cgn2rd7H4jGaiXC$WmvbNzC2JONo
w2oe07AZUEVktds9$P5A0N40Ray0YJav9UHlwZiq2HO80DC5Ghqh4tJ_wFRvrJH4ke3Pnwj7wBBUHspzuaY_JmYYV_UHCicVeBUwOYGi6cavQWHboiFK8BZ2
ETY5RjHHT_yLdY1E4vwNsIKbP81mbwBzg554gU6804Ll0qBiyrQ73eF_x5ivFiIfODsLk5sdxmb_BREq9fmrIFkNK$dj62PU4VIBrknm0BOHwfV7gg43P$Zj
6tMY4Dm44Z1ltHEpbUQZXBBQ63ptXI8v5KMSeyfetO3QDtFlzGKYNO_y6xVSYx7bldLycrf6jyK6TUopjyvO4Z4qmoMGbse8aZhzQ$9fvD7
"""  # 5124

"""Graph Max Cut Env"""


class GraphMaxCutEnv:
    def __init__(self, graph_key: str = 'g14', gpu_id: int = -1, num_limit: int = 0):
        device = th.device(f'cuda:{gpu_id}' if th.cuda.is_available() and gpu_id >= 0 else 'cpu')

        '''read graph from txt'''
        map_graph_to_node_edge = {
            'g14': (800, 4694),
            'g15': (800, 4661),
            'g22': (2000, 19990),
            'g49': (3000, 6000),
            'g50': (3000, 6000),
            'g70': (10000, 9999),
        }
        assert graph_key in map_graph_to_node_edge
        txt_path = f"./graph_set_{graph_key}.txt"
        with open(txt_path, 'r') as file:
            lines = file.readlines()
            lines = [[int(i1) for i1 in i0.split()] for i0 in lines]
        num_nodes, num_edges = lines[0]
        edge_to_n0_n1_dist = [(i[0] - 1, i[1] - 1, i[2]) for i in lines[1:]]

        '''
        n0: index of node0
        n1: index of node1
        dt: distance between node0 and node1
        p0: the probability of node0 is in set, (1-p0): node0 is in another set
        p1: the probability of node1 is in set, (1-p1): node1 is in another set
        '''

        n0_to_n1s = [[] for _ in range(num_nodes)]  # 将 node0_id 映射到 node1_id
        n0_to_dts = [[] for _ in range(num_nodes)]  # 将 mode0_id 映射到 node1_id 与 node0_id 的距离
        for n0, n1, dist in edge_to_n0_n1_dist:
            n0_to_n1s[n0].append(n1)
            n0_to_dts[n0].append(dist)
        n0_to_n1s = [th.tensor(node1s, dtype=th.long, device=device) for node1s in n0_to_n1s]
        n0_to_dts = [th.tensor(node1s, dtype=th.long, device=device) for node1s in n0_to_dts]  # dists == 1
        assert num_nodes == len(n0_to_n1s)
        assert num_nodes == len(n0_to_dts)
        assert num_edges == sum([len(n0_to_n1) for n0_to_n1 in n0_to_n1s])
        assert num_edges == sum([len(n0_to_dt) for n0_to_dt in n0_to_dts])

        '''取出前num_limit个节点'''
        if 0 < num_limit < num_nodes:
            _n0_to_n1s = []
            _n0_to_dts = []
            for n0 in range(num_limit):
                n1s = n0_to_n1s[n0]
                dts = n0_to_dts[n0]

                _n1s = []
                _dts = []
                assert len(n1s) == len(dts)
                for j in range(len(n1s)):
                    n1 = n1s[j]
                    dt = dts[j]
                    if n1 < num_limit:
                        _n1s.append(n1)
                        _dts.append(dt)

                _n0_to_n1s.append(_n1s)
                _n0_to_dts.append(_dts)
            n0_to_n1s = [th.tensor(node1s, dtype=th.long, device=device) for node1s in _n0_to_n1s]
            n0_to_dts = [th.tensor(node1s, dtype=th.long, device=device) for node1s in _n0_to_dts]  # dists == 1
            assert len(n0_to_dts) == len(n0_to_n1s)

        self.num_nodes = len(n0_to_n1s)
        self.num_edges = sum([len(n0_to_n1) for n0_to_n1 in n0_to_n1s])
        self.device = device

        '''高性能计算'''
        n0_ids = []
        n1_ids = []
        for i, n1s in enumerate(n0_to_n1s):
            n0_ids.extend([i, ] * n1s.shape[0])
            n1_ids.extend(n1s)
        self.n0_ids = th.tensor(n0_ids, dtype=th.long, device=device).unsqueeze(0)
        self.n1_ids = th.tensor(n1_ids, dtype=th.long, device=device).unsqueeze(0)
        self.env_is = th.zeros(self.num_edges, dtype=th.long, device=device).unsqueeze(0)

    def get_objectives(self, probs):
        p0s, p1s = self.get_p0s_p1s(probs)
        return -(p0s + p1s - 2 * p0s * p1s).sum(1)

    def get_scores(self, probs):
        p0s, p1s = self.get_p0s_p1s(probs)
        return (p0s ^ p1s).sum(1)

    def get_p0s_p1s(self, probs):
        num_envs = probs.shape[0]
        if num_envs != self.env_is.shape[0]:
            self.n0_ids = self.n0_ids[0].repeat(num_envs, 1)
            self.n1_ids = self.n1_ids[0].repeat(num_envs, 1)
            self.env_is = self.env_is[0:1] + th.arange(num_envs, device=self.device).unsqueeze(1)

        p0s = probs[self.env_is, self.n0_ids]
        p1s = probs[self.env_is, self.n1_ids]
        return p0s, p1s

    def get_rand_probs(self, num_envs: int):
        return th.rand((num_envs, self.num_nodes), dtype=th.float32, device=self.device)

    @staticmethod
    def convert_prob_to_bool(p0s, thresh=0.5):
        return (p0s > thresh).to(th.int8)

    def node_prob_bool_to_str(self, x_bool: TEN) -> str:
        x_int = int(''.join([str(i) for i in x_bool.tolist()]), 2)
        x_str = bin_int_to_str(x_int)
        x_str = '\n'.join([x_str[i:i + 120] for i in range(0, len(x_str), 120)])
        return x_str.zfill(math.ceil(self.num_nodes // 6 + 1))

    def node_prob_str_to_bool(self, x_str: str) -> TEN:
        x_b64: str = x_str.replace('\n', '')
        x_int: int = bin_str_to_int(x_b64)
        return self.int_to_bool(x_int)

    def int_to_bool(self, x_int: int) -> TEN:
        x_bin: str = bin(x_int)[2:]
        x_bool = th.zeros(self.num_nodes, dtype=th.int8)
        x_bool[-len(x_bin):] = th.tensor([int(i) for i in x_bin], dtype=th.int8)
        return x_bool


def bin_int_to_str(decimal: bin):
    base_digits = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz_$"
    base_num = len(base_digits)
    if decimal == 0:
        return base_digits[0]

    base = ""
    while decimal > 0:
        remainder = decimal % base_num
        base = base_digits[remainder] + base
        decimal //= base_num
    return base


def bin_str_to_int(base: str):
    base_digits = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz_$"
    base_len = len(base)
    base_num = len(base_digits)

    decimal = 0
    for i in range(base_len):
        digit = base_digits.index(base[i])
        power = base_len - 1 - i
        decimal += digit * (base_num ** power)
    return decimal


def check_env():
    th.manual_seed(0)
    num_envs = 6

    env = GraphMaxCutEnv(graph_key='g14', num_limit=12)
    # env = GraphMaxCutEnv(graph_key='g70')

    probs = env.get_rand_probs(num_envs=num_envs)
    print(env.get_objectives(probs))

    for thresh in th.linspace(0, 1, 8):
        objs = env.get_objectives(probs=env.convert_prob_to_bool(probs, thresh))
        print(f"{thresh.item():6.3f}  {objs.numpy()}")

    best_score = -th.inf
    best_theta = None
    for _ in range(8):
        probs = env.get_rand_probs(num_envs=num_envs)
        thetas = env.convert_prob_to_bool(probs)
        scores = env.get_scores(thetas)
        max_score, max_id = th.max(scores, dim=0)
        if max_score > best_score:
            best_score = max_score
            best_theta = thetas[max_id]
            print(f"best_score {best_score}  best_theta \n{env.node_prob_bool_to_str(best_theta)}")

    print()
    print(f"best_score {best_score}  best_theta \n{env.node_prob_bool_to_str(best_theta)}")


def check_theta():
    graph_key, num_limit, theta = 'g14', 0, ThetaG14
    graph_key, num_limit, theta = 'g14', 128, ThetaG14L128_Opti

    env = GraphMaxCutEnv(graph_key=graph_key, num_limit=num_limit)
    best_theta = env.node_prob_str_to_bool(theta)
    best_score = env.get_scores(best_theta.unsqueeze(0)).squeeze(0)
    print(f"NumNodes {env.num_nodes}  NumEdges {env.num_edges}")
    print(f"score {best_score}  theta \n{env.node_prob_bool_to_str(best_theta)}")


def check_convert_between_b10_and_b64():
    print()
    i = 2 ** 800
    j = bin_int_to_str(i)
    print(len(str(j)), j)
    i = bin_str_to_int(j)
    print(len(str(i)), i)
    b = bin(i)
    print(len(str(b)), b)

    print()
    i = 2 ** 5000
    j = bin_int_to_str(i)
    print(len(str(j)), j)
    i = bin_str_to_int(j)
    print(len(str(i)), i)
    b = bin(i)
    print(len(str(b)), b)


def convert_between_str_and_bool():
    gpu_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    num_envs = 1
    graph_key = 'g14'
    # graph_key = 'g70'

    env = GraphMaxCutEnv(graph_key=graph_key, gpu_id=gpu_id)

    x_prob = env.get_rand_probs(num_envs=num_envs)[0]
    x_bool = env.convert_prob_to_bool(x_prob)

    x_str = env.node_prob_bool_to_str(x_bool)
    print(x_str)
    x_bool = env.node_prob_str_to_bool(x_str)

    assert all(x_bool == env.convert_prob_to_bool(x_prob))


def exhaustion_search():
    import numpy as np
    from tqdm import tqdm

    gpu_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    num_envs = 2 ** 14
    th.manual_seed(0)
    graph_key = 'g14'

    save_dir = f"./exhaustion_{graph_key}"
    os.makedirs(save_dir, exist_ok=True)

    th.set_grad_enabled(False)
    # for num_limit in range(10, 32, 2):
    for num_limit in range(30, 34, 2):
        env = GraphMaxCutEnv(graph_key=graph_key, gpu_id=gpu_id, num_limit=num_limit)
        dim = env.num_nodes

        best_score = -th.inf
        best_theta = None
        all_score = []

        _num_envs = min(2 ** dim, num_envs)
        num_iter = 2 ** (dim - 1)
        print(f"NumNodes {env.num_nodes}  NumEdges {env.num_edges}  Total search num_iter {num_iter}")
        time.sleep(0.1)
        i_iter = tqdm(range(0, num_iter, _num_envs), ascii=True)
        for i in i_iter:
            thetas = [env.int_to_bool(i + j) for j in range(_num_envs)]
            thetas = th.stack(thetas).to(env.device)
            scores = env.get_scores(thetas)

            max_score, max_id = th.max(scores, dim=0)
            if max_score > best_score:
                best_score = max_score
                best_theta = thetas[max_id]
                i_iter.set_description(
                    f"best_score {best_score:6.0f}  best_theta {env.node_prob_bool_to_str(best_theta)}")
                print()

            all_score.append(scores)
        i_iter.close()

        all_score = th.hstack(all_score).data.cpu().numpy().astype(int)
        save_path = f'{save_dir}/exhaustion_{graph_key}_{num_limit:06}.npz'
        np.savez_compressed(save_path, arr=all_score)
        print(f"NumNodes {env.num_nodes}  NumEdges {env.num_edges}  NumSearch {2 ** dim}")
        print(f"best_score {best_score:6.0f}  best_theta {env.node_prob_bool_to_str(best_theta)}")


def exhaustion_search_result():
    import numpy as np

    graph_key = 'g14'
    save_dir = f"./exhaustion_{graph_key}"
    # for num_limit in range(10, 32, 2):
    for num_limit in range(10, 32, 2):
        save_path = f'{save_dir}/exhaustion_{graph_key}_{num_limit:06}.npz'
        scores = np.load(save_path)['arr']

        env = GraphMaxCutEnv(graph_key=graph_key, num_limit=num_limit)

        best_score = np.max(scores)
        best_index = int(np.argmax(scores))

        num_best = (scores == best_score).sum()
        best_theta = env.int_to_bool(best_index)
        best_theta = env.node_prob_bool_to_str(best_theta)
        print(f"ThetaG{graph_key[1:]}L{num_limit} = \"{best_theta}\"  # best_score {best_score}  count 2*{num_best}")

        # import matplotlib.pyplot as plt
        # plt.plot(scores)
        # plt.plot((0, scores.shape[0]), (best_score, best_score))
        # plt.grid()
        # plt.show()


"""Optimize with update theta"""


class OptimizerOpti(nn.Module):
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


def run_v1_update_theta_by_grad():
    gpu_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    num_envs = 2 ** 6
    graph_key = 'g14'
    # graph_key = 'g70'

    '''hyper-parameters'''
    lr = 1e-3
    opt_num = 2 ** 12
    eval_gap = 2 ** 6

    '''init task'''
    env = GraphMaxCutEnv(graph_key=graph_key, gpu_id=gpu_id)
    probs = env.get_rand_probs(num_envs=num_envs)
    probs.requires_grad_(True)

    '''loop'''
    best_score = -th.inf
    best_theta = probs
    start_time = time.time()
    for i in range(1, opt_num + 1):
        obj = env.get_objectives(probs).mean()
        obj.backward()

        grads = probs.grad.data
        probs.data.add_(-lr * grads).clip_(0, 1)

        if i % eval_gap == 0:
            thetas = env.convert_prob_to_bool(probs)
            scores = env.get_scores(thetas)
            used_time = time.time() - start_time
            print(f"|{used_time:9.0f}  {i:6}  {obj.item():9.3f}  {scores.max().item():9.0f}")

            max_score, max_id = th.max(scores, dim=0)
            if max_score > best_score:
                best_score = max_score
                best_theta = thetas[max_id]
                print(f"best_score {best_score}  best_theta \n{env.node_prob_bool_to_str(best_theta)}")

    print()
    print(f"best_score {best_score}  best_theta \n{env.node_prob_bool_to_str(best_theta)}")


def run_v2_update_theta_by_adam():
    gpu_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    num_envs = 2 ** 4
    graph_key = 'g14'
    # graph_key = 'g70'

    '''hyper-parameters'''
    lr = 1e-3
    opt_num = 2 ** 12
    eval_gap = 2 ** 6

    '''init task'''
    env = GraphMaxCutEnv(graph_key=graph_key, gpu_id=gpu_id)
    probs = env.get_rand_probs(num_envs=num_envs)
    probs.requires_grad_(True)

    '''init opti'''
    opt_base = th.optim.Adam([probs, ], lr=lr)

    '''loop'''
    best_score = -th.inf
    best_theta = env.convert_prob_to_bool(probs[0])
    start_time = time.time()
    for i in range(1, opt_num + 1):
        obj = env.get_objectives(probs).mean()
        opt_base.zero_grad()
        obj.backward()
        opt_base.step()

        probs.data.clip_(0, 1)

        if i % eval_gap == 0:
            thetas = env.convert_prob_to_bool(probs)
            scores = env.get_scores(thetas)
            used_time = time.time() - start_time
            print(f"|{used_time:9.0f}  {i:6}  {obj.item():9.3f}  {scores.max().item():9.0f}")

            max_score, max_id = th.max(scores, dim=0)
            if max_score > best_score:
                best_score = max_score
                best_theta = thetas[max_id]
                print(f"best_score {best_score}  best_theta \n{env.node_prob_bool_to_str(best_theta)}")

    print()
    print(f"best_score {best_score}  best_theta \n{env.node_prob_bool_to_str(best_theta)}")


def run_v3_update_theta_by_opti():
    gpu_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    num_envs = 2 ** 6
    graph_key = 'g14'
    # graph_key = 'g70'

    '''hyper-parameters'''
    lr = 1e-3
    mid_dim = 2 ** 8
    num_layers = 1
    seq_len = 2 ** 5
    reset_gap = 2 ** 6

    opt_num = int(2 ** 16 / num_envs)
    eval_gap = 2 ** 1

    '''init task'''
    env = GraphMaxCutEnv(graph_key=graph_key, gpu_id=gpu_id)
    dim = env.num_nodes
    probs = env.get_rand_probs(num_envs=num_envs)
    probs.requires_grad_(True)
    obj = None
    hidden0 = None
    hidden1 = None

    '''init opti'''
    device = th.device(f'cuda:{gpu_id}' if th.cuda.is_available() and gpu_id >= 0 else 'cpu')
    opt_opti = OptimizerOpti(inp_dim=dim, mid_dim=mid_dim, out_dim=dim, num_layers=num_layers).to(device)
    opt_base = th.optim.Adam(opt_opti.parameters(), lr=lr)

    '''loop'''
    best_score = -th.inf
    best_theta = env.convert_prob_to_bool(probs[0])
    start_time = time.time()
    for i in range(1, opt_num + 1):
        if i % reset_gap == 0:
            probs = env.get_rand_probs(num_envs=num_envs)
            probs.requires_grad_(True)
            obj = None
            hidden0 = None
            hidden1 = None

        prob_ = probs.clone()
        updates = []

        for j in range(seq_len):
            obj = env.get_objectives(probs).mean()
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
        obj_ = env.get_objectives(prob_).mean()

        opt_base.zero_grad()
        obj_.backward()
        opt_base.step()

        probs.data[:] = prob_

        if i % eval_gap == 0:
            thetas = env.convert_prob_to_bool(probs)
            scores = env.get_scores(thetas)
            used_time = time.time() - start_time
            print(f"|{used_time:9.0f}  {i:6}  {obj.item():9.3f}  "
                  f"max_score {scores.max().item():9.0f}  "
                  f"best_score {best_score}")

            max_score, max_id = th.max(scores, dim=0)
            if max_score > best_score:
                best_score = max_score
                best_theta = thetas[max_id]

        if i % eval_gap * 256:
            print(f"best_score {best_score}  best_theta \n{env.node_prob_bool_to_str(best_theta)}")

    print()
    print(f"best_score {best_score}  best_theta \n{env.node_prob_bool_to_str(best_theta)}")


def run_v4_update_theta_by_opti():
    gpu_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    num_envs = 2 ** 7
    graph_key, num_limit = 'g14', 0

    '''hyper-parameters'''
    lr = 1e-3
    mid_dim = 2 ** 7
    num_layers = 2
    seq_len = 2 ** 6
    reset_gap = 2 ** 6
    reset_num = num_envs // reset_gap

    opt_num = int(2 ** 14)
    eval_gap = 2 ** 1

    '''init task'''
    env = GraphMaxCutEnv(graph_key=graph_key, gpu_id=gpu_id, num_limit=num_limit)
    dim = env.num_nodes
    probs = env.get_rand_probs(num_envs=num_envs)
    probs.requires_grad_(True)
    obj = None
    hidden0 = [th.zeros((num_layers, num_envs, mid_dim), dtype=th.float32, device=env.device) for _ in range(2)]
    hidden1 = [th.zeros((num_layers, num_envs * dim, 8), dtype=th.float32, device=env.device) for _ in range(2)]

    '''init opti'''
    device = th.device(f'cuda:{gpu_id}' if th.cuda.is_available() and gpu_id >= 0 else 'cpu')
    opt_opti = OptimizerOpti(inp_dim=dim, mid_dim=mid_dim, out_dim=dim, num_layers=num_layers).to(device)
    opt_base = th.optim.Adam(opt_opti.parameters(), lr=lr)

    '''loop'''
    best_score = -th.inf
    best_theta = env.convert_prob_to_bool(probs[0])
    start_time = time.time()
    for i in range(1, opt_num + 1):  # ring reset
        k0 = (i % reset_gap) * reset_num
        k1 = k0 + reset_num
        probs.data[k0:k1] = env.get_rand_probs(reset_num)
        hidden0[0].data[:, k0:k1] = 0
        hidden0[1].data[:, k0:k1] = 0
        hidden1[0].data[:, k0 * dim:k1 * dim] = 0
        hidden1[1].data[:, k0 * dim:k1 * dim] = 0

        prob_ = probs.clone()
        updates = []

        for j in range(seq_len):
            obj = env.get_objectives(probs).mean()
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
        obj_ = env.get_objectives(prob_).mean()

        opt_base.zero_grad()
        obj_.backward()
        opt_base.step()

        probs.data[:] = prob_

        thetas = env.convert_prob_to_bool(probs)
        scores = env.get_scores(thetas)
        used_time = time.time() - start_time
        max_score, max_id = th.max(scores, dim=0)
        if max_score > best_score:
            best_score = max_score
            best_theta = thetas[max_id]
            print(f"best_score {best_score}  best_theta \n{env.node_prob_bool_to_str(best_theta)}")

        if i % eval_gap == 0:
            print(f"|{used_time:9.0f}  {i:6}  {obj.item():9.3f}  "
                  f"score {scores.max().item():9.0f}  {best_score}")
            if i % (eval_gap * 256) == 0:
                print(f"best_score {best_score}  best_theta \n{env.node_prob_bool_to_str(best_theta)}")


"""Optimize with generate theta"""


class OptimizerGene(nn.Module):
    def __init__(self, inp_dim, mid_dim, out_dim, num_layers):
        super().__init__()
        self.inp_dim = inp_dim
        self.mid_dim = mid_dim
        self.out_dim = out_dim
        self.num_layers = num_layers

        self.rnn1 = nn.LSTM(1, mid_dim, num_layers=num_layers)
        self.mlp1 = nn.Sequential(nn.Linear(mid_dim, 1), nn.Sigmoid())

    def forward(self, inp, hid=None):
        tmp, hid = self.rnn1(inp, hid)
        out = self.mlp1(tmp)
        return out, hid


def generate_theta_by_auto_regression(num_envs, device, dim, opt_opti, if_train=True):
    hidden = None
    sample = th.zeros((num_envs, 1), dtype=th.float32, device=device)

    samples = []
    logprobs = []
    entropies = []

    samples.append(sample)
    for _ in range(dim - 1):
        node_prob, hidden = opt_opti(sample.unsqueeze(0), hidden)
        dist = Bernoulli(node_prob.squeeze(0))
        sample = dist.sample()

        samples.append(sample)
        if if_train:
            logprobs.append(dist.log_prob(sample))
            entropies.append(dist.entropy())

    samples = th.stack(samples).squeeze(2)
    thetas = samples.permute(1, 0).to(th.int)

    if if_train:
        logprobs = th.stack(logprobs).squeeze(2).sum(0)
        logprobs = logprobs - logprobs.mean()

        entropies = th.stack(entropies).squeeze(2).mean(0)
    return thetas, logprobs, entropies


def run_v1_generate_theta_by_auto_regression():
    gpu_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    num_envs = 2 ** 8
    # graph_key, num_limit = 'g14', sys.maxsize
    graph_key, num_limit = 'g14', 28

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
    env = GraphMaxCutEnv(graph_key=graph_key, gpu_id=gpu_id, num_limit=num_limit)
    dim = env.num_nodes

    '''init opti'''
    device = th.device(f'cuda:{gpu_id}' if th.cuda.is_available() and gpu_id >= 0 else 'cpu')
    opt_opti = OptimizerGene(inp_dim=dim, mid_dim=mid_dim, out_dim=dim, num_layers=num_layers).to(device)
    opt_base = th.optim.Adam(opt_opti.parameters(), lr=lr)

    '''loop'''
    best_score = -th.inf
    best_theta = env.get_rand_probs(num_envs=1)[0]
    start_time = time.time()
    for i in range(num_opt):
        alpha = (math.cos(i * math.pi / alpha_period) + 1) / 2
        thetas, logprobs, entropies = generate_theta_by_auto_regression(num_envs, device, dim, opt_opti, if_train=True)
        scores = env.get_scores(probs=thetas).detach().to(th.float32)
        scores = (scores - scores.min()) / (scores.std() + 1e-4)

        obj_probs = logprobs.exp()
        obj = -((obj_probs / obj_probs.mean()) * scores + (alpha * alpha_weight) * entropies).mean()

        opt_base.zero_grad()
        obj.backward()
        opt_base.step()

        if i % eval_gap == 0:
            _thetas, _, _ = generate_theta_by_auto_regression(num_envs, device, dim, opt_opti, if_train=False)
            _scores = env.get_scores(_thetas)

            thetas = th.vstack((thetas, _thetas))
            scores = th.hstack((scores, _scores))

        max_score, max_id = th.max(scores, dim=0)
        if max_score > best_score:
            best_score = max_score
            best_theta = thetas[max_id]
            print(f"best_score {best_score}  best_theta \n{env.node_prob_bool_to_str(best_theta)}")

        if i % print_gap == 0:
            used_time = time.time() - start_time
            print(f"|{used_time:9.0f}  {i:6}  {obj.item():9.3f}  "
                  f"score {scores.max().item():6.0f}  {best_score:6.0f}  "
                  f"entropy {entropies.mean().item():6.3f}  alpha {alpha:5.3f}")

            if i % (print_gap * 256) == 0:
                print(f"best_score {best_score}  best_theta \n{env.node_prob_bool_to_str(best_theta)}")


if __name__ == '__main__':
    # check_env()
    # check_theta()
    # exhaustion_search()
    # exhaustion_search_result()

    # run_v1_update_theta_by_grad()
    # run_v2_update_theta_by_adam()
    # run_v3_update_theta_by_opti()
    run_v4_update_theta_by_opti()
    # run_v1_generate_theta_by_auto_regression()
