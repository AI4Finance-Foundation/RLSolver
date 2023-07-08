import sys
import time
import torch as th
import torch.nn as nn

TEN = th.Tensor
ThetaGset_14 = """
BYazUs1eWYN9rkM72XZVmSlFWkztsLbW1oQvepFN1ncCSwepfwZqfPkX94Wj72s83xu3ogvwW3OqwCDhlUBqQIVr5PWKK@m9oxFFj4ggJ2Iwz3cQQCWYYhsS
pl!bxDGo7kLtG3
"""  # 3025
"""
BHak9ztUcX9XcYhD!w61dHdiNl8SPEc969fwLIK3NChqc1J6uDC5CSuQSgzcifjw6JjyENQ!G0UMBGbVqefGnzhtbe4P3ShJXTB!Zi1Y6c2S@qLv4RTkm7bI
3Xay!wxOG70GyL
"""  # 3017

ThetaGset_70 = """
Mk@CGftMyn1hBJtZJ8JboOSf@DJDxG2sSqM68H75c!8TqFWAHT4TaxU2wU8RMCTsD9aP3nUO8EN6Q6nZsiooW!N7iRQvYzw6co8iAdmbDBthJRehiqXj1gzZ
SwDSCnO4aQcEJtNOAuWNUu1Vjw8B3xP7egjg08!5TtD5A4L@mJmzU@JOU5JXWt5MqRJw8NqRavCN8aJxQXvsaeOKs9MloHShxQc5MR0D4dB9nXE8@By3FwkJ
HTOu5du!gvbRGQhOjjw@W59zsAc0l7fxApcKU8NRaCBrOYuzoD8qebH0cvSWWE9vc9JJCTv6VrJTXUdLcKKrxJpQEbMdIGcZeQoDG07Wjsz!3tmM3PIpDx5C
vJcyuu03bPfoqHrTv82HNIxrp9b4NeIKl!rBvkaBN!4ZNgXkoqc7BeSTar0nglBCBsFh6L1HdW@nk6eOQNdTQuOwXw2kquu1vlywGiQxNaGwUZ@6xExrZzyd
5wnnBgIbnlJstOjmX17Dn0n74D!hFAVcBelJKbnSUz0KwxCkfDDobKG1tb9QbF8@W57HI@NSStS4Ie2N3T9zVjOSm5Z3@aGrhQM7mYF@oe5H5JnKjcCbNPPd
X0x33QOJjOhdZyE@JdFi3Fhm0uIX92cj0ACAiNO8Ji7LiDpT9bQi5p1Cqjan@8rGQ@PVKTWmwR6NOJLGU@cOdbOqhqnpEEX6HIC4AtEmd2wwiLp6rY5f4xCn
lOBgzoOzlGjPhib2HyVGZTTxCi3wdA!dBbDeWjBMyB9Ph@VrkdceNabC1MJuoFmB6KfkixB7TZY5D@fxUlkC1KsyNp1NDhv9ogW@PlhRFP6kACHmTQbNH@Y@
0LTzBSUaF7votcNavjP4cqh4bD5iHR7IaZpXY4as91Bb3usfxCAjSK4KmyFbLW8dtxcaLHynnsw@RkKylb8BUnqtp0ifyJ7BB1!qUPuTxak2BV9Bb0DBtzpG
YxLiUKiaG4qO7D88eD34HXSKJYAYzmSSYo@TosuLc1EXao1MqupvsKAih4PaiWytP@LsQ6piJ3i9AMMhoOc0i72wWHGl!ttbD55emBXC3G0bd0qaASmj3P9H
qKJ9nNe2fnRCkTPwEODN8IDvjPQzEDUD6FfJqgSjzUmjJqIbug9N7R5Hukc9aTZVCV9b0t!JYny9yNeLaRyRgalVvpQ8AOo3Sl17X@BQ2c3d8R4MMNEbh8Cm
WwTkNp1PpEXjn7REugoJkhaDdTFwHnSTGUvxwN4RZom9H88PdaQZ!8GS27rr0MQmPezJLMQOq5EtYPTH8YI@@JzcaGG@MqxC1nHREtQkshM@gw5lX9MCTYXy
6CyoAHKjeOfu3n2J@ZFKAXEAbk8AiTk5JeRv6js0CRYIANMFQr0rE3T!6Pb51TREuoDZx6tH6LLeR2z94ki!Twiif!eRMsmfoLe6YiQsGmk5agRlysPUILjC
OdiFbtRRd!8VniBOE56X2SUlZIBwl6L9qFFEqeGIAEVvA0Ls81aHDoP!7Fs5PsSpYN2VuF2n7wl!LbO0Jpw1SPuXU@ntGCZeEfSL1uxwALYR6pfHqqEDZ@jt
G3WiENwEEjBv3ROzleajhLLaGDz3hSI5FUWco8po3YDaN3Pv9QUiXY!8G7fci7HlvnV8m1pGt8UGdeyzt85YEjE0wyWQl2oIkjr9a@Jp5NH
"""  # 5124

"""Graph Max Cut Env"""


class GraphMaxCutEnv:
    def __init__(self, graph_key: str = 'gset_70', gpu_id: int = -1):
        device = th.device(f'cuda:{gpu_id}' if th.cuda.is_available() and gpu_id >= 0 else 'cpu')

        '''read graph from txt'''
        map_graph_to_node_edge = {'gset_14': (800, 4694),
                                  'gset_15': (800, 4661),
                                  'gset_22': (2000, 19990),
                                  'gset_49': (3000, 6000),
                                  'gset_50': (3000, 6000),
                                  'gset_70': (10000, 9999), }
        assert graph_key in map_graph_to_node_edge
        txt_path = f"./data/maxcut/{graph_key}.txt"
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
        adjacency_matrix = th.zeros(num_nodes, num_nodes).to(device)
        for n0, n1, dist in edge_to_n0_n1_dist:
            n0_to_n1s[n0].append(n1)
            n0_to_dts[n0].append(dist)
            adjacency_matrix[n0][n1] = dist
            adjacency_matrix[n1][n0] = dist

        n0_to_n1s = [th.tensor(node1s, dtype=th.long, device=device) for node1s in n0_to_n1s]
        n0_to_dts = [th.tensor(node1s, dtype=th.long, device=device) for node1s in n0_to_dts]  # dists == 1

        assert num_nodes == len(n0_to_n1s)
        assert num_nodes == len(n0_to_dts)
        assert num_edges == sum([len(n0_to_n1) for n0_to_n1 in n0_to_n1s])
        assert num_edges == sum([len(n0_to_dt) for n0_to_dt in n0_to_dts])

        self.num_nodes = len(n0_to_n1s)
        self.num_edges = sum([len(n0_to_n1) for n0_to_n1 in n0_to_n1s])
        self.n0_to_n1s = n0_to_n1s
        self.adjacency_matrix = adjacency_matrix
        self.sparsity = self.num_edges / (self.num_nodes * self.num_nodes)
        self.device = device

        '''为了高性能计算，删掉了 n0_to_n1s 的空item'''
        v2_ids = [i for i, n1 in enumerate(n0_to_n1s) if n1.shape[0] > 0]
        self.v2_ids = v2_ids
        self.v2_n0_to_n1s = [n0_to_n1s[idx] for idx in v2_ids]
        self.v2_num_nodes = len(v2_ids)

    def get_neighbor_nodes(self, node: int):
        res = th.where(self.adjacency_matrix[node] > 0)[0]
        res = [int(i) for i in res]
        return res

    def get_objective(self, p0s):
        assert p0s.shape[-1] == self.num_nodes
        num_envs = p0s.shape[0]

        sum_dts = []
        for env_i in range(num_envs):
            p0 = p0s[env_i]
            n0_to_p1 = []
            for n1 in self.n0_to_n1s:
                p1 = p0[n1]
                n0_to_p1.append(p1)

            sum_dt = []
            for _p0, _p1 in zip(p0, n0_to_p1):
                # `_p0 * (1-_p1)` node_0 属于这个集合 且 node1 属于那个集合的概率
                # `_p1 * (1-_p0)` node_1 属于这个集合 且 node0 属于那个集合的概率
                # dt = _p0 * (1-_p1) + _p1 * (1-_p0)  # 等价于以下一行代码，相加计算出了这条边两端的节点分别属于两个集合的概率
                dt = _p0 + _p1 - 2 * _p0 * _p1
                # 此计算只能算出的局部梯度，与全局梯度有差别，未考虑无向图里节点间的复杂关系，但是没关系，我们不是直接用梯度去下降
                sum_dt.append(dt.sum(dim=0))
            sum_dt = th.stack(sum_dt).sum(dim=-1)
            sum_dts.append(sum_dt)
        sum_dts = th.hstack(sum_dts)
        return -sum_dts

    def get_objectives(self, p0s):
        assert p0s.shape[-1] == self.num_nodes
        n0s_to_p1 = self.get_n0s_to_p1(p0s)
        sum_dts = self.get_sum_dts_by_p0s_float(p0s, n0s_to_p1)
        return -sum_dts

    def get_scores(self, p0s):
        n0s_to_p1 = self.get_n0s_to_p1(p0s)
        sum_dts = self.get_sum_dts_by_p0s_int(p0s, n0s_to_p1)
        return sum_dts

    def get_n0s_to_p1(self, p0s):
        n0s_to_p1 = []
        num_envs = p0s.shape[0]

        env_is = th.arange(num_envs, device=self.device)
        for n1 in self.v2_n0_to_n1s:
            num_n1 = n1.shape[0]
            env_js = env_is.repeat(num_n1, 1).T.reshape(num_envs * num_n1)
            n1s = n1.repeat(num_envs)
            p1s = p0s[env_js, n1s].reshape(num_envs, num_n1)
            n0s_to_p1.append(p1s)
        return n0s_to_p1

    def get_sum_dts_by_p0s_float(self, p0s, n0s_to_p1):  # 计算节点不一定属于某个集合的分割距离，算出浮点数
        num_envs = p0s.shape[0]

        v2_p0s = p0s[:, self.v2_ids]
        v2_num_nodes = len(self.v2_ids)
        sum_dts = th.zeros((num_envs, v2_num_nodes), dtype=th.float32, device=self.device)
        for node_i in range(v2_num_nodes):
            _p0 = v2_p0s[:, node_i].unsqueeze(1)
            _p1 = n0s_to_p1[node_i]

            dt = _p0 + _p1 - 2 * _p0 * _p1
            sum_dts[:, node_i] = dt.sum(dim=-1)
        return sum_dts.sum(dim=-1)

    def get_sum_dts_by_p0s_int(self, p0s, n0s_to_p1):  # 计算节点一定属于某个集合的分割距离，算出正整数
        num_envs = p0s.shape[0]

        v2_p0s = p0s[:, self.v2_ids]
        v2_num_nodes = len(self.v2_ids)
        sum_dts = th.zeros((num_envs, v2_num_nodes), dtype=th.float32, device=self.device)
        for node_i in range(v2_num_nodes):
            _p0 = v2_p0s[:, node_i].unsqueeze(1)
            _p1 = n0s_to_p1[node_i]

            dt = _p0 ^ _p1
            sum_dts[:, node_i] = dt.sum(dim=-1)
        return sum_dts.sum(dim=-1)

    def get_rand_probs(self, num_envs: int):
        device = self.device
        return th.rand((num_envs, self.num_nodes), dtype=th.float32, device=device)

    @staticmethod
    def convert_prob_to_bool(p0s, thresh=0.5):
        return (p0s > thresh).to(th.int8)

    @staticmethod
    def node_prob_bool_to_str(x_bool: TEN) -> str:
        # x_bool = env.convert_prob_to_bool(x)
        x_int = int(''.join([str(i) for i in x_bool.tolist()]), 2)
        x_b64 = bin_int_to_str(x_int)
        x_str = '\n'.join([x_b64[i:i + 120] for i in range(0, len(x_b64), 120)])
        return x_str

    def node_prob_str_to_bool(self, x_str: str) -> TEN:
        x_b64: str = x_str.replace('\n', '')
        x_int: int = bin_str_to_int(x_b64)
        x_bin: str = bin(x_int)[2:]
        x_bool = th.zeros(self.num_nodes, dtype=th.int8)
        x_bool[-len(x_bin):] = th.tensor([int(i) for i in x_bin], dtype=th.int8)
        return x_bool


def bin_int_to_str(decimal: bin):
    base_digits = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789!@"
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
    base_digits = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789!@"
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

    # env = GraphMaxCutEnv(graph_key='gset_14')
    env = GraphMaxCutEnv(graph_key='gset_70')

    probs = env.get_rand_probs(num_envs=num_envs)
    print(env.get_objective(probs))
    print(env.get_objectives(probs))

    for thresh in th.linspace(0, 1, 8):
        objs = env.get_objectives(p0s=env.convert_prob_to_bool(probs, thresh))
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
    env = GraphMaxCutEnv(graph_key='gset_14')
    best_theta = env.node_prob_str_to_bool(ThetaGset_14)
    # env = GraphMaxCutEnv(graph_key='gset_70')
    # best_theta = env.node_prob_str_to_bool(ThetaGset_70)

    print(best_theta.shape)
    best_score = env.get_scores(best_theta.unsqueeze(0)).squeeze(0)
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
    graph_key = 'gset_14'
    # graph_key = 'gset_70'

    env = GraphMaxCutEnv(graph_key=graph_key, gpu_id=gpu_id)

    x_prob = env.get_rand_probs(num_envs=num_envs)[0]
    x_bool = env.convert_prob_to_bool(x_prob)

    x_str = env.node_prob_bool_to_str(x_bool)
    print(x_str)
    x_bool = env.node_prob_str_to_bool(x_str)

    assert all(x_bool == env.convert_prob_to_bool(x_prob))


"""Optimize with sequences"""


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


def train_optimizer_level1_update_theta_by_grad():
    gpu_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    num_envs = 2 ** 6
    graph_key = 'gset_14'
    # graph_key = 'gset_70'

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


def train_optimizer_level2_update_theta_by_adam():
    gpu_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    num_envs = 2 ** 4
    graph_key = 'gset_14'
    # graph_key = 'gset_70'

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
    best_theta = probs
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


def train_optimizer_level3_update_theta_by_opti():
    gpu_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    num_envs = 2 ** 4
    graph_key = 'gset_14'
    # graph_key = 'gset_70'

    '''hyper-parameters'''
    lr = 1e-3
    mid_dim = 2 ** 8
    num_layers = 1
    seq_len = 2 ** 4

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
    best_theta = probs
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
            print(f"|{used_time:9.0f}  {i:6}  {obj.item():9.3f}  {scores.max().item():9.0f}  best_score {best_score:9.0f}")
            

            max_score, max_id = th.max(scores, dim=0)
            if max_score > best_score:
                best_score = max_score
                best_theta = thetas[max_id]
                print(f"best_score {best_score}  best_theta \n{env.node_prob_bool_to_str(best_theta)}")

    print()
    print(f"best_score {best_score}  best_theta \n{env.node_prob_bool_to_str(best_theta)}")


if __name__ == '__main__':
    # check_env()
    # check_theta()
    # train_optimizer_level1_update_theta_by_grad()
    # train_optimizer_level2_update_theta_by_adam()
    train_optimizer_level3_update_theta_by_opti()
