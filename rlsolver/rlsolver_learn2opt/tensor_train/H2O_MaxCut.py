import sys
import time
import torch as th
import torch.nn as nn

TEN = th.Tensor
MapGraphToNodeEdge = {'g14': (800, 4694),
                      'g15': (800, 4661),
                      'g22': (2000, 19990),
                      'g49': (3000, 6000),
                      'g50': (3000, 6000),
                      'g70': (10000, 9999), }

"""Graph Max Cut Env"""


class GraphMaxCutEnv:
    def __init__(self, num_envs=8, graph_key: str = 'g70', device=th.device('cpu')):
        assert graph_key in MapGraphToNodeEdge

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

        self.num_envs = num_envs
        self.num_nodes = len(n0_to_n1s)
        self.num_edges = sum([len(n0_to_n1) for n0_to_n1 in n0_to_n1s])
        self.n0_to_n1s = n0_to_n1s
        self.device = device

        '''为了高性能计算，删掉了 n0_to_n1s 的空item'''
        v2_ids = [i for i, n1 in enumerate(n0_to_n1s) if n1.shape[0] > 0]
        self.v2_ids = v2_ids
        self.v2_n0_to_n1s = [n0_to_n1s[idx] for idx in v2_ids]
        self.v2_num_nodes = len(v2_ids)

    def get_objective(self, p0s):
        assert p0s.shape == (self.num_envs, self.num_nodes)

        sum_dts = []
        for env_i in range(self.num_envs):
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

    def get_objectives_v1(self, p0s):  # version 1
        device = p0s.device
        num_envs = self.num_envs

        n0s_to_p1 = []
        env_is = th.arange(self.num_envs, device=device)
        for n1 in self.n0_to_n1s:
            num_n1 = n1.shape[0]
            if num_n1 == 0:  # 为了高性能计算，可将 n0_to_n1s 的空item 删掉
                p1s = th.zeros((num_envs, 0), dtype=th.float32, device=device)
            else:
                env_js = env_is.repeat(num_n1, 1).T.reshape(num_envs * num_n1)
                n1s = n1.repeat(num_envs)
                p1s = p0s[env_js, n1s].reshape(num_envs, num_n1)
            n0s_to_p1.append(p1s)

        num_nodes = self.num_nodes
        sum_dts = th.zeros((num_envs, num_nodes), dtype=th.float32, device=device)
        for node_i in range(num_nodes):
            _p0 = p0s[:, node_i].unsqueeze(1)
            _p1 = n0s_to_p1[node_i]

            dt = _p0 + _p1 - 2 * _p0 * _p1
            sum_dts[:, node_i] = dt.sum(dim=-1)
        return -sum_dts.sum(dim=-1)

    def get_objectives(self, p0s):  # version 2
        n0s_to_p1 = self.get_n0s_to_p1(p0s)
        sum_dts = self.get_sum_dts_by_p0s_float(p0s, n0s_to_p1)
        return -sum_dts

    def get_scores(self, p0s):  # version 2
        n0s_to_p1 = self.get_n0s_to_p1(p0s)
        sum_dts = self.get_sum_dts_by_p0s_int(p0s, n0s_to_p1)
        return sum_dts

    def get_n0s_to_p1(self, p0s):
        n0s_to_p1 = []
        env_is = th.arange(self.num_envs, device=self.device)
        for n1 in self.v2_n0_to_n1s:
            num_n1 = n1.shape[0]
            env_js = env_is.repeat(num_n1, 1).T.reshape(self.num_envs * num_n1)
            n1s = n1.repeat(self.num_envs)
            p1s = p0s[env_js, n1s].reshape(self.num_envs, num_n1)
            n0s_to_p1.append(p1s)
        return n0s_to_p1

    def get_sum_dts_by_p0s_float(self, p0s, n0s_to_p1):  # 计算节点不一定属于某个集合的分割距离，算出浮点数
        v2_p0s = p0s[:, self.v2_ids]
        v2_num_nodes = len(self.v2_ids)
        sum_dts = th.zeros((self.num_envs, v2_num_nodes), dtype=th.float32, device=self.device)
        for node_i in range(v2_num_nodes):
            _p0 = v2_p0s[:, node_i].unsqueeze(1)
            _p1 = n0s_to_p1[node_i]

            dt = _p0 + _p1 - 2 * _p0 * _p1
            sum_dts[:, node_i] = dt.sum(dim=-1)
        return sum_dts.sum(dim=-1)

    def get_sum_dts_by_p0s_int(self, p0s, n0s_to_p1):  # 计算节点一定属于某个集合的分割距离，算出正整数
        v2_p0s = p0s[:, self.v2_ids]
        v2_num_nodes = len(self.v2_ids)
        sum_dts = th.zeros((self.num_envs, v2_num_nodes), dtype=th.float32, device=self.device)
        for node_i in range(v2_num_nodes):
            _p0 = v2_p0s[:, node_i].unsqueeze(1)
            _p1 = n0s_to_p1[node_i]

            dt = _p0 ^ _p1
            sum_dts[:, node_i] = dt.sum(dim=-1)
        return sum_dts.sum(dim=-1)

    def get_rand_p0s(self):
        device = self.device
        return th.rand((self.num_envs, self.num_nodes), dtype=th.float32, device=device)

    @staticmethod
    def convert_prob_to_bool(p0s, thresh=0.5):
        return (p0s > thresh).to(th.int8)

    @staticmethod
    def convert_prob_to_float01(p0s, thresh=0.5):
        return (p0s > thresh).to(th.int8)

    @staticmethod
    def node_prob_bool_to_str(x_bool: TEN) -> str:
        # x_bool = env.convert_prob_to_bool(x)
        x_int = int(''.join([str(i) for i in x_bool.tolist()]), 2)
        x_b64 = int_b10_to_str_b64(x_int)
        x_str = '\n'.join([x_b64[i:i + 120] for i in range(0, len(x_b64), 120)])
        return x_str

    @staticmethod
    def node_prob_str_to_bool(x_str: str) -> TEN:
        x_b64 = x_str.replace('\n', '')
        x_int = str_b64_to_int_b10(x_b64)
        x_bool = th.tensor([int(i) for i in bin(x_int)[2:]], dtype=th.int8)
        return x_bool


def check_env():
    th.manual_seed(0)
    env = GraphMaxCutEnv(num_envs=6, graph_key='g14')

    p0s = env.get_rand_p0s()
    print(env.get_objective(p0s))
    print(env.get_objectives_v1(p0s))
    print(env.get_objectives(p0s))

    for thresh in th.linspace(0, 1, 32):
        objs = env.get_objectives(p0s=env.convert_prob_to_bool(p0s, thresh))
        print(f"{thresh.item():6.3f}  {objs.numpy()}")


def int_b10_to_str_b64(decimal):
    base64_digits = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789!@"

    if decimal == 0:
        return base64_digits[0]

    base64 = ""
    while decimal > 0:
        remainder = decimal % 64
        base64 = base64_digits[remainder] + base64
        decimal //= 64

    return base64


def str_b64_to_int_b10(base64):
    base64_digits = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789!@"
    base64_length = len(base64)
    decimal = 0

    for i in range(base64_length):
        digit = base64_digits.index(base64[i])
        power = base64_length - 1 - i
        decimal += digit * (64 ** power)

    return decimal


def check_convert_between_b10_and_b64():
    print()
    i = 2 ** 800
    j = int_b10_to_str_b64(i)
    print(len(str(j)), j)
    i = str_b64_to_int_b10(j)
    print(len(str(i)), i)
    b = bin(i)
    print(len(str(b)), b)

    print()
    i = 2 ** 5000
    j = int_b10_to_str_b64(i)
    print(len(str(j)), j)
    i = str_b64_to_int_b10(j)
    print(len(str(i)), i)
    b = bin(i)
    print(len(str(b)), b)


def convert_between_str_and_bool():
    gpu_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    num_envs = 1
    graph_key = 'g14'
    # graph_key = 'g70'

    device = th.device(f'cuda:{gpu_id}' if th.cuda.is_available() and gpu_id >= 0 else 'cpu')

    env = GraphMaxCutEnv(num_envs=num_envs, graph_key=graph_key, device=device)

    x_prob = env.get_rand_p0s()[0]
    x_bool = env.convert_prob_to_bool(x_prob)

    x_str = env.node_prob_bool_to_str(x_bool)
    print(x_str)
    x_bool = env.node_prob_str_to_bool(x_str)

    assert all(x_bool == env.convert_prob_to_bool(x_prob))


"""Learn to optimize with Hamilton term"""


class OptimizerTask(nn.Module):
    def __init__(self, thetas):
        super().__init__()
        self.thetas = nn.Parameter(thetas.detach(), requires_grad=True)

    def forward(self):
        return self.thetas

    def do_regularization(self):
        self.thetas.data.clamp_(0, 1)


class NnSeqBnMLP(nn.Module):
    def __init__(self, inp_dim, mid_dim, out_dim):
        super(NnSeqBnMLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(inp_dim, mid_dim), nn.BatchNorm1d(mid_dim), nn.ReLU(),
            nn.Linear(mid_dim, out_dim),
        )

    def forward(self, seq):
        d0, d1, d2 = seq.shape
        inp = seq.view(d0 * d1, -1)
        out = self.mlp(inp)
        return out.view(d0, d1, -1)


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


class OptimizerOpti4(nn.Module):
    def __init__(self, inp_dim, mid_dim, out_dim, num_layers):
        super().__init__()
        self.inp_dim = inp_dim
        self.mid_dim = mid_dim
        self.out_dim = out_dim
        self.num_layers = num_layers

        self.rnn0 = nn.LSTM(inp_dim * 2, mid_dim, num_layers=num_layers)
        self.mlp0 = nn.Linear(mid_dim, out_dim)
        self.rnn1 = nn.LSTM(2, 8, num_layers=num_layers)
        self.mlp1 = nn.Linear(8, 1)

    def forward(self, inp, the, hid0=None, hid1=None):
        tmp0, hid0 = self.rnn0(th.cat((inp, the), dim=2), hid0)
        out0 = self.mlp0(tmp0)

        d0, d1, d2 = inp.shape
        inp1 = inp.reshape(d0, d1 * d2, 1)
        the1 = the.reshape(d0, d1 * d2, 1)
        tmp1, hid1 = self.rnn1(th.cat((inp1, the1), dim=2), hid1)
        out1 = self.mlp1(tmp1).reshape(d0, d1, d2)

        out = out0 + out1
        return out, hid0, hid1


def train_optimizer_level1_update_theta_by_grad():
    gpu_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    num_envs = 2 ** 6
    graph_key = 'g14'
    # graph_key = 'g70'

    device = th.device(f'cuda:{gpu_id}' if th.cuda.is_available() and gpu_id >= 0 else 'cpu')

    '''hyper-parameters'''
    lr = 1e-3
    opt_num = 2 ** 12
    eval_gap = 2 ** 6

    '''init task'''
    env = GraphMaxCutEnv(num_envs=num_envs, graph_key=graph_key, device=device)

    thetas = env.get_rand_p0s()
    thetas.requires_grad_(True)

    '''init opti'''
    # opt_base = th.optim.Adam([thetas, ], lr=lr)

    for i in range(1, opt_num + 1):
        obj = env.get_objectives(thetas).mean()

        # opt_base.zero_grad()
        obj.backward()
        # opt_base.step()

        grad_s = thetas.grad.data
        thetas.data.add_(-lr * grad_s).clip_(0, 1)

        if i % eval_gap == 0:
            scores = env.get_scores(env.convert_prob_to_bool(thetas))
            score = scores.max()

            print(f"{i:6}  {obj.item():9.3f}  {score:9.3f}")


def train_optimizer_level2_update_theta_by_adam():
    gpu_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    num_envs = 2 ** 6
    graph_key = 'g14'
    # graph_key = 'g70'

    device = th.device(f'cuda:{gpu_id}' if th.cuda.is_available() and gpu_id >= 0 else 'cpu')

    '''hyper-parameters'''
    lr = 1e-3
    opt_num = 2 ** 12
    eval_gap = 2 ** 6

    '''init task'''
    env = GraphMaxCutEnv(num_envs=num_envs, graph_key=graph_key, device=device)

    thetas = env.get_rand_p0s()
    thetas.requires_grad_(True)

    '''init opti'''
    opt_base = th.optim.Adam([thetas, ], lr=lr)

    for i in range(1, opt_num + 1):
        obj = env.get_objectives(thetas).mean()

        opt_base.zero_grad()
        obj.backward()
        opt_base.step()

        thetas.data.clip_(0, 1)

        if i % eval_gap == 0:
            scores = env.get_scores(env.convert_prob_to_bool(thetas))
            score = scores.max()

            print(f"{i:6}  {obj.item():9.3f}  {score:9.3f}")


def train_optimizer_level3_update_theta_by_opti():
    gpu_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    num_envs = 2 ** 6
    graph_key = 'g14'
    # graph_key = 'g70'

    device = th.device(f'cuda:{gpu_id}' if th.cuda.is_available() and gpu_id >= 0 else 'cpu')

    '''hyper-parameters'''
    lr = 1e-3
    mid_dim = 2 ** 8
    num_layers = 1

    opt_num = 2 ** 3

    print(f"gpu_id {gpu_id}, num_envs {num_envs}, graph_key {graph_key}, opt_num {opt_num}, ")

    eval_gap = 2 ** 6
    train_times = 2 ** 12

    '''init task'''
    env = GraphMaxCutEnv(num_envs=num_envs, graph_key=graph_key, device=device)
    dim = env.num_nodes

    thetas = env.get_rand_p0s()
    thetas.requires_grad_(True)
    obj = None
    hidden0 = None
    hidden1 = None

    '''init opti'''
    opt_opti = OptimizerOpti(inp_dim=dim, mid_dim=mid_dim, out_dim=dim, num_layers=num_layers).to(device)
    opt_base = th.optim.Adam(opt_opti.parameters(), lr=lr)

    for j in range(1, train_times + 1):
        thetas_ = thetas.clone()
        updates = []

        for i in range(opt_num):
            score = env.get_objectives(thetas)
            obj = score.mean()
            obj.backward()

            grad_s = thetas.grad.data
            update, hidden0, hidden1 = opt_opti(grad_s.unsqueeze(0), hidden0, hidden1)
            update = (update.squeeze_(0) - grad_s) * lr
            updates.append(update)
            thetas.data.add_(update).clip_(0, 1)
        hidden0 = [h.detach() for h in hidden0]
        hidden1 = [h.detach() for h in hidden1]

        updates = th.stack(updates, dim=0)
        thetas_ = (thetas_ + updates.mean(0)).clip(0, 1)
        obj_ = env.get_objectives(thetas_).mean()

        opt_base.zero_grad()
        obj_.backward()
        opt_base.step()

        thetas.data[:] = thetas_

        if j % eval_gap == 0:
            scores = env.get_scores(env.convert_prob_to_bool(thetas))
            score = scores.max()

            print(f"{j:6}  {obj.item():9.3f}  {score:9.3f}")


def train_optimizer_level4_update_theta_by_opti():
    gpu_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    num_envs = 2 ** 8
    graph_key = 'g14'
    # graph_key = 'g70'

    device = th.device(f'cuda:{gpu_id}' if th.cuda.is_available() and gpu_id >= 0 else 'cpu')

    '''hyper-parameters'''
    lr = 1e-3
    mid_dim = 2 ** 8
    num_layers = 1

    opt_num = 2 ** 3

    print(f"gpu_id {gpu_id}, num_envs {num_envs}, graph_key {graph_key}, opt_num {opt_num}, ")

    eval_gap = 2 ** 6
    train_times = 2 ** 12

    '''init task'''
    env = GraphMaxCutEnv(num_envs=num_envs, graph_key=graph_key, device=device)
    dim = env.num_nodes

    thetas = env.get_rand_p0s()
    thetas.requires_grad_(True)
    hidden0 = None
    hidden1 = None

    '''init opti'''
    opt_opti = OptimizerOpti(inp_dim=dim, mid_dim=mid_dim, out_dim=dim, num_layers=num_layers).to(device)
    opt_base = th.optim.Adam(opt_opti.parameters(), lr=lr)

    for j in range(1, train_times + 1):
        thetas_ = thetas.clone()
        updates = []

        for i in range(opt_num):
            # obj = env.get_objectives(thetas).mean()
            # obj.backward()
            #
            # grad_s = thetas.grad.data
            temp_s = env.convert_prob_to_float01(thetas)  # todo
            update, hidden0, hidden1 = opt_opti(temp_s.unsqueeze(0), hidden0, hidden1)
            update = (update.squeeze_(0)) * lr
            updates.append(update)
            thetas.data.add_(update).clip_(0, 1)
        hidden0 = [h.detach() for h in hidden0]
        hidden1 = [h.detach() for h in hidden1]

        updates = th.stack(updates, dim=0)
        thetas_ = (thetas_ + updates.mean(0)).clip(0, 1)

        obj = env.get_objectives(thetas_).mean()

        opt_base.zero_grad()
        obj.backward()
        opt_base.step()

        thetas.data[:] = thetas_

        if j % eval_gap == 0:
            scores = env.get_scores(env.convert_prob_to_bool(thetas))
            score = scores.max()

            print(f"{j:6}  {obj.item():9.3f}  {score:9.3f}")


if __name__ == '__main__':
    # train_optimizer_level3_update_theta_by_opti()
    convert_between_str_and_bool()
