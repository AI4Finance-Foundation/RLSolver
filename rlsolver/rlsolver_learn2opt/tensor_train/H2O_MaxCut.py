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


class OptimizerOpti0(nn.Module):
    def __init__(self, inp_dim, mid_dim, out_dim, num_layers):
        super().__init__()
        self.inp_dim = inp_dim
        self.mid_dim = mid_dim
        self.out_dim = out_dim
        self.num_layers = num_layers

        self.rnn = nn.LSTM(inp_dim, mid_dim, num_layers=num_layers)
        self.mlp = nn.Linear(mid_dim, out_dim)
        # self.mlp = NnSeqBnMLP(inp_dim=mid_dim, mid_dim=mid_dim, out_dim=out_dim)

    def forward(self, inp, hid=None):
        tmp, hid = self.rnn(inp, hid)
        out = self.mlp(tmp)
        return out, hid


class OptimizerOpti(nn.Module):
    def __init__(self, inp_dim, mid_dim, out_dim, num_layers):
        super().__init__()
        self.inp_dim = inp_dim
        self.mid_dim = mid_dim
        self.out_dim = out_dim
        self.num_layers = num_layers

        self.rnn0 = nn.LSTM(inp_dim, mid_dim, num_layers=num_layers)
        self.mlp0 = NnSeqBnMLP(inp_dim=mid_dim, mid_dim=mid_dim, out_dim=out_dim)
        self.rnn1 = nn.LSTM(1, mid_dim, num_layers=num_layers)
        self.mlp1 = NnSeqBnMLP(inp_dim=mid_dim, mid_dim=mid_dim, out_dim=1)

    def forward(self, inp, hid0=None, hid1=None):
        tmp0, hid0 = self.rnn0(inp, hid0)
        out0 = self.mlp0(tmp0)

        d0, d1, d2 = inp.shape
        inp1 = inp.reshape(d0, d1 * d2, 1)
        tmp1, hid1 = self.rnn1(inp1, hid1)
        out1 = self.mlp1(tmp1).view(d0, d1, d2)

        out = out0 + out1
        return out, hid0, hid1


def train_optimizer_level1():
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

    xs = env.get_rand_p0s()

    '''init opti'''
    opt_task = OptimizerTask(thetas=xs).to(device)
    opt_base = th.optim.Adam(opt_task.parameters(), lr=lr)

    for i in range(1, opt_num + 1):
        thetas = opt_task()
        opt_task.do_regularization()

        obj = env.get_objectives(thetas).mean()
        all_loss = obj

        opt_base.zero_grad()
        all_loss.backward()
        opt_base.step()

        if i % eval_gap == 0:
            scores = env.get_scores(env.convert_prob_to_bool(thetas))
            score = scores.max()

            print(f"{i:6}  {obj.item():9.3f}  {score:9.3f}")


def train_optimizer_level2():
    gpu_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    num_envs = 2 ** 6
    graph_key = 'g14'
    # graph_key = 'g70'

    device = th.device(f'cuda:{gpu_id}' if th.cuda.is_available() and gpu_id >= 0 else 'cpu')

    '''hyper-parameters'''
    lr = 1e-3
    # mid_dim = 2 ** 4
    # num_layers = 2

    # unroll = 2 ** 3
    opt_num = 2 ** 12
    eval_gap = 2 ** 5

    '''init task'''
    env = GraphMaxCutEnv(num_envs=num_envs, graph_key=graph_key, device=device)
    # dim = env.num_nodes

    xs = env.get_rand_p0s()

    '''init opti'''
    opt_task = OptimizerTask(thetas=xs).to(device)
    # opt_opti = OptimizerOpti(inp_dim=dim, mid_dim=mid_dim, out_dim=dim, num_layers=num_layers).to(device)
    # opt_base = th.optim.Adam(opt_opti.parameters(), lr=lr)

    for i in range(1, opt_num + 1):
        thetas = opt_task()
        opt_task.do_regularization()

        obj = env.get_objectives(thetas).mean()
        # obj_list.append(obj)
        # if i % unroll == 0:
        #     all_loss = th.stack(obj_list[-unroll:]).mean()
        #
        #     opt_base.zero_grad()
        #     all_loss.backward()
        #     opt_base.step()

        all_loss = obj
        all_loss.backward()

        thetas.data.add_(-lr * thetas.grad.data)
        # opt_base.step()
        # opt_base.zero_grad()

        if i % eval_gap == 0:
            scores = env.get_scores(env.convert_prob_to_bool(thetas))
            score = scores.max()

            print(f"{i:6}  {obj.item():9.3f}  {score:9.3f}")


def train_optimizer_level3():
    gpu_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    num_envs = 2 ** 6
    graph_key = 'g14'
    # graph_key = 'g70'

    device = th.device(f'cuda:{gpu_id}' if th.cuda.is_available() and gpu_id >= 0 else 'cpu')

    '''hyper-parameters'''
    lr = 1e-3
    mid_dim = 2 ** 4
    num_layers = 1

    # unroll = 2 ** 3
    opt_num = 2 ** 8  # larger is better, until reached memory limit.
    eval_gap = 2 ** 1
    train_times = 2 ** 9

    '''init task'''
    env = GraphMaxCutEnv(num_envs=num_envs, graph_key=graph_key, device=device)
    dim = env.num_nodes

    xs = env.get_rand_p0s()

    '''init opti'''
    opt_task = OptimizerTask(thetas=xs).to(device)
    opt_opti = OptimizerOpti(inp_dim=dim, mid_dim=mid_dim, out_dim=dim, num_layers=num_layers).to(device)
    opt_base = th.optim.Adam(opt_opti.parameters(), lr=lr)

    thetas = opt_task.thetas
    opt_task.do_regularization()
    obj = env.get_objectives(thetas).mean()
    obj.backward()

    for j in range(1, train_times + 1):
        hidden0 = None
        hidden1 = None
        for i in range(opt_num):
            grad_s = thetas.grad.data
            update, hidden0, hidden1 = opt_opti(grad_s.unsqueeze(0), hidden0, hidden1)

            thetas.data.add_(-lr * (grad_s + update.squeeze(0)))
            opt_task.do_regularization()
            obj = env.get_objectives(thetas).mean()

            opt_base.zero_grad()
            obj.backward()
            opt_base.step()

        if j % eval_gap == 0:
            scores = env.get_scores(env.convert_prob_to_bool(thetas))
            score = scores.max()

            print(f"{j:6}  {obj.item():9.3f}  {score:9.3f}")


if __name__ == '__main__':
    # train_optimizer_level1()
    # train_optimizer_level2()
    train_optimizer_level3()
