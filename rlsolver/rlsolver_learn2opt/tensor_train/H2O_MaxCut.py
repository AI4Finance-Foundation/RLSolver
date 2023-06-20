
import sys
import time
import torch as th
import torch.nn as nn

try:  # ObjectiveMISO requires functorch.vmap
    from functorch import vmap
except ImportError:
    vmap = None  # Run MISO need

TEN = th.Tensor
MapGraphToNodeEdge = {'g14': (800, 4694),
                      'g15': (800, 4661),
                      'g22': (2000, 19990),
                      'g49': (3000, 6000),
                      'g50': (3000, 6000),
                      'g70': (10000, 9999), }

"""Learn To Optimize + Hamilton Term

想要使用 pytorch 的 functorch.vmap，就要先安装 functorch

1. 先安装 conda

2. 打开终端并输入以下命令以添加阿里云的镜像源：
conda config --add channels https://mirrors.aliyun.com/anaconda/pkgs/free/
conda config --add channels https://mirrors.aliyun.com/anaconda/pkgs/main/
conda config --add channels https://mirrors.aliyun.com/anaconda/cloud/pytorch/
conda config --add channels https://mirrors.aliyun.com/anaconda/cloud/conda-forge/

3. 然后，输入以下命令创建一个新的 conda 环境并安装 functorch：
conda create --name myenv
conda activate myenv
conda install pytorch torchvision torchaudio cpuonly -c pytorch
pip install functorch

4. 最后，运行以下命令以确认是否安装成功：
python -c "import functorch; print(functorch.__version__)"
"""


class GraphMaxCutEnv:
    def __init__(self, num_envs=8, graph_key: str = 'g70', device=th.device('cpu')):
        assert graph_key in MapGraphToNodeEdge

        txt_path = f"./gset_{graph_key}.txt"
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
        p1: the probability of node0 is in set, (1-p1): node0 is in another set
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
                # dt = _p0 * (1-_p1) + _p1 * (1-_p0)  # 等价于以下一行代码
                dt = _p0 + _p1 - 2 * _p0 * _p1
                sum_dt.append(dt.sum(dim=0))
            sum_dt = th.stack(sum_dt).sum(dim=-1)
            sum_dts.append(sum_dt)
        sum_dts = th.hstack(sum_dts)
        return sum_dts

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
        return sum_dts.sum(dim=-1)

    def get_objectives(self, p0s):  # version 2
        n0s_to_p1 = self.get_n0s_to_p1(p0s)
        sum_dts = self.get_sum_dts_by_p0s_float(p0s, n0s_to_p1)
        return sum_dts

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

    def get_sum_dts_by_p0s_float(self, p0s, n0s_to_p1):
        v2_p0s = p0s[:, self.v2_ids]
        v2_num_nodes = len(self.v2_ids)
        sum_dts = th.zeros((self.num_envs, v2_num_nodes), dtype=th.float32, device=self.device)
        for node_i in range(v2_num_nodes):
            _p0 = v2_p0s[:, node_i].unsqueeze(1)
            _p1 = n0s_to_p1[node_i]

            dt = _p0 + _p1 - 2 * _p0 * _p1
            sum_dts[:, node_i] = dt.sum(dim=-1)
        return sum_dts.sum(dim=-1)

    def get_sum_dts_by_p0s_int(self, p0s, n0s_to_p1):
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
    env = GraphMaxCutEnv(num_envs=6, graph_key='g70')

    p0s = env.get_rand_p0s()
    print(env.get_objective(p0s))
    print(env.get_objectives_v1(p0s))
    print(env.get_objectives(p0s))

    for thresh in th.linspace(0, 1, 32):
        objs = env.get_objectives(p0s=env.convert_prob_to_bool(p0s, thresh))
        print(f"{thresh.item():6.3f}  {objs.numpy()}")


class ObjectiveTask:
    def __init__(self, num_envs, num_evals, device):
        self.num_envs = num_envs
        self.num_eval = num_evals
        self.device = device

        self.env = GraphMaxCutEnv(num_envs=num_envs, graph_key='g14', device=device)
        self.dim = self.env.num_nodes

    def get_objectives(self, xs) -> TEN:
        return -self.env.get_objectives(xs)

    def get_scores(self, xs) -> TEN:
        return self.env.get_scores(xs)

    @staticmethod
    def get_norm(xs):
        return xs.clip(0, 1)

    def get_init_xs(self, num: int):
        return th.rand((num, self.dim), dtype=th.float32, device=self.device)


class OptimizerTask(nn.Module):
    def __init__(self, num, dim, device, thetas=None):
        super().__init__()
        self.num = num
        self.dim = dim
        self.device = device

        with th.no_grad():
            if thetas is None:
                thetas = th.randn((self.num, self.dim), requires_grad=True, device=device)
                thetas = (thetas - thetas.mean(dim=-1, keepdim=True)) / (thetas.std(dim=-1, keepdim=True) + 1e-6)
                thetas = thetas.clamp(-3, +3)
            else:
                thetas = thetas.clone().detach()
                assert thetas.shape[0] == num
        self.register_buffer('thetas', thetas.requires_grad_(True))

    def re_init(self, num, thetas=None):
        self.__init__(num=num, dim=self.dim, device=self.device, thetas=thetas)

    def get_outputs(self):
        return self.thetas


class OptimizerOpti(nn.Module):
    def __init__(self, inp_dim: int, hid_dim: int):
        super().__init__()
        self.inp_dim = inp_dim
        self.hid_dim = hid_dim
        self.num_rnn = 2

        self.activation = nn.Tanh()
        self.recurs1 = nn.GRUCell(inp_dim, hid_dim)
        self.recurs2 = nn.GRUCell(hid_dim, hid_dim)
        self.output0 = nn.Linear(hid_dim * self.num_rnn, inp_dim)
        layer_init_with_orthogonal(self.output0, std=0.1)

    def forward(self, inp0, hid_):
        hid1 = self.activation(self.recurs1(inp0, hid_[0]))
        hid2 = self.activation(self.recurs2(hid1, hid_[1]))

        hid = th.cat((hid1, hid2), dim=1)
        out = self.output0(hid)
        return out, (hid1, hid2)


def layer_init_with_orthogonal(layer, std=1.0, bias_const=1e-6):
    th.nn.init.orthogonal_(layer.weight, std)
    th.nn.init.constant_(layer.bias, bias_const)


def opt_loop(
        obj_task: ObjectiveTask,
        opt_opti: OptimizerOpti,
        opt_task: OptimizerTask,
        opt_base: th.optim,
        num_opt: int,
        unroll: int,
        device: th.device,
        if_train: bool = True,
):
    if if_train:
        opt_opti.train()
        num = obj_task.num_envs
    else:
        opt_opti.eval()
        num = obj_task.num_eval

    thetas = obj_task.get_init_xs(num=num)
    opt_task.re_init(num=num, thetas=thetas)

    opt_task.zero_grad()

    hid_dim = opt_opti.hid_dim
    hid_state1 = [th.zeros((num, hid_dim), device=device) for _ in range(opt_opti.num_rnn)]

    outputs_list = []
    losses_list = []
    all_losses = []

    th.set_grad_enabled(True)
    for iteration in range(1, num_opt + 1):
        outputs = opt_task.get_outputs()
        outputs = obj_task.get_norm(outputs)

        losses = obj_task.get_objectives(outputs)
        loss = losses.mean()
        loss.backward(retain_graph=True)

        all_losses.append(losses)

        '''record for selecting best output'''
        outputs_list.append(outputs.clone())
        losses_list.append(losses.clone())

        '''params update with gradient'''
        thetas = opt_task.thetas
        gradients = thetas.grad.detach().clone().requires_grad_(True)

        updates, hid_states2 = opt_opti(gradients, hid_state1)

        result = thetas + updates
        result = obj_task.get_norm(result)
        result.retain_grad()
        result_params = {'thetas': result}

        if if_train:
            if iteration % unroll == 0:
                # all_loss = th.min(th.stack(all_losses[iteration - unroll:iteration]), dim=0)[0].mean()
                all_loss = th.stack(all_losses[iteration - unroll:iteration]).mean()
                opt_base.zero_grad()
                all_loss.backward()
                opt_base.step()

                opt_task.re_init(num=num)
                opt_task.load_state_dict(result_params)
                opt_task.zero_grad()

                hid_state1 = [ten.detach().clone().requires_grad_(True) for ten in hid_states2]
            else:
                opt_task.thetas = result_params['thetas']

                hid_state1 = hid_states2
        else:
            opt_task.re_init(num=num)
            opt_task.load_state_dict(result_params)
            opt_task.zero_grad()

            hid_state1 = [ten.detach().clone().requires_grad_(True) for ten in hid_states2]

    th.set_grad_enabled(False)

    '''record for selecting best output'''
    losses_list = th.stack(losses_list)
    min_losses, ids = th.min(losses_list, dim=0)

    outputs_list = th.stack(outputs_list)
    best_outputs = outputs_list[ids, th.arange(num, device=device)]

    return best_outputs, min_losses


"""run"""


def train_optimizer():
    gpu_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0

    '''train'''
    train_times = 2 ** 10
    num = 2 ** 3  # batch_size
    lr = 8e-4
    unroll = 4  # step of Hamilton Term
    num_opt = 16
    hid_dim = 2 ** 7

    '''eval'''
    eval_gap = 32

    '''init task'''
    device = th.device(f'cuda:{gpu_id}' if th.cuda.is_available() and gpu_id >= 0 else 'cpu')
    obj_task = ObjectiveTask(num_envs=num, num_evals=num, device=device)
    dim = obj_task.dim

    '''init opti'''
    opt_task = OptimizerTask(num=num, dim=dim, device=device)
    opt_opti = OptimizerOpti(inp_dim=dim, hid_dim=hid_dim).to(device)
    opt_base = th.optim.Adam(opt_opti.parameters(), lr=lr)

    '''loop'''
    print('training start')
    start_time = time.time()
    for i in range(train_times + 1):
        opt_loop(
            obj_task=obj_task, opt_task=opt_task, opt_opti=opt_opti,
            num_opt=num_opt, device=device, unroll=unroll, opt_base=opt_base, if_train=True)

        if i % eval_gap == 1:
            best_results, min_losses = opt_loop(
                obj_task=obj_task, opt_task=opt_task, opt_opti=opt_opti,
                num_opt=num_opt, device=device, unroll=unroll, opt_base=opt_base, if_train=False)

            time_used = round((time.time() - start_time))
            print(f"{'H2O':>8} {repr(best_results)} {repr(min_losses)}   "
                  f"TimeUsed {time_used:9}")
    print('training stop')


if __name__ == '__main__':
    train_optimizer()
    # check_env()

from H2O_MaxCut import *


class OptimizerTask(nn.Module):
    def __init__(self, thetas):
        super().__init__()
        self.thetas = nn.Parameter(thetas.detach(), requires_grad=True)

    def forward(self):
        return self.thetas

    def do_regularization(self):
        with th.no_grad():
            self.thetas[:] = self.thetas.clip(0, 1)


class NnSeqBnMLP(nn.Module):
    def __init__(self, inp_dim, mid_dim, out_dim):
        super(NnSeqBnMLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(inp_dim, mid_dim), nn.BatchNorm1d(mid_dim), nn.ReLU(),
            nn.Linear(mid_dim, out_dim),
        )
        layer_init_with_orthogonal(self.mlp[-1], std=0.1)

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

        self.rnn = nn.LSTM(inp_dim, mid_dim, num_layers=num_layers)
        self.mlp = NnSeqBnMLP(inp_dim=mid_dim, mid_dim=mid_dim, out_dim=out_dim)

    def forward(self, inp, hid=None):
        tmp, hid = self.rnn(inp, hid)
        out = self.mlp(tmp)
        return out, hid


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
    lr = 1e-2
    mid_dim = 2 ** 4
    num_layers = 2

    # unroll = 2 ** 3
    opt_num = 2 ** 12
    eval_gap = 2 ** 5

    '''init task'''
    env = GraphMaxCutEnv(num_envs=num_envs, graph_key=graph_key, device=device)
    dim = env.num_nodes

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

    # if gpu_id in {7, }:
    #     graph_key = 'g70'
    # elif gpu_id in {6, 5, 0, -1}:
    #     graph_key = 'g14'
    # else:
    #     raise ValueError(f"GPU_ID {gpu_id}")

    device = th.device(f'cuda:{gpu_id}' if th.cuda.is_available() and gpu_id >= 0 else 'cpu')

    '''hyper-parameters'''
    lr = 1e-2
    mid_dim = 2 ** 4
    num_layers = 2

    # unroll = 2 ** 3
    opt_num = 2 ** 12
    eval_gap = 2 ** 5

    '''init task'''
    env = GraphMaxCutEnv(num_envs=num_envs, graph_key=graph_key, device=device)
    dim = env.num_nodes

    '''init opti'''
    # opt_task = OptimizerTask(thetas=xs).to(device)
    thetas = env.get_rand_p0s().requires_grad_(True)

    opt_opti = OptimizerOpti(inp_dim=dim, mid_dim=mid_dim, out_dim=dim, num_layers=num_layers).to(device)
    opt_base = th.optim.Adam(opt_opti.parameters(), lr=lr)

    hidden = None
    for i in range(1, opt_num + 1):
        obj = env.get_objectives(thetas).mean()
        obj.backward()
        # obj_list.append(obj)
        # if i % unroll == 0:
        #     all_loss = th.stack(obj_list[-unroll:]).mean()
        #
        #     opt_base.zero_grad()
        #     all_loss.backward()
        #     opt_base.step()

        grad_s = thetas.grad.data
        update, hidden = opt_opti(grad_s.unsqueeze(0), hidden)
        update = update.squeeze(0)

        thetas.data = (thetas + update).clip(0, 1)

        opt_base.zero_grad()
        obj = env.get_objectives(thetas).mean()
        obj.backward()
        opt_base.step()

        if i % eval_gap == 0:
            scores = env.get_scores(env.convert_prob_to_bool(thetas))
            score = scores.max()

            print(f"{i:6}  {obj.item():9.3f}  {score:9.3f}")


if __name__ == '__main__':
    train_optimizer_level1()
    # train_optimizer_level2()
    # train_optimizer_level3()

