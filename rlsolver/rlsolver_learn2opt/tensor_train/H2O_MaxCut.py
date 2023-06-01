import sys
import time
import torch as th
import torch.nn as nn

try:  # ObjectiveMISO requires functorch.vmap
    from functorch import vmap
except ImportError:
    vmap = None  # Run MISO need

TEN = th.Tensor

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


class ObjectiveTask:
    def __init__(self, *args):
        self.num = None
        self.num_eval = None
        self.dims = None
        self.args = ()

    def get_args_for_train(self):
        return self.args

    def get_args_for_eval(self):
        return self.args

    @staticmethod
    def get_objectives(*_args) -> TEN:
        return th.zeros()

    @staticmethod
    def get_norm(x):
        return x

    def get_thetas(self, num: int):
        return None


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
        obj_args = obj_task.get_args_for_train()
        num = obj_task.num
    else:
        opt_opti.eval()
        obj_args = obj_task.get_args_for_eval()
        num = obj_task.num_eval

    thetas = obj_task.get_thetas(num=num)
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

        losses = obj_task.get_objectives(outputs, *obj_args)
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
    best_outputs = outputs_list[ids.squeeze(1), th.arange(num, device=device)]

    return best_outputs, min_losses


"""run"""


class BaseEnv:
    def __init__(self, num_nodes: int = 20, num_envs: int = 128, device: th.device = th.device("cuda:0"),
                 episode_length: int = 6):
        self.num_nodes = num_nodes
        self.num_envs = num_envs
        self.device = device
        self.episode_length = episode_length
        self.x = th.rand(self.num_envs, self.num_nodes).to(self.device)
        self.best_x = None
        self.calc_obj_for_two_graphs_vmap = th.vmap(self.calc_obj_for_two_graphs, in_dims=(0, 0))
        self.adjacency_matrix = None
        self.num_steps = None

    def load_graph(self, file_name: str):
        import numpy as np  # todo not elegant
        self.adjacency_matrix = th.as_tensor(np.load(file_name), device=self.device)

    def reset(self, add_noise_for_best_x=False, sample_ratio_envs=0.2, sample_ratio_nodes=0.2):
        if add_noise_for_best_x and self.best_x is not None:
            e = max(1, int(sample_ratio_envs * self.num_envs))
            n = max(1, int(sample_ratio_nodes * self.num_nodes))
            indices_envs = th.randint(0, self.num_envs, size=(e,))  # indices of selected envs/rows
            indices_nodes = th.randint(0, self.num_nodes, size=(n,))
            # noise = th.randn(n, self.num_nodes).to(self.device)
            noise = th.rand(self.num_envs, self.num_nodes).to(self.device)
            mask = th.zeros(self.num_envs, self.num_nodes, dtype=th.bool).to(self.device)
            mask[indices_envs, indices_nodes.unsqueeze(1)] = True
            noise = th.mul(noise, mask).to(self.device)

            mask2 = th.zeros(self.num_envs, self.num_nodes, dtype=th.bool).to(self.device)
            mask2[indices_envs, :] = True
            add_noise_for_best_x = th.mul(self.best_x.repeat(self.num_envs, 1), mask2).to(self.device)

            mask3 = th.ones(self.num_envs, self.num_nodes, dtype=th.bool).to(self.device)
            mask3[indices_envs, :] = False
            x = th.mul(th.rand(self.num_envs, self.num_nodes), mask3).to(self.device)

            self.x = x + add_noise_for_best_x + noise
            self.x[0, :] = self.best_x  # the first row is best_x, no noise
        else:
            self.x = th.rand(self.num_envs, self.num_nodes).to(self.device)
        self.num_steps = 0
        return self.x

    def generate_symmetric_adjacency_matrix(self, sparsity: float):  # sparsity for binary
        upper_triangle = th.mul(th.rand(self.num_nodes, self.num_nodes).triu(diagonal=1),
                                (th.rand(self.num_nodes, self.num_nodes) < sparsity).int().triu(diagonal=1))
        adjacency_matrix = upper_triangle + upper_triangle.transpose(-1, -2)
        return adjacency_matrix  # num_env x self.N x self.N

    # make sure that mu1 and mu2 are different tensors. If they are the same, use get_cut_value_one_tensor
    def calc_obj_for_two_graphs(self, mu1: TEN, mu2: TEN):
        pass

    def calc_obj_for_one_graph(self, mu: TEN):
        pass


class MaxCutEnv(BaseEnv):
    def __init__(self, num_nodes=20, num_envs=128, device=th.device("cuda:0"), episode_length=6):
        super(MaxCutEnv, self).__init__(num_nodes, num_envs, device, episode_length)
        self.adjacency_matrix = None

    # make sure that mu1 and mu2 are different tensors. If they are the same, use calc_obj_for_one_graph
    def calc_obj_for_two_graphs(self, mu1: TEN, mu2: TEN):
        obj1 = self.calc_obj_for_one_graph(mu1)
        obj2 = self.calc_obj_for_one_graph(mu2)

        mu1_reshaped = mu1.reshape(-1, self.num_nodes, 1)
        mu2_reshaped = mu2.reshape(-1, self.num_nodes, 1)
        mu2_transposed = mu2_reshaped.transpose(-1, -2)
        mu1_complement = 1 - mu1_reshaped
        mu2_complement = 1 - mu2_reshaped

        term1 = mu1_reshaped @ mu2_complement.transpose(-1, -2) * self.adjacency_matrix
        term2 = mu1_complement @ mu2_transposed * self.adjacency_matrix

        result = obj1 + obj2 + term1 + term2
        return result

    def calc_obj_for_one_graph(self, mu: TEN):
        mu_reshaped = mu.reshape(-1, self.num_nodes, 1)
        mu_complement = 1 - mu_reshaped
        mu_complement_transposed = mu_complement.transpose(-1, -2)

        mu_product = th.matmul(mu_reshaped, mu_complement_transposed)
        term = th.mul(mu_product, self.adjacency_matrix)
        cut = term.flatten().sum(dim=-1) / self.num_envs
        return cut


class ObjectiveMISO(ObjectiveTask):
    def __init__(self, num, dims, device):
        super(ObjectiveMISO, self).__init__()
        num_envs = num
        num_nodes = dims
        episode_length = 30

        self.env = MaxCutEnv(num_nodes=num_nodes, num_envs=num_envs, device=device, episode_length=episode_length)
        # https://github.com/AI4Finance-Foundation/ElegantRL_Solver/blob/main/
        # rlsolver/rlsolver_learn2opt/np_complete_problems/envs/maxcut_env.py#L10

        self.num = num
        self.dim = th.prod(th.tensor(dims)).item()
        self.args = ()
        self.device = device

        self.dims = dims

        self.get_objective_vmap = vmap(self.get_objective, in_dims=(0, 0), out_dims=0)

    def get_args_for_train(self):
        return self.args

    def get_args_for_eval(self):
        return self.args

    def get_objective(self, w: TEN, h: TEN, noise: float = 1.) -> TEN:
        x = w
        num_envs = self.num
        num_nodes = self.dim

        if h is None:
            obj = self.env.calc_obj_for_one_graph(x.reshape(num_envs, num_nodes))
        else:
            obj = self.env.calc_obj_for_two_graphs_vmap(h.reshape(num_envs, num_nodes), x.reshape(num_envs, num_nodes))
            obj = obj.sum() * (-0.2)
        return obj

    def get_objectives(self, thetas: TEN, hs: TEN) -> TEN:
        ws = thetas
        return self.get_objective_vmap(ws, hs)

    @staticmethod
    def get_norm(x):
        return x / x.norm(dim=1, keepdim=True)

    @staticmethod
    def load_from_disk(device):
        import pickle
        with open(f'./K8N8Samples=100.pkl', 'rb') as f:
            h_evals = th.as_tensor(pickle.load(f), dtype=th.cfloat, device=device)
            assert h_evals.shape == (100, 8, 8)

        h_evals = th.stack((h_evals.real, h_evals.imag), dim=1)
        assert h_evals.shape == (100, 2, 8, 8)
        return h_evals

    @staticmethod
    def get_result_of_mmse(h, p) -> TEN:  # MMSE beamformer
        h = h[0] + 1j * h[1]
        k, n = h.shape
        eye_mat = th.eye(n, dtype=h.dtype, device=h.device)
        w = th.linalg.solve(eye_mat * k / p + th.conj(th.transpose(h, 0, 1)) @ h, th.conj(th.transpose(h, 0, 1)))
        w = w / (th.norm(w, dim=0, keepdim=True) * k ** 0.5)  # w.shape == [K, N]
        return th.stack((w.real, w.imag), dim=0)  # return.shape == [2, K, N]


def train_optimizer():
    gpu_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0

    '''train'''
    train_times = 2 ** 10
    num = 2 ** 10  # batch_size
    lr = 8e-4
    unroll = 16  # step of Hamilton Term
    num_opt = 256
    hid_dim = 2 ** 7

    '''eval'''
    eval_gap = 2 ** 7

    '''init task'''
    device = th.device(f'cuda:{gpu_id}' if th.cuda.is_available() and gpu_id >= 0 else 'cpu')
    dims = (2, 8, 8)
    obj_task = ObjectiveMISO(num=num, dims=dims, device=device)
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
