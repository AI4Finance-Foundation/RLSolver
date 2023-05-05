import os
import sys
import time
import torch as th
import torch.nn as nn
from tqdm import tqdm

from TNCO_env import TensorNetworkEnv  # get_nodes_list
from L2O_H_term import ObjectiveTask, OptimizerTask, OptimizerOpti
from L2O_H_term import opt_train, opt_eval

from TNCO_env import NodesSycamoreN53M20

TEN = th.Tensor

NodesList, BanEdges = NodesSycamoreN53M20, 0

WarmUpSize = 2 ** 14
BufferSize = 2 ** 20
BufferRate = 0.25  # Buffer1Size = Buffer2Size * BufferCoff
BatchSize = 2 ** 9

NumRepeats = 8
IterMaxStep = 2 ** 12
IterMinLoss = 2 ** -4
EmaGamma = 0.98
NoiseRatio = 1
Dims = (512, 256, 512, 256)

GPU_ID = int(sys.argv[1]) if len(sys.argv) > 1 else 0


# if GPU_ID in {0, }:  # 22.22
#     IterMinLoss = 2 ** -4
#     NoiseRatio = 1.0
# if GPU_ID in {1, }:  # 25.45
#     IterMinLoss = 2 ** -3
#     NoiseRatio = 1.0
# if GPU_ID in {2, }:  # 22.85
#     IterMinLoss = 2 ** -4
#     NoiseRatio = 0.25
# if GPU_ID in {3, }:  # 24.079
#     IterMinLoss = 2 ** -3
#     NoiseRatio = 0.25
#
# if GPU_ID in {4, 5}:  # 25, 25， 21.86  | 26, 21， 21.128
#     IterMinLoss = 2 ** -3
# if GPU_ID in {6, }:  # 25, 25， 22.482
#     IterMinLoss = 2 ** -4
# if GPU_ID in {7, }:  # 26.234
#     NoiseRatio = 0.5


def build_mlp(dims: [int], activation: nn = None, if_raw_out: bool = True) -> nn.Sequential:
    """
    build MLP (MultiLayer Perceptron)

    dims: the middle dimension, `dims[-1]` is the output dimension of this network
    activation: the activation function
    if_remove_out_layer: if remove the activation function of the output layer.
    """
    if activation is None:
        activation = nn.ReLU
    net_list = []
    for i in range(len(dims) - 1):
        net_list.extend([nn.Linear(dims[i], dims[i + 1]), activation()])
    if if_raw_out:
        del net_list[-1]  # delete the activation function of the output layer to keep raw output
    return nn.Sequential(*net_list)


def layer_init_with_orthogonal(layer, std=1.0, bias_const=1e-6):
    th.nn.init.orthogonal_(layer.weight, std)
    th.nn.init.constant_(layer.bias, bias_const)


class ObjModel(nn.Module):
    def __init__(self, inp_dim, out_dim, dims=(256, 256, 256)):
        super().__init__()
        self.net = build_mlp(dims=[inp_dim, *dims, out_dim], activation=nn.Tanh, if_raw_out=True)
        layer_init_with_orthogonal(self.net[-1], std=0.1)

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:  # for L2O
    def __init__(self, max_size: int, state_dim: int, gpu_id: int = 0):
        self.p = 0  # pointer
        self.if_full = False
        self.cur_size = 0
        self.max_size = max_size
        self.device = th.device(f"cuda:{gpu_id}" if (th.cuda.is_available() and (gpu_id >= 0)) else "cpu")

        self.thetas = th.empty((max_size, state_dim), dtype=th.float32, device=self.device)
        self.scores = th.empty((max_size, 1), dtype=th.float32, device=self.device)

    def update(self, items: [TEN]):
        states, scores = items
        # assert thetas.shape == (warm_up_size, self.dim)
        # assert scores.shape == (warm_up_size, 1)

        p = self.p + scores.shape[0]  # pointer
        if p > self.max_size:
            self.if_full = True
            p0 = self.p
            p1 = self.max_size
            p2 = self.max_size - self.p
            p = p - self.max_size

            self.thetas[p0:p1], self.thetas[0:p] = states[:p2], states[-p:]
            self.scores[p0:p1], self.scores[0:p] = scores[:p2], scores[-p:]
        else:
            self.thetas[self.p:p] = states
            self.scores[self.p:p] = scores
        self.p = p
        self.cur_size = self.max_size if self.if_full else self.p

    def sample(self, batch_size: int) -> [TEN]:
        ids = th.randint(self.cur_size - 1, size=(batch_size,), requires_grad=False)
        return self.thetas[ids], self.scores[ids]

    def save_or_load_history(self, cwd: str, if_save: bool):
        item_paths = (
            (self.thetas, f"{cwd}/buffer_thetas.pth"),
            (self.scores, f"{cwd}/buffer_scores.pth"),
        )

        if if_save:
            print(f"| buffer.save_or_load_history(): Save {cwd}    cur_size {self.cur_size}")
            for item, path in item_paths:
                if self.cur_size == self.p:
                    buf_item = item[:self.cur_size]
                else:
                    buf_item = th.vstack((item[self.p:self.cur_size], item[0:self.p]))
                th.save(buf_item.half(), path)  # save float32 as float16

        elif all([os.path.isfile(path) for item, path in item_paths]):
            max_sizes = []
            for item, path in item_paths:
                buf_item = th.load(path, map_location=th.device('cpu')).float()  # load float16 as float32
                print(f"| buffer.save_or_load_history(): Load {path}    {buf_item.shape}")

                max_size = buf_item.shape[0]
                max_size = min(self.max_size, max_size)
                item[:max_size] = buf_item[-max_size:]  # load
                max_sizes.append(max_size)

            assert all([max_size == max_sizes[0] for max_size in max_sizes])
            self.cur_size = max_sizes[0]
            self.p = self.cur_size
            self.if_full = self.cur_size == self.max_size

    def get_cur_ten(self, var_name: str) -> TEN:
        ten = getattr(self, var_name)
        return ten[:self.cur_size]

    def empty_and_reset_pointer(self):
        self.p = 0
        self.cur_size = 0
        self.if_full = False


def collect_buffer_history(if_remove: bool = False):
    max_size = 2 ** 18
    save_dir0 = 'task_TNCO'
    save_dirs = [save_dir for save_dir in os.listdir('') if save_dir[:9] == 'task_TNCO']

    states_ary = []
    scores_ary = []
    for save_dir in save_dirs:
        states_path = f"{save_dir}/replay_buffer_states.pth"
        scores_path = f"{save_dir}/replay_buffer_scores.pth"

        if_all_exists = all([os.path.isfile(path) for path in (states_path, scores_path)])
        if not if_all_exists:
            print(f"FileExist? [states, scores] {if_all_exists}")
            continue

        states = th.load(states_path, map_location=th.device('cpu')).half()
        scores = th.load(scores_path, map_location=th.device('cpu')).half()
        states_ary.append(states)
        scores_ary.append(scores)

        os.remove(states_path) if if_remove else None
        os.remove(scores_path) if if_remove else None
        print(f"Load {save_dir:12}    num_samples {scores.shape[0]}")

    states_ary = th.vstack(states_ary)
    scores_ary = th.vstack(scores_ary)

    sort = -scores_ary.squeeze(1).argsort()[:max_size]  # notice negative symbol here.
    states_ary = states_ary[sort]
    scores_ary = scores_ary[sort]

    os.makedirs(save_dir0, exist_ok=True)
    th.save(states_ary, f"{save_dir0}/replay_buffer_states.pth")
    th.save(scores_ary, f"{save_dir0}/replay_buffer_scores.pth")
    print(f"Save {save_dir0:12}    num_samples {scores_ary.shape[0]}")
    print(f"sort max {scores_ary[+0].item():9.3f}")
    print(f"sort min {scores_ary[-1].item():9.3f}")


class ObjectiveTNCO(ObjectiveTask):
    def __init__(self, dim, device):
        super(ObjectiveTNCO, self).__init__()
        self.device = device
        self.args = ()

        self.env = TensorNetworkEnv(nodes_list=NodesList, ban_edges=BanEdges, device=device)
        self.dim = self.env.num_edges - self.env.ban_edges
        print(f"ObjectiveTNCO.dim {self.dim} != dim {dim}") if self.dim != dim else None

        self.obj_model1 = ObjModel(inp_dim=self.dim, out_dim=1, dims=Dims).to(device)
        self.obj_model0 = ObjModel(inp_dim=self.dim, out_dim=1, dims=Dims).to(device)

        self.optimizer = th.optim.Adam(self.obj_model1.parameters(), lr=1e-4)
        self.criterion = nn.MSELoss()
        self.batch_size = BatchSize
        self.ema_loss = IterMinLoss * 1.1
        self.ema_step = 0
        self.keep_score = -th.inf
        self.best_score = th.inf

        gpu_id = -1 if self.device.index is None else self.device.index

        """init buffer"""
        self.save_path = f'./task_TNCO_{gpu_id:02}'
        os.makedirs(self.save_path, exist_ok=True)

        '''build buffer'''
        buf_max_size = BufferSize
        warm_up_size = WarmUpSize
        buf_rate0 = BufferRate
        buf_rate1 = 1 - BufferRate
        self.buffer1 = ReplayBuffer(max_size=int(buf_max_size * buf_rate1), state_dim=self.dim, gpu_id=gpu_id)
        self.buffer0 = ReplayBuffer(max_size=int(buf_max_size * buf_rate0), state_dim=self.dim, gpu_id=gpu_id)
        self.buffer1.save_or_load_history(cwd=self.save_path, if_save=False)

        if self.buffer1.cur_size < warm_up_size:  # warm_up
            thetas, scores = self.random_generate_input_output(warm_up_size=warm_up_size, if_tqdm=True)
            self.buffer1.update(items=(thetas, scores))

        self.save_and_check_buffer()

        iter_max_step = IterMaxStep * 2 ** 3
        while self.ema_loss > IterMinLoss:
            self.pbar = tqdm(total=iter_max_step, ascii=True)
            self.fast_train_obj_model(iter_max_step=iter_max_step)
            self.pbar.close()

    def get_objective(self, theta, *args) -> TEN:
        num_repeats = NumRepeats
        iter_max_step = IterMaxStep
        with th.no_grad():
            thetas = theta.repeat(num_repeats, 1)
            thetas[1:] += th.randn_like(thetas[1:]) * NoiseRatio
            thetas = self.get_norm(thetas)  # shape == (warm_up_size, self.dim)
            scores = self.get_objectives_without_grad(thetas).unsqueeze(1)  # shape == (warm_up_size, 1)
            self.buffer1.update(items=(thetas, scores))

        self.fast_train_obj_model(iter_max_step=iter_max_step)

        self.hard_update(self.obj_model0, self.obj_model1)
        objective = self.obj_model0(theta)
        return objective

    def get_objectives_without_grad(self, thetas, *_args) -> TEN:
        # assert theta.shape[0] == self.env.num_edges
        with th.no_grad():
            log10_multiple_times = self.env.get_log10_multiple_times(edge_sorts=thetas.argsort(dim=1))
        return log10_multiple_times

    @staticmethod
    def get_norm(x):
        return (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + 1e-6)

    def random_generate_input_output(self, warm_up_size: int = 512, if_tqdm: bool = False):
        batch_size = 512
        print(f"TNCO | random_generate_input_output: num_warm_up={warm_up_size}")
        thetas = th.randn((warm_up_size, self.dim), dtype=th.float32, device=self.device)
        thetas = ((thetas - thetas.mean(dim=1, keepdim=True)) / (thetas.std(dim=1, keepdim=True) + 1e-6))

        scores = th.zeros((warm_up_size, 1), dtype=th.float32, device=self.device)

        i_iter = range(0, warm_up_size, batch_size)
        i_iter = tqdm(i_iter, ascii=True) if if_tqdm else i_iter
        for i in i_iter:
            j = i + batch_size
            scores[i:j, 0] = self.get_objectives_without_grad(thetas[i:j])

        return thetas, scores

    def fast_train_obj_model(self, iter_max_step: int):
        iter_min_loss = IterMinLoss

        ema_loss = iter_min_loss * 1.1  # Exponential Moving Average (EMA) loss value
        ema_gamma = EmaGamma

        ratio1 = self.buffer1.cur_size / (self.buffer1.cur_size + self.buffer0.cur_size)
        ratio0 = self.buffer0.cur_size / (self.buffer1.cur_size + self.buffer0.cur_size)

        batch_size1 = int(self.batch_size * ratio1)
        batch_size0 = int(self.batch_size * ratio0)

        pbar = self.pbar
        iter_step = 0
        for iter_step in range(1, iter_max_step + 1):
            inputs1, labels1 = self.buffer1.sample(batch_size1)
            inputs0, labels0 = self.buffer0.sample(batch_size0)
            inputs = th.vstack((inputs1, inputs0))
            labels = th.vstack((labels1, labels0))

            outputs = self.obj_model1(inputs)
            loss = self.criterion(outputs, labels)

            self.optimizer.zero_grad()
            loss.backward(retain_graph=True)
            self.optimizer.step()

            ema_loss = ema_gamma * ema_loss + (1 - ema_gamma) * loss.item()

            pbar.n = iter_step
            pbar.set_description(f"Loss {ema_loss:9.3e}")
            pbar.update(0)

            if ema_loss < iter_min_loss:
                break

        self.ema_loss = ema_loss
        self.ema_step = iter_step

    def save_and_check_buffer(self):
        buffer_size = self.buffer1.cur_size + self.buffer0.cur_size

        buffer = ReplayBuffer(max_size=buffer_size, state_dim=self.dim, gpu_id=-1)
        buffer.update(items=(self.buffer1.get_cur_ten('thetas'), self.buffer1.get_cur_ten('scores')))
        buffer.update(items=(self.buffer0.get_cur_ten('thetas'), self.buffer0.get_cur_ten('scores')))
        buffer.save_or_load_history(cwd=self.save_path, if_save=True)

        '''move data to buffer1 buffer0'''
        states = buffer.get_cur_ten('thetas')
        scores = buffer.get_cur_ten('scores')
        del buffer

        self.keep_score = th.quantile(scores.squeeze(1), q=BufferRate).item()
        self.buffer1.empty_and_reset_pointer()
        self.buffer0.empty_and_reset_pointer()
        mask1 = scores.squeeze(1) > self.keep_score
        mask0 = ~mask1
        self.buffer1.update(items=(states[mask1], scores[mask1]))
        self.buffer0.update(items=(states[mask0], scores[mask0]))

        thetas0 = self.buffer0.get_cur_ten('thetas')
        scores0 = self.buffer0.get_cur_ten('scores')
        min_score = scores0.min().item()
        avg_score = scores0.mean().item()
        max_score = scores0.max().item()
        print(f"num_score: {scores0.shape[0]}")
        print(f"min_score: {min_score:9.3f}")
        print(f"avg_score: {avg_score:9.3f} ± {scores0.std(dim=0).item():9.3f}")
        print(f"max_score: {max_score:9.3f}")

        if min_score < self.best_score:
            self.best_score = min_score
            print(f"best_result: {self.best_score:9.3f}    best_sort:\n"
                  f"{thetas0[scores0.argmin()].argsort().cpu().detach().numpy()}")

    @staticmethod
    def hard_update(target_net: th.nn.Module, current_net: th.nn.Module):
        for tar, cur in zip(target_net.parameters(), current_net.parameters()):
            tar.data.copy_(cur.data)

    def get_args_for_train(self):
        scores = self.buffer0.get_cur_ten('scores').squeeze(1)
        self.keep_score = th.quantile(scores, q=BufferRate).item()
        return ()

    def get_args_for_eval(self):
        return self.get_args_for_train()


def unit_test__objective_tnco():
    gpu_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    device = th.device(f'cuda:{gpu_id}' if th.cuda.is_available() and gpu_id >= 0 else 'cpu')

    obj_task = ObjectiveTNCO(dim=0, device=device)
    obj_task.get_objective(theta=th.rand(obj_task.dim, dtype=th.float32, device=obj_task.device))
    obj_task.save_and_check_buffer()


"""trainable objective function"""


def train_optimizer():
    gpu_id = GPU_ID

    '''train'''
    train_times = 2 ** 12
    lr = 2e-4
    unroll = 16
    num_opt = 128
    hid_dim = 64

    '''eval'''
    eval_gap = 2 ** 3
    import numpy as np
    np.set_printoptions(linewidth=120)

    print('start training')
    device = th.device(f'cuda:{gpu_id}' if th.cuda.is_available() and gpu_id >= 0 else 'cpu')

    dim = 414  # set by env.num_edges
    obj_task = ObjectiveTNCO(dim=dim, device=device)
    dim = obj_task.env.num_edges

    opt_task = OptimizerTask(dim=dim, device=device)
    opt_opti = OptimizerOpti(hid_dim=hid_dim).to(device)
    opt_base = th.optim.Adam(opt_opti.parameters(), lr=lr)

    '''loop'''
    start_time = time.time()
    for i in range(train_times + 1):
        obj_task.pbar = tqdm(total=NumRepeats, ascii=True)

        for _ in range(eval_gap):
            opt_train(obj_task=obj_task, opt_task=opt_task, opt_opti=opt_opti,
                      num_opt=num_opt, device=device, unroll=unroll, opt_base=opt_base)

        best_result, min_loss = opt_eval(obj_task=obj_task, opt_opti=opt_opti, opt_task=opt_task,
                                         num_opt=num_opt * 2, device=device)

        with th.no_grad():
            edge_sorts = best_result.argsort().unsqueeze(0)
            scores = obj_task.env.get_log10_multiple_times(edge_sorts=edge_sorts)
            score = scores.squeeze(0)

            time_used = time.time() - start_time
            obj_task.pbar.set_description(f"Loss {obj_task.ema_loss:9.3e}  Step {obj_task.ema_step:6}  "
                                          f"{i:>9}  {score:9.3f}  {min_loss.item():9.3e}  {time_used:9.3e} sec")
        obj_task.pbar.close()

        '''save'''
        if i % 4 == 0:
            obj_task.save_and_check_buffer()

    obj_task.save_and_check_buffer()


if __name__ == '__main__':
    # unit_test__objective_tnco()
    train_optimizer()
    # collect_buffer_history(if_remove=False)
