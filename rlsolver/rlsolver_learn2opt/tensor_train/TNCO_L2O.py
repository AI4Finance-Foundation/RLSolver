import sys
import torch
import torch as th
import torch.nn as nn
import torch.optim as optim
import numpy as np
from time import time

from TNCO_env import TensorNetworkEnv, SycamoreNodesN53M12

TEN = th.Tensor
ENV = TensorNetworkEnv(nodes_list=SycamoreNodesN53M12)


def get_cumulative_reward(theta: TEN) -> float:
    env = ENV
    assert theta.shape[0] == env.num_edges
    log10_multiple_times = env.get_log10_multiple_times(edge_argsort=theta.argsort())
    return log10_multiple_times


"""trainable objective function"""


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
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)


class MLP(nn.Module):
    def __init__(self, inp_dim, out_dim, dims=(256, 256, 256)):
        super().__init__()
        self.net = build_mlp(dims=[inp_dim, *dims, out_dim], activation=nn.Tanh)
        layer_init_with_orthogonal(self.net[-1], std=0.1)

    def forward(self, x):
        return self.net(x)


theta_list = []  # todo 设置成全局变量，建议每次训练后，独立地保存并收集这些数据，因为训练 trainable objective function 的数据越多越好
score_list = []  # todo 设置成全局变量，建议每次训练后，独立地保存并收集这些数据，因为训练 trainable objective function 的数据越多越好
gpu_id = int(sys.argv[1]) if len(sys.argv) > 1 else -1
DEVICE = th.device(f'cuda:{gpu_id}' if th.cuda.is_available() and gpu_id >= 0 else 'cpu')
obj_model = MLP(inp_dim=ENV.num_edges, out_dim=1, dims=(256, 256, 256))
optimizer = optim.SGD(obj_model.parameters(), lr=1e-4)
criterion = nn.MSELoss()
batch_size = 64
StartTrainSize = batch_size * 2 ** 8


def get_objective(theta: TEN) -> TEN:
    score = get_cumulative_reward(theta)
    theta_list.append(theta)
    score_list.append(score)

    theta_tensor = torch.stack(theta_list).to(DEVICE)
    score_tensor = torch.tensor(score_list).to(DEVICE)
    train_size = score_tensor.shape[0]

    if train_size > StartTrainSize:
        for epoch in range(128):  # fast_fit_it
            # indices = torch.randint(train_size, size=(batch_size,), device=device) # 训练集样本数量远大于 batch_size，就用这个
            indices = torch.randperm(train_size, device=DEVICE)[:batch_size]  # 训练集样本数量没有远大于 batch_size，就用这个

            inputs = theta_tensor[indices]
            labels = score_tensor[indices]

            outputs = obj_model(inputs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    objective = obj_model(theta)
    return objective


"""Learn To Optimize"""


def opt_train(obj_task, obj_args, opt_opti, opt_task, num_opt, device, unroll, opt_base):
    opt_opti.train()
    opt_task.zero_grad()

    n_params = 0
    for name, p in opt_task.get_register_params():
        n_params += int(np.prod(p.size()))
    hc_state1 = th.zeros(4, n_params, opt_opti.hid_dim, device=device)

    all_losses_ever = []
    all_losses = None

    torch.set_grad_enabled(True)
    for iteration in range(1, num_opt + 1):
        output = opt_task.get_output()
        loss = obj_task(*obj_args, output)
        loss.backward(retain_graph=True)

        if all_losses is None:
            all_losses = loss
        else:
            all_losses += loss
        all_losses_ever.append(loss.data.cpu().numpy())

        i = 0
        result_params = {}
        hc_state2 = th.zeros(4, n_params, opt_opti.hid_dim, device=device)
        for name, p in opt_task.get_register_params():
            hid_dim = int(np.prod(p.size()))
            gradients = p.grad.view(hid_dim, 1).detach().clone().requires_grad_(True)

            j = i + hid_dim
            hc_part = hc_state1[:, i:j]
            updates, new_hidden, new_cell = opt_opti(gradients, hc_part[0:2], hc_part[2:4])

            hc_state2[0, i:j] = new_hidden[0]
            hc_state2[1, i:j] = new_hidden[1]
            hc_state2[2, i:j] = new_cell[0]
            hc_state2[3, i:j] = new_cell[1]

            result = p + updates.view(*p.size())
            result_params[name] = result / result.norm()
            result_params[name].retain_grad()

            i = j

        if iteration % unroll == 0:

            if len(score_list) > StartTrainSize:  # todo 等数据积攒够多了，再启用 opt_base 去训练
                opt_base.zero_grad()
                all_losses.backward()
                opt_base.step()

            all_losses = None

            opt_task.re_init()
            opt_task.load_state_dict(result_params)
            opt_task.zero_grad()

            hc_state1 = hc_state2.detach().clone().requires_grad_(True)

        else:
            for name, p in opt_task.get_register_params():
                set_attr(opt_task, name, result_params[name])

            hc_state1 = hc_state2
    torch.set_grad_enabled(False)
    return all_losses_ever


def opt_eval(obj_task, obj_args, opt_opti, opt_task, num_opt, device):
    opt_opti.eval()

    n_params = 0
    for name, param in opt_task.get_register_params():
        n_params += int(np.prod(param.size()))
    hc_state1 = th.zeros(4, n_params, opt_opti.hid_dim, device=device)

    best_res = None
    min_loss = torch.inf

    torch.set_grad_enabled(True)
    for _ in range(num_opt):
        output = opt_task.get_output()
        loss = obj_task(*obj_args, output)
        loss.backward(retain_graph=True)

        result_params = {}
        hc_state2 = th.zeros(4, n_params, opt_opti.hid_dim, device=device)

        i = 0
        for name, param in opt_task.get_register_params():
            param_dim = int(np.prod(param.size()))
            gradients = param.grad.view(param_dim, 1).detach().clone().requires_grad_(True)

            j = i + param_dim
            hc_part = hc_state1[:, i:j]
            updates, new_hidden, new_cell = opt_opti(gradients, hc_part[0:2], hc_part[2:4])

            hc_state2[0, i:j] = new_hidden[0]
            hc_state2[1, i:j] = new_hidden[1]
            hc_state2[2, i:j] = new_cell[0]
            hc_state2[3, i:j] = new_cell[1]

            result = param + updates.view(*param.size())
            result_params[name] = result / result.norm()

            i = j

        opt_task.re_init()
        opt_task.load_state_dict(result_params)
        opt_task.zero_grad()

        hc_state1 = hc_state2.detach().clone().requires_grad_(True)

        if loss < min_loss:
            best_res = opt_task.get_output()
            min_loss = loss
    torch.set_grad_enabled(False)
    return best_res, min_loss.item()


def set_attr(obj, attr, val):
    attrs = attr.split('.')
    for attr in attrs[:-1]:
        obj = getattr(obj, attr)
    setattr(obj, attrs[-1], val)


class OptimizerTask(nn.Module):
    def __init__(self, dim, device):
        super().__init__()
        self.dim = dim
        self.device = device

        self.register_buffer('theta', th.zeros(dim, requires_grad=True, device=device))

    def re_init(self):
        self.__init__(dim=self.dim, device=self.device)

    def get_register_params(self):
        return [('theta', self.theta)]

    def forward(self, target):
        batch_theta = self.theta.unsqueeze(0)
        return target.objective_function(batch_theta)  # 有空把这个改成并行的

    def get_output(self):
        return self.theta


class OptimizerLSTM(nn.Module):
    def __init__(self, hid_dim=20):
        super().__init__()
        self.hid_dim = hid_dim
        self.recurs1 = nn.LSTMCell(1, hid_dim)
        self.recurs2 = nn.LSTMCell(hid_dim, hid_dim)
        self.output = nn.Linear(hid_dim, 1)

    def forward(self, inp0, hid0, cell):
        hid1, cell1 = self.recurs1(inp0, (hid0[0], cell[0]))
        hid2, cell2 = self.recurs2(hid1, (hid0[1], cell[1]))
        return self.output(hid2), (hid1, hid2), (cell1, cell2)


"""run"""


def train_optimizer():
    dim = ENV.num_edges  # 如果是TensorTrain 里的List结构，共100个节点，则有99条边

    '''train'''
    train_times = 2 ** 9
    lr = 1e-4  # 要更小一些，甚至前期先不训练。
    unroll = 16
    num_opt = 64  # 要更大一些
    hid_dim = 40

    '''eval'''
    eval_gap = 128

    print('start training')
    device = th.device(f'cuda:{gpu_id}' if th.cuda.is_available() and gpu_id >= 0 else 'cpu')

    obj_task = get_objective
    opt_task = OptimizerTask(dim=dim, device=device)
    opt_opti = OptimizerLSTM(hid_dim=hid_dim).to(device)
    opt_base = optim.Adam(opt_opti.parameters(), lr=lr)

    start_time = time()
    '''loop'''
    for i in range(train_times + 1):
        opt_train(obj_task=obj_task, obj_args=(), opt_task=opt_task, opt_opti=opt_opti,
                  num_opt=num_opt, device=device, unroll=unroll, opt_base=opt_base)

        if i % eval_gap == 0:
            best_result, min_loss = opt_eval(
                obj_task=obj_task, obj_args=(), opt_opti=opt_opti, opt_task=opt_task,
                num_opt=num_opt * 2, device=device
            )
            time_used = time() - start_time

            log10_multiple_times = ENV.get_log10_multiple_times(edge_argsort=best_result.squeeze(0).argsort())
            print(f"{i:>9}    {log10_multiple_times:9.3f}    {min_loss:9.3e}    TimeUsed {time_used:9.0f}")

    # todo 训练结束后，一定要时刻保存 我们的 score_list 以及 theta_list，比避免保存的时候因为重名覆盖掉上一次训练的，积攒数据
    # torch.save(torch.stack(theta_list), './theta_list.pth')
    # torch.save(torch.stack(score_list), './score_list.pth')


if __name__ == '__main__':
    train_optimizer()
