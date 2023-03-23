import sys
import torch
import torch as th
import torch.nn as nn
import torch.optim as optim
import numpy as np
from time import time

TEN = th.Tensor

"""Tensor Train"""
from rlsolver.rlsolver_learn2opt.tensor_train.env_tt import *


# https://github.com/AI4Finance-Foundation/ElegantRL_Solver/blob/e500897738a8561027559a780be9b5fae0f936be/rlsolver/rlsolver_learn2opt/tensor_train/env_tt.py#L10


class Env:
    def __init__(self, N=100, episode_length=6, num_env=4096, max_dim=2, epsilon=0.9,
                 device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
        self.N = N
        self.device = device
        self.num_env = num_env
        self.episode_length = episode_length
        self.max_dim = max_dim
        self.mask_state = th.zeros(self.N + 2, self.N + 2).to(self.device)
        self.mask_state[1, 1] = 1
        for i in range(2, self.N + 1):
            self.mask_state[i, i - 1] = 1
            self.mask_state[i, i] = 1
        self.mask_state = self.mask_state.reshape(-1).repeat(1, self.num_env).reshape(self.num_env, self.N + 2,
                                                                                      self.N + 2).to(self.device)
        with open(f"test_data_tensor_train_N={N}.pkl", 'rb') as f:
            import pickle as pkl
            self.test_state = pkl.load(f)
        self.permute_base = th.as_tensor([i for i in range(self.N - 1)]).repeat(1, self.num_env).reshape(self.num_env,
                                                                                                         -1).to(
            self.device)
        self.zero = th.zeros(self.N - 1).to(self.device)
        self.epsilon = epsilon
        self.mask = th.ones(self.num_env, self.N - 1).to(self.device)

    def reset(self, test=False):
        if test:
            self.num_env = self.test_state.shape[0]
        else:
            self.num_env = self.permute_base.shape[0]
        self.state = torch.randint(0, self.max_dim, (self.num_env, self.N + 2, self.N + 2), device=self.device).to(
            torch.float32)
        self.state = th.mul(self.state, self.mask_state[:self.num_env])
        self.state += th.ones_like(self.state)
        self.reward = th.zeros(self.num_env, self.episode_length).to(self.device)
        self.reward_no_prob = th.zeros(self.num_env, self.episode_length).to(self.device)
        self.if_test = test
        self.start = th.as_tensor([i for i in range(self.N)]).repeat(1, self.num_env).reshape(self.num_env, -1).to(
            self.device) + 1
        self.end = th.as_tensor([i for i in range(self.N)]).repeat(1, self.num_env).reshape(self.num_env, -1).to(
            self.device) + 1
        self.mask = th.ones(self.num_env, self.N - 1).to(self.device)
        if test:
            self.state = self.test_state
        self.num_steps = 0
        self.done = False
        initial_action = th.rand(self.num_env, self.N - 1).to(self.device)
        initial_action /= initial_action.sum(dim=-1, keepdim=True)
        return (self.state, self.start, self.end, self.mask, initial_action)

    def step(self, action):
        action = action.unsqueeze(0)
        # print(action.shape, action)
        reward = 0
        reward_no_prob = 0
        mask = deepcopy(self.mask)
        action_mask = th.mul(mask, action)
        action_mask = action_mask / action_mask.sum(dim=-1, keepdim=True)
        # if self.if_test:
        # print(self.num_steps, action_mask[0].detach().cpu().numpy(), self.reward_no_prob[0].detach().cpu().numpy(), self.epsilon)
        for k in range(action.shape[0]):
            r = 0
            r_no_prob = 0
            print(k)
            print(self.state.shape)
            state = self.state[k]
            x = th.rand(1).item()
            if x > self.epsilon or self.if_test:
                selected_edge_id = th.max(action_mask[k], dim=-1)[1].item()
            else:
                selected_edge_id = th.randint(low=0, high=self.N - 1, size=(1, 1)).item()
            self.mask[k, selected_edge_id] = 0
            r = 1
            for j in range(self.start[k, selected_edge_id], self.end[k, selected_edge_id] + 1):
                r *= (state[j, j] * state[j, self.start[k, selected_edge_id] - 1] * state[
                    self.end[k, selected_edge_id] + 1, j])
            for j in range(self.start[k, selected_edge_id + 1], self.end[k, selected_edge_id + 1] + 1):
                r *= (state[j, j] * state[j, self.start[k, selected_edge_id + 1] - 1] * state[
                    self.end[k, selected_edge_id + 1] + 1, j])
            r /= state[self.start[k, selected_edge_id + 1], self.start[k, selected_edge_id + 1] - 1]
            start_new = min(self.start[k, selected_edge_id], self.start[k, selected_edge_id + 1])
            end_new = max(self.end[k, selected_edge_id], self.end[k, selected_edge_id + 1])
            for __ in range(start_new, end_new + 1):
                self.start[k, __ - 1] = start_new
                self.end[k, __ - 1] = end_new
            r_no_prob = r
            r = r * action_mask[k, selected_edge_id]
            reward = reward + r
            reward_no_prob += r_no_prob
            self.reward[k, self.num_steps] = r
            self.reward_no_prob[k, self.num_steps] = r_no_prob.detach()
        self.num_steps += 1
        self.done = True if self.num_steps >= self.episode_length else False
        if self.done and self.if_test:
            action_mask_ = th.mul(self.mask, action)
            # print(self.num_steps, action_mask_[0].detach().cpu().numpy(), self.reward_no_prob[0].detach().cpu().numpy())
        return (self.state, self.start, self.end, self.mask, action.detach()), reward, self.done

    def get_state(self, action):
        N = self.N
        s = deepcopy(self.state.reshape(-1))
        start = deepcopy(self.start.reshape(-1))
        end = deepcopy(self.end.reshape(-1))
        mask_1 = deepcopy(self.mask.reshape(-1))
        input_dim = (N + 2) * (N + 2) + N + N + (N - 1) + (N - 1)
        state = th.zeros(N - 1, input_dim)
        for i in range(N - 1):
            curr_mask = th.zeros(N - 1)
            curr_mask[i] = 1
            state[i] = th.cat((s, start, end, mask_1, curr_mask), dim=-1)
        return state.to(self.device)

    def get_loss(self, action):
        action = action.unsqueeze(0)
        reward = 0
        reward_no_prob = 0
        mask = deepcopy(self.mask)
        print(mask)
        print(action)

        action_mask = th.mul(mask, action)
        print(action_mask)
        print(self.state)
        action_mask = action_mask / action_mask.sum(dim=-1, keepdim=True)

        for k in range(action.shape[0]):
            for selected_edge_id in mask[k]:

                selected_edge_id = int(selected_edge_id)
                if selected_edge_id == 0:
                    continue
                # print(k, action_mask[k], selected_edge_id)
                r = 0
                r_no_prob = 0
                state = self.state[k]
                r = 1
                for j in range(self.start[k, selected_edge_id], self.end[k, selected_edge_id] + 1):
                    r *= (state[j, j] * state[j, self.start[k, selected_edge_id] - 1] * state[
                        self.end[k, selected_edge_id] + 1, j])
                for j in range(self.start[k, selected_edge_id + 1], self.end[k, selected_edge_id + 1] + 1):
                    r *= (state[j, j] * state[j, self.start[k, selected_edge_id + 1] - 1] * state[
                        self.end[k, selected_edge_id + 1] + 1, j])
                r /= state[self.start[k, selected_edge_id + 1], self.start[k, selected_edge_id + 1] - 1]
                start_new = min(self.start[k, selected_edge_id], self.start[k, selected_edge_id + 1])
                end_new = max(self.end[k, selected_edge_id], self.end[k, selected_edge_id + 1])
                for __ in range(start_new, end_new + 1):
                    self.start[k, __ - 1] = start_new
                    self.end[k, __ - 1] = end_new
                r_no_prob = r
                r = r * action_mask[k, selected_edge_id]
                reward = reward + r
                reward_no_prob += r_no_prob
                self.reward[k, self.num_steps] = r.detach()
                self.reward_no_prob[k, self.num_steps] = r_no_prob.detach()
                print("get_loss reward: ", selected_edge_id, reward)
        print("get_loss: ", reward)
        return reward


def get_cumulative_reward(theta: TEN) -> float:
    assert theta.shape == (99,)  # # 如果是TensorTrain 里的List结构，共100个节点，则有99条边

    env = Env()  # 你需要在Env 的 __init__() 里添加一个 self.multiple_times
    env.reset()
    for edge_i in theta.argsort():
        env.step(map_edge_i_to_action(edge_i))  # 你需要自己写一个将 收缩的边 edge_i 映射到 action 的函数
    return env.multiple_times  # self.multiple_times 这个数值在 env.step 里面会不断地累加


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


theta_list = []  # 设置成全局变量，建议每一次训练后，都独立地保存并收集这些数据，因为训练 trainable objective function 的数据越多越好
score_list = []  # 设置成全局变量，建议每一次训练后，都独立地保存并收集这些数据，因为训练 trainable objective function 的数据越多越好
gpu_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
DEVICE = th.device(f'cuda:{gpu_id}' if th.cuda.is_available() and gpu_id >= 0 else 'cpu')
obj_model = MLP(inp_dim=99, out_dim=1, dims=(256, 256, 256))
optimizer = optim.SGD(obj_model.parameters(), lr=1e-4)
criterion = nn.MSELoss()
batch_size = 64


def get_objective(theta: TEN) -> TEN:
    score = get_cumulative_reward(theta)
    theta_list.append(theta)
    score_list.append(score)

    theta_tensor = torch.stack(theta_list).to(DEVICE)
    score_tensor = torch.tensor(score_list).to(DEVICE)
    train_size = score_tensor.shape[0]

    if train_size > batch_size:
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


def opt_train(obj_func, obj_args, opt_lstm, opt_task, num_opt, device, unroll, opt_base):
    opt_lstm.train()
    opt_task.zero_grad()

    n_params = 0
    for name, p in opt_task.get_register_params():
        n_params += int(np.prod(p.size()))
    hc_state1 = th.zeros(4, n_params, opt_lstm.hid_dim, device=device)

    all_losses_ever = []
    all_losses = None

    torch.set_grad_enabled(True)
    for iteration in range(1, num_opt + 1):
        output = opt_task.get_output()
        loss = obj_func(*obj_args, output)
        loss.backward(retain_graph=True)

        if all_losses is None:
            all_losses = loss
        else:
            all_losses += loss
        all_losses_ever.append(loss.data.cpu().numpy())

        i = 0
        result_params = {}
        hc_state2 = th.zeros(4, n_params, opt_lstm.hid_dim, device=device)
        for name, p in opt_task.get_register_params():
            hid_dim = int(np.prod(p.size()))
            gradients = p.grad.view(hid_dim, 1).detach().clone().requires_grad_(True)

            j = i + hid_dim
            hc_part = hc_state1[:, i:j]
            updates, new_hidden, new_cell = opt_lstm(gradients, hc_part[0:2], hc_part[2:4])

            hc_state2[0, i:j] = new_hidden[0]
            hc_state2[1, i:j] = new_hidden[1]
            hc_state2[2, i:j] = new_cell[0]
            hc_state2[3, i:j] = new_cell[1]

            result = p + updates.view(*p.size())
            result_params[name] = result / result.norm()
            result_params[name].retain_grad()

            i = j

        if iteration % unroll == 0:
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


def opt_eval(obj_func, obj_args, opt_iter, opt_task, num_opt, device):
    opt_iter.eval()

    n_params = 0
    for name, p in opt_task.get_register_params():
        n_params += int(np.prod(p.size()))
    hc_state1 = th.zeros(4, n_params, opt_iter.hid_dim, device=device)

    best_res = None
    min_loss = torch.inf

    torch.set_grad_enabled(True)
    for _ in range(num_opt):
        output = opt_task.get_output()
        loss = obj_func(*obj_args, output)
        loss.backward(retain_graph=True)

        result_params = {}
        hc_state2 = th.zeros(4, n_params, opt_iter.hid_dim, device=device)

        i = 0
        for name, p in opt_task.get_register_params():
            param_dim = int(np.prod(p.size()))
            gradients = p.grad.view(param_dim, 1).detach().clone().requires_grad_(True)

            j = i + param_dim
            hc_part = hc_state1[:, i:j]
            updates, new_hidden, new_cell = opt_iter(gradients, hc_part[0:2], hc_part[2:4])

            hc_state2[0, i:j] = new_hidden[0]
            hc_state2[1, i:j] = new_hidden[1]
            hc_state2[2, i:j] = new_cell[0]
            hc_state2[3, i:j] = new_cell[1]

            result = p + updates.view(*p.size())
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
    return best_res, min_loss


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
    gpu_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0

    '''init'''
    dim = 99  # 如果是TensorTrain 里的List结构，共100个节点，则有99条边

    '''train'''
    train_times = 1000
    lr = 1e-4  # 要更小一些，甚至前期先不训练。
    unroll = 16
    num_opt = 64  # 要更大一些
    hid_dim = 40

    '''eval'''
    eval_gap = 128

    print('start training')
    device = th.device(f'cuda:{gpu_id}' if th.cuda.is_available() and gpu_id >= 0 else 'cpu')

    obj_net = get_objective  # 这里换成一个可以根据 99条边 估计出 张量收缩后的乘法个数的 可导神经网络。我们要最小化它输出的乘法次数
    # 还有，最好让 可导的神经网络拟合的乘法个数是  log10(乘法个数)，这个数值是已经求和过了的，所以可以直接套一个log10(...)
    opt_obj = OptimizerTask(dim=dim, device=device)
    opt_rnn = OptimizerLSTM(hid_dim=hid_dim).to(device)
    opt_org = optim.Adam(opt_rnn.parameters(), lr=lr)

    start_time = time()
    print("Start training")

    '''loop'''
    for i in range(train_times + 1):
        opt_train(obj_func=obj_net, obj_args=(), opt_task=opt_obj, opt_lstm=opt_rnn,
                  num_opt=num_opt, device=device, unroll=unroll, opt_base=opt_org)

        if i % eval_gap == 0:
            best_result, min_loss = opt_eval(
                obj_func=obj_net, obj_args=(), opt_iter=opt_rnn, opt_task=opt_obj,
                num_opt=num_opt * 2, device=device
            )
            time_used = time() - start_time
            print(f"{i:>9}    {min_loss}    TimeUsed {time_used:9}")


if __name__ == '__main__':
    train_optimizer()
