import sys
import torch
import torch as th
import torch.nn as nn
import numpy as np
# from functorch import vmap
from copy import deepcopy
from beta3 import *

np.set_printoptions(suppress=True)
gpu_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
DEVICE = th.device(f'cuda:{gpu_id}' if th.cuda.is_available() and gpu_id >= 0 else 'cpu')


def objective_function0(ten):
    env = Env(device=DEVICE)

    env.reset()
    for a in ten:
        env.step(a)
    return env.multiple_num


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


batch_size = 128
# x = np.random.rand(batch_size, NumEdges)
# y = np.random.rand(batch_size, 1)
obj_model = MLP(NumEdges, 256, 1)
xs = []
ys = []
optimizer = optim.SGD(obj_model.parameters(), lr=0.1)
criterion = nn.MSELoss()


def fast_fit_it(x, y):
    try:
        xs.extend(x)
        ys.extend(y)
    except:
        xs.append(x)
        ys.append(y)

    for epoch in range(128):
        inputs = torch.as_tensor(trans(xs)).to(DEVICE)
        labels = torch.as_tensor(trans(ys)).to(DEVICE)

        # 前向传播
        outputs = obj_model(inputs)
        loss = criterion(outputs, labels)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def objective_function(ten):
    detach_obj = objective_function0(ten)
    fast_fit_it(ten, detach_obj)
    obj = obj_model(ten)
    return obj


class Env:
    def __init__(self, episode_length=6, num_env=4, max_dim=2, epsilon=0.9,
                 device=None):
        self.device = device
        self.num_env = num_env
        self.episode_length = episode_length
        self.max_dim = max_dim
        # self.mask_state = th.zeros(self.N + 2, self.N + 2).to(self.device)
        # self.mask_state[1, 1] = 1
        # for i in range(2, self.N + 1):
        #     self.mask_state[i, i - 1] = 1
        #     self.mask_state[i, i] = 1
        # self.mask_state = self.mask_state.reshape(-1).repeat(1, self.num_env).reshape(self.num_env, self.N + 2,
        #                                                                               self.N + 2).to(self.device)
        # with open(f"test_data_tensor_train_N={N}.pkl", 'rb') as f:
        #     import pickle as pkl
        #     self.test_state = pkl.load(f)
        # self.permute_base = th.as_tensor([i for i in range(self.N - 1)]).repeat(1, self.num_env).reshape(self.num_env,
        #                                                                                                  -1).to(
        #     self.device)
        # self.zero = th.zeros(self.N - 1).to(self.device)
        self.epsilon = epsilon
        # self.mask = th.ones(self.num_env, self.N - 1).to(self.device)
        self.multiple_num = None

    def reset(self):
        with open('tensor_network_n53.txt', 'r') as f:
            data = f.readlines()
            data = eval(get_data(data))
        self.data = torch.tensor(data, device=DEVICE)
        self._reset()

    def reset0(self, test=False):
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
        reward = 0
        reward_no_prob = 0
        mask = deepcopy(self.mask)
        action_mask = th.mul(mask, action)
        action_mask = action_mask / action_mask.sum(dim=-1, keepdim=True)

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
            # for j in range(self.start[k, selected_edge_id], self.end[k, selected_edge_id] + 1):
            #     r *= (state[j, j] * state[j, self.start[k, selected_edge_id] - 1] * state[
            #         self.end[k, selected_edge_id] + 1, j])
            # for j in range(self.start[k, selected_edge_id + 1], self.end[k, selected_edge_id + 1] + 1):
            #     r *= (state[j, j] * state[j, self.start[k, selected_edge_id + 1] - 1] * state[
            #         self.end[k, selected_edge_id + 1] + 1, j])
            self.update_edge_and_node()

            r /= state[self.start[k, selected_edge_id + 1], self.start[k, selected_edge_id + 1] - 1]
            start_new = min(self.start[k, selected_edge_id], self.start[k, selected_edge_id + 1])
            end_new = max(self.end[k, selected_edge_id], self.end[k, selected_edge_id + 1])
            for i in range(start_new, end_new + 1):
                self.start[k, i - 1] = start_new
                self.end[k, i - 1] = end_new
            r_no_prob = r
            # r = r * action_mask[k, selected_edge_id]
            reward = reward + r
            reward_no_prob += r_no_prob
            self.reward[k, self.num_steps] = r.item()
            self.reward_no_prob[k, self.num_steps] = r_no_prob.item()
        self.num_steps += 1
        self.done = self.num_steps >= self.episode_length
        if self.done and self.if_test:
            action_mask_ = th.mul(self.mask, action)
            # print(self.num_steps, action_mask_[0].detach().cpu().numpy(), self.reward_no_prob[0].detach().cpu().numpy())
        # return (self.state, self.start, self.end, self.mask, action.detach()), reward, self.done
        return None

    def step0(self, action):
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

    def update_edge_and_node(self):
        self.multiple_num = self.get_num()
        for i in self.data:
            map_function, ii = i
            for j in i:
                map_function(i, j, 'set')
        return self.multiple_num

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

    def get_loss0(self, action):
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


def get_data():
    path = './sycamore.pkl'
    # with open(path, 'wb') as f:
    #     pickle.dump(obj, f)

    # 从文件中加载对象
    import pickle
    with open(path, 'rb') as f:
        obj = pickle.load(f)

    string = repr(obj)
    return string[string.find('[['):, string.rfind(']]')]
