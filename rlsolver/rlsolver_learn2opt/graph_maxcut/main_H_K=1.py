import torch as th
import torch.nn as nn
from copy import deepcopy
import functorch
import numpy as np
from torch import Tensor
class Opt_net(nn.Module):
    def __init__(self, N, hidden_layers):
        super(Opt_net, self).__init__()
        self.N = N
        self.hidden_layers = hidden_layers
        self.lstm = nn.LSTM(self.N, self.hidden_layers,batch_first=True)

    def forward(self, configuration, hidden_state, cell_state):
        x, (h, c) = self.lstm(configuration, (hidden_state, cell_state))
        return x.sigmoid(), h, c

class MaxcutEnv():
    def __init__(self, N = 20, num_env=128, device=th.device("cuda:0"), episode_length=6):
        self.N = N
        self.num_env = num_env
        self.device = device
        self.episode_length = episode_length
        self.get_cut_value_tensor = functorch.vmap(self.get_cut_value, in_dims=(0, 0))
        self.adjacency_matrix = None

    def load_graph(self, file_name):
        self.adjacency_matrix = th.as_tensor(np.load(file_name), device=self.device)

    def reset(self):
        self.configuration = th.rand(num_env, self.N).to(self.device)
        self.num_steps = 0
        return self.configuration

    # def step(self, configuration):
    #     self.configuration = configuration   # num_env x N x 1
    #     self.reward = self.get_cut_value_tensor(self.configuration)
    #     self.num_steps +=1
    #     self.done = True if self.num_steps >= self.episode_length else False
    #     return  self.configuration.detach(), self.reward, self.done

    def generate_adjacency_symmetric_matrix(self, sparsity): # sparsity for binary
        upper_triangle = th.mul(th.rand(self.N, self.N).triu(diagonal=1), (th.rand(self.N, self.N) < sparsity).int().triu(diagonal=1))
        adjacency_matrix = upper_triangle + upper_triangle.transpose(-1, -2)
        return adjacency_matrix # num_env x self.N x self.N

    def get_cut_value_and_indicators_for_one_tensor(self, mu: Tensor):
        # rand = th.randn(mu.size())
        indicators = mu >= 0.5
        n = len(indicators)
        matrix = th.zeros([n, n], dtype=th.int)
        for i in range(n):
            for j in range(n):
                if indicators[i] == indicators[j]:
                    matrix[i, j] = 0
                else:
                    matrix[i, j] = 1
        matrix *= self.adjacency_matrix
        cut = matrix.sum() / 2
        return cut, indicators


    def get_cut_value(self, mu1, mu2):
        return th.mul(th.matmul(mu_1.reshape(self.N, 1), (1 - mu_2.reshape(-1, self.N, 1)).transpose(-1, -2)), self.adjacency_matrix).flatten().sum(dim=-1)
        mu1_ = mu1.reshape(self.N, 1)
        mu2_ = 1 - mu2
        cut1, indicators1 = self.get_cut_value_and_indicators_for_one_tensor(mu1)
        cut2, indicators2 = self.get_cut_value_and_indicators_for_one_tensor(mu2)
        cut = cut1 + cut2 + (indicators1 != indicators2).sum()
        # for i in range(len(indicators1)):
        #     if indicators1[i] != indicators2[i]:
        #         cut += 1
        # cut2 = th.mul(th.matmul(mu_1.reshape(self.N, 1), (1 - mu_2.reshape(-1, self.N, 1)).transpose(-1, -2)), self.adjacency_matrix).flatten().sum(dim=-1)
        return cut


def train(N, num_env, device, opt_net, optimizer, episode_length, hidden_layer_size):
    env_maxcut = MaxcutEnv(N=N, num_env=num_env, device=device, episode_length=episode_length)
    env_maxcut.load_graph("./data/gset_G14.npy")
    h_init = th.zeros(1, num_env, hidden_layer_size).to(device)
    c_init = th.zeros(1, num_env, hidden_layer_size).to(device)
    for epoch in range(100000):
        prev_h, prev_c = h_init.clone(), c_init.clone()
        loss = 0
        loss_list = th.zeros(episode_length * num_env).to(device)
        action_prev = env_maxcut.reset()
        gamma0 = 0.98
        gamma = gamma0 ** episode_length
        for step in range(episode_length):
            action, h, c = opt_net(action_prev.reshape(num_env, 1, N), prev_h, prev_c)

            #action = action.reshape(num_env, N)
            l = env_maxcut.get_cut_value_tensor(action.reshape(num_env, N), action.reshape(num_env, N))
            loss_list[num_env*(step):num_env*(step+1)] = l.detach()
            loss -= l.sum()
            l = env_maxcut.get_cut_value_tensor(action_prev, action.reshape(num_env, N))
            loss -= 0.05 * l.sum()
            action_prev = action.detach()
            #prev_h, prev_c = h.detach(), c.detach()
            gamma /= gamma0


            if (step + 1) % 5 == 0:
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()
                loss = 0
                #h, c = h_init.clone(), c_init.clone()
            prev_h, prev_c = h.detach(), c.detach()

        if epoch % 50 == 0:
            print("train:",  loss_list.max().item())
            h, c = h_init, c_init
            # print(h_init.mean(), c_init.mean())
            loss = 0
            #loss_list = []
            loss_list = th.zeros(episode_length * num_env * 2).to(device)
            action = env_maxcut.reset()
            for step in range(episode_length * 2):
                action, h, c = opt_net(action.detach().reshape(num_env, 1, N), h, c)
                action = action.reshape(num_env, N)
                a = action.detach()
                a = (a>0.5).to(th.float32)
                # print(a)
                # assert 0
                l = env_maxcut.get_cut_value_tensor(a, a)
                loss_list[num_env*(step):num_env*(step+1)] = l.detach()
                #if (step + 6) % 2 == 0:
                    #optimizer.zero_grad()
                    #loss.backward()
                    #optimizer.step()
                    #loss = 0
                    #h, c = h_init.clone(), c_init.clone()

                # loss -= l
                # for i in range(num_env):
                #     for j in range(N):
                #         if a[i][j] > 0.5:
                #             a[i][j] = 1
                #         else:
                #             a[i][j] = 0
                #     l = env_maxcut.get_cut_value(a[i], a[i])
                #     loss_list.append(l)
                #     loss -= l

            print("train:",  loss_list.max().item())




if __name__ == "__main__":
    N = 800
    hidden_layer_size = N
    learning_rate = 1e-5
    num_env=1024
    episode_length = 20
    gpu_id = 0
    device = th.device(f"cuda:{gpu_id}" if (th.cuda.is_available() and (gpu_id >= 0)) else "cpu")
    th.manual_seed(0)
    opt_net = Opt_net(N, hidden_layer_size).to(device)
    optimizer = th.optim.Adam(opt_net.parameters(), lr=learning_rate)

    train(N, num_env, device, opt_net, optimizer, episode_length, hidden_layer_size)

