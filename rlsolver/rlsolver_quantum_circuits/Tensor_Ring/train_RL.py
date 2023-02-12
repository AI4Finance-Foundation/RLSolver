import torch
import torch as th
import torch.nn as nn
import numpy as np
from functorch import vmap
from copy import deepcopy
from tqdm import tqdm
# 取消科学计数法
np.set_printoptions(suppress=True)


class Env():
    def __init__(self, N=6, episode_length=6, num_env=4096, max_dim=2, epsilon=0.9, device=torch.device("cuda:0")):
        self.N = N
        self.device = device
        self.num_env = num_env
        self.episode_length = episode_length
        self.max_dim = max_dim
        with open(f"test_data_tensor_ring_N={N}.pkl", 'rb') as f:
            import pickle as pkl
            self.test_state = pkl.load(f)
        # self.zero = th.zeros(self.N - 1).to(self.device)
        self.state = th.zeros(self.num_env, self.N + 1, self.N + 1).to(self.device)
        x_index = th.arange(1, self.N + 1, dtype=th.int64, device=self.device)
        self.state[:, x_index, x_index] = 2
        self.state[:, N, 1] = 2
        self.mask = th.ones(self.num_env, self.N + 1, self.N + 1).to(self.device)
        self.indicator = th.ones(self.num_env, self.N).to(self.device)
        self.epsilon = epsilon
        self.reward = th.zeros(self.num_env, self.N + 1).to(self.device)

    def reset(self, test=False):
        self.state = th.zeros(self.num_env, self.N + 1, self.N + 1).to(self.device)
        x_index = th.arange(1, self.N + 1, dtype=th.int64, device=self.device)
        self.state[:, x_index, x_index] = 2
        self.state[:, N, 1] = 2
        self.mask = th.zeros(self.num_env, self.N + 1, self.N + 1).to(self.device)
        self.mask[:, x_index, x_index] = 1
        self.reward = th.zeros(self.num_env, self.N+1).to(self.device)
        self.reward_no_prob = th.zeros(self.num_env, self.episode_length).to(self.device)
        self.if_test = test
        self.mask = th.ones(self.num_env, self.N + 1, self.N + 1).to(self.device)
        if test:
            self.state = self.test_state
        self.num_steps = 0
        self.done = False
        initial_action = th.rand(self.num_env, self.N).to(self.device)
        initial_action /= initial_action.sum(dim=-1, keepdim=True)
        self.indicator = th.ones(self.num_env, self.N).to(self.device)
        return (self.state, self.mask, initial_action)

    def step(self, action):
        # action: (num_env, N)
        # mask: [num_env, N+1, N+1]
        reward_no_prob = 0  # 直接选概率最大的，贪心的reward
        valid_action = action * self.indicator
        valid_action = valid_action / valid_action.sum(dim=-1, keepdim=True)
        selected_edge = torch.argmax(valid_action, dim=-1)
        for env_i in range(self.num_env):
            r = 1.
            edge = selected_edge[env_i]
            first_node = self.N if ((edge + 1) % self.N == 0) else (edge + 1) % self.N
            second_node = self.N if ((edge + 2) % self.N == 0) else (edge + 2) % self.N
            first_node_cluster = [first_node]
            second_node_cluster = [second_node]
            for i in range(self.N):
                current_node = self.N if ((first_node - i) % self.N == 0) else (first_node - i) % self.N
                if self.mask[env_i, current_node, first_node] == 1:
                    first_node_cluster.append(current_node)
                else:
                    break
            for i in range(self.N):
                current_node = self.N if ((second_node - i) % self.N == 0) else (second_node - i) % self.N
                if self.mask[env_i, current_node, second_node] == 1:
                    second_node_cluster.append(current_node)
                else:
                    break
            for i in range(self.N):
                current_node = self.N if ((second_node + i) % self.N == 0) else (second_node + i) % self.N
                if self.mask[env_i, first_node, current_node] == 1:
                    second_node_cluster.append(current_node)
                else:
                    break
            for i in range(self.N):
                current_node = self.N if ((second_node - i) % self.N == 0) else (second_node - i) % self.N
                if self.mask[env_i, second_node, second_node] == 1:
                    second_node_cluster.append(current_node)
                else:
                    break
            for node in first_node_cluster:
                r *= torch.prod(self.state[env_i, node, 1:node])
                r *= torch.prod(self.state[env_i, node:, node])
                r *= self.state[env_i, node, node]
            for node in second_node_cluster:
                r *= torch.prod(self.state[env_i, node, 1:node])
                r *= torch.prod(self.state[env_i, node:, node])
                r *= self.state[env_i, node, node]
            for node_i in range(len(first_node_cluster)):
                for node_j in range(len(second_node_cluster)):
                    r /= self.state[env_i, first_node_cluster[node_i], second_node_cluster[node_j]]
                    r /= self.state[env_i, second_node_cluster[node_j], first_node_cluster[node_i]]
                    self.mask[env_i, first_node_cluster[node_i], second_node_cluster[node_j]] = 1
                    self.mask[env_i, second_node_cluster[node_j], first_node_cluster[node_i]] = 1

            reward_no_prob += r
            self.reward_no_prob[env_i, self.num_steps] = r
            self.reward[env_i, edge] = r

            self.indicator[env_i, edge] = 0
            self.state[env_i, first_node, second_node] = 1
            self.state[env_i, second_node, first_node] = 1
        self.done = True if torch.sum(self.indicator) == self.num_env else False


        return (self.state, self.mask, action), self.reward_no_prob, self.done


class Policy_Net(nn.Module):
    def __init__(self, mid_dim=1024, N=6, ):
        super(Policy_Net, self).__init__()
        self.N = N + 1
        self.action_dim = N
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.mid_dim = mid_dim
        self.net = nn.Sequential(
            nn.Linear((N + 1) * (N + 1) * 2, mid_dim * 2),
            nn.Linear(mid_dim * 2, mid_dim * 2), nn.ReLU(),
            nn.Linear(mid_dim * 2, mid_dim * 2), nn.ReLU(),
            nn.Linear(mid_dim * 2, self.action_dim),
        )
        self.output_layer = nn.Softmax().to(self.device)

    def forward(self, state):
        s, mask, previous_action = state
        action = self.output_layer(self.net(th.cat((s.reshape(s.shape[0], -1), mask.reshape(s.shape[0], -1)), dim=-1)))
        return action


def train_curriculum_learning(policy_net, optimizer, device, N=6, num_epochs=100000000, num_env=100, gamma=0.9,
                              best_reward=4e+31, if_wandb=False):
    env = Env(N=N, device=device, num_env=num_env, episode_length=N)
    for epoch in tqdm(range(num_epochs)):
        test = False
        if epoch % 10 == 0:
            test = True
        state = env.reset(test)
        loss = 0
        env.epsilon = max(0.5, 0.5 + 0.5 * (1 - epoch / 500))
        while (1):
            action = policy_net(state)
            next_state, reward_no_prob, done = env.step(action)
            state = next_state
            if done and test == False:
                g = 1
                discounted_reward = th.zeros(num_env).to(device)
                loss_ = 0.
                for i in range(N - 1, -1, -1):
                    discounted_reward = discounted_reward + env.reward_no_prob[:, i]
                    loss_ = loss_ + env.reward[:, i + 1] / env.reward_no_prob[:, i] * discounted_reward
                    discounted_reward = discounted_reward * gamma
                loss = loss_.mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                break
            if done and test == True:
                best_reward = min(best_reward, env.reward_no_prob.sum().item() / env.num_env)
                print(env.reward.sum().item() / env.num_env, env.reward_no_prob.sum().item() / env.num_env, best_reward,
                      epoch)
                # print(best_reward, epoch)
                if if_wandb:
                    wandb.log({"flops": env.reward, "flops_no_prob": env.reward_no_prob})
                break


if __name__ == "__main__":
    with th.enable_grad():
        N = 6

        mid_dim = 256
        learning_rate = 5e-5

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        policy_net = Policy_Net(mid_dim=mid_dim, N=N).to(device)
        policy_net.train()
        optimizer = torch.optim.Adam(policy_net.parameters(), lr=learning_rate)
        if_wandb = False
        if if_wandb:
            import wandb

            wandb.init(
                project='classical_simulation',
                entity="beamforming",
                sync_tensorboard=True,
            )
        train_curriculum_learning(policy_net, optimizer, N=N, device=device, if_wandb=if_wandb)
