import torch
import torch as th
import torch.nn as nn
import numpy as np
from functorch import vmap
from copy import deepcopy


class Env():
    def __init__(self, N=4, episode_length=6, num_env=4096, device=torch.device("cuda:0")):
        self.N = N
        self.device = device
        self.num_env = num_env
        self.episode_length = episode_length
        self.epsilon = 1
        self.test_state = torch.randint(0, 9, (self.num_env, self.N + 2, self.N + 2), device=self.device).to(
            torch.float32)
        self.mask = th.zeros(self.N + 2, self.N + 2).to(self.device)
        self.mask[1, 1] = 1
        for i in range(2, self.N + 1):
            self.mask[i, i - 1] = 1
            self.mask[i, i] = 1
        self.mask = self.mask.reshape(-1).repeat(1, self.num_env).reshape(self.num_env, self.N + 2, self.N + 2).to(
            self.device)
        with open("test_4.pkl", 'rb') as f:
            import pickle as pkl
            self.test_state = pkl.load(f)
        self.permute_base = th.as_tensor([i for i in range(self.N - 1)]).repeat(1, self.num_env).reshape(self.num_env,
                                                                                                         -1).to(
            self.device)
        self.start = th.as_tensor([i for i in range(self.N)]).repeat(1, self.num_env).reshape(self.num_env, -1).to(
            self.device)
        self.end = th.as_tensor([i for i in range(self.N)]).repeat(1, self.num_env).reshape(self.num_env, -1).to(
            self.device)
        self.zero = th.zeros(self.N - 1).to(self.device)

    def reset(self, test=False):
        self.state = torch.randint(0, 9, (self.num_env, self.N + 2, self.N + 2), device=self.device).to(torch.float32)
        self.state = th.mul(self.state, self.mask)
        self.state += th.ones_like(self.state)
        self.if_test = test
        if test:
            self.state = self.test_state
        self.num_steps = 0
        self.done = False
        initial_action = th.rand(self.num_env, self.N - 1).to(self.device)
        initial_action /= initial_action.sum(dim=-1, keepdim=True)
        return (self.state, initial_action)

    def step(self, action):
        reward = 0
        reward_no_prob = 0
        if self.if_test:
            print(action[:2])
        for k in range(action.shape[0]):
            for c in range(2):
                if self.if_test:
                    sorted, indices = action[k].sort(descending=True)
                    permute = indices
                else:
                    permute = self.permute_base[k, th.randperm(self.N - 1)]
                r = 0
                r_no_prob = 0
                p = 1
                state = self.state[k]
                start = deepcopy(self.start[k]) + 1
                end = deepcopy(self.end[k]) + 1
                for i in permute:
                    p *= action[k, i]
                    tmp = 1
                    for j in range(start[i], end[i] + 1):
                        tmp *= (state[j, j] * state[j, start[i] - 1] * state[end[i] + 1, j])
                    for j in range(start[i + 1], end[i + 1] + 1):
                        tmp *= (state[j, j] * state[j, start[i + 1] - 1] * state[end[i + 1] + 1, j])
                    tmp / state[start[i + 1], start[i + 1] - 1]
                    start_new = min(start[i], start[i + 1])
                    end_new = max(end[i], end[i + 1])
                    for __ in range(start_new, end_new + 1):
                        start[__ - 1] = start_new
                        end[__ - 1] = end_new
                    r += tmp * p
                    r_no_prob += tmp
                reward += r
                reward_no_prob += r_no_prob


        self.num_steps += 1
        self.reward = reward / self.num_env
        self.reward_no_prob = reward_no_prob / self.num_env
        self.done = True if self.num_steps >= self.episode_length else False
        return (self.state, action.detach()), reward, self.done


class Policy_Net(nn.Module):
    def __init__(self, mid_dim=1024, N=4, ):
        super(Policy_Net, self).__init__()
        self.N = N + 2
        self.action_dim = N - 1
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.mid_dim = mid_dim
        self.net = nn.Sequential(
            nn.Linear(self.N * self.N + N - 1, mid_dim * 2),
            nn.Linear(mid_dim * 2, mid_dim * 2), nn.ReLU(),
            nn.Linear(mid_dim * 2, mid_dim * 2), nn.ReLU(),
            nn.Linear(mid_dim * 2, self.action_dim),
        )
        self.output_layer = nn.Softmax().to(self.device)

    def forward(self, state):
        action = self.output_layer(self.net(th.cat((state[0].reshape(state[0].shape[0], -1), state[1]), dim=-1)))
        return action


def train_curriculum_learning(policy_net, optimizer, device, N=4, num_epochs=100000000, num_env=128):
    env = Env(N=N, device=device, num_env=num_env, episode_length=1)
    for epoch in range(num_epochs):
        test = False
        if epoch % 2 == 0:
            test = True
        state = env.reset(test)
        loss = 0
        # print(state[0].shape, state[1].shape)
        while (1):
            action = policy_net(state)
            next_state, reward, done = env.step(action)
            loss += reward
            state = next_state
            if done and test == False:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                break
            if done and test == True:
                print(env.reward)
                print(env.reward_no_prob)
                wandb.log({"flops": env.reward, "flops_no_prob": env.reward_no_prob})
                break


if __name__ == "__main__":
    N = 4

    mid_dim = 256
    learning_rate = 5e-5

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    policy_net = Policy_Net(mid_dim=mid_dim, N=N).to(device)
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=learning_rate)
    import wandb

    wandb.init(
        project='classical_simulation',
        entity="beamforming",
        sync_tensorboard=True,
    )

    train_curriculum_learning(policy_net, optimizer, N=N, device=device, )
