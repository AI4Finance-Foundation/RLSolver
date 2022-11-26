import torch
import torch.nn as nn
import numpy as np
from functorch import vmap

class ClassicalSimulationEnv():
    def __init__(self, N=4, episode_length=6, num_env=4096, device=torch.device("cuda:0")):
        self.N = N  
        self.device = device
        self.num_env = num_env
        self.episode_length = episode_length
        self.get_vec_reward = vmap(self.get_reward, in_dims = (0, 0), out_dims = (0, 0))
        self.epsilon = 1
        self.test = False
        

    def reset(self,):
        self.adjacency_matrix =  torch.randint(low_value=1,high_value=10,size=(self.num_env, self.N, self.N))
        self.outer_dim = torch.randint(low_value=1, high_value=10, size=(self.num_env, self.N))
        self.num_steps = 0
        self.done = False
        return self.dim
    
    def step(self, action ):
        reward = self.adjacency_matrix * 
        self.reward = reward
        self.num_steps += 1
        self.done = True if self.num_steps >= self.episode_length else False
        return self.dim, self.reward, self.done, reward.detach()

class Policy_Net_ClassicalSimulation(nn.Module):
    def __init__(self, mid_dim=1024, N=4, ):
        super(Policy_Net_ClassicalSimulation, self).__init__()
        self.N = N
        self.action_dim = 2
        self.theta_0 = nn.Linear(self.K * 2, self.encode_dim)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.mid_dim = mid_dim
        self.net = nn.Sequential(
        nn.Linear(5, mid_dim * 2),
        nn.Linear(mid_dim * 2, mid_dim * 2), nn.ReLU(),
        nn.Linear(mid_dim * 2, mid_dim * 2), nn.ReLU(),
        nn.Linear(mid_dim * 2, self.action_dim),
        )
        self.output_layer = nn.Sigmoid().to(self.device)

    def forward(self, state):
        action = self.sigmoid(self.net(state))
        action = action / (action[:, 0] + action[:, 1])
        return action

def train_curriculum_learning(policy_net_ClassicalSimulation, optimizer, device, N=4,  num_epochs=100000000, num_env=512):
    env_ClassicalSimulation_ = ClassicalSimulationEnv(N=N, device=device, num_env=num_env, episode_length=1)
    for epoch in range(num_epochs):
        state = env_ClassicalSimulation_.reset()
        loss = 0
        while(1):
            action = policy_net_ClassicalSimulation(state)
            next_state, reward, done, _ = env_ClassicalSimulation_.step(action)
            loss += reward
            state = next_state
            if done:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                break
                
            
if __name__  == "__main__":
    N = 4   
    
    mid_dim = 256
    learning_rate = 5e-5
    
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    policy_net_ClassicalSimulation = Policy_Net_ClassicalSimulation(mid_dim= mid_dim, N=N).to(device)
    optimizer = torch.optim.Adam(policy_net_ClassicalSimulation.parameters(), lr=learning_rate)
    
    train_curriculum_learning(policy_net_ClassicalSimulation, optimizer,N=N,  device=device, )
