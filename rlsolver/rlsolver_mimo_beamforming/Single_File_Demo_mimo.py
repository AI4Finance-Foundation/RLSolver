import os
import torch
import sys
import torch
import numpy as np
from functorch import vmap

class MIMOEnv():
    def __init__(self, K=4, N=4, P=10, noise_power=1, episode_length=6, num_env=4096, device=torch.device("cuda:0"), reward_mode='sl', snr = 10):
        self.N = N  # #antennas
        self.K = K  # #users
        self.P = P  # Power
        self.noise_power = noise_power
        self.device = device
        self.num_env = num_env
        self.episode_length = episode_length
        self.get_vec_sum_rate = vmap(self.get_sum_rate, in_dims = (0, 0), out_dims = (0, 0))
        self.num_x = 1000
        self.epsilon = 1
        self.snr = snr
        self.reward_mode = reward_mode
        self.test = False

    def reset(self,):
        self.vec_H = self.generate_channel_batch(self.N, self.K, self.num_env, self.subspace_dim, self.basis_vectors)
        self.mat_H = (self.vec_H[:, :self.K * self.N] + self.vec_H[:, self.K * self.N:] * 1.j).reshape(-1, self.K, self.N)
        self.mat_H = self.mat_H 
        
        vec_W = torch.randn((self.mat_H.shape[0], self.K* self.K), dtype=torch.cfloat, device=self.device)
        vec_W = vec_W / torch.norm(vec_W, dim=1, keepdim=True)
        self.mat_W = vec_W.reshape(-1, self.K, self.N)
        HW = torch.bmm(self.mat_H, self.mat_W.transpose(-1, -2))
        self.num_steps = 0
        self.done = False
        return (self.mat_H, self.mat_W, self.P, HW)
    def step(self, action ):
        sum_rate = 0
        sum_rate, HW = self.get_vec_sum_rate(self.mat_H, action)
        self.reward = sum_rate
        self.mat_W = action.detach()
        self.num_steps += 1
        self.done = True if self.num_steps >= self.episode_length else False
        return (self.mat_H, self.mat_W, self.P, HW.detach()), self.reward, self.done, sum_rate.detach()

    def generate_channel_batch(self, N, K, batch_size):
        vec_channel = torch.randn(2 * batch_size * N * K, device=self.device)
        return vec_channel / np.sqrt(2)
    def get_sum_rate(self, H, W):
        HW = torch.matmul(H, W.T)
        S = torch.abs(HW.diag()) ** 2
        I = torch.sum(torch.abs(HW)**2, dim=-1) - torch.abs(HW.diag()) ** 2
        N = 1
        SINR = S / (I + N)
        reward = torch.log2(1 + SINR).sum(dim=-1)
        return reward, HW

    
reward_mode = ['empirical', 'analytical', 'supervised_mmse', 'rl', 'supervised_mmse_curriculum']
def train_curriculum_learning(policy_net_mimo, optimizer, save_path, device, K=4, N=4, P=10, noise_power=1, num_epochs=100000000, num_env=512):
    env_mimo_relay = MIMOEnv(K=K, N=N, P=P, noise_power=noise_power, device=device, num_env=num_env, reward_mode=reward_mode[int(sys.argv[1])], episode_length=1)
    for epoch in range(num_epochs):
        state = env_mimo_relay.reset()
        policy_net_mimo.previous = torch.randn(1, num_env, policy_net_mimo.mid_dim * 2, device=device)
        loss = 0
        sr = 0
        while(1):
            action = policy_net_mimo(state)
            next_state, reward, done, _ = env_mimo_relay.step(action)
            loss -= reward
            state = next_state
            if done:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                break
                
            
if __name__  == "__main__":
    N = 4   # number of antennas
    K = 4   # number of users
    SNR = 10 # "10_15_20"
    P = 20 ** (SNR / 10)
    
    mid_dim = 2048
    noise_power = 1
    learning_rate = 5e-5
    cwd = f"{reward_mode[int(sys.argv[1])]}_N{N}K{K}SNR{SNR}"
    env_name = f"N{N}K{K}SNR{SNR}_mimo_beamforming"
    save_path = None
    
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    policy_net_mimo = Policy_Net_MIMO(mid_dim= mid_dim, K=K, N=N, P=P).to(device)
    optimizer = torch.optim.Adam(policy_net_mimo.parameters(), lr=learning_rate)
    
    train_curriculum_learning(policy_net_mimo, optimizer, K=K, N=N, save_path=save_path, device=device, P=P, noise_power=noise_power)
