import torch as th
import numpy as np
from baseline_mmse import compute_mmse_beamformer

class MIMOEnv():
    def __init__(self, K=4, N=4, P=10, noise_power=1, episode_length=6, num_env=4, device=th.device("cuda:0")):
        self.N = N  # #antennas
        self.K = K  # #users
        self.P = P  # Power
        self.noise_power = noise_power
        self.basis_vectors, _ = th.linalg.qr(th.rand(2 * self.K * self.N, 2 * self.K * self.N, dtype=th.float))
        self.subspace_dim = 1
        self.num_env = num_env
        self.device = device
        self.episode_length = episode_length
        
    def reset(self,):
        if self.subspace_dim <= 2 * self.K * self.N:
            self.vec_H = self.generate_channel_batch(self.N, self.K, self.num_env, self.subspace_dim, self.basis_vectors).to(self.device)
        else:
            self.vec_H = th.randn(self.num_env, 2 * self.K * self.N, dtype=th.cfloat).to(self.device)
        self.mat_H = (self.vec_H[:, :self.K * self.N] + self.vec_H[:, self.K * self.N:] * 1.j).reshape(-1, self.K, self.N)
        self.mat_W = compute_mmse_beamformer(self.mat_H).to(self.device)
        self.num_steps = 0
        self.done = False
        return (self.mat_H, self.mat_W)

    def step(self, action):
        self.mat_W = action
        HW = th.bmm(self.mat_H, self.mat_W.transpose(-1, -2))
        S = th.abs(th.diagonal(HW, dim1=-2, dim2=-1))**2
        I = th.sum(th.abs(HW)**2, dim=-1) - th.abs(th.diagonal(HW, dim1=-2, dim2=-1))**2
        N = 1
        SINR = S/(I+N)
        self.reward=  th.log2(1+SINR).sum(dim=-1).unsqueeze(-1)
        self.mat_W = self.mat_W.detach()
        self.num_steps += 1
        self.done = True if self.num_steps >= self.episode_length else False
        return (self.mat_H, self.mat_W), self.reward, self.done
    
    def generate_channel_batch(self, N, K, batch_size, subspace_dim, basis_vectors):
        coordinates = th.randn(batch_size, subspace_dim, 1)
        basis_vectors_batch = basis_vectors[:subspace_dim].T.repeat(batch_size, 1).reshape(-1, 2 * K * N, subspace_dim)
        vec_channel = th.bmm(basis_vectors_batch, coordinates).reshape(-1 ,2 * K * N) * (( 2 * K * N / subspace_dim) ** 0.5)
        return  (N * K) ** 0.5 * (vec_channel / vec_channel.norm(dim=1, keepdim = True))