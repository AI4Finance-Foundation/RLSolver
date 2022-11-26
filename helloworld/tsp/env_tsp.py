import torch
import numpy as np
import pickle as pkl
from functorch import vmap
class TSPEnv():
    def __init__(self, N=4, episode_length=6, num_env=4096, device=torch.device("cuda:0")):
        self.N = N  # #antennas
        self.basis_vectors, _ = torch.linalg.qr(torch.rand(self.N * 2, self.N * 2, dtype=torch.float))
        self.subspace_dim = 1
        self.num_env = num_env
        self.device = device
        self.episode_length = episode_length
        self.shape_holder = self.zeros(self.num_env, 1)
        self.get_reward_vec = vmap(self.get_reward, in_dims = (0, 0), out_dims = (0))
        self.generate_graph_cl_vec = vmap(self.generate_graph_cl, in_dims = 0, out_dims = 0)
        self.generate_graph_rand_vec = vmap(self.generate_graph_rand, in_dims = 0, out_dims = 0)
        self.generate_W_rand_vec = vmap(self.generate_W_rand, in_dims=0, out_dims=0)
        self.diag = 1e6 * torch.eye(self.N)

    def reset(self, test=False):
        if test:
            with open("./N=15Samples156.pkl", 'rb') as f:
                self.mat_H = torch.as_tensor(pkl.load(f), dtype=torch.float).to(self.device)
            self.shape_holder = self.zeros(self.mat_H.shape[0], 1)
        else:
            self.shape_holder = self.zeros(self.num_env, 1)
            if self.subspace_dim <= self.N:
                self.mat_H = self.generate_graph_cl_vec(self.shape_holder)
            else:
                self.mat_H = self.generate_graph_rand_vec(self.shape_holder)
            
        self.mat_W = self.generate_W_rand_vec(self.shape_holder)
        self.num_steps = 0
        self.done = False
        return (self.mat_H, self.mat_W)
    
    def step(self, action ):
        self.mat_W = action
        self.reward = self.get_vec_reward(self.mat_H, self.mat_W)
        self.mat_W = self.mat_W.detach()
        self.num_steps += 1
        self.done = True if self.num_steps >= self.episode_length else False
        return (self.mat_H, self.mat_W), self.reward, self.done

    def generate_graph_cl(self, shape):
        coordinates = torch.rand(self.subspace_dim, 1, device=self.device)
        vec_H = torch.matmul(self.basis_vectors[:self.subspace_dim].T, coordinates)
        tmp_Hx = torch.mat_mul(vec_H[:self.N], torch.ones(1, self.N, device=self.device))
        tmp_Hy = torch.mat_mul(vec_H[self.N:], torch.ones(1, self.N, device=self.device))
        mat_H = torch.sqrt((tmp_Hx - tmp_Hx.T) ** 2 + (tmp_Hy - tmp_Hy.T) ** 2) + self.diagself.diag
        return mat_H

    def generate_graph_rand(self, shape):
        vec_H = torch.rand(self.N * 2) * 2 - 1
        tmp_Hx = torch.mat_mul(vec_H[:self.N], torch.ones(1, self.N, device=self.device))
        tmp_Hy = torch.mat_mul(vec_H[self.N:], torch.ones(1, self.N, device=self.device))
        mat_H = torch.sqrt((tmp_Hx - tmp_Hx.T) ** 2 + (tmp_Hy - tmp_Hy.T) ** 2) + self.diagself.diag
        return mat_H
    
    def generate_W_rand(self, shape):
        mat_W = torch.rand(self.N, self.N)
        mat_W = mat_W.softmax(dim=0)
        mat_W = mat_W.softmax(dim=1)
        return mat_W
        
    def get_reward(self, H, W):
        return torch.mul(H, W.T).mean()