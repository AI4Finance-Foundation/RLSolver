import torch as th
import numpy as np


class MaxcutEnv():
    def __init__(self, N = 20, num_env=4096, device=th.device("cuda:0"), episode_length=6):
        self.N = N
        self.state_dim = self.N * self.N + self.N # adjacency mat + configuration
        #self.adjacency_matrix
        self.basis_vectors, _ = th.linalg.qr(th.randn(self.N * self.N, self.N * self.N, dtype=th.float))
        self.num_env = num_env
        self.device = device
        self.sparsity = 0.005
        self.subspace_dim = 1
        self.episode_length = episode_length
    
    def reset(self, if_test=False, test_adjacency_matrix=None):
        if if_test:
            self.adjacency_matrix = test_adjacency_matrix.to(self.device)
        else:
            self.adjacency_matrix = self.generate_adjacency_symmetric_matrix_batch(if_binary=False, sparsity = self.sparsity).to(self.device)
        self.configuration = th.rand(self.adjacency_matrix.shape[0], self.N).to(self.device).to(self.device)
        self.num_steps = 0
        return self.adjacency_matrix, self.configuration
    
    def step(self, configuration):
        self.configuration = configuration   # num_env x N x 1
        self.reward = self.get_cut_value(self.configuration)
        next_state = (self.adjacency_matrix, self.configuration.detach())
        self.num_steps +=1
        self.done = True if self.num_steps >= self.episode_length else False
        return next_state, self.reward, self.done

    def generate_adjacency_symmetric_matrix_batch(self, sparsity, if_binary, CL=False, graph_type='ER'): # sparsity for binary
        if if_binary:
            self.upper_triangle = (th.rand(self.num_env, self.N, self.N) < sparsity).int().triu(diagonal=1)
        else:
            self.upper_triangle = th.mul(th.rand(self.num_env, self.N, self.N).triu(diagonal=1), (th.rand(self.num_env, self.N, self.N) < self.sparsity).int().triu(diagonal=1))
        
        self.adjacency_matrix = self.upper_triangle + self.upper_triangle.transpose(-1, -2)
        return self.adjacency_matrix # num_env x self.N x self.N
        
    
    def get_cut_value(self,configuration):
        return th.mul(th.bmm(configuration.reshape(-1, self.N, 1), (1 - configuration.reshape(-1, self.N, 1)).transpose(-1, -2)), self.adjacency_matrix).flatten(start_dim=1).sum(dim=-1)
        
