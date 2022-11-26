import torch
import functorch
import numpy as np


class MaxcutEnv():
    def __init__(self, N = 20, num_env=4096, device=torch.device("cuda:0"), episode_length=6):
        self.N = N
        self.state_dim = self.N * self.N + self.N # adjacency mat + configuration
        self.basis_vectors, _ = torch.linalg.qr(torch.randn(self.N * self.N, self.N * self.N, dtype=torch.float))
        self.num_env = num_env
        self.device = device
        self.sparsity = 0.005
        self.subspace_dim = 1
        self.episode_length = episode_length
        self.get_cut_value_tensor = functorch.vmap(self.get_cut_value, in_dims=(0, 0))
        self.generate_adjacency_symmetric_matrix_tensor = functorch.vmap(self.generate_adjacency_symmetric_matrix, in_dims=0)
    
    def reset(self, if_test=False, test_adjacency_matrix=None):
        if if_test:
            self.adjacency_matrix = test_adjacency_matrix.to(self.device)
        else:
            self.adjacency_matrix = self.generate_adjacency_symmetric_matrix_batch(if_binary=False, sparsity = self.sparsity).to(self.device)
        self.configuration = torch.rand(self.adjacency_matrix.shape[0], self.N).to(self.device).to(self.device)
        self.num_steps = 0
        return self.adjacency_matrix, self.configuration
    
    def step(self, configuration):
        self.configuration = configuration   # num_env x N x 1
        self.reward = self.get_cut_value_tensor(self.adjacency_matrix, self.configuration)
        self.num_steps +=1
        self.done = True if self.num_steps >= self.episode_length else False
        return (self.adjacency_matrix, self.configuration.detach()), self.reward, self.done

    def generate_adjacency_symmetric_matrix(self, sparsity): # sparsity for binary
        upper_triangle = torch.mul(torch.rand(self.N, self.N).triu(diagonal=1), (torch.rand(self.N, self.N) < sparsity).int().triu(diagonal=1))
        adjacency_matrix = upper_triangle + upper_triangle.transpose(-1, -2)
        return adjacency_matrix # num_env x self.N x self.N
        
    
    def get_cut_value(self,  adjacency_matrix, configuration):
        return torch.mul(torch.matmul(configuration.reshape(self.N, 1), (1 - configuration.reshape(-1, self.N, 1)).transpose(-1, -2)), adjacency_matrix).flatten().sum(dim=-1)
        
