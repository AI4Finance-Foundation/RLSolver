import torch as th

class MaxcutEnv():
    def __init__(self, N = 20, num_env=20, device=th.device("cuda:0"), episode_length=6):
        self.N = N
        self.state_dim = self.N * self.N + self.N # adjacency mat + configuration
        self.adjacency_matrix
        self.num_env = num_env
        self.device = device
        self.subspace_dim = 1
        self.episode_length = episode_length
    
    def reset(self,):
        self.configuration = th.rand(self.num_env, self.N).to(self.device)
        self.adjacency_matrix = self.generate_adjacency_symmetric_matrix_batch()
        return self.adjacency_matrix, self.configuration
    
    def step(self, configuration):
        self.configuration = configuration   # num_env x N x 1
        self.reward = self.get_cut_value(self.configuration)
        next_state = (self.adjacency_matrix, self.configuration)
        self.num_steps +=1
        self.done = True if self.num_steps >= self.episode_length else False
        return next_state, self.reward, self.done

    def generate_adjacency_symmetric_matrix_batch(self, CL, graph_type, sparsity, if_binary, ): # sparsity for binary
        if if_binary:
            self.upper_triangle = (th.rand(self.num_env, self.N, self.N) < sparsity).int().triu(diagonal=1)
        else:
            self.upper_triangle = (th.rand(self.num_env, self.N, self.N)).triu(diagonal=1)
        self.adjacency_matrix = self.upper_triangle + self.upper_triangle.transpoer(-1, -2)
        return self.adjacency_matrix # num_env x self.N x self.N
        
    
    def get_cut_value(self,):
        return th.mul(th.bmm(self.configuration, (1 - self.configuration).T), self.adjacency_matrix).flatten().sum()
        