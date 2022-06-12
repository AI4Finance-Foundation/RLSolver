import torch as th
import numpy as np
class Maxcut:
    def __init__(self, adjacency_matrix):
        self.adjacency_matrix = adjacency_matrix
        self.n = adjacency_matrix.shape[0]
        self.state = np.random.rand(self.n)
    
    def reset(self, ):
        self.state = np.random.rand(self.n )
        return self.state
    
    def step(self, action):
        self.state = (self.state + action).clip(0,1)
        reward = self.get_reward(self.clip)
        return self.state, 
        None
    def get_reward(self,):
        for i in range()