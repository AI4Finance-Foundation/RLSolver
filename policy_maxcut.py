import torch as th
import numpy as np
class Maxcut:
    def __init__(self, adjacency_matrix):
        self.adjacency_matrix = adjacency_matrix
        self.N = adjacency_matrix.shape[0]
        self.state = np.random.rand(self.N)
    
    def reset(self, ):
        self.state = np.random.rand(self.N )
        return self.state
    
    def step(self, action):
        self.state = (self.state + action).clip(0,1)
        reward = self.get_reward(self.clip)
        done = False
        return self.state, reward, done, {}

    def calc_H(self, configure):
        H = 0
        for i in range(self.N):
            for j in range(self.N):
                H -= configure[i] * (1 - configure[j]) * self.adjacency[i,j]
        return H