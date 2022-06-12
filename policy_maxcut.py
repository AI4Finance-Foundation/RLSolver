import torch as th
import numpy as np
class Maxcut:
    def __init__(self, adjacency_matrix):
        self.adjacency_matrix = adjacency_matrix
        self.n = adjacency_matrix.shape[0]
        self.state = np.random.rand(self.n)
    
    def reset(self, ):
        self.state = np.random.rand(() )
    
    def step(self, ):
        None
    def get_reward(self, tate):
        for i in range()