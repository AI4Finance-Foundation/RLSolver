import numpy as np
from copy import deepcopy


class TSPEnv():
    def __init__(self, N=50, ):
        self.N = N
        self.move_cost = -1
        self.invalid_action_penalty = -100
        self.nodes = np.arange(self.N)
        self.step_limit = 2 * self.N
        self.state_dim = self.N ** 2    # (self.N ** 2 + 1) - > (self.N ** 2), done
        self.action_dim = self.N
        self.reset()
    
    def reset(self):
        self.step_count = 0
        self.generate_connections() # ER & power Law, Curriculum Learning (dimension)
        self.current_node = np.random.choice(self.nodes) # indicator vector
        self.visit_log = {n: 0 for n in self.nodes} 
        self.visit_log[self.current_node] += 1    
        self.state = self.update_state() # return indicator vector
        self.done = False
        return self.state  
    
    def step(self, action):
        done = False
        connections = self.node_dict[self.current_node] # action N^2
        #adj_matrix[current_node][action.where(=1)]
        if action not in connections:      # 2-D plane, complete graph
            reward = self.invalid_action_cost
        else:
            self.current_node = action
            reward = self.move_cost      # adjacency_matrix enough
            self.visit_log[self.current_node] += 1
            
        self.state = self.update_state()
        self.step_count += 1
        unique_visits = sum([1 if v > 0 else 0 
            for v in self.visit_log.values()])
        done = True if ... else False
        # if unique_visits >= self.N:
        #     done = True
        #     reward += 1000
        # if self.step_count >= self.step_limit:
        #     done = True
            
        return self.state, reward, done, {}
        
    def update_state(self):
        node_connections = self.adjacency_matrix.copy()
        visited = np.array([bool(min(v, 1))
            for v in self.visit_log.values()])
        node_connections[:, visited] = -1
        node_connections[np.where(self.adjacency_matrix==0)] = 0

        connections = node_connections.flatten().astype(int)
        obs = np.hstack([self.current_node, connections])
        state = obs.copy()
        return state
        
    def generate_connections(self):
        node_dict = {}
        for n in range(self.N):
            connections = np.random.randint(2, self.N - 1)
            node_dict[n] = np.sort(
               np.random.choice(self.nodes[np.where(self.nodes!=n)],
                                 size=connections, replace=False))
        # Get unique, bi-directional connections
        for k, v in node_dict.items():
            for k1, v1 in node_dict.items():
                if k == k1:
                    continue
                if k in v1 and k1 not in v:
                    v = np.append(v, k1)

            node_dict[k] = np.sort(v.copy())
        self.node_dict = deepcopy(node_dict)
        self.generate_adjacency_matrix()
    
    def generate_adjacency_matrix(self):
        self.adjacency_matrix = np.zeros((self.N, self.N))
        for k, v in self.node_dict.items():
            self.adjacency_matrix[k][v] += 1
        self.adjacency_matrix.astype(int)