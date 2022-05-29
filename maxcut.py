import numpy as np
import torch.nn as nn
import torch
from torch import Tensor
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
import sys


def build_mlp(mid_dim: int, num_layer: int, input_dim: int, output_dim: int):  # MLP (MultiLayer Perceptron)
    assert num_layer >= 1
    net_list = list()
    if num_layer == 1:
        net_list.extend([nn.Linear(input_dim, output_dim), ])
    else:  # elif num_layer >= 2:
        net_list.extend([nn.Linear(input_dim, mid_dim), nn.ReLU()])
        for _ in range(num_layer - 2):
            net_list.extend([nn.Linear(mid_dim, mid_dim), nn.ReLU()])
        net_list.extend([nn.Linear(mid_dim, output_dim), ])
    return nn.Sequential(*net_list)

class Policy(nn.Module):
    def __init__(self, mid_dim: int, num_layer: int, state_dim: int, action_dim: int):
        super().__init__()
        self.net = build_mlp(mid_dim, num_layer, input_dim=state_dim, output_dim=action_dim)
        self.explore_noise_std = 0.1  # standard deviation of exploration action noise
        self.log_sqrt_2pi = np.log(np.sqrt(2 * np.pi))

    def forward(self, state: Tensor) -> Tensor:
        return self.net(state).sigmoid()  # action

   

class DHN:
    def __init__(self, mid_dim=256, num_layer=3, state_dim=10, 
                 action_dim=10, gamma=0.9, gpu_id=0, learning_rate=3e-4,
                 G=10, N=10, T=10, clip_grad_norm = 3.0, adjacency_mat=None):
        self.gamma = gamma
        self.device = torch.device(f"cuda:{gpu_id}" if (torch.cuda.is_available() and (gpu_id >= 0)) else "cpu")
        self.learning_rate = learning_rate
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.adjacency_mat = adjacency_mat
        self.policy = Policy(mid_dim, num_layer, state_dim * state_dim, action_dim).to(self.device)
        self.N = N
        self.T = T
        self.clip_grad_norm = clip_grad_norm
        
        self.G = G
        self.optimizer = torch.optim.Adam(self.policy.parameters(), self.learning_rate)
        self.replay_buffer = []
        self.obj_record = []
                
                
    def explore(self,):
        traj = []
        for _ in range(self.N):
            
            v = np.random.randint(0, self.action_dim)
            
            while True:
                non_zero_edge = np.where(self.adjacency_mat[v] == 1)[0]
                if non_zero_edge.shape[0] == 0:
                    v = np.random.randint(0, self.action_dim)
                else:
                    break
                
            traj.append(v)
            
            for i in range(self.T):
                non_zero_edge = np.where(self.adjacency_mat[v] == 1)[0]
                #print(non_zero_edge)
                #print(np.random.choice(non_zero_edge))
                v = np.random.choice(non_zero_edge)
                traj.append(v)
            
            self.replay_buffer.append(traj)
        
    def train(self,):
        for _ in range(self.G):
            s = torch.as_tensor(self.adjacency_mat, dtype=torch.float32)
            s = torch.flatten(s)
            s = s.reshape([1, self.state_dim * self.state_dim]).to(self.device)
            p = self.policy(s)
            #print(p.shape) 
            obj = 0
            
            for i in range(self.state_dim):
                for j in range(self.state_dim):
                    obj += p[0,i] * (1 - p[0,j]) * self.adjacency_mat[i][j]
            '''
            obj_1 = 0
            l = min(100, len(self.replay_buffer))
            l_ = np.random.randint(0, len(self.replay_buffer), [l])
            for j in l_:
                traj = self.replay_buffer[j]
                for i in range(1,len(traj)):
                    obj_1 += p[0,traj[i - 1]] * (1 - p[0,traj[i]])
                    obj_1 += (1 - p[0,traj[i - 1]]) * p[0,traj[i]]
            obj_1 = obj_1 / len(self.replay_buffer)
            obj += obj_1
            '''
            obj = -obj
            #print(obj.item())
            self.obj_record.append(obj.item())
            self.optimizer.zero_grad()
            print(obj.grad)
            obj.backward()
            clip_grad_norm_(parameters=self.optimizer.param_groups[0]["params"], max_norm=self.clip_grad_norm)
            self.optimizer.step()
    
    def save(self,):
        
        #print(self.obj_record)
        import os
        file_list = os.listdir()
        if '{}'.format(self.action_dim) not in file_list:
            os.mkdir('{}'.format(self.action_dim))
        file_list = os.listdir('./{}/'.format(self.action_dim))
        
        exp_id = 0

        for name in file_list:
            exp_id_ = int(name)
            if exp_id_+1 > exp_id:
                exp_id = exp_id_ + 1
        print("Objective function is:\n")
        print([self.obj_record[i] for i in range(0, len(self.obj_record, 100))])
        #import plotext as plt
        #plt.plot(self.obj_record)
        #plt.title("Exp {} Obj ".format(exp_id))
        #plt.show()
        print("Finished experiment {}.".format(exp_id))


        os.mkdir('./{}/{}/'.format(self.action_dim, exp_id))
        t = self.action_dim
        path = './{}/{}/policy.pth'.format(self.action_dim, exp_id)
        matrix_file = ''
        torch.save(self.policy.state_dict(), path)
        matrix_file = './{}/{}/adjacency.npy'.format(self.action_dim, exp_id)
        np.save(matrix_file, self.adjacency_mat)
        obj_record = np.array(self.obj_record)

        np.save('./{}/{}/obj.npy'.format(self.action_dim,exp_id), obj_record)
        

    def load(self, path_1, path_2):
        self.policy.load_state_dict(torch.load(path_1, map_location=torch.device('cpu')))
        self.adjacency_mat = np.load(path_2)
        
    def test(self,):
        s = torch.as_tensor(self.adjacency_mat, dtype=torch.float32)
        s = torch.flatten(s)
        s = s.reshape([1, self.state_dim * self.state_dim]).to(self.device)
        p = self.policy(s)
        
        return p.detach().cpu().numpy() 

def get_adjacency_matrix(size=10):
    '''generate a binary symmetric matrix'''
    mat = np.random.randint(0, 2, (size, size))
    for i in range(size):
        for j in range(0, i):
            mat[j,i] = mat[i,j]
    #mat ^= mat.T
    return mat

def run(seed=1, gpu_id = 0, v_num = 10):
    import time
    np.random.seed(seed + int(time.time()))
    s = v_num
    mat = get_adjacency_matrix(s)
    agent = DHN(adjacency_mat = mat, state_dim=s, action_dim=s, gpu_id=gpu_id)
    try:
        for i in tqdm(range(100)):
            #agent.explore()
            agent.train()
        agent.save()
    except KeyboardInterrupt:
        agent.save()
        exit()
    
    
if __name__ =='__main__':
    GPU_ID = int(sys.argv[1]) if len(sys.argv) > 1 else 0  # >=0 means GPU ID, -1 means CPU
    v_num = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    #ENV_ID = int(sys.argv[3]) if len(sys.argv) > 3 else 1  
    while True:
        run(1, GPU_ID, v_num)
