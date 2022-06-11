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
        net_list.extend([nn.Linear(input_dim, int(input_dim / 4) ), nn.ReLU()])
        #for _ in range(num_layer - 2):
        net_list.extend([nn.Linear(int(input_dim / 4), int(input_dim / 10) ), nn.ReLU()])
        #net_list.extend([nn.Linear(int(input_dim / 4), int(input_dim / 8) ), nn.ReLU()])
        net_list.extend([nn.Linear(int(input_dim / 10), int(output_dim * 5)), nn.ReLU()])
        net_list.extend([nn.Linear(int(output_dim * 5), output_dim), ])
    return nn.Sequential(*net_list)


class Policy(nn.Module):
    def __init__(self, mid_dim: int, num_layer: int, state_dim: int, action_dim: int):
        super().__init__()
        self.net = build_mlp(mid_dim, num_layer, input_dim=state_dim, output_dim=action_dim)

    def forward(self, state: Tensor, beta):
        return (self.net(state) * beta).sigmoid()

class Policy_softmax(nn.Module):
    def __init__(self, mid_dim: int, num_layer: int, state_dim: int, action_dim: int):
        super().__init__()
        self.action_dim = action_dim        
        self.net = build_mlp(mid_dim, num_layer, input_dim=state_dim, output_dim=action_dim * 2)
        self.softmax = nn.Softmax(dim = 2)

    def forward(self, state: Tensor, beta) -> Tensor:
        mid = self.net(state).reshape((1, self.action_dim, 2)) * beta
        return self.softmax(mid)

class DHN:
    def __init__(self, mid_dim=256, num_layer=3, state_dim=10, 
                 action_dim=10, gamma=0.999, gpu_id=0, learning_rate=3e-4,
                 G=2, N=1, T=1000, clip_grad_norm = 3.0, adjacency_mat=None):
        self.gamma = gamma
        self.device = torch.device(f"cuda:{gpu_id}" if (torch.cuda.is_available() and (gpu_id >= 0)) else "cpu")
        self.learning_rate = learning_rate
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.adjacency_mat = adjacency_mat
        self.policy = Policy(mid_dim, num_layer, state_dim * state_dim, action_dim * 10).to(self.device)
        self.N = N
        self.T = T
        self.clip_grad_norm = clip_grad_norm
        self.exp_id = -1
        self.G = G
        self.optimizer = torch.optim.Adam(self.policy.parameters(), self.learning_rate)
        self.replay_buffer = []
        self.obj_record = []
        self.beta = 0.919375
        self.explore_rate = 1

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
            p = self.policy(s, self.beta)
            #p
            #rint(p.shape) 
            obj = 0
            obj_1 = 0
            for i in range(self.state_dim):
                for j in range(self.state_dim):
                    obj += p[0, i] * (1 - p[0, j]) * self.adjacency_mat[i][j]

                # if np.random.rand() < 0.005:
                #    continue
                for j in range(self.state_dim):
                    #obj += p[0,i,0] * p[0,j,1] * self.adjacency_mat[i][j] * 10
                    print(p) 
                    obj_1 += p[0, i] * (1 - p[0, j]) * self.adjacency_mat[i][j] * 10000 #* self.dist[i] * self.dist[j]
            
            #obj_1 = 0 
            print("befor explore loss is ", -obj_1.item())
            
            '''
            l = min(self.N, len(self.replay_buffer))
            l_ = np.random.randint(0, len(self.replay_buffer), [l])
            
            obj_2 = 0
            for j in l_:
                traj = self.replay_buffer[j]
                g = 1
                for i in range(1,len(traj)):
                    #assert 0
                    obj_2 += p[0,traj[i - 1]] * (1 - p[0,traj[i]]) * g * self.adjacency_mat[traj[i - 1]][traj[i]] * 10000 * self.dist[traj[i - 1]] * self.dist[traj[i]]
                    obj_2 += (1 - p[0,traj[i - 1]]) * p[0,traj[i]] * g* self.adjacency_mat[traj[i - 1]][traj[i]] * 10000 * self.dist[traj[i - 1]] * self.dist[traj[i]]
                    g *= self.gamma
            #obj_1 = obj_1
            #obj += obj_1
            obj_1 += obj_2 * 0.1
            '''
            print("After explore dist loss is ",-obj_1.item())
            print("unif loss is ", -obj.item())
            obj_1 = -(obj_1) 
            #obj.retain_grad()
            #print("loss is ", obj_1.item())
            self.obj_record.append(obj.item())
            self.optimizer.zero_grad()
            #print(obj.grad)
            #obj.retain_grad()
            obj_1.backward()
            #obj.retain_grad()
            #print(obj.grad)
            #clip_grad_norm_(parameters=self.optimizer.param_groups[0]["params"], max_norm=self.clip_grad_norm)
            self.optimizer.step()
    
    def save(self,):
        import os
        #print(self.obj_record)
        exp_id = 0
        if self.exp_id == -1:
            
            file_list = os.listdir()
            if '{}'.format(self.action_dim) not in file_list:
                os.mkdir('{}'.format(self.action_dim))
            file_list = os.listdir('./{}/'.format(self.action_dim))

            for name in file_list:
                exp_id_ = int(name)
                if exp_id_+1 > exp_id:
                    exp_id = exp_id_ + 1
            self.exp_id = exp_id
            os.mkdir('./{}/{}/'.format(self.action_dim, exp_id))
        exp_id = self.exp_id
        #print("Objective function is:\n")
        #print([self.obj_record[i] for i in range(0, len(self.obj_record), 100)])
        #import plotext as plt
        #plt.plot(self.obj_record)
        #plt.title("Exp {} Obj ".format(exp_id))
        #plt.show()
        print("Finished experiment {}.".format(exp_id))


        #os.mkdir('./{}/{}/'.format(self.action_dim, exp_id))
        t = self.action_dim
        path = './{}/{}/policy.pth'.format(self.action_dim, exp_id)
        matrix_file = ''
        torch.save(self.policy.state_dict(), path)
        matrix_file = './{}/{}/adjacency.npy'.format(self.action_dim, exp_id)
        np.save(matrix_file, self.adjacency_mat)
        obj_record = np.array(self.obj_record)

        np.save('./{}/{}/obj.npy'.format(self.action_dim,exp_id), obj_record)
        

    def load(self, n, exp_id):
        #self.policy.load_state_dict(torch.load('./' + str(n) + '/' + str(exp_id) + '/policy.pth', map_location=torch.device(self.device)))
        self.adjacency_mat = np.load('./' + str(n) + '/' + str(exp_id) + '/adjacency.npy')
        
    def test(self,):
        s = torch.as_tensor(self.adjacency_mat, dtype=torch.float32)
        s = torch.flatten(s)
        s = s.reshape([1, self.state_dim * self.state_dim]).to(self.device)
        p = self.policy(s, 10.855375)
        
        return p.detach().cpu().numpy() 

def gen_adjacency_matrix_unweighted(n=10, p=0.5):
    '''generate a binary symmetric matrix'''
    mat = np.random.rand(n, n)
    dist = np.zeros(n)
    for i in range(n):
        for j in range(0, i + 1):
            if mat[i,j] <= p:
                mat[i, j] = 1
            else:
                mat[i, j] = 0
            
            mat[j,i] = mat[i,j] # symmetric
        mat[i, i] = 0

    E = 0
    for i in range(n):
        for j in range(n):
            dist[i] += mat[i,j]
            E += mat[i,j]
    dist = dist / E
    return mat, dist
def cal_dist(mat):
    n = mat.shape[0]
    dist = np.zeros(n)
    E = 0
    for i in range(n):
        for j in range(n):
            dist[i] += mat[i,j]
            E += mat[i,j]
    dist = dist / E
    return dist




def gen_adjacency_matrix_weighted(n=10, p=0.5):
    '''generate a weighted symmetric matrix'''
    mat = np.random.rand(n, n)

    for i in range(n):
        for j in range(0, i + 1):
            if mat[i,j] > p:
                mat[i, j] = 0
            mat[j,i] = mat[i,j] # symmetric
        mat[i, i] = 0 
    return mat

def star(N=10):
    mat = np.zeros((N,N))
    for i in range(1,N):
        mat[0, i] = 1
        mat[i, 0] = 1
    return mat

def run(seed=1, gpu_id = 0, v_num = 10):
    import time
    np.random.seed(seed + int(time.time()))
    s = v_num
    mat,dist = gen_adjacency_matrix_unweighted(s, 0.05)
    #mat =[ [0, 1, 1],[1, 0, 0], [1, 0, 0]]
    #mat = star(v_num)
    #print(mat)
    agent = DHN(adjacency_mat = mat, state_dim=s, action_dim=s, gpu_id=gpu_id)
    agent.load(100, 20)
    agent.dist = cal_dist(agent.adjacency_mat)
    try:
        for i in tqdm(range(10000)):
            #agent.explore()
            agent.train()
            agent.beta =( i) / 1000.0 + 0.919375
            agent.explore_rate = 1 - (i * 5 / 1000)
            if i % 10 == 0:
                agent.save()
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
