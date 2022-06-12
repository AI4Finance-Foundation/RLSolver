import numpy as np
import torch.nn as nn
import torch
from torch import Tensor
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from elegantrl.agents.AgentPPO import AgentPPO
from elegantrl.agents.AgentBase import AgentBase
from elegantrl.agents.AgentDQN import AgentDQN
from elegantrl.train.config import Arguments
import sys
from elegantrl.train.evaluator import Evaluator
from elegantrl.train.replay_buffer import ReplayBuffer, ReplayBufferList
from copy import deepcopy
from utils import gen_adjacency_matrix_unweighted, gen_adjacency_matrix_weighted
import os
import time
def kwargs_filter(func, kwargs: dict) -> dict:
    import inspect  # Python built-in package

    sign = inspect.signature(func).parameters.values()
    sign = set([val.name for val in sign])

    common_args = sign.intersection(kwargs.keys())
    filtered_kwargs = {key: kwargs[key] for key in common_args}
    return filtered_kwargs
class Maxcut:    
    def __init__(self, N=10, adjacency=None):
        self.N = N
        self.configure = None
        self.adjacency = adjacency
        self.maxstep = self.N * 1
        self.max_step = self.N * 1
        f = open('./617.npy', 'rb') 
        self.initial_adjacency = np.random.randint(0, 2, self.N)#np.load(f)[0]
        self.if_discrete = True
        self.s = 0
        self.record = []
        self.record_ = []
        self.min_H = 0
        self.min_configuration = np.random.randint(0,2,self.N)
        
    def reset(self,):
        #print("Reset:")
        self.s = 0
        #self.configure = self.initial_adjacency#randint(0,2, self.N)
        #f = open('./617.npy', 'rb')
        
        self.configure = deepcopy(self.min_configuration)#np.random.randint(0, 2, self.N)#self.initial_adjacency
        #self.configure = np.array([0, 0, 0])
        self.H = self.calc_H(self.configure)
        #print(self.configure)
        #assert 0
        self.record.append(self.configure)
        return self.configure
            
    def step(self, action=None):
        #print("#################### step ###############")
        #print(action)
        #time.sleep(0.1)
        configure = self.configure
        next_configure = self.configure
        
        next_configure = (next_configure + action) % 2
        #next_configure[action] = (a + 1) % 2
        
        #print("action: ", action, "current configure: ", self.configure)
        #next_configure[action] = (a + 1) % 2
        #print("After flipping: ", next_configure)
        #time.sleep(0.1)
        #time.sleep(1)
        H = self.calc_H(next_configure)
        reward = (self.H - H) * 10
        self.H = H
        #print(self.calc_H(self.configure))
        self.configure = next_configure
        #print(next_configure, reward)
        self.s+=1
        #print(self.configure)
        self.record.append(self.configure)
        #assert 0
        #print("Step ", self.s, next_configure, reward)
        if self.H < self.min_H:
            self.min_H = deepcopy(self.H)
            self.min_configuration = deepcopy(self.configure)

        if self.s >= self.max_step:
            self.record_.append(self.record)
            self.record = []
            return next_configure, reward, True, {} 
        #print(next_configure, reward)
        return next_configure, reward, False, {}

    def calc_H(self, configure):
        H = 0
        for i in range(self.N):
            for j in range(self.N):
                H -= configure[i] * (1 - configure[j]) * self.adjacency[i,j]
        return H


def init_args(N, adjacency):
    env_func = Maxcut
    # env_func = gym.make
    env_args = {
        'N': N,
        'adjacency': adjacency,
        
    }
    
    args = Arguments(AgentDQN, env_func=env_func, env_args=env_args)
    args.env_num = 1
    args.state_dim = N
    args.action_dim = N
    args.target_return = 100000000
    args.num_layer = 3
    args.net_dim = 2 ** 10
    args.batch_size = int(args.net_dim)
    
    args.worker_num = 4
    args.target_step = args.batch_size
    args.repeat_times = 2 ** 5
    args.reward_scale = 2 ** -3

    args.learning_rate = 3e-4
    args.clip_grad_norm = 10.0
    args.gamma = 0.985

    args.lambda_gae_adv = 0.94
    args.if_use_gae = True
    args.ratio_clip = 0.4
    args.lambda_entropy = 2 ** -4

    '''H-term'''
    args.h_term_sample_rate = 2 ** -2
    args.h_term_drop_rate = 2 ** -3
    args.eval_times = 2 ** 1
    args.if_allow_break = False
    args.break_step = int(2e7)
    args.if_discrete = True
    return args

def init_buffer(args: Arguments, gpu_id: int) :
    if args.if_off_policy:
        buffer = ReplayBuffer(gpu_id=gpu_id,
                              max_capacity=args.max_memo,
                              state_dim=args.state_dim,
                              action_dim=args.action_dim if args.if_discrete else args.action_dim, )
        buffer.save_or_load_history(args.cwd, if_save=False)

    else:
        buffer = ReplayBufferList()
    return buffer


def init_evaluator(args: Arguments, gpu_id: int) -> Evaluator:
    eval_func = args.eval_env_func if getattr(args, "eval_env_func") else args.env_func
    eval_args = args.eval_env_args if getattr(args, "eval_env_args") else args.env_args
    eval_env = build_env(args.env, eval_func, eval_args)
    evaluator = Evaluator(cwd=args.cwd, agent_id=gpu_id, eval_env=eval_env, args=args)
    return evaluator

def build_env(env=None, env_func=None, env_args: dict = None):  # [ElegantRL.2021.12.12]
    if env is not None:
        env = deepcopy(env)
    elif env_func.__module__ == 'gym.envs.registration':
        import gym
        gym.logger.set_level(40)  # Block warning
        env = env_func(id=env_args['env_name'])
        env.env_num = 1
        env.env_name = env_args['env_name']
    else:
        env = env_func(**kwargs_filter(env_func.__init__, env_args.copy()))

    for attr_str in ('state_dim', 'action_dim', 'max_step', 'if_discrete', 'target_return'):
        if (not hasattr(env, attr_str)) and (attr_str in env_args):
            setattr(env, attr_str, env_args[attr_str])
    return env

def init_agent(args: Arguments, gpu_id: int, env=None) -> AgentBase:
    agent = args.agent_class(args.net_dim, args.state_dim, args.action_dim, gpu_id=gpu_id, args=args)
    agent.save_or_load_agent(args.cwd, if_save=False)

    if env is not None:
        '''assign `agent.states` for exploration'''
        if args.env_num == 1:
            state = env.reset()
            assert isinstance(state, np.ndarray) or isinstance(state, torch.Tensor)
            #print(args.state_dim)
            #assert 0
            assert state.shape in {(args.state_dim,), args.state_dim}
            states = [state, ]
        else:
            states = env.reset()
            assert isinstance(states, torch.Tensor)
            assert states.shape == (args.env_num, args.state_dim)
        agent.states = states
    return agent

def star(N=10):
    mat = np.zeros((N,N))
    for i in range(1,N):
        mat[0, i] = 1
        mat[i, 0] = 1
    return mat

def run(seed=1, gpu_id = 0, v_num = 100):
    import time
    np.random.seed(seed + int(time.time()))
    s = v_num
    mat = gen_adjacency_matrix_unweighted(s, 0.4)
    with open("./adjacency.npy", 'rb') as f:
        mat = np.load(f)
    #with open(f"./{v_num}/{1}/adjacency.npy", 'rb') as f:
    #    mat = np.load(f)
    #mat = star(v_num)
    #mat = np.array([[0, 1, 1], [1, 0, 0], [1, 0, 0]])
    #agent = DHN(adjacency_mat = mat, state_dim=s, action_dim=s, gpu_id=gpu_id)
    #agent.load(100,20)
    args = init_args(s, mat, )
    env = build_env(args.env, args.env_func, args.env_args)
    env.initial_adjacency = mat
    #agent = AgentPPO(256, v_num, v_num, gpu_id = gpu_id, args = args)
    agent = init_agent(args, gpu_id, env)
    buffer = init_buffer(args, gpu_id)
    evaluator = init_evaluator(args, gpu_id)
    
    env = build_env(args.env, args.env_func, args.env_args)
    agent.state = env.reset()
    if args.if_off_policy:
        trajectory = agent.explore_env(env, args.target_step)
        buffer.update_buffer([trajectory, ])

    '''start training'''
    cwd = args.cwd
    break_step = args.break_step
    target_step = args.target_step
    if_allow_break = args.if_allow_break
    del args

    if_train = True
    while if_train:
        trajectory = agent.explore_env(env, target_step)
        print(trajectory[0].shape)
        steps, r_exp = buffer.update_buffer([trajectory, ])
        with open("configuration_record.npy", 'wb') as f:
            np.save(f, np.array(env.record_))
        torch.set_grad_enabled(True)
        logging_tuple = agent.update_net(buffer)
        torch.set_grad_enabled(False)
        print(env.min_H, env.min_configuration)
        
        
        evaluator.eval_env = env
        (if_reach_goal, if_save) = evaluator.evaluate_save_and_plot(agent.act, steps, r_exp, logging_tuple)
        dont_break = not if_allow_break
        not_reached_goal = not if_reach_goal
        stop_dir_absent = not os.path.exists(f"{cwd}/stop")
        if_train = (
                (dont_break or not_reached_goal)
                and evaluator.total_step <= break_step
                and stop_dir_absent
        )
    print(f'| UsedTime: {time.time() - evaluator.start_time:.0f} | SavedDir: {cwd}')

    agent.save_or_load_agent(cwd, if_save=True)

    buffer.get_state_norm(
        cwd=cwd,
        state_avg=getattr(env, 'neg_state_avg', 0),
        state_std=getattr(env, 'div_state_std', 1),
    )
    if hasattr(buffer, 'save_or_load_history'):
        print(f"| LearnerPipe.run: ReplayBuffer saving in {cwd}")
        buffer.save_or_load_history(cwd, if_save=True)
    
    
if __name__ =='__main__':
    GPU_ID = int(sys.argv[1]) if len(sys.argv) > 1 else 0  # >=0 means GPU ID, -1 means CPU
    v_num = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    #ENV_ID = int(sys.argv[3]) if len(sys.argv) > 3 else 1  
    while True:
        run(1, GPU_ID, v_num)

