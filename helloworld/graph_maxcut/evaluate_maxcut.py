import torch as th
import numpy as np
from net_maxcut import Policy_Net_Maxcut
from env_maxcut import MaxcutEnv


def evaluator(policy_net_maxcut, N=4, device=th.device("cuda:0")):
    env_maxcut = MaxcutEnv( N=N, device=device, num_env=1)
    import pickle as pkl
    with open ("s2v_dqn_benchmark/gtype-erdos_renyi-nrange-15-n_graph-164-p-0.15-m-0-w-float-0-1-cnctd-0-seed-2.pkl", 'rb') as f:
        test_adjacency_matrix = th.as_tensor(np.array(pkl.load(f, encoding = 'latin1')), dtype=th.float).to(device)
    state = env_maxcut.reset(if_test=True, test_adjacency_matrix=test_adjacency_matrix)
    cut_value = th.zeros(state[0].shape[0], env_maxcut.episode_length)
    while(1):
        action = policy_net_maxcut(state)
        action = th.where(action >= 0.5, 1.0, 0.)
        next_state, reward, done = env_maxcut.step(action)
        cut_value[:, env_maxcut.num_steps-1] = reward
        state = next_state
        if done:
            break
    print(cut_value[0])
    print(cut_value.max(dim=1)[0][0])
    return cut_value.max(dim=1)[0].mean().item()
    
if __name__  == "__main__":
    N = 4   # number of nodes
    
    device=th.device("cuda:0" if th.cuda.is_available() else "cpu")
    policy_net_maxcut = Policy_Net_Maxcut(N=N).to(device)
    
    evaluator(policy_net_maxcut, N=N, device=device)
