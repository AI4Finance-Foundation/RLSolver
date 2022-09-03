import torch as th
from net_maxcut import Policy_Net_Maxcut
from maxcut import MaxcutEnv

def test(policy_net_maxcut, N=4, device=th.device("cpu")):
    env_maxcut = MaxcutEnv( N=N, device=device, num_env=1000)
    import pickle as pkl
    state = env_maxcut.reset()
    cut_value = th.zeros(state[0].shape[0], env_maxcut.episode_length, 1)
    while(1):
        action = policy_net_maxcut(state)
        next_state, reward, done = env_maxcut.step(action)
        cut_value[:, env_maxcut.num_steps-1] = reward
        state = next_state
        if done:
            break
    print(f"test_sum_rate: {cut_value.max(dim=1)[0].mean().item()}")
    
if __name__  == "__main__":
    N = 4   # number of nodes
    
    device=th.device("cuda:0" if th.cuda.is_available() else "cpu")
    policy_net_maxcut = Policy_Net_Maxcut().to(device)
    
    test(policy_net_maxcut, N=N, device=device)