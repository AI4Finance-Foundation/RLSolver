import os
import torch as th
from envs.mimo_beamforming.env_net_mimo import Policy_Net_MIMO
from envs.mimo_beamforming.env_mimo import generate_channel_batch

def test_curriculum_learning(policy_net_mimo, test_path, device, K=4, N=4, P=10, noise_power=1):
    episode_length = 6
    with open(test_path, 'rb') as f:
        mat_H = th.as_tensor(pkl.load(f)).to(device).to(th.cfloat)
    mat_W = policy_net_mimo.calc_mmse(mat_H).to(device)
    sum_rate = th.zeros[mat_H.shape[0], episode_length, 1].to(device)
    for step in range(episode_length):
        mat_W = policy_net_mimo(mat_H, mat_W)
        sum_rate[:, step] = policy_net_mimo.calc_sum_rate(mat_H, mat_W)
    
    print(f" test sum_rate on 120 samples: {sum_rate.reshape(-1, episode_length).max(dim=1)[0].mean():.3f}}")

if __name__  == "__main__":
    N = 4   # number of antennas
    K = 4   # number of users
    P = 10  # power constraint
    noise_power = 1
    
    device=th.device("cuda:0" if th.cuda.is_available() else "cpu")
    policy_net_mimo = Policy_Net_MIMO().to(device)
    trained_model_path = "rl_cl_sum_rate_9.77_trained_network.pth"
    policy_net_mimo.load_state_dict(th.load(trained_model_path, map_location=device))
    test_path = "Channel_K=4_N=4_P=10_Samples=120_Optimal=9.9.pkl"
    
    test_curriculum_learning(policy_net_mimo, K=K, N=N, device=device, P=P, noise_power=noise_power, test_path=test_channel_path)