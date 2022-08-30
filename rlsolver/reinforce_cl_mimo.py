import os
import torch as th
from envs.mimo_beamforming.env_net_mimo import Policy_Net_MIMO
from envs.mimo_beamforming.env_mimo import generate_channel_batch

def train_curriculum_learning( policy_net_mimo, optimizer, save_path, K=4, N=4, total_power=10, noise_power=1, num_training_epochs=40000,
                num_subspace_update_gap=400, num_save_model_gap=1000, episode_length=5, subspace_dim=1, batch_size=4, 
                device=th.device("cuda:0" if th.cuda.is_available() else "cpu")):
    # generate basis vectors of an N x K space, using QR decomposition
    basis_vectors, _ = th.linalg.qr(th.rand(2 * K * N, 2 * K * N, dtype=th.float))
    
    for epoch in range(num_training_epochs):
        if (epoch + 1) % num_subspace_update_gap == 0 and subspace_dim < 2 * K * N:
            subspace_dim +=1
        if subspace_dim < 2 * K * N:
            vec_H = generate_channel_batch(N, K, batch_size, subspace_dim, basis_vectors).to(device)
        else:
            vec_H = th.randn(batch_size, 2 * K * N, dtype=th.cfloat).to(device)
        mat_H = (vec_H[:, :K * N] + vec_H[:, K * N:] * 1.j).reshape(-1, K, N)
        mat_W = policy_net_mimo.calc_mmse(mat_H).to(device)
        loss = 0
        for _ in range(episode_length):
            mat_W = policy_net_mimo(mat_H, mat_W)
            loss -= policy_net_mimo.calc_sum_rate(mat_H, mat_W).sum()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f" training_loss: {loss.mean().item():.3f} | gpu memory: {th.cuda.memory_allocated():3d}")
        if epoch % num_save_model_gap == 0:
            th.save(policy_net_mimo.state_dict(), save_path + f"{epoch}.pth")
            
def get_experiment_path(env_name):
    file_list = os.listdir()
    if env_name not in file_list:
        os.mkdir(env_name)
    file_list = os.listdir('./{}/'.format(env_name))
    max_exp_id = 0
    for exp_id in file_list:
        if int(exp_id) + 1 > max_exp_id:
            max_exp_id = int(exp_id) + 1
    os.mkdir('./{}/{}/'.format(env_name, max_exp_id))
    return f"./{env_name}/{max_exp_id}/"

if __name__  == "__main__":
    
    K = 4   # number of users
    N = 4   # number of antennas
    mid_dim = 512
    learning_rate = 5e-5
    
    env_name = "mimo_beamforming"
    save_path = get_experiment_path(env_name) # folder to save the trained policy net
    device=th.device("cuda:0" if th.cuda.is_available() else "cpu")
    policy_net_mimo = Policy_Net_MIMO(mid_dim).to(device)
    optimizer = th.optim.Adam(policy_net_mimo.parameters(), lr=learning_rate)
    
    try:
        train_curriculum_learning(policy_net_mimo, optimizer, K=K, N=N, save_path=save_path)
        th.save(policy_net_mimo.state_dict(), save_path + "policy_net_mimo_1.pth")  # number your result policy net
    except KeyboardInterrupt:
        th.save(policy_net_mimo.state_dict(), save_path + "policy_net_mimo_1.pth")  # number your result policy net
        exit()
