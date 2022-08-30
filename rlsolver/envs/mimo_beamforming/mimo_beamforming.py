from net import Policy_Net_MIMO
import torch as th
import time
import os

def train_mimo( policy_net_mimo, optimizer, curriculum_base_vectors, num_users=4, num_antennas=4, total_power=10, noise_power=1, num_training_epochs=40000,
                num_subspace_update_gap=400, num_save_model_gap=1000, episode_length=5, fullspace_dim=32, subspace_dim=1, batch_size=4096, 
                device=th.device("cuda:0" if th.cuda.is_available() else "cpu")):
    
    for epoch in range(num_training_epochs):
        if (epoch + 1) % num_subspace_update_gap == 0 and subspace_dim < fullspace_dim:
            subspace_dim +=1
            vec_H = generate_channel_batch(num_antennas, num_users, fullspace_dim, batch_size, subspace_dim, curriculum_base_vectors).to(device)
        else:
            vec_H = th.randn(batch_size, fullspace_dim, dtype=th.cfloat).to(device)
        mat_H = (vec_H[:, :num_users * num_antennas] + vec_H[:, num_users * num_antennas:] * 1.j).reshape(-1, num_users, num_antennas)
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
            th.save(policy_net_mimo.state_dict(), save_path+f"{epoch}.pth")

def generate_channel_batch(num_antennas, num_users, fullspace_dim, batch_size, subspace_dim, base_vectors):
    coordinates = th.randn(batch_size, subspace_dim, 1)
    base_vectors_batch = base_vectors[:subspace_dim].T.repeat(batch_size, 1).reshape(batch_size, base_vectors.shape[1], fullspace_dim)
    vec_channel = th.bmm(, coordinates).reshape(-1 ,2 * num_users * num_antennas) * (( 32 / subspace_dim) ** 0.5)
    return  (num_antennas * num_users) ** 0.5 * (vec_channel / vec_channel.norm(dim=1, keepdim = True))

def get_experiment_path(file_name):
    file_list = os.listdir()
    if file_name not in file_list:
        os.mkdir(file_name)
    file_list = os.listdir('./{}/'.format(file_name))
    max_exp_id = 0
    for exp_id in file_list:
        if int(exp_id) + 1 > max_exp_id:
            max_exp_id = int(exp_id) + 1
    os.mkdir('./{}/{}/'.format(file_name, max_exp_id))
    return f"./{file_name}/{max_exp_id}/"

if __name__  == "__main__":
    
    num_users = 4
    num_antennas = 4
    fullspace_dim = 2 * num_users * num_antennas
    curriculum_base_vectors, _ = th.linalg.qr(th.rand(fullspace_dim, fullspace_dim, dtype=th.float))
    mid_dim = 512
    learning_rate=5e-5
    file_name = "mimo_beamforming"
    device=th.device("cuda:0" if th.cuda.is_available() else "cpu")
    policy_net_mimo = Policy_Net_MIMO(mid_dim).to(device)
    optimizer = th.optim.Adam(policy_net_mimo.parameters(), lr=learning_rate)
    save_path = get_experiment_path(file_name)
    try:
        train_mimo(policy_net_mimo, optimizer, curriculum_base_vectors=curriculum_base_vectors, num_users=num_users, num_antennas=num_antennas, fullspace_dim=fullspace_dim)
    except KeyboardInterrupt:
        th.save(policy_net_mimo.state_dict(), save_path+"0.pth")
        exit()
    th.save(policy_net_mimo.state_dict(), save_path+"0.pth")