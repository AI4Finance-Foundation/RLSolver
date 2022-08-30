import pickle as pkl
import torch as th
import time
import os
from net import Net_MIMO

def train_mimo(net_mimo, optimizer, curriculum_base_vectors, num_users=4, num_antennas=4, total_power=10, noise_power=1, num_training_epochs=40000,
               num_subspace_update_gap=400, num_save_model_gap = 1000, episode_length=5, fullspace_dim=32, cur_subspace=1, learning_rate=5e-5,
               batch_size=8192, mid_dim=512, current_step=1, device=th.device("cuda:0" if th.cuda.is_available() else "cpu")):
    
    for i in range(num_training_epochs):
        if current_step % num_subspace_update_gap == 0 and cur_subspace < 32:
            cur_subspace +=1
            channel = compute_channel(num_antennas, num_users, fullspace_dim, batch_size, total_power,current_step, cur_subspace, curriculum_base_vectors).to(device)
        else:
            channel = th.randn(batch_size, num_antennas, num_users, dtype=th.cfloat).to(device)
        
        net_input = th.cat((th.as_tensor(channel.real).reshape(-1, num_users * num_antennas), th.as_tensor(channel.imag).reshape(-1, num_users * num_antennas)), 1).to(device)
        input_w_unflatten = net_mimo.calc_mmse(channel).to(device)
        input_w= th.cat((th.as_tensor(input_w_unflatten.real).reshape(-1, num_users * num_antennas), th.as_tensor(input_w_unflatten.imag).reshape(-1, num_users * num_antennas)), 1).to(device)
        loss = 0
        for _ in range(episode_length):
            input_hw_unflatten = th.bmm(channel, input_w_unflatten.transpose(1,2).conj())
            input_hw = th.cat((th.as_tensor(input_hw_unflatten.real).reshape(-1, num_users * num_antennas), th.as_tensor(input_hw_unflatten.imag).reshape(-1, num_users * num_antennas)), 1).to(device)
            net_output = net_mimo(th.cat((net_input, input_w, input_hw), 1), input_hw_unflatten, channel)
            output_w = (net_output.reshape(batch_size, 2, num_users * num_antennas)[:, 0] + net_output.reshape(batch_size, 2, num_users * num_antennas)[:, 1] * 1j).reshape(-1, num_users, num_antennas)
            loss -= net_mimo.calc_wsr(channel, output_w, noise_power).sum()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f" training_loss: {loss.item():.3f} | gpu memory: {th.cuda.memory_allocated():3d}")
        if current_step % num_save_model_gap == 0:
            th.save(net_mimo.state_dict(), save_path+f"{current_step}.pth")
        current_step += 1

def compute_channel(num_antennas, num_users, fullspace_dim, batch_size , cur_subspace, base_vectors):
    coordinates = th.randn(batch_size, cur_subspace, 1)
    channel = th.bmm(base_vectors[:cur_subspace].T.repeat(batch_size, 1).reshape(batch_size, base_vectors.shape[1], fullspace_dim), coordinates).reshape(-1 ,2 * num_users * num_antennas) * (( 32 / cur_subspace) ** 0.5) * (num_antennas * num_users) ** 0.5
    channel = (channel / channel.norm(dim=1, keepdim = True)).reshape(-1, 2, num_users, num_antennas)
    return (channel[:, 0] + channel[:, 1] * 1.j).reshape(-1, num_users, num_antennas)

def get_experiment_path(file_name):
    file_list = os.listdir()
    if file_name not in file_list:
        os.mkdir(file_name)
    file_list = os.listdir('./{}/'.format(file_name))
    max_exp_id = 0
    for exp_id in file_list:
        if int(exp_id) + 1 > max_exp_id:
            max_exp_id = int(exp_id) + 1
    os.mkdir('./{}/{}/'.format(file_name, exp_id))
    return f"./{file_name}/{max_exp_id}/"

if __name__  == "__main__":
    
    num_users = 4
    num_antennas = 4
    total_power = 10 
    noise_power = 1
    num_training_epochs = 40000 
    num_update_each_subspace = 400
    num_epoch_test = 1000
    episode_length = 5
    fullspace_dim = 2 * num_users * num_antennas
    curriculum_base_vectors, _ = th.linalg.qr(th.rand(fullspace_dim, fullspace_dim, dtype=th.float))
    learning_rate = 5e-5
    batch_size = 8192
    mid_dim = 512
    cur_subspace = 1
    current_step = 1
    file_name = f"lr_{learning_rate}_bs_{batch_size}_middim_{mid_dim}"
    device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
    net_mimo = Net_MIMO(mid_dim).to(device)
    optimizer = th.optim.Adam(mmse_net.parameters(), lr=learning_rate)
    save_path = get_experiment_path(file_name)
    
    print("start of session")
    start_of_time = time.time()
    
    try:
        train(net_mimo, optimizer, curriculum_base_vectors=curriculum_base_vectors)
    
    except KeyboardInterrupt:
        th.save(net_mimo.state_dict(), save_path+"0.pth")
        exit()
    th.save(net_mimo.state_dict(), save_path+"0.pth")
    print("Training took:", time.time()-start_of_time)