import torch as th
import numpy as np
import copy
from copy import deepcopy
from gnn_net import gnn_net
import time
import wandb
from tqdm import tqdm
from utils import generate_channel, compute_weighted_sum_rate, save


# Set variables 
num_users = 4
num_antennas = 4
selected_users = [0,1,2,3] # Note that we select all the users.

total_power = 10 # power constraint
noise_power = 1
num_training_epochs = 40000
num_test_samples = 120

curriculum_base_vectors, _ = th.linalg.qr(th.randn(32,32,dtype=th.float))
num_update_each_subspace = 400
num_subspace = 2 * num_users * num_antennas
num_epoch_test = 5
cur_subspace_num = 1

total_steps = 0
learning_rate = 5e-5
batch_size = 8192
gamma = 0.99
mid_dim = 512
num_loop = 5

cwd = f"DROPOUT_bs={batch_size}_lr={learning_rate}_middim={mid_dim}"
config = {
    'method': 'wsr_unsupervised_CL_drop_out',
    'learning_rate': learning_rate,
    'batch_size': batch_size,
    'mid_dim' : mid_dim,
    'total_power': total_power,
    'num_loop': num_loop
}
wandb_ = True
if wandb_:
    wandb.init(
        project='wsr' + '_unsupervised_' + f'{total_power}',
        entity="beamforming",
        sync_tensorboard=True,
        config=config,
        name=cwd,
        monitor_gym=True,
        save_code=True,
    )

device = th.device("cuda:0")
gnn_net = gnn_net(mid_dim)
device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
gnn_net.to(device)
optimizer = th.optim.Adam(gnn_net.parameters(), lr=learning_rate)
scheduler = th.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=gamma)
print("start of session")
start_of_time = time.time()

import scipy.io
mat = scipy.io.loadmat('K=4_N=4_id=100_csinoise=0.mat')
H = mat["Hnoisy"]


for i in tqdm(range(num_training_epochs)):
    weight = 1
    batch_for_training = []
    initial_transmitter_precoder_batch = []
    gnn_net_input = []
    input_w = []
    channel = []
    
    # Expand subspace by one-dimension
    if (total_steps+1 ) % num_update_each_subspace == 0:
        subspace +=1
    if subspace > num_subspace:
        subspace = num_subspace
    total_steps += 1

    # Generate training samples
    for ii in range(batch_size):
        channel.append(generate_channel(num_antennas, num_users, total_power,total_steps, subspace, curriculum_base_vectors, np.array([H[:, :, 0]]), fullspace=True))
    
    
    channel = th.as_tensor(channel).to(device)
    gnn_net_input = th.cat((th.as_tensor(channel.real).reshape(-1, 16), th.as_tensor(channel.imag).reshape(-1, 16)), 1).to(device)        
    
    w = gnn_net.calc_mmse(channel).to(th.cfloat)
    input_w= th.cat((th.as_tensor(w.real).reshape(-1, 16), th.as_tensor(w.imag).reshape(-1, 16)), 1).to(device)
    
    input_hw_unflatten = th.bmm(channel, w.transpose(1,2).conj())
    input_hw= th.cat((th.as_tensor(input_hw_unflatten.real).reshape(-1, 16), th.as_tensor(input_hw_unflatten.imag).reshape(-1, 16)), 1).to(device)
    
    gnn_net_output = gnn_net(th.cat((gnn_net_input, input_w, input_hw), 1), input_hw_unflatten, channel)
    output_w = (gnn_net_output.reshape(batch_size, 2, 16)[:, 0] + gnn_net_output.reshape(batch_size, 2, 16)[:, 1] * 1j).reshape(-1, 4, 4)
    
    obj = 0
    obj -= gnn_net.calc_wsr(th.as_tensor(np.array(channel)).to(device), output_w, noise_power, selected_users)
    
    for _ in range(num_loop):
        gnn_net_output = gnn_net(th.cat((gnn_net_input, gnn_net_output.detach(), input_hw.detach()), 1), input_hw_unflatten, )
        output_w = (gnn_net_output.reshape(batch_size, 2, 16)[:, 0] + gnn_net_output.reshape(batch_size, 2, 16)[:, 1] * 1j).reshape(-1, 4, 4)
        input_hw_unflatten = th.bmm(tmp, output_w.transpose(1,2).conj())
        input_hw= th.cat((th.as_tensor(input_hw_unflatten.real).reshape(-1, 16), th.as_tensor(input_hw_unflatten.imag).reshape(-1, 16)), 1).to(device)
        obj -= weight * gnn_net.calc_wsr(th.as_tensor(np.array(channel)).to(device), output_w, noise_power, selected_users)
        weight += 0.2

    if i % num_epoch_test == 0:
        WSR_last = []
        WSR_mean = []
        WSR_max = []
        
        for ii_ in range(num_test_samples):       
            wsr = []
            channel = th.as_tensor(np.array(generate_channel(num_antennas, num_users, total_power, total_steps, subspace, curriculum_base_vectors, H[:, :, ii_], fullspace=True))).to(device)
            gnn_net_input = th.cat((th.as_tensor(gnn_net_input.real).reshape(-1, 16), th.as_tensor(gnn_net_input.imag).reshape(-1, 16)), 1)
            gnn_net_input = th.as_tensor(gnn_net_input, dtype=th.float32).to(device)
            
            input_hw_unflatten = th.bmm(gnn_net_input, input_w.transpose(1,2).conj())
            input_hw = th.cat((th.as_tensor(input_hw_unflatten.real).reshape(-1, 16), th.as_tensor(input_hw_unflatten.imag).reshape(-1, 16)), 1)
            input_hw = th.as_tensor(input_hw, dtype=th.float32).to(device)
            input_w = th.cat((th.as_tensor(input_w.real).reshape(-1, 16), th.as_tensor(input_w.imag).reshape(-1, 16)), 1)
            input_w = th.as_tensor(input_w, dtype=th.float32).to(device)
            
            output = gnn_net(th.cat((gnn_net_input, input_w, input_hw), 1), input_hw_unflatten, gnn_net_input)
            output_w = (output[0].detach().cpu().numpy().reshape(2, 16)[0] + output[0].detach().cpu().numpy().reshape(2, 16)[1] * 1j).reshape(4,4)
            
            wsr.append(compute_weighted_sum_rate(channel, output_w, noise_power, selected_users))
            for _ in range(num_loop):
                output = gnn_net(th.cat((gnn_net_input, output.detach(), input_hw.detach()), 1), input_hw_unflatten)
                output_w = (output.reshape(-1, 2, 16)[:, 0] + output.reshape(-1, 2, 16)[:, 1] * 1j).reshape(-1, 4, 4)
                
                input_hw_unflatten = th.bmm(gnn_net_input, output_w.transpose(1,2).conj())
                input_hw= th.cat((th.as_tensor(input_hw_unflatten.real).reshape(-1, 16), th.as_tensor(input_hw_unflatten.imag).reshape(-1, 16)), 1).to(device)
                output_w  = output_w.detach().cpu().numpy().reshape(4,4)
                wsr.append(compute_weighted_sum_rate(channel, output_w, noise_power, selected_users))              
            wsr = np.array(wsr)
            WSR_last.append(wsr[-1])
            WSR_mean.append(wsr.mean())
            WSR_max.append(wsr.max())
    if wandb_:
        wandb.log({'wsr': obj.sum().item() / batch_size, '120_samples': np.mean(WSR_last), '120_max_wsr:': np.mean(WSR_max), '120_mean_wsr:': np.mean(WSR_mean)})
    print("traing_loss  wsr: ", obj.sum().item() / batch_size, " 120 samples: ",np.mean(WSR_last))
    optimizer.zero_grad()
    obj.sum().backward()
    optimizer.step()   
save(gnn_net, cwd)