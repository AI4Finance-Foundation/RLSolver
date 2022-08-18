import torch as th
import numpy as np
import os
def compute_channel(num_antennas, num_users, num_subspace, curriculum_base, H = None, fullspace=False):
    channel = np.zeros((num_users, num_antennas)) + 1j*np.zeros((num_users, num_antennas))
    vector_curriculum = th.randn(num_subspace)
    H_CL = th.matmul(curriculum_base[:num_subspace].T, vector_curriculum).reshape(2,4,4)
    
    for i in range(num_users):
        path_loss = 0 # path loss is 0 dB by default, otherwise it is drawn randomly from a uniform distribution (N.B. it is different for each user)
        if H.shape[0] != 1:
            result_real = H.real[i,:].reshape(-1, 1)
            result_imag  = H.imag[i, :].reshape(-1, 1)
        else:
            if fullspace: #or (num_subspace >=32 and total_steps > 12801):
                result_real = np.sqrt(10**(path_loss/10))*np.sqrt(0.5)*np.random.normal(size = (num_antennas,1))
                result_imag = np.sqrt(10**(path_loss/10))*np.sqrt(0.5)*np.random.normal(size = (num_antennas,1))        
            else:
                result_real = H_CL[0][i].reshape(-1, 1).numpy()
                result_imag = H_CL[1][i].reshape(-1, 1).numpy()

        channel[i,:] = np.reshape(result_real,(1,num_antennas)) + 1j*np.reshape(result_imag, (1,num_antennas))
        
    return channel


def compute_weighted_sum_rate(channel, precoder, noise_power, selected_users):
    result = 0
    nr_of_users = np.size(channel,0)
    for user_index in range(nr_of_users):
        if user_index in selected_users:
            user_sinr = compute_sinr(channel, precoder, noise_power, user_index, selected_users)
        result = result + np.log2(1 + user_sinr)
    return result

def compute_sinr(channel, precoder, noise_power, user_id, selected_users):
    nr_of_users = np.size(channel,0)
    numerator = (np.absolute(np.matmul(channel[user_id,:], precoder[user_id,:])))**2

    inter_user_interference = 0
    for user_index in range(nr_of_users):
        if user_index != user_id and user_index in selected_users:
            inter_user_interference = inter_user_interference + (np.absolute(np.matmul(channel[user_id,:],precoder[user_index,:])))**2
    denominator = noise_power + inter_user_interference

    result = numerator/denominator
    return result

def save(net,folder_name):
    file_list = os.listdir()

    if folder_name not in file_list:
        os.mkdir(folder_name)

    file_list = os.listdir('./{}/'.format(folder_name))

    exp_id = 0

    for name in file_list:
        exp_id_ = int(name)
        if exp_id_+1 > exp_id:
            exp_id = exp_id_ + 1
    
    os.mkdir('./{}/{}/'.format(folder_name, exp_id))
    path = './{}/{}/net.pth'.format(folder_name, exp_id)
    th.save(net.state_dict(), path)
    print("Finished experiment {}, {}.".format(folder_name, exp_id))
