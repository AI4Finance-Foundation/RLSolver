import os
import torch as th

def generate_channel_batch(N, K, batch_size, subspace_dim, basis_vectors):
    coordinates = th.randn(batch_size, subspace_dim, 1)
    basis_vectors_batch = basis_vectors[:subspace_dim].T.repeat(batch_size, 1).reshape(-1, 2 * K * N, subspace_dim)
    vec_channel = th.bmm(basis_vectors_batch, coordinates).reshape(-1 ,2 * K * N) * (( 2 * K * N / subspace_dim) ** 0.5)
    return  (N * K) ** 0.5 * (vec_channel / vec_channel.norm(dim=1, keepdim = True))

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