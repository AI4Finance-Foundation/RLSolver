# 用于生成Tensor-Ring类型的邻接矩阵

import torch as th
import torch
device = th.device("cuda:0")
# 根据张量网络中张量数量设定
N = 100
# Env数量
num_env = 100
max_dim = 2
test_state = torch.ones((num_env, N + 2, N + 2), device=device).to(torch.float32)
mask = th.zeros(N + 2, N + 2).to(device)
mask[1, 1] = 1
for i in range(2, N + 1):
    mask[i, i-1] = 1
    mask[i, i] = 1

mask = mask.reshape(-1).repeat(1, num_env).reshape(num_env, N + 2, N + 2).to(device)
test_state = th.mul(test_state, mask)
test_state += th.ones_like(test_state)
test_state[N,1] = 2
with open(f"test_data_tensor_train_N={N}.pkl", 'wb') as f:
    import pickle as pkl
    pkl.dump(test_state, f)
