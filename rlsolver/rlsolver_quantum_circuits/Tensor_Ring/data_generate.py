# 用于生成Tensor-Ring类型的邻接矩阵

import torch as th
import torch
device = th.device("cpu")
# 根据张量网络中张量数量设定
N = 100
# Env数量
num_env = 100
max_dim = 2
test_state = th.ones((N+1, N+1), dtype=th.float32, device=device)
x_index = th.arange(1, N+1, dtype=th.int64, device=device)
y_index = th.arange(1, N+1, dtype=th.int64, device=device)
test_state[x_index, y_index] = 2
x_index = th.arange(2, N+1, dtype=th.int64, device=device)
y_index = th.arange(1, N, dtype=th.int64, device=device)
test_state[x_index, y_index] = 2
test_state = test_state.unsqueeze(0).repeat(num_env, 1, 1)
with open(f"test_data_tensor_ring_N={N}.pkl", 'wb') as f:
    import pickle as pkl
    pkl.dump(test_state, f)
