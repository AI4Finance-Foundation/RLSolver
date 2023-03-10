import pickle as pkl
import torch as th
from copy import deepcopy
import numpy as np

device = th.device("cuda:0")
reward = 0
N = 127
np.set_printoptions(suppress=True)
with open(f"test_data_tensor_ring_N={N}.pkl", 'rb') as f:
    a = pkl.load(f).detach().cpu()
num_env = a.shape[0]
num_samples = 500
stend_ = th.zeros((N, N), device=device).to(th.float32)
for i in range(1, N):
    stend_[i, i] = 1
stend_ = stend_.reshape(-1).repeat(1, num_env).reshape(num_env, N, N).to(device)
reward = th.zeros(num_env, num_samples)
permute_record = th.zeros(num_env, num_samples, N - 1)
min_best = 4e+31
np.set_printoptions(suppress=True)
for k in range(num_env):
    best_reward = 4e+31
    for permute_i in range(num_samples):
        permute = th.randperm(N - 1)
        rtp = 0
        state = deepcopy(a[k])
        stend = deepcopy(stend_[k])
        for i in permute:
            r = 1
            if ((th.div(i, 2, rounding_mode='floor') + 1) * 2 < N):
                r = r * state[(th.div(i, 2, rounding_mode='floor') + 1) * 2, th.div(i, 2, rounding_mode='floor') + 1] \
                    * state[(th.div(i, 2, rounding_mode='floor') + 1) * 2 + 1, th.div(i, 2, rounding_mode='floor') + 1] \
                    * state[(th.div(i, 2, rounding_mode='floor') + 1), th.div((th.div(i, 2, rounding_mode='floor') + 1), 2, rounding_mode='floor')]
            if ((i + 2) * 2 < N):
                r = r * state[(i + 2) * 2, (i + 2)] * state[(i + 2) * 2 + 1, (i + 2)]
            if ((i + 2) * 2 < N and stend[th.div(i, 2, rounding_mode='floor') + 1, th.div(i, 2, rounding_mode='floor')] == 1):
                r = r * state[(i + 2) * 2, th.div(i, 2, rounding_mode='floor') + 2] * state[(i + 2) * 2 + 1, th.div(i, 2, rounding_mode='floor') + 1]
            if ((i + 3) * 2 < N and stend[th.div(i, 2, rounding_mode='floor') + 2, th.div(i, 2, rounding_mode='floor')] == 1):
                r = r * state[(i + 3) * 2, th.div(i, 2, rounding_mode='floor') + 2] * state[(i + 3) * 2 + 1, th.div(i, 2, rounding_mode='floor') + 1]
            stend[i + 1, th.div(i, 2, rounding_mode='floor')] = 1
            state[i + 2, th.div(i, 2, rounding_mode='floor') + 1] = 1
            rtp += r
        reward[k, permute_i] = rtp
        permute_record[k, permute_i] = permute
        best_reward = min(best_reward, rtp)
    min_best = min(best_reward, min_best)
    best_reward_str = str(best_reward.numpy())
    min_best_str = str(min_best.numpy())
    print(best_reward_str, min_best_str)
print(reward.min(dim=-1)[0].mean().numpy())
