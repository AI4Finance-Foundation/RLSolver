import pickle as pkl
import torch as th
from copy import deepcopy
import numpy as np

device = th.device("cuda:0")
reward = 0
# 根据张量网络中张量数量设定
N = 4
with open(f"test_data_tensor_ring_N={N}.pkl", 'rb') as f:
    a = pkl.load(f).detach().cpu()
num_env = a.shape[0]
# 每个env中包含的sample数量
num_samples = 500
start_ = th.as_tensor([i for i in range(N)]).repeat(1, num_env).reshape(num_env, -1).to(device)
end_ = th.as_tensor([i for i in range(N)]).repeat(1, num_env).reshape(num_env, -1).to(device)
reward = th.zeros(num_env, num_samples)
permute_record = th.zeros(num_env, num_samples, N)
min_best = 4e+31
np.set_printoptions(suppress=True)
for k in range(num_env):
    best_reward = 4e+31
    for permute_i in range(num_samples):
        permute = th.randperm(N)
        rtp = 0
        state = deepcopy(a[k])
        start = deepcopy(start_[k]) + 1
        end = deepcopy(end_[k]) + 1
        for i in (permute[:-2]):
            r = 1
            print(state)
            if (i == N-1):
                first_node = (i + 1) % N
                second_node = (i + 2) % N
            else:
                first_node = (i + 2) % N
                second_node = (i + 1) % N
            if (first_node == 0):    first_node = N
            if (second_node == 0):    second_node = N

            if start[i] == 1 or end[i] == N:
                r = r * state[N, 1]
            if start[(i + 1) % N] == 1 or end[(i + 1) % N] == N:
                r = r * state[N, 1]
            tmp1 = start[i]
            tmp2 = start[(i + 1) % N]
            for i in range(N):
                if ((start[i] == tmp1) or (start[i] == tmp2)):
                    r = r * (state[i + 1, i + 1] * state[i + 1, i] * state[i + 2, i + 1])
            r /= 2
            state[first_node, second_node] = 1
            s1 = 0 + start[i]
            s2 = 0 + start[(i + 1) % N]
            start_new = min(start[i], start[(i + 1) % N])
            end_new = max(end[i], end[(i + 1) % N])
            for i in range(N):
                if ((start[i] == s1) or (start[i] == s2)):
                    start[i] = start_new
                    end[i] = end_new

            rtp += r
        reward[k, permute_i] = rtp
        # print(permute, permute_i)
        permute_record[k, permute_i] = permute
        best_reward = min(best_reward, rtp - 2 ** (N + 1))
    min_best = min(best_reward, min_best)

    # print(reward[k], permute_record[k, reward[k].min(dim=-1)[1]], best_reward.numpy(), min_best.numpy())
    best_reward_str = str(best_reward.numpy())
    min_best_str = str(min_best.numpy())
    # 分别输出  当前env中最优的reward、所有env中最优的reward
    print(best_reward_str, min_best_str)
# 所有env中最优reward的平均值
print(reward.min(dim=-1)[0].mean().numpy())
# with open("record_r_baseline_random.pkl", "wb") as f:
#     import pickle as pkl
#
#     pkl.dump(reward, f)
# with open("record_permute_baseline_random.pkl", "wb") as f:
#     import pickle as pkl
#
#     pkl.dump(permute_record, f)
