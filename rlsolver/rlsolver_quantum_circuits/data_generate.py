import torch as th
import torch
device = th.device("cuda:0")
N = 4
num_env = 100
test_state = torch.randint(0,9, (num_env, N + 2, N + 2), device=device).to(torch.float32)
mask = th.zeros(N + 2, N + 2).to(device)
mask[1, 1] = 1
for i in range(2, N + 1):
    mask[i, i-1] = 1
    mask[i, i] = 1

mask = mask.reshape(-1).repeat(1, num_env).reshape(num_env, N + 2, N + 2).to(device)
test_state = th.mul(test_state, mask)
test_state += th.ones_like(test_state)
test_state[0, 2, 1] = 1
test_state[0, 1, 1] = 1
test_state[0, 2, 2] = 2
test_state[0, 3, 3] = 2
test_state[0, 4, 4] = 2
test_state[0, 3, 2] = 1
test_state[0, 4, 3] = 1
print(test_state[0])
with open(f"test_{N}.pkl", 'wb') as f:
    import pickle as pkl
    pkl.dump(test_state, f)
