import numpy as np
import torch as th
file1 = open(".data/Gset/G14.txt", 'r')
Lines = file1.readlines()
 
count = 0
for line in Lines:
    count += 1
    s = line.split()
    if count== 1:
        N = int(s[0])
        edge = int(s[1])
        adjacency = th.zeros(N, N)
    else:
        i = int(s[0])
        j = int(s[1])
        w = int(s[2])
        adjacency[i-1, j-1] = w
        adjacency[j-1, i-1] = w
sparsity=edge / (N * N)
np.save(f"./data/N{N}Sparsity{sparsity}.npy", adjacency)
# adjacency = th.as_tensor(np.load("N800Sparsity0.007.npy"))
# print(adjacency.shape, adjacency.sum())