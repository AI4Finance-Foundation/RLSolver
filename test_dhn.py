from  maxcut import DHN
import sys
s = int(sys.argv[1]) if len(sys.argv) > 1 else 100
agent = DHN(state_dim=s, action_dim=s)
agent.load("./policy_{}.pth".format(s), "./adjacency_{}.npy".format(s)) 
p = agent.test()
print(p)
import numpy as np
p_around = np.round(p)
p_around.shape
p_around = p_around.astype(int)
print(p_around)
cnt = 0
cnt_2 = 0
A  =  np.load("adjacency_{}.npy".format(s))
B = A
for i in range(s):
    for j in range(s):
       if B[i][j] ==1:
           cnt_2 += 1
       if B[i][j] == 1 and p_around[0,i]^p_around[0,j]:
           cnt += 1
           
print(cnt / 2)

print(cnt_2 / 2)
