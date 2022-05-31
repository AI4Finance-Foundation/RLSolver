from  maxcut import DHN
import sys
import numpy as np
s = int(sys.argv[1]) if len(sys.argv) > 1 else 100
exp_id = int(sys.argv[2]) if len(sys.argv) > 2 else 0
A  =  np.load("./{}/{}/adjacency.npy".format(s, exp_id))
print("Adjacency matrix: \n", A)
agent = DHN(state_dim=s, action_dim=s)
agent.load(s, exp_id)
#agent.load("./{}/{}/policy.pth".format(s,exp_id), "./{}/{}/adjacency.npy".format(s, exp_id))
p = agent.test()
print("Configuration output: ", p)
p_around = np.round(p)
p_around.shape
p_around = p_around.astype(int)
print("Rounded configuration: ", p_around)
#np.save("./617.npy", p_around)
cnt = 0
cnt_2 = 0
A  =  np.load("./{}/{}/adjacency.npy".format(s, exp_id))
B = A
for i in range(s):
    for j in range(s):
       if B[i][j] ==1:
           cnt_2 += 1
       if B[i][j] == 1 and p_around[0,i]^p_around[0,j]:
           cnt += 1
           
print("Maxcut Solution: ", cnt / 2)

print("Toal edges in the graph: ", cnt_2 / 2)
