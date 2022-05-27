M = -1000
ans = []
import networkx as nx 
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import sys

s = int(sys.argv[1]) if len(sys.argv) > 1 else 100
A  =  np.load("adjacency_{}.npy".format(s))
l = A.shape[0]
p = [0 for i in range(l)]
def func(i ):
    global M
    global p
    global ans
    
    if i == l:
        cnt = 0
        for i in range(l):
            for j in range(l):
                if p[i] ^ p[j] and A[i][j] == 1:
                    cnt += 1
        cnt = cnt / 2
        
        if cnt > M:
            print( cnt)
            ans = []
            ans.append(np.copy(p))
            M = cnt
        elif cnt == M:
            ans.append(np.copy(p))
        return
    func(i + 1)
    p[i] = 1
    func(i + 1)
    p[i] = 0
    
if __name__ == '__main__':
    func(0)
    s = 100
    A = np.load("adjacency_{}.npy".format(s))
    p = [0 for i in range(s)]
    print(ans, M)
