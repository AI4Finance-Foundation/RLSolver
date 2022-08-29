from th3 import *
K = 4
N = 2
total_power = 10
_, t, H, ZF = compute_channel(N, K, total_power, np.array([[1]]))
MMSE(H, 10, 1)
