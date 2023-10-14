import matplotlib.pyplot as plt
import scipy.io

import torch as th
import pickle as pkl
n_grad = 20
N = 100
fig, ax = plt.subplots(ncols=1, nrows=4,figsize =(8, 6))
x_list = [_ for _ in range(N)]
plt.tight_layout()
fig.subplots_adjust(hspace=0.2, wspace = 0.08)
# with open(f"recove_signal_{n_grad}.pkl", 'rb') as f:
with open(f"933_{n_grad}.pkl",'rb') as f:
    signal = pkl.load(f)
with open(f"metric.pkl", 'rb') as f:
    metric = pkl.load(f)
original_data = signal[:64]
mdic = {"a": original_data.numpy(), "label": "experiment"}
scipy.io.savemat("signal_identity.mat", mdic)

original_data = metric[:64]
mdic = {"a": original_data.numpy(), "label": "experiment"}
scipy.io.savemat("metric.mat", mdic)
assert 0

lasso = scipy.io.loadmat("lasso_recovery.mat")
lasso = lasso["signal_recons"]
generated_data_optimized = signal[64:]
for row in range(4):
    ax[row].plot(x_list, (original_data[row]), label='original')
    ax[row].set_ylim(-1,1)
    ax[row].legend(loc="upper left")

fig.savefig(f"origin_signal_{n_grad}_dct.png", bbox_inches='tight')

fig, ax = plt.subplots(ncols=1, nrows=4,figsize =(8, 6))
x_list = [_ for _ in range(N)]
plt.tight_layout()
fig.subplots_adjust(hspace=0.2, wspace = 0.08)
for row in range(4):
    ax[row].plot(x_list, (generated_data_optimized[row]), label = 'F + grad steps (M=50)')    
    ax[row].set_ylim(-1,1)
    ax[row].legend(loc="upper left")
fig.savefig(f"recovery_signal_{n_grad}_dct.png", bbox_inches='tight')

# fig, ax = plt.subplots(ncols=1, nrows=4,figsize =(8, 6))
# x_list = [_ for _ in range(784)]
# plt.tight_layout()
# fig.subplots_adjust(hspace=0.2, wspace = 0.08)
# for row in range(4):
#     ax[row].plot(x_list, (lasso[row]), label = 'LASSO')    
#     ax[row].set_ylim(-1,1)
#     ax[row].legend(loc="upper left")


# fig.savefig(f"recovery_signal_lasso.png", bbox_inches='tight')


