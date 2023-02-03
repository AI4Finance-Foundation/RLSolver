import matplotlib.pyplot as plt
fig, ax = plt.subplots(ncols=2, nrows=2,figsize =(16, 3))
x_list = [_ for _ in range(784)]
plt.tight_layout()
fig.subplots_adjust(hspace=0.2, wspace = 0.08)
import scipy.io


import pickle as pkl
n_grad = 20
with open(f"recove_signal_{n_grad}.pkl", 'rb') as f:
    signal = pkl.load(f)

original_data = signal[:64]
mdic = {"a": original_data.numpy(), "label": "experiment"}
scipy.io.savemat("signal.mat", mdic)
lasso = scipy.io.loadmat("lasso_recovery.mat")
lasso = lasso["signal_recons"]
generated_data_optimized = signal[64:]
for row in range(1):
    ax[row, 0].plot(x_list, (original_data[row]), label='original')
    ax[row, 1].plot(x_list, (generated_data_optimized[row]), label = 'F + grad steps (M=300)')
    ax[row, 0].set_ylim(0,1)
    ax[row, 1].set_ylim(0,1)
    
    ax[row,0].legend(loc="upper left")
    ax[row,1].legend(loc="upper left")
    
    
for row in range(1,2):
    ax[row, 0].plot(x_list, (original_data[row-1]), label='original')
    ax[row, 1].plot(x_list, (lasso[row-1]), label = 'Lasso (M=300)')
    ax[row, 0].set_ylim(0,1)
    ax[row, 1].set_ylim(0,1)
    ax[row, 0].legend(loc="upper left")
    ax[row, 1].legend(loc="upper left")

fig.savefig(f"recovery_signal_lasso_vs_dcs.png", bbox_inches='tight')

