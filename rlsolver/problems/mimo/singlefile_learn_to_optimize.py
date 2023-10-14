import sys
import pickle as pkl

import torch
import torch as th
import torch.nn as nn
import torch.optim as optim
import numpy as np
from time import time

TEN = th.Tensor

'''train'''


def mmse_beamformers(h, p):
    k, n = h.shape
    eye_mat = th.eye(n, dtype=h.dtype, device=h.device)
    w = th.linalg.solve(eye_mat * k / p + th.conj(th.transpose(h, 0, 1)) @ h, th.conj(th.transpose(h, 0, 1)))
    w = w / th.norm(w, dim=0, keepdim=True) / k ** 0.5
    return w


"""utils"""


def opt_train(obj_func, obj_args, opt_iter, opt_task, num_opt, device, unroll, opt_base):
    opt_iter.train()
    opt_task.zero_grad()

    n_params = 0
    for name, p in opt_task.get_register_params():
        n_params += int(np.prod(p.size()))
    hc_state1 = th.zeros(4, n_params, opt_iter.hid_dim, device=device)

    all_losses_ever = []
    all_losses = None

    torch.set_grad_enabled(True)
    for iteration in range(1, num_opt + 1):
        output = opt_task.get_output()
        loss = obj_func(*obj_args, output)
        loss.backward(retain_graph=True)

        if all_losses is None:
            all_losses = loss
        else:
            all_losses += loss
        all_losses_ever.append(loss.data.cpu().numpy())

        i = 0
        result_params = {}
        hc_state2 = th.zeros(4, n_params, opt_iter.hid_dim, device=device)
        for name, p in opt_task.get_register_params():
            hid_dim = int(np.prod(p.size()))
            gradients = p.grad.view(hid_dim, 1).detach().clone().requires_grad_(True)

            j = i + hid_dim
            hc_part = hc_state1[:, i:j]
            updates, new_hidden, new_cell = opt_iter(gradients, hc_part[0:2], hc_part[2:4])

            hc_state2[0, i:j] = new_hidden[0]
            hc_state2[1, i:j] = new_hidden[1]
            hc_state2[2, i:j] = new_cell[0]
            hc_state2[3, i:j] = new_cell[1]

            result = p + updates.view(*p.size())
            result_params[name] = result / result.norm()
            result_params[name].retain_grad()

            i = j

        if iteration % unroll == 0:
            opt_base.zero_grad()
            all_losses.backward()
            opt_base.step()

            all_losses = None

            opt_task.re_init()
            opt_task.load_state_dict(result_params)
            opt_task.zero_grad()

            hc_state1 = hc_state2.detach().clone().requires_grad_(True)

        else:
            for name, p in opt_task.get_register_params():
                set_attr(opt_task, name, result_params[name])

            hc_state1 = hc_state2
    torch.set_grad_enabled(False)
    return all_losses_ever


def opt_eval(obj_func, obj_args, opt_iter, opt_task, num_opt, device):
    opt_iter.eval()

    n_params = 0
    for name, p in opt_task.get_register_params():
        n_params += int(np.prod(p.size()))
    hc_state1 = th.zeros(4, n_params, opt_iter.hid_dim, device=device)

    best_res = None
    min_loss = torch.inf

    torch.set_grad_enabled(True)
    for _ in range(num_opt):
        output = opt_task.get_output()
        loss = obj_func(*obj_args, output)
        loss.backward(retain_graph=True)

        result_params = {}
        hc_state2 = th.zeros(4, n_params, opt_iter.hid_dim, device=device)

        i = 0
        for name, p in opt_task.get_register_params():
            param_dim = int(np.prod(p.size()))
            gradients = p.grad.view(param_dim, 1).detach().clone().requires_grad_(True)

            j = i + param_dim
            hc_part = hc_state1[:, i:j]
            updates, new_hidden, new_cell = opt_iter(gradients, hc_part[0:2], hc_part[2:4])

            hc_state2[0, i:j] = new_hidden[0]
            hc_state2[1, i:j] = new_hidden[1]
            hc_state2[2, i:j] = new_cell[0]
            hc_state2[3, i:j] = new_cell[1]

            result = p + updates.view(*p.size())
            result_params[name] = result / result.norm()

            i = j

        opt_task.re_init()
        opt_task.load_state_dict(result_params)
        opt_task.zero_grad()

        hc_state1 = hc_state2.detach().clone().requires_grad_(True)

        if loss < min_loss:
            best_res = opt_task.get_output()
            min_loss = loss
    torch.set_grad_enabled(False)
    return best_res, min_loss


def set_attr(obj, attr, val):
    attrs = attr.split('.')
    for attr in attrs[:-1]:
        obj = getattr(obj, attr)
    setattr(obj, attrs[-1], val)


def objective_function(h: TEN, w: TEN, noise: float = 1.) -> TEN:
    hw = h @ w
    abs_hw_squared = th.abs(hw) ** 2
    signal = th.diagonal(abs_hw_squared)
    interference = abs_hw_squared.sum(dim=-1) - signal
    sinr = signal / (interference + noise)
    return -th.log2(1 + sinr).sum()


class OptimizerTask(nn.Module):
    def __init__(self, dim, device):
        super().__init__()
        self.dim = dim
        self.device = device

        self.N, self.K = dim
        self.register_buffer('theta', th.zeros((2, self.N, self.K), requires_grad=True, device=device))

    def re_init(self):
        self.__init__(dim=self.dim, device=self.device)

    def get_register_params(self):
        return [('theta', self.theta)]

    def forward(self, target):
        w = self.theta[0] + 1j * self.theta[1]
        return target.objective_function(w)

    def get_output(self):
        return self.theta[0] + 1j * self.theta[1]


class OptimizerFunc(nn.Module):
    def __init__(self, hid_dim=20):
        super().__init__()
        self.hid_dim = hid_dim
        self.recurs1 = nn.LSTMCell(1, hid_dim)
        self.recurs2 = nn.LSTMCell(hid_dim, hid_dim)
        self.output = nn.Linear(hid_dim, 1)

    def forward(self, inp0, hid0, cell):
        hid1, cell1 = self.recurs1(inp0, (hid0[0], cell[0]))
        hid2, cell2 = self.recurs2(hid1, (hid0[1], cell[1]))
        return self.output(hid2), (hid1, hid2), (cell1, cell2)


"""run"""


def train_optimizer():
    gpu_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0

    '''init'''
    dim = 8, 8

    '''train'''
    train_times = 1000
    lr = 1e-3
    unroll = 16
    num_opt = 64
    hid_dim = 40

    '''eval'''
    eval_gap = 128
    p_eval = [1, 10, 100]

    print('start training')
    device = th.device(f'cuda:{gpu_id}' if th.cuda.is_available() and gpu_id >= 0 else 'cpu')

    with open(f'./K{dim[0]}N{dim[1]}Samples=100.pkl', 'rb') as f:
        h_evals = th.as_tensor(pkl.load(f), dtype=th.cfloat, device=device)

    obj_func = objective_function
    opt_func = OptimizerFunc(hid_dim=hid_dim).to(device)
    opt_task = OptimizerTask(dim=dim, device=device)
    opt_base = optim.Adam(opt_func.parameters(), lr=lr)

    time_start = time()
    print(f"{'':>8} {'SNR=00':>9} {'SNR=10':>9} {'SNR=20':>9}")

    '''get loss_mmse for evaluation'''
    loss_mmse_list = []
    for p in p_eval:
        loss_mmse = []
        for h in h_evals:
            h_scale = h * (p ** 0.5)
            w_mmse = mmse_beamformers(h, p)
            loss_mmse.append(-objective_function(h_scale, w_mmse).item())
        loss_mmse = sum(loss_mmse) / len(loss_mmse)
        loss_mmse_list.append(loss_mmse)
    print(f"{'MMSE':>8} {loss_mmse_list[0]:>9.3f} {loss_mmse_list[1]:>9.3f} {loss_mmse_list[2]:>9.3f}")

    '''loop'''
    for i in range(train_times + 1):
        p = 10 ** (np.random.rand() + 1)
        h_scale = (p ** 0.5) * th.randn(dim, dtype=th.cfloat, device=device)
        opt_train(obj_func=obj_func, obj_args=(h_scale,), opt_task=opt_task, opt_iter=opt_func,
                  num_opt=num_opt, device=device, unroll=unroll, opt_base=opt_base)

        if i % eval_gap == 0:
            loss_opts = []
            for i_p, p in enumerate(p_eval):
                loss_opt = []
                for h in h_evals:
                    opt_func.eval()
                    h_scale = h * (p ** 0.5)
                    best_result, min_loss = opt_eval(
                        obj_func=obj_func, obj_args=(h_scale,), opt_iter=opt_func, opt_task=opt_task,
                        num_opt=num_opt * 2, device=device
                    )

                    loss_opt.append(-min_loss.item())
                loss_opts.append(sum(loss_opt) / len(loss_opt))
            time_used = round((time() - time_start))
            print(f"{'L2O':>8} {loss_opts[0]:>9.3f} {loss_opts[1]:>9.3f} {loss_opts[2]:>9.3f}    "
                  f"TimeUsed {time_used:9}")


if __name__ == '__main__':
    train_optimizer()
