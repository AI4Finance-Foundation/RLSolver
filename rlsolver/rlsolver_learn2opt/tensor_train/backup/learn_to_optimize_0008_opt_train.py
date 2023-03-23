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


def opt_train(dim, opt_net, obj_task, opt_task,
              num_opt, device, unroll, opt_base=None):
    opt_net.train()

    target = opt_task(dim, device)
    target.zero_grad()

    n_params = 0
    for name, p in target.get_register_params():
        n_params += int(np.prod(p.size()))
    hidden_states = [th.zeros(n_params, opt_net.hid_dim, device=device) for _ in range(2)]
    cell_states = [th.zeros(n_params, opt_net.hid_dim, device=device) for _ in range(2)]

    all_losses_ever = []
    all_losses = None

    torch.set_grad_enabled(True)
    for iteration in range(1, num_opt + 1):
        loss = target(obj_task)
        loss.backward(retain_graph=True)

        if all_losses is None:
            all_losses = loss
        else:
            all_losses += loss
        all_losses_ever.append(loss.data.cpu().numpy())

        offset = 0
        result_params = {}
        hidden_states2 = [th.zeros(n_params, opt_net.hid_dim, device=device) for _ in range(2)]
        cell_states2 = [th.zeros(n_params, opt_net.hid_dim, device=device) for _ in range(2)]
        for name, p in target.get_register_params():
            cur_sz = int(np.prod(p.size()))
            gradients = p.grad.view(cur_sz, 1).detach().clone().requires_grad_(True)

            updates, new_hidden, new_cell = opt_net(
                gradients,
                [h[offset:offset + cur_sz] for h in hidden_states],
                [c[offset:offset + cur_sz] for c in cell_states]
            )
            for i in range(len(new_hidden)):
                hidden_states2[i][offset:offset + cur_sz] = new_hidden[i]
                cell_states2[i][offset:offset + cur_sz] = new_cell[i]

            temp = p + updates.view(*p.size())
            result_params[name] = temp / th.norm(temp)

            result_params[name].retain_grad()

            offset += cur_sz

        if iteration % unroll == 0:
            opt_base.zero_grad()
            all_losses.backward()
            opt_base.step()

            all_losses = None

            target = opt_task(dim, device)
            target.load_state_dict(result_params)
            target.zero_grad()

            hidden_states = [v.detach().clone().requires_grad_(True) for v in hidden_states2]
            cell_states = [v.detach().clone().requires_grad_(True) for v in cell_states2]

        else:
            for name, p in target.get_register_params():
                set_attr(target, name, result_params[name])

            hidden_states = hidden_states2
            cell_states = cell_states2
    torch.set_grad_enabled(False)
    return all_losses_ever


def opt_eval(dim, opt_net, target, opt_task, num_opt, device):
    opt_net.eval()

    meta_optimizer = opt_task(dim, device)

    n_params = 0
    for name, p in meta_optimizer.get_register_params():
        n_params += int(np.prod(p.size()))
    hc_state1 = th.zeros(4, n_params, opt_net.hid_dim, device=device)

    best_res = None
    min_loss = torch.inf

    torch.set_grad_enabled(True)
    for _ in range(num_opt):
        loss = meta_optimizer(target)
        loss.backward(retain_graph=True)

        result_params = {}
        hc_state2 = th.zeros(4, n_params, opt_net.hid_dim, device=device)

        i = 0
        for name, p in meta_optimizer.get_register_params():
            param_dim = int(np.prod(p.size()))
            gradients = p.grad.view(param_dim, 1).detach().clone().requires_grad_(True)

            j = i + param_dim
            hc_part = hc_state1[:, i:j]
            updates, new_hidden, new_cell = opt_net(gradients, hc_part[0:2], hc_part[2:4])

            hc_state2[0, i:j] = new_hidden[0]
            hc_state2[1, i:j] = new_hidden[1]
            hc_state2[2, i:j] = new_cell[0]
            hc_state2[3, i:j] = new_cell[1]

            result = p + updates.view(*p.size())
            result_params[name] = result / result.norm()

            i = j

        meta_optimizer = opt_task(dim, device)
        meta_optimizer.load_state_dict(result_params)
        meta_optimizer.zero_grad()

        hc_state1 = hc_state2.detach().clone().requires_grad_(True)

        if loss < min_loss:
            best_res = meta_optimizer.get_output()
            min_loss = loss
    torch.set_grad_enabled(False)
    return best_res, min_loss


def do_test(dim, opt_net, obj_task, opt_task, num_opt, p_eval, test_data, device):
    loss_opt = []
    loss_mmse = []

    n_test = test_data.shape[0]
    for p in p_eval:
        for h in test_data:
            h_scaled = h * (p ** 0.5)
            target = obj_task(h_scaled)

            opt_net.eval()
            best_result, min_loss = opt_eval(dim=dim, opt_net=opt_net, target=target, opt_task=opt_task,
                                             num_opt=num_opt, device=device)
            loss_opt.append(min_loss.cpu().numpy())

            w_mmse = mmse_beamformers(h, p)
            loss_mmse.append(target.objective_function(w_mmse).cpu().numpy())

    loss_opt = -np.array(loss_opt).reshape((len(p_eval), n_test)).mean(axis=1)
    loss_mmse = -np.array(loss_mmse).reshape((len(p_eval), n_test)).mean(axis=1)
    return loss_opt, loss_mmse


def set_attr(obj, attr, val):
    attrs = attr.split('.')
    for attr in attrs[:-1]:
        obj = getattr(obj, attr)
    setattr(obj, attrs[-1], val)


class ObjectiveMISO:  # 必须把 objective 改成一个 class，因为要传入 run_opt
    def __init__(self, h: TEN):
        self.h = h  # hall

    def get_objective(self, w: TEN, noise: float = 1.) -> TEN:
        hw = self.h @ w
        abs_hw_squared = th.abs(hw) ** 2
        signal = th.diagonal(abs_hw_squared)
        interference = abs_hw_squared.sum(dim=-1) - signal
        sinr = signal / (interference + noise)
        return -th.log2(1 + sinr).sum()


class OptimizerMISO(nn.Module):
    def __init__(self, dim, device):
        super().__init__()
        self.N, self.K = dim
        self.register_buffer('theta', th.zeros((2, self.N, self.K), requires_grad=True, device=device))

    def get_register_params(self):
        return [('theta', self.theta)]

    def forward(self, target):
        w = self.theta[0] + 1j * self.theta[1]
        return target.objective_function(w)

    def get_output(self):
        return self.theta[0] + 1j * self.theta[1]


class OptimizerMeta(nn.Module):
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
    obj_task = ObjectiveMISO
    opt_task = OptimizerMISO

    '''train'''
    train_times = 1000
    lr = 1e-3
    unroll = 16
    num_opt = 64  # important hyperparameter
    hid_dim = 40

    '''eval'''
    eval_gap = 64
    p_eval = [1, 10, 100]
    n_eval = 10

    device = th.device(f'cuda:{gpu_id}' if th.cuda.is_available() and gpu_id >= 0 else 'cpu')

    with open(f'./K{dim[0]}N{dim[1]}Samples=100.pkl', 'rb') as f:
        test_data = th.as_tensor(pkl.load(f), dtype=th.cfloat, device=device)

    opt_net = OptimizerMeta(hid_dim=hid_dim).to(device)
    opt_base = optim.Adam(opt_net.parameters(), lr=lr)

    time_start = time()
    print('start training loop')
    for i in range(train_times + 1):
        p = 10 ** (np.random.rand() + 1)
        h_scaled = (p ** 0.5) * th.randn(dim, dtype=th.cfloat, device=device)

        opt_train(dim=dim, opt_net=opt_net, obj_task=obj_task(h_scaled), opt_task=opt_task,
                  num_opt=num_opt, device=device, unroll=unroll, opt_base=opt_base)

        if i % eval_gap == 0:
            loss_opt, loss_mmse = do_test(dim=dim, opt_net=opt_net, obj_task=obj_task, opt_task=opt_task,
                                          num_opt=num_opt, test_data=test_data[:n_eval], p_eval=p_eval, device=device)
            time_used = round((time() - time_start))

            print(f"\n{'Test SNR (dB)':>16} {'LSTM':>9} {'MMSE':>9}   TimeUsed {time_used:>9}")
            for isnr, snr in enumerate((10 * np.log10(p_eval)).astype(int)):
                print(f"{snr:>16} {loss_opt[isnr]:9.3f} {loss_mmse[isnr]:9.3f}")

    do_test(dim=dim, opt_net=opt_net, obj_task=obj_task, opt_task=opt_task,
            num_opt=num_opt, test_data=test_data, p_eval=p_eval, device=device)


if __name__ == '__main__':
    train_optimizer()
