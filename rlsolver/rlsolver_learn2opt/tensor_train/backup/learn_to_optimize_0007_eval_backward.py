import sys
import pickle as pkl
import torch as th
import torch.nn as nn
import torch.optim as optim
import numpy as np
from time import time

TEN = th.Tensor

'''train'''


def var_detach_retain_grad(v):
    var = v.clone().detach()
    var.requires_grad_(True).retain_grad()
    return var


def mmse_beamformers(h, p):
    k, n = h.shape
    eye_mat = th.eye(n, dtype=h.dtype, device=h.device)
    w = th.linalg.solve(eye_mat * k / p + th.conj(th.transpose(h, 0, 1)) @ h, th.conj(th.transpose(h, 0, 1)))
    w = w / th.norm(w, dim=0, keepdim=True) / k ** 0.5
    return w


"""utils"""


def opt_train(dim, opt_net, objective, meta_optimizer_class,
              optim_it, device, unroll=1, meta_opt=None):
    should_train = meta_opt is not None
    if should_train:
        opt_net.train()
        meta_opt.zero_grad()
    else:
        opt_net.eval()
        unroll = 1

    meta_optimizer = meta_optimizer_class(dim).to(device)

    n_params = 0
    for name, p in meta_optimizer.get_register_params():
        n_params += int(np.prod(p.size()))
    hidden_states = [th.zeros(n_params, opt_net.hid_dim, device=device) for _ in range(2)]
    cell_states = [th.zeros(n_params, opt_net.hid_dim, device=device) for _ in range(2)]
    all_losses_ever = []
    if should_train:
        meta_opt.zero_grad()
    all_losses = None
    for iteration in range(1, optim_it + 1):
        loss = meta_optimizer(objective)

        if all_losses is None:
            all_losses = loss
        else:
            all_losses += loss

        all_losses_ever.append(loss.data.cpu().numpy())
        loss.backward(retain_graph=should_train)

        offset = 0
        result_params = {}
        hidden_states2 = [th.zeros(n_params, opt_net.hid_dim, device=device) for _ in range(2)]
        cell_states2 = [th.zeros(n_params, opt_net.hid_dim, device=device) for _ in range(2)]
        for name, p in meta_optimizer.get_register_params():
            cur_sz = int(np.prod(p.size()))
            gradients = var_detach_retain_grad(p.grad.view(cur_sz, 1))
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
            if should_train:
                meta_opt.zero_grad()
                all_losses.backward()
                meta_opt.step()

            all_losses = None

            meta_optimizer = meta_optimizer_class(dim).to(device)
            meta_optimizer.load_state_dict(result_params)
            meta_optimizer.zero_grad()
            hidden_states = [var_detach_retain_grad(v) for v in hidden_states2]
            cell_states = [var_detach_retain_grad(v) for v in cell_states2]

        else:
            for name, p in meta_optimizer.get_register_params():
                set_attr(meta_optimizer, name, result_params[name])
            hidden_states = hidden_states2
            cell_states = cell_states2

    return all_losses_ever


def opt_eval(dim, opt_net, objective, meta_optimizer_class,
             optim_it, device):
    opt_net.eval()

    meta_optimizer = meta_optimizer_class(dim).to(device)

    n_params = 0
    for name, p in meta_optimizer.get_register_params():
        n_params += int(np.prod(p.size()))
    hidden_states = [th.zeros(n_params, opt_net.hid_dim, device=device) for _ in range(2)]
    cell_states = [th.zeros(n_params, opt_net.hid_dim, device=device) for _ in range(2)]
    all_losses_ever = []
    for iteration in range(1, optim_it + 1):
        loss = meta_optimizer(objective)

        all_losses_ever.append(loss.data.cpu().numpy())
        loss.backward(retain_graph=False)  # todo ?????????????

        offset = 0
        result_params = {}
        hidden_states2 = [th.zeros(n_params, opt_net.hid_dim, device=device) for _ in range(2)]
        cell_states2 = [th.zeros(n_params, opt_net.hid_dim, device=device) for _ in range(2)]
        for name, p in meta_optimizer.get_register_params():
            cur_sz = int(np.prod(p.size()))
            gradients = var_detach_retain_grad(p.grad.view(cur_sz, 1)) # todo
            # gradients = p.grad.view(cur_sz, 1).detach()

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

        meta_optimizer = meta_optimizer_class(dim).to(device)
        meta_optimizer.load_state_dict(result_params)
        meta_optimizer.zero_grad()
        hidden_states = [var_detach_retain_grad(v) for v in hidden_states2]
        cell_states = [var_detach_retain_grad(v) for v in cell_states2] # todo
        # hidden_states = [v.detach() for v in hidden_states2]
        # cell_states = [v.detach() for v in cell_states2]

    return all_losses_ever


def do_test(dim, opt_net, objective, meta_optimizer_class, optim_it, p_test, test_data, device):
    loss_list = []
    loss_mmse = []
    n_test = test_data.shape[0]
    for P in p_test:
        for H in test_data:
            h_scaled = H * (P ** 0.5)
            target = objective(h_scaled)

            opt_net.eval()
            rnn = opt_eval(dim, opt_net, target, meta_optimizer_class,
                           optim_it, device)  # todo
            loss_list.append(rnn)

            w_mmse = mmse_beamformers(H, P)
            loss_mmse.append(target.objective_function(w_mmse).cpu().numpy())

    loss_list = -np.array(loss_list).reshape((len(p_test), n_test, optim_it))
    losses = np.mean(loss_list[:, :, -1], axis=-1)
    loss_mmse = -np.array(loss_mmse).reshape((len(p_test), n_test))
    loss_mmse = np.mean(loss_mmse, axis=-1)
    print(f"{'Test SNR (dB)':<16} {'LSTM':>9} {'MMSE':>9}")

    for isnr, snr in enumerate((10 * np.log10(p_test)).astype(int)):
        print(f"{snr:<16} {losses[isnr]:9.3f} {loss_mmse[isnr]:9.3f}")
    return losses


def train_optimizer():
    dim = 8, 8
    p_test = [1, 10, 100]
    n_train_epochs = 1000
    test_every = 100
    objective = ObjectiveSunRate
    meta_optimizer_class = MetaOptimizerMISO

    lr = 1e-3
    unroll = 20
    optim_it = 100
    n_test = 10

    hidden_sz = 40
    device = DEVICE

    n, k = dim

    with open(f'./K{k}N{n}Samples=100.pkl', 'rb') as f:
        test_data = th.as_tensor(pkl.load(f), dtype=th.cfloat, device=device)

    opt_net = Optimizer(hidden_sz=hidden_sz).to(device)
    meta_opt = optim.Adam(opt_net.parameters(), lr=lr)

    time_start = time()
    print('start training loop')
    for epoch in range(n_train_epochs + 1):

        # generate sample
        p = 10 ** (np.random.rand() + 1)
        h_scaled = (p ** 0.5) * th.randn(n, k, dtype=th.cfloat, device=DEVICE)

        opt_train(dim, opt_net, objective(h_scaled), meta_optimizer_class,
                  optim_it, device, unroll, meta_opt)

        if epoch % test_every == 0:
            do_test(dim, opt_net, objective, meta_optimizer_class=meta_optimizer_class,
                    optim_it=200, test_data=test_data[:n_test], p_test=p_test, device=device)

            print('=' * 60)
            print(f'epoch {epoch}    seconds per epoch {round((time() - time_start) / test_every, 2),}')
            time_start = time()

    do_test(dim, opt_net, objective, meta_optimizer_class=meta_optimizer_class,
            optim_it=200, test_data=test_data[:n_test], p_test=p_test, device=device)


class ObjectiveSunRate:  # 必须把 objective 改成一个 class，因为要传入 run_opt
    def __init__(self, h: TEN):
        self.h = h  # hall

    def get_objective(self, w: TEN, noise: float = 1.) -> TEN:
        hw = self.h @ w
        abs_hw_squared = th.abs(hw) ** 2
        signal = th.diagonal(abs_hw_squared)
        interference = abs_hw_squared.sum(dim=-1) - signal
        sinr = signal / (interference + noise)
        return -th.log2(1 + sinr).sum()


def set_attr(obj, attr, val):
    attrs = attr.split('.')
    for attr in attrs[:-1]:
        obj = getattr(obj, attr)
    setattr(obj, attrs[-1], val)


class MetaOptimizerMISO(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.N, self.K = dim
        self.register_buffer('theta', th.zeros((2, self.N, self.K), device=DEVICE, requires_grad=True))

    def get_register_params(self):
        return [('theta', self.theta)]

    def forward(self, target):
        w = self.theta[0] + 1j * self.theta[1]
        return target.objective_function(w)

    def output(self):
        return self.theta[0] + 1j * self.theta[1]


class Optimizer(nn.Module):
    def __init__(self, hidden_sz=20):
        super().__init__()
        self.hidden_sz = hidden_sz
        self.recurs = nn.LSTMCell(1, hidden_sz)
        self.recurs2 = nn.LSTMCell(hidden_sz, hidden_sz)
        self.output = nn.Linear(hidden_sz, 1)

    def forward(self, inp, hidden, cell):
        hidden0, cell0 = self.recurs(inp, (hidden[0], cell[0]))
        hidden1, cell1 = self.recurs2(hidden0, (hidden[1], cell[1]))
        return self.output(hidden1), (hidden0, hidden1), (cell0, cell1)


if __name__ == '__main__':
    gpu_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    DEVICE = th.device(f'cuda:{gpu_id}' if th.cuda.is_available() and gpu_id >= 0 else 'cpu')

    train_optimizer()
