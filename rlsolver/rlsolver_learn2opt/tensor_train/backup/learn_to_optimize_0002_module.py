import pickle as pkl

import torch as th
import torch.nn as nn
import torch.optim as optim
import numpy as np
import numpy.linalg as la
import functools

'''module'''


class MetaModule(nn.Module):
    # adopted from: Adrien Ecoffet https://github.com/AdrienLE
    def parameters(self, **kwargs):
        for name, param in self.named_params(self):
            yield param

    def named_parameters(self, **kwargs):
        for name, param in self.named_params(self):
            yield name, param

    @staticmethod
    def named_leaves():
        return []

    @staticmethod
    def named_submodules():
        return []

    def named_params(self, curr_module=None, memo=None, prefix=''):
        if memo is None:
            memo = set()

        if hasattr(curr_module, 'named_leaves'):
            for name, p in curr_module.named_leaves():
                if p is not None and p not in memo:
                    memo.add(p)
                    yield prefix + ('.' if prefix else '') + name, p
        else:
            # for name, p in curr_module._parameters.items():
            for name, p in getattr(curr_module, '_parameters').items():
                if p is not None and p not in memo:
                    memo.add(p)
                    yield prefix + ('.' if prefix else '') + name, p

        for mname, module in curr_module.named_children():
            submodule_prefix = prefix + ('.' if prefix else '') + mname
            for name, p in self.named_params(module, memo, submodule_prefix):
                yield name, p

    def update_params(self, lr_inner, first_order=False, source_params=None, detach=False):
        if source_params is not None:
            for tgt, src in zip(self.named_params(self), source_params):
                name_t, param_t = tgt
                # name_s, param_s = src
                # grad = param_s.grad
                # name_s, param_s = src
                grad = src
                if first_order:
                    grad = grad.detach().data.requires_grad_(True)
                tmp = param_t - lr_inner * grad
                self.set_param(self, name_t, tmp)
        else:
            for name, param in self.named_params(self):
                if not detach:
                    grad = param.grad
                    if first_order:
                        grad.data.requires_grad_(True)
                    tmp = param - lr_inner * grad
                    self.set_param(self, name, tmp)
                else:
                    param = param.detach_()
                    self.set_param(self, name, param)

    def set_param(self, curr_mod, name, param):
        if '.' in name:
            n = name.split('.')
            module_name = n[0]
            rest = '.'.join(n[1:])
            for name, mod in curr_mod.named_children():
                if module_name == name:
                    self.set_param(mod, rest, param)
                    break
        else:
            setattr(curr_mod, name, param)

    def detach_params(self):
        for name, param in self.named_params(self):
            self.set_param(self, name, param.detach())

    def copy(self, other, same_var=False):
        for name, param in other.named_params():
            if not same_var:
                param = param.data.clone().requires_grad_(True)
            self.set_param(self, name, param)


'''train'''


def mmse_beamformers(h, p):
    k, n = h.shape
    w = la.solve(np.eye(n) * k / p + h.T.conj() @ h, h.T.conj())
    w = w / la.norm(w, axis=0, keepdims=True) / np.sqrt(k)

    return w


"""utils"""


def run_opt(dim, opt_net, objective, meta_optimizer_class,
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
    for name, p in meta_optimizer.parameters():
        n_params += int(np.prod(p.size()))
    hidden_states = [th.zeros(n_params, opt_net.hid_dim, requires_grad=True, device=device) for _ in range(2)]
    cell_states = [th.zeros(n_params, opt_net.hid_dim, requires_grad=True, device=device) for _ in range(2)]
    all_losses_ever = []

    all_losses = None
    for iteration in range(1, optim_it + 1):
        loss = meta_optimizer(objective)

        if all_losses is None:
            all_losses = loss
        else:
            all_losses += loss

        all_losses_ever.append(loss.data.cpu().numpy().copy())
        loss.backward(retain_graph=should_train)

        offset = 0
        result_params = {}
        hidden_states2 = [th.zeros(n_params, opt_net.hid_dim, requires_grad=True, device=device) for _ in range(2)]
        cell_states2 = [th.zeros(n_params, opt_net.hid_dim, requires_grad=True, device=device) for _ in range(2)]

        print(';;1', dict(meta_optimizer.parameters()))
        for name, p in meta_optimizer.parameters():
            cur_sz = int(np.prod(p.size()))
            # We do this so the gradients are disconnected from the graph but we still get
            # gradients from the rest
            gradients = p.grad.view(cur_sz, 1).requires_grad_(True)
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
            print(';;2', dict(meta_optimizer.parameters()))
            print(';;3', result_params)
            meta_optimizer.load_state_dict(result_params)
            meta_optimizer.zero_grad()
            hidden_states = [v.requires_grad_(True) for v in hidden_states2]
            cell_states = [v.requires_grad_(True) for v in cell_states2]

        else:
            for name, p in meta_optimizer.named_parameters():
                setattr(meta_optimizer, name, result_params[name])
            hidden_states = hidden_states2
            cell_states = cell_states2

    return all_losses_ever


def run():
    dim = 8, 8
    n, k = dim
    gpu_id = 2
    lr = 1e-3
    n_train_epochs = 1000
    unroll = 20
    test_every = 20
    p_test = [1, 10, 100]

    device = th.device(f'cuda:{gpu_id}' if th.cuda.is_available() and gpu_id >= 0 else 'cpu')

    opt_net = Optimizer(hidden_sz=40).to(device)
    meta_optimizer = OptimizerForMISO(dim).to(device)
    optim_it = 100

    optimizer = optim.Adam(opt_net.parameters(), lr=lr)
    n_params = 0
    for name, p in meta_optimizer.parameters():
        n_params += int(np.prod(p.size()))

    '''eval'''
    with open(f'./K{k}N{n}Samples=100.pkl', 'rb') as f:
        test_data = th.as_tensor(pkl.load(f), dtype=th.cfloat)

    for epoch in range(1, n_train_epochs + 1):
        p = 10 ** (np.random.rand() + 1)  # MISO
        h_scaled = (p ** 0.5) * th.randn(n, k, dtype=th.cfloat, device=device)
        objective = ObjectiveSumRateForMISO(h_scaled)

        run_opt(dim, opt_net, objective, OptimizerForMISO,
                optim_it, device, unroll, optimizer)

        '''test'''
        if epoch % test_every:
            loss = []
            loss_mmse = []

            n_test = test_data
            for p in p_test:
                for h in test_data:
                    h_scaled = h * (p ** 0.5)
                    objective = ObjectiveSumRateForMISO(h_scaled)
                    rnn = run_opt(dim, opt_net, objective, OptimizerForMISO,
                                  optim_it, device, unroll=1, meta_opt=None)
                    loss.append(rnn)

                    wmmse = mmse_beamformers(h.cpu().numpy(), p)
                    loss_mmse.append(objective.get_loss(th.tensor(wmmse, dtype=th.cfloat, device=device)).cpu().numpy())

            loss = -np.array(loss).reshape((len(p_test), n_test, optim_it))
            loss_avg_final = np.mean(loss[:, :, -1], axis=-1)
            loss_mmse = -np.array(loss_mmse).reshape((len(p_test), n_test))
            loss_mmse = np.mean(loss_mmse, axis=-1)
            print(f"{'Test SNR (dB)':<16}{'LSTM':9}{'MMSE':9}")

            for isnr, snr in enumerate((10 * np.log10(p_test)).astype(int)):
                print(f"{snr:<16}{loss_avg_final[isnr]:9}{loss_mmse[isnr]:9}")


class ObjectiveSumRateForMISO:
    def __init__(self, h, **_kwargs):
        self.h = h  # hall

    def get_loss(self, w):
        hw = self.h @ w
        abs_hw_squared = th.abs(hw) ** 2
        signal = th.diagonal(abs_hw_squared)
        interference = th.sum(abs_hw_squared, dim=-1) - signal
        noise = 1
        sinr = signal / (interference + noise)
        return -th.log2(1 + sinr).sum()


class OptimizerForMISO(MetaModule):
    def __init__(self, dim):
        super().__init__()
        self.N, self.K = dim
        self.theta = nn.Parameter(th.zeros(2 * self.N * self.K))

    def forward(self, target):
        w = self.theta.reshape((2, self.N, self.K))
        w = w[0] + 1j * w[1]
        return target.objective_function(w)


class Optimizer(nn.Module):
    def __init__(self, preproc=False, hidden_sz=20, preproc_factor=10.0):
        super().__init__()
        self.hidden_sz = hidden_sz
        if preproc:
            self.recurs = nn.LSTMCell(2, hidden_sz)
        else:
            self.recurs = nn.LSTMCell(1, hidden_sz)
        self.recurs2 = nn.LSTMCell(hidden_sz, hidden_sz)
        self.output = nn.Linear(hidden_sz, 1)
        self.preproc = preproc
        self.preproc_factor = preproc_factor
        self.preproc_threshold = np.exp(-preproc_factor)

    def forward(self, inp, hidden, cell):
        device = inp.device
        if self.preproc:
            # Implement preproc described in Appendix A

            # Note: we do all this work on tensors, which means
            # the gradients won't propagate through inp. This
            # should be ok because the algorithm involves
            # making sure that inp is already detached.
            inp = inp.data
            inp2 = th.zeros(inp.size()[0], 2).to(device)
            keep_grads = (th.abs(inp) >= self.preproc_threshold).squeeze()
            inp2[:, 0][keep_grads] = (th.log(th.abs(inp[keep_grads]) + 1e-8) / self.preproc_factor).squeeze()
            inp2[:, 1][keep_grads] = th.sign(inp[keep_grads]).squeeze()

            inp2[:, 0][~keep_grads] = -1
            inp2[:, 1][~keep_grads] = (float(np.exp(self.preproc_factor)) * inp[~keep_grads]).squeeze()
            inp = th.tensor(inp2, device=device)
        hidden0, cell0 = self.recurs(inp, (hidden[0], cell[0]))
        hidden1, cell1 = self.recurs2(hidden0, (hidden[1], cell[1]))
        return self.output(hidden1), (hidden0, hidden1), (cell0, cell1)


if __name__ == '__main__':
    run()
