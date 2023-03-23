import sys
import pickle as pkl
import torch as th
import torch.nn as nn
import torch.optim as optim
import numpy as np
from time import time
import numpy.linalg as la
import functools

'''module'''


def to_var(x, requires_grad=True):
    # if torch.cuda.is_available():
    #     x = x.cuda()
    # return th.tensor(x, requires_grad=requires_grad).to(DEVICE) # todo
    return x.clone().requires_grad_(requires_grad)


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
                    grad = to_var(grad.detach().data)
                tmp = param_t - lr_inner * grad
                self.set_param(self, name_t, tmp)
        else:
            for name, param in self.named_params(self):
                if not detach:
                    grad = param.grad
                    if first_order:
                        grad = to_var(grad.detach().data)
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
                param = to_var(param.data.clone(), requires_grad=True)
            self.set_param(self, name, param)


'''train'''


def to_cuda(v):
    return v.to(DEVICE)


def detach_var(v):
    # var = th.tensor(v.data, requires_grad=True, device=DEVICE)  # todo
    var = v.clone().detach().requires_grad_(True)
    var.retain_grad()
    return var


def mmse_beamformers(h, p):
    k, n = h.shape
    w = la.solve(np.eye(n) * k / p + h.T.conj() @ h, h.T.conj())
    w = w / la.norm(w, axis=0, keepdims=True) / np.sqrt(k)

    return w


"""utils"""


def re_setattr(obj, attr, val):
    pre, _, post = attr.rpartition('.')
    return setattr(re_getattr(obj, pre) if pre else obj, post, val)


def re_getattr(obj, attr, *args):
    return functools.reduce(lambda _obj, _attr: getattr(_obj, _attr, *args),
                            [obj] + attr.split('.'))


def run_opt(dim, opt_net, objective, meta_optimizer_class,
            optim_it, device, unroll=1, meta_opt=None):
    # def do_fit(dim, opt_net, meta_opt, objective, meta_optimizer_class, unroll, optim_it, should_train=True):
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

        all_losses_ever.append(loss.data.cpu().numpy().copy())
        loss.backward(retain_graph=should_train)

        offset = 0
        result_params = {}
        hidden_states2 = [th.zeros(n_params, opt_net.hid_dim, device=device) for _ in range(2)]
        cell_states2 = [th.zeros(n_params, opt_net.hid_dim, device=device) for _ in range(2)]
        for name, p in meta_optimizer.get_register_params():
            cur_sz = int(np.prod(p.size()))
            # We do this so the gradients are disconnected from the graph but we still get
            # gradients from the rest
            gradients = detach_var(p.grad.view(cur_sz, 1))
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

            meta_optimizer = meta_optimizer_class(dim).to(device)  # todo?
            meta_optimizer.load_state_dict(result_params)
            meta_optimizer.zero_grad()
            hidden_states = [detach_var(v) for v in hidden_states2]
            cell_states = [detach_var(v) for v in cell_states2]

        else:
            for name, p in meta_optimizer.get_register_params():
                re_setattr(meta_optimizer, name, result_params[name])
            assert len(list(meta_optimizer.get_register_params()))
            hidden_states = hidden_states2
            cell_states = cell_states2

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
            # rnn = forward_pass(dim, opt_net, target, meta_optimizer_class, optim_it)
            rnn = run_opt(dim, opt_net, target, meta_optimizer_class,
                          optim_it, device)
            loss_list.append(rnn)

            wmmse = mmse_beamformers(H.cpu().numpy(), P)
            loss_mmse.append(target.objective_function(th.tensor(wmmse, dtype=th.cfloat, device=DEVICE)).cpu().numpy())

    loss_list = -np.array(loss_list).reshape((len(p_test), n_test, optim_it))
    losses = np.mean(loss_list[:, :, -1], axis=-1)
    loss_mmse = -np.array(loss_mmse).reshape((len(p_test), n_test))
    loss_mmse = np.mean(loss_mmse, axis=-1)
    print(f"{'Test SNR (dB)':<16} {'LSTM':9} {'MMSE':9}")

    for isnr, snr in enumerate((10 * np.log10(p_test)).astype(int)):
        print(f"{snr:<16} {losses[isnr]:9.3f} {loss_mmse[isnr]:9.3f}")
    return losses


def train_optimizer():
    dim = 8, 8
    p_test = [1, 10, 100]
    n_train_epochs = 1000
    test_every = 100
    objective = SumRateObjective
    meta_optimizer_class = MetaOptimizerMISO
    lr = 1e-3
    preproc = False
    unroll = 20
    optim_it = 100

    hidden_sz = 40
    device = DEVICE

    n, k = dim

    with open(f'./K{k}N{n}Samples=100.pkl', 'rb') as f:
        test_data = th.as_tensor(pkl.load(f), dtype=th.cfloat, device=device)

    opt_net = Optimizer(preproc=preproc, hidden_sz=hidden_sz).to(device)
    meta_opt = optim.Adam(opt_net.parameters(), lr=lr)

    time_start = time()
    for epoch in range(n_train_epochs + 1):

        # generate sample
        p = 10 ** (np.random.rand() + 1)
        h_scaled = (p ** 0.5) * th.randn(n, k, dtype=th.cfloat, device=DEVICE)
        # perform one training epoch
        # loss = do_fit(dim, opt_net, meta_opt, objective(h_scaled), meta_optimizer_class, unroll, optim_it,
        #               should_train=True)
        run_opt(dim, opt_net, objective(h_scaled), meta_optimizer_class,
                optim_it, device, unroll, meta_opt)

        if epoch % test_every == 0:
            do_test(dim, opt_net, objective, meta_optimizer_class=meta_optimizer_class,
                    optim_it=200, test_data=test_data, p_test=p_test, device=device)

            print('=' * 60)
            print(f'epoch {epoch}    seconds per epoch {round((time() - time_start) / test_every, 2),}')
            time_start = time()


class SumRateObjective:
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


class MetaOptimizerMISO(MetaModule):
    def __init__(self, dim):
        super().__init__()
        self.N, self.K = dim
        self.register_buffer('theta', to_cuda(to_var(th.zeros(2 * self.N * self.K, device=DEVICE), requires_grad=True)))
        # self.theta = self.theta.to(device)
        # for name, p in self.all_named_parameters():
        #     p = p.to(device)

    def forward(self, target):
        w = self.theta.reshape((2, self.N, self.K))
        w = w[0] + 1j * w[1]
        return target.objective_function(w)

    def all_named_parameters(self):
        return [('theta', self.theta)]


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
        if self.preproc:
            # Implement preproc described in Appendix A

            # Note: we do all this work on tensors, which means
            # the gradients won't propagate through inp. This
            # should be ok because the algorithm involves
            # making sure that inp is already detached.
            inp = inp.data
            inp2 = to_cuda(th.zeros(inp.size()[0], 2))
            keep_grads = (th.abs(inp) >= self.preproc_threshold).squeeze()
            inp2[:, 0][keep_grads] = (th.log(th.abs(inp[keep_grads]) + 1e-8) / self.preproc_factor).squeeze()
            inp2[:, 1][keep_grads] = th.sign(inp[keep_grads]).squeeze()

            inp2[:, 0][~keep_grads] = -1
            inp2[:, 1][~keep_grads] = (float(np.exp(self.preproc_factor)) * inp[~keep_grads]).squeeze()
            inp = to_cuda(th.tensor(inp2))
        hidden0, cell0 = self.recurs(inp, (hidden[0], cell[0]))
        hidden1, cell1 = self.recurs2(hidden0, (hidden[1], cell[1]))
        return self.output(hidden1), (hidden0, hidden1), (cell0, cell1)


if __name__ == '__main__':
    gpu_id = int(sys.argv[1])
    DEVICE = th.device(f'cuda:{gpu_id}' if th.cuda.is_available() and gpu_id >= 0 else 'cpu')

    train_optimizer()
