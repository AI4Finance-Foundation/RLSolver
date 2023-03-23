import pickle as pkl
import torch as th
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import copy
import os
import numpy as np
from time import time
import numpy.linalg as la
import functools

'''module'''


def to_var(x, requires_grad=True):
    # if torch.cuda.is_available():
    #     x = x.cuda()
    return Variable(x, requires_grad=requires_grad)


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
    if USE_CUDA:
        return v.cuda()
    return v


def detach_var(v):
    var = to_cuda(Variable(v.data, requires_grad=True))
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


def do_fit(dim, opt_net, meta_opt, objective, meta_optimizer_class, unroll, optim_it, should_train=True):
    if should_train:
        opt_net.train()
    else:
        opt_net.eval()
        unroll = 1

    meta_optimizer = to_cuda(meta_optimizer_class(dim))
    n_params = 0
    for name, p in meta_optimizer.get_register_params():
        n_params += int(np.prod(p.size()))
    hidden_states = [to_cuda(Variable(th.zeros(n_params, opt_net.hid_dim))) for _ in range(2)]
    cell_states = [to_cuda(Variable(th.zeros(n_params, opt_net.hid_dim))) for _ in range(2)]
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
        hidden_states2 = [to_cuda(Variable(th.zeros(n_params, opt_net.hid_dim))) for _ in range(2)]
        cell_states2 = [to_cuda(Variable(th.zeros(n_params, opt_net.hid_dim))) for _ in range(2)]
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

            meta_optimizer = to_cuda(meta_optimizer_class(dim))
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


def forward_pass(dim, opt_net, target, meta_optimizer_class, optim_it):
    opt_net.eval()

    meta_optimizer = to_cuda(meta_optimizer_class(dim))
    n_params = 0
    for name, p in meta_optimizer.get_register_params():
        n_params += int(np.prod(p.size()))

    hidden_states = [to_cuda(Variable(th.zeros(n_params, opt_net.hid_dim))) for _ in range(2)]
    cell_states = [to_cuda(Variable(th.zeros(n_params, opt_net.hid_dim))) for _ in range(2)]
    all_losses_ever = []
    all_losses = None
    for iteration in range(1, optim_it + 1):
        loss = meta_optimizer(target)

        if all_losses is None:
            all_losses = loss
        else:
            all_losses += loss

        all_losses_ever.append(loss.data.cpu().numpy().copy())
        loss.backward()

        offset = 0
        result_params = {}
        hidden_states2 = [to_cuda(Variable(th.zeros(n_params, opt_net.hid_dim))) for _ in range(2)]
        cell_states2 = [to_cuda(Variable(th.zeros(n_params, opt_net.hid_dim))) for _ in range(2)]
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

        meta_optimizer = to_cuda(meta_optimizer_class(dim))
        meta_optimizer.load_state_dict(result_params)
        meta_optimizer.zero_grad()
        hidden_states = [detach_var(v) for v in hidden_states2]
        cell_states = [detach_var(v) for v in cell_states2]

    return all_losses_ever


def do_test(dim, best_loss, opt_net, objective, meta_optimizer_class, optim_it, p_test, test_data):
    loss = []
    loss_mmse = []
    n_test = test_data.shape[0]
    for P in p_test:
        for H in test_data:
            h_scaled = H * (P ** 0.5)
            target = objective(h_scaled)
            rnn = forward_pass(dim, opt_net, target, meta_optimizer_class, optim_it)
            loss.append(rnn)

            wmmse = mmse_beamformers(H.cpu().numpy(), P)
            loss_mmse.append(target.objective_function(th.tensor(wmmse, dtype=th.cfloat, device=DEVICE)).cpu().numpy())

    loss = -np.array(loss).reshape((len(p_test), n_test, optim_it))
    loss_avg_final = np.mean(loss[:, :, -1], axis=-1)
    loss_mmse = -np.array(loss_mmse).reshape((len(p_test), n_test))
    loss_mmse = np.mean(loss_mmse, axis=-1)
    print(('{:<15}' + '{:<10}' * 3).format(*['Test SNR (dB)', 'LSTM', 'LSTM*', 'MMSE']))

    for isnr, snr in enumerate((10 * np.log10(p_test)).astype(int)):
        print(('{:<15}' + '{:<10f}' * 3).format(*[snr, loss_avg_final[isnr], best_loss[isnr], loss_mmse[isnr]]))
    return loss_avg_final, loss_mmse, loss


def load_test_data(n, k, n_test, save_path):
    if n == 4 and k == 4:
        with open("./Channel_K=4_N=4_P=10_Samples=100_Optimal=9.8.pkl", 'rb') as f:
            test_data = to_cuda(th.as_tensor(pkl.load(f), dtype=th.cfloat))[:n_test]  # .transpose(-1,-2)
    elif n == 8 and k == 8:
        with open("./K8N8Samples=100.pkl", 'rb') as f:
            test_data = to_cuda(th.as_tensor(pkl.load(f), dtype=th.cfloat))[:n_test]
    elif k == 16:
        test_data = to_cuda(th.as_tensor(np.load('./HallN16K16.npy'), dtype=th.cfloat))
    else:
        test_data = th.randn(5, n, k, dtype=th.cfloat, device=DEVICE)
        th.save(test_data, save_path + 'test_data.pth')
    return test_data


def train_optimizer(dim, run_id, objective, meta_optimizer_class, test_every=20, preproc=False, unroll=20, optim_it=100,
                    lr=0.001, hidden_sz=20, load_net_path=None, n_test=10, save_path=None, p_eval=None, p_test=None,
                    n_train_epochs=1000):
    n, k = dim
    test_data = load_test_data(n, k, n_test, save_path)

    opt_net = to_cuda(Optimizer(preproc=preproc, hidden_sz=hidden_sz))
    # summary(opt_net)
    meta_opt = optim.Adam(opt_net.parameters(), lr=lr)
    if load_net_path is not None:
        opt_net.load_state_dict(th.load(load_net_path))

    best_loss, _, _ = do_test(dim, [0] * len(p_test), opt_net, objective, meta_optimizer_class=meta_optimizer_class,
                              optim_it=200,
                              test_data=test_data, p_test=p_test)

    history = {
        'test_loss': [[] for _ in p_test],
        'train_loss': []
    }

    for il, l in enumerate(best_loss):
        history['test_loss'][il].append(l)
    best_net_path = None
    ip_eval = np.where(p_eval == np.array(p_test))[0].item()
    time_start = time()
    for epoch in range(1, n_train_epochs + 1):

        # Update learning rate
        try:
            meta_opt.param_groups[0]['lr'] = LR
        except Exception as error:
            print(f"failed to load lr. meta_opt.param_groups[0]['lr'] = LR   {error}")
            pass

        # generate sample
        p = 10 ** (np.random.rand() + 1)
        h_scaled = (p ** 0.5) * th.randn(n, k, dtype=th.cfloat, device=DEVICE)
        # perform one training epoch
        loss = do_fit(dim, opt_net, meta_opt, objective(h_scaled), meta_optimizer_class, unroll, optim_it,
                      should_train=True)

        history['train_loss'].append(-np.mean(loss))
        if epoch % test_every == 0:
            print('=' * 60)
            print('epoch', epoch)
            loss, _, _ = do_test(dim, best_loss, opt_net, objective, meta_optimizer_class=meta_optimizer_class,
                                 optim_it=200,
                                 test_data=test_data, p_test=p_test)
            print('lr = {:.2e}'.format(meta_opt.param_groups[0]['lr']))
            print('id', run_id)
            print('k = ' + str(k) + ', n = ' + str(n))
            print(round((time() - time_start) / test_every, 2), 'seconds per epoch')

            for il, l in enumerate(loss):
                history['test_loss'][il].append(l)

            np.save(save_path + 'history.npy', history)
            if loss[ip_eval] > best_loss[ip_eval]:
                savename = 'epoch{}_loss={:.2f}'.format(epoch, loss[ip_eval])
                print('checkpoint:', save_path + savename)
                best_loss = loss
                best_net = copy.deepcopy(opt_net.state_dict())
                th.save(best_net, save_path + savename)
                best_net_path = save_path + savename

            time_start = time()

    return best_loss, best_net_path


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
            inp = to_cuda(Variable(inp2))
        hidden0, cell0 = self.recurs(inp, (hidden[0], cell[0]))
        hidden1, cell1 = self.recurs2(hidden0, (hidden[1], cell[1]))
        return self.output(hidden1), (hidden0, hidden1), (cell0, cell1)


def get_cwd(env_name, dim):
    n, k = dim
    # file_list = os.listdir()
    # if env_name not in file_list:
    os.makedirs(env_name, exist_ok=True)
    env_name = env_name + '/n' + str(n) + 'k' + str(k)
    os.makedirs(env_name, exist_ok=True)

    file_list = os.listdir('./{}/'.format(env_name))
    max_exp_id = 0
    for exp_id in file_list:
        if exp_id == '.DS_Store':
            pass
        elif int(exp_id) + 1 > max_exp_id:
            max_exp_id = int(exp_id) + 1
    os.mkdir('./{}/{}/'.format(env_name, max_exp_id))
    return f"./{env_name}/{max_exp_id}/", max_exp_id


def test_optimizer(dim, p_test, optim_it, n_test, objective, meta_optimizer_class, preproc=False, hidden_sz=20,
                   load_net_path=None, save_path=None):
    n, k = dim
    test_data = load_test_data(n, k, n_test, save_path)
    # opt_net = w(Optimizer(preproc=preproc))
    opt_net = to_cuda(Optimizer(preproc=preproc, hidden_sz=hidden_sz))
    # summary(opt_net)
    if load_net_path is not None:
        opt_net.load_state_dict(th.load(load_net_path))

    _, _, loss = do_test(dim, [0] * len(p_test), opt_net, objective, meta_optimizer_class, optim_it,
                         test_data=test_data,
                         p_test=p_test)
    return loss


def do_test_sumrate_vs_snr():
    env_name = f"nets"
    save_path, run_id = get_cwd(env_name, DIM)
    p_test = np.logspace(0, 2, 5)
    n_test = 100
    n = k = 64
    if k == 4:
        with open("./Channel_K=4_N=4_P=10_Samples=100_Optimal=9.8.pkl", 'rb') as f:
            test_data = to_cuda(th.as_tensor(pkl.load(f), dtype=th.cfloat))[:n_test]  # .transpose(-1,-2)
            # test_data = th.randn(100,N,k, dtype=th.cfloat)
    elif k == 8:
        with open("./K8N8Samples=100.pkl", 'rb') as f:
            test_data = to_cuda(th.as_tensor(pkl.load(f), dtype=th.cfloat))[:n_test]
    elif k == 16:
        test_data = to_cuda(th.as_tensor(np.load('./HallN16K16.npy'), dtype=th.cfloat))
    else:
        test_data = th.randn(100, n, k, dtype=th.cfloat)

    print(repr(test_data))
    path = 'path/to/network.pth'
    res = test_optimizer((n, k), p_test=p_test, optim_it=100, objective=SumRateObjective, n_test=n_test,
                         meta_optimizer_class=MetaOptimizerMISO, hidden_sz=40, load_net_path=path, save_path=save_path)
    np.save('./results.npy', res)


def run():
    p_eval = 100
    p_test = [1, 10, 100]

    objective = SumRateObjective
    meta_optimizer_class = MetaOptimizerMISO

    env_name = f"nets"
    save_path, run_id = get_cwd(env_name, DIM)

    hidden_sz = 40

    loss, path = train_optimizer(
        dim=DIM,
        run_id=run_id,
        objective=objective,
        meta_optimizer_class=meta_optimizer_class,
        test_every=100,
        hidden_sz=hidden_sz,
        lr=LR,
        load_net_path=None,
        n_test=10,
        save_path=save_path,
        p_test=p_test,
        p_eval=p_eval,
        n_train_epochs=1000)

    test_optimizer(
        dim=DIM,
        p_test=np.logspace(0, 2, 5),
        optim_it=100,
        n_test=100,
        objective=objective,
        meta_optimizer_class=meta_optimizer_class,
        hidden_sz=hidden_sz,
        load_net_path=path
    )


if __name__ == '__main__':
    USE_CUDA = False
    if USE_CUDA:
        DEVICE = th.device('cuda:0')
        th.cuda.set_device(DEVICE)
    else:
        DEVICE = th.device('cpu')

    # run()

    DIM = (8, 8)  # (N,K)
    LR = 1e-3

    run()
