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
    hidden_states = [th.zeros(n_params, opt_net.hidden_sz, device=device) for _ in range(2)]
    cell_states = [th.zeros(n_params, opt_net.hidden_sz, device=device) for _ in range(2)]
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
        hidden_states2 = [th.zeros(n_params, opt_net.hidden_sz, device=device) for _ in range(2)]
        cell_states2 = [th.zeros(n_params, opt_net.hidden_sz, device=device) for _ in range(2)]
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
    hidden_states = [th.zeros(n_params, opt_net.hidden_sz, device=device) for _ in range(2)]
    cell_states = [th.zeros(n_params, opt_net.hidden_sz, device=device) for _ in range(2)]
    all_losses_ever = []
    for iteration in range(1, optim_it + 1):
        loss = meta_optimizer(objective)

        all_losses_ever.append(loss.data.cpu().numpy())
        loss.backward(retain_graph=False)  # todo ?????????????

        offset = 0
        result_params = {}
        hidden_states2 = [th.zeros(n_params, opt_net.hidden_sz, device=device) for _ in range(2)]
        cell_states2 = [th.zeros(n_params, opt_net.hidden_sz, device=device) for _ in range(2)]
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


def do_test(dim, opt_net, objective, meta_optimizer_class, optim_it, test_data, device):
    loss_list = []
    n_test = test_data.shape[0]
    for H in test_data:
        h_scaled = H
        target = objective(h_scaled)

        opt_net.eval()
        rnn = opt_eval(dim, opt_net, target, meta_optimizer_class,
                        optim_it, device)  # todo
        loss_list.append(rnn)

    loss_list = -np.array(loss_list).reshape((1, n_test, optim_it))
    losses = np.mean(loss_list[:, :, -1], axis=-1)
    print(f"{'Test Max Cut':<16} {'LSTM':>9}")
    print(f"{losses[0]:9.3f}")
    return losses


def train_optimizer():
    dim = 8
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
    sparsity = 0.2
    n = dim

    try:
        with open(f'./N{n}Samples=100.pkl', 'rb') as f:
            test_data = th.as_tensor(pkl.load(f), dtype=th.float, device=device)
    except Exception as e:
        test_data = th.zeros(100, n, n, device=device)
        for i in range(100):
            upper_triangle = th.mul(th.rand(n, n).triu(diagonal=1), (th.rand(n, n) < sparsity).int().triu(diagonal=1))
            test_data[i] = upper_triangle + upper_triangle.transpose(-1, -2)
        with open(f'./N{n}Samples=100.pkl', 'wb') as f:
            pkl.dump(test_data.detach().cpu(), f)

    opt_net = Optimizer(hidden_sz=hidden_sz).to(device)
    meta_opt = optim.Adam(opt_net.parameters(), lr=lr)

    time_start = time()
    print('start training loop')
    for epoch in range(n_train_epochs + 1):

        # generate sample
        upper_triangle = th.mul(th.rand(n, n).triu(diagonal=1), (th.rand(n, n) < sparsity).int().triu(diagonal=1))
        adjacency_matrix = upper_triangle + upper_triangle.transpose(-1, -2)

        opt_train(dim, opt_net, objective(adjacency_matrix), meta_optimizer_class,
                  optim_it, device, unroll, meta_opt)

        if epoch % test_every == 0:
            do_test(dim, opt_net, objective, meta_optimizer_class=meta_optimizer_class,
                    optim_it=200, test_data=test_data[:n_test], device=device)

            print('=' * 60)
            print(f'epoch {epoch}    seconds per epoch {round((time() - time_start) / test_every, 2),}')
            time_start = time()

    do_test(dim, opt_net, objective, meta_optimizer_class=meta_optimizer_class,
            optim_it=200, test_data=test_data[:n_test], device=device)


class ObjectiveSunRate:  # 必须把 objective 改成一个 class，因为要传入 run_opt
    def __init__(self, adjacency_matrix: TEN):
        self.adjacency_matrix = adjacency_matrix  # hall

    def get_objective(self, configuration: TEN, noise: float = 1.) -> TEN:
        return th.mul(th.matmul(configuration.reshape(self.N, 1), (1 - configuration.reshape(-1, self.N, 1)).transpose(-1, -2)), self.adjacency_matrix).flatten().sum(dim=-1)



def set_attr(obj, attr, val):
    attrs = attr.split('.')
    for attr in attrs[:-1]:
        obj = getattr(obj, attr)
    setattr(obj, attrs[-1], val)


class MetaOptimizerMISO(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.N = dim
        self.register_buffer('theta', th.ones((self.N) / 2, device=DEVICE, requires_grad=True))

    def get_register_params(self):
        return [('theta', self.theta)]

    def forward(self, target):
        w = self.theta
        return target.get_objective(w)

    def output(self):
        return self.theta


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
