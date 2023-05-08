import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
import operator
from itertools import islice
import os
from collections import OrderedDict
import collections.abc as container_abcs
import functools


def to_var(x, requires_grad=True):
    # if torch.cuda.is_available():
    #     x = x.cuda()
    return Variable(x, requires_grad=requires_grad)


def calc_file_name(front: str, id2: int, val: int, end: str):
    return front + str(id2) + "_" + str(val) + end + "pkl"


# e.g., gset14_345.pkl, front = "gset", end = ".", new_val = 500, then output is gset14_500.pkl
def remove_files_less_equal_new_val(dir: str, front: str, end: str, new_val: int):
    files = os.listdir(dir)
    max_val = -np.inf
    prev_val = -np.inf
    for f in files:
        if front in f:
            id2 = int(f.split(front)[1].split("_")[0])
            if end in f.split(str(id2))[1]:
                val = int(f.split('_')[1].split(end)[0])
                if val > max_val:
                    max_val = val
                prev_file_name = calc_file_name(front, id2, prev_val, end)
                if prev_file_name in files and val > prev_val:
                    prev_file_name = dir + "/" + prev_file_name
                    if os.path.isfile(prev_file_name):
                        os.remove(prev_file_name)
                prev_val = val

    if new_val >= max_val:
        max_val_file_name = calc_file_name(front, id2, max_val, end)
        max_val_file_name = dir + "/" + max_val_file_name
        if os.path.isfile(max_val_file_name):
            os.remove(max_val_file_name)


class MetaModule(nn.Module):
    # adopted from: Adrien Ecoffet https://github.com/AdrienLE
    def parameters(self):
        for name, param in self.named_params(self):
            yield param

    def named_parameters(self):
        for name, param in self.named_params(self):
            yield name, param

    def named_leaves(self):
        return []

    def named_submodules(self):
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
            for name, p in curr_module._parameters.items():
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
            self.set_param(name, param)


class MetaLinear(MetaModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        ignore = nn.Linear(*args, **kwargs)

        self.register_buffer('weight', to_var(ignore.weight.data, requires_grad=True))
        self.register_buffer('bias', to_var(ignore.bias.data, requires_grad=True))
        self.in_features = ignore.weight.size(1)
        self.out_features = ignore.weight.size(0)

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)

    def named_leaves(self):
        return [('weight', self.weight), ('bias', self.bias)]


class MetaConv2d(MetaModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        ignore = nn.Conv2d(*args, **kwargs)

        self.stride = ignore.stride
        self.padding = ignore.padding
        self.dilation = ignore.dilation
        self.groups = ignore.groups

        self.register_buffer('weight', to_var(ignore.weight.data, requires_grad=True))

        if ignore.bias is not None:
            self.register_buffer('bias', to_var(ignore.bias.data, requires_grad=True))
        else:
            self.register_buffer('bias', None)

    def forward(self, x):
        return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

    def named_leaves(self):
        return [('weight', self.weight), ('bias', self.bias)]


class MetaConvTranspose2d(MetaModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        ignore = nn.ConvTranspose2d(*args, **kwargs)

        self.stride = ignore.stride
        self.padding = ignore.padding
        self.dilation = ignore.dilation
        self.groups = ignore.groups

        self.register_buffer('weight', to_var(ignore.weight.data, requires_grad=True))

        if ignore.bias is not None:
            self.register_buffer('bias', to_var(ignore.bias.data, requires_grad=True))
        else:
            self.register_buffer('bias', None)

    def forward(self, x, output_size=None):
        output_padding = self._output_padding(x, output_size)
        return F.conv_transpose2d(x, self.weight, self.bias, self.stride, self.padding,
                                  output_padding, self.groups, self.dilation)

    def named_leaves(self):
        return [('weight', self.weight), ('bias', self.bias)]


class MetaBatchNorm2d(MetaModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        ignore = nn.BatchNorm2d(*args, **kwargs)

        self.num_features = ignore.num_features
        self.eps = ignore.eps
        self.momentum = ignore.momentum
        self.affine = ignore.affine
        self.track_running_stats = ignore.track_running_stats

        if self.affine:
            self.register_buffer('weight', to_var(ignore.weight.data, requires_grad=True))
            self.register_buffer('bias', to_var(ignore.bias.data, requires_grad=True))

        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(self.num_features))
            self.register_buffer('running_var', torch.ones(self.num_features))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)

    def forward(self, x):
        return F.batch_norm(x, self.running_mean, self.running_var, self.weight, self.bias,
                            self.training or not self.track_running_stats, self.momentum, self.eps)

    def named_leaves(self):
        return [('weight', self.weight), ('bias', self.bias)]


class MetaSequential(MetaModule):
    r"""A sequential container.
    Modules will be added to it in the order they are passed in the constructor.
    Alternatively, an ordered dict of modules can also be passed in.

    To make it easier to understand, here is a small example::

        # Example of using Sequential
        model = MetaSequential(
                  MetaConv2d(1,20,5),
                  nn.ReLU(),
                  MetaConv2d(20,64,5),
                  nn.ReLU()
                )

        # Example of using Sequential with OrderedDict
        model = MetaSequential(OrderedDict([
                  ('conv1', MetaConv2d(1,20,5)),
                  ('relu1', nn.ReLU()),
                  ('conv2', MetaConv2d(20,64,5)),
                  ('relu2', nn.ReLU())
                ]))
    """

    def __init__(self, *args):
        super(MetaSequential, self).__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)

    def _get_item_by_idx(self, iterator, idx):
        """Get the idx-th item of the iterator"""
        size = len(self)
        idx = operator.index(idx)
        if not -size <= idx < size:
            raise IndexError('index {} is out of range'.format(idx))
        idx %= size
        return next(islice(iterator, idx, None))

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return self.__class__(OrderedDict(list(self._modules.items())[idx]))
        else:
            return self._get_item_by_idx(self._modules.values(), idx)

    def __setitem__(self, idx, module):
        key = self._get_item_by_idx(self._modules.keys(), idx)
        return setattr(self, key, module)

    def __delitem__(self, idx):
        if isinstance(idx, slice):
            for key in list(self._modules.keys())[idx]:
                delattr(self, key)
        else:
            key = self._get_item_by_idx(self._modules.keys(), idx)
            delattr(self, key)

    def __len__(self):
        return len(self._modules)

    def __dir__(self):
        keys = super(nn.Sequential, self).__dir__()
        keys = [key for key in keys if not key.isdigit()]
        return keys

    def forward(self, input):
        for module in self._modules.values():
            input = module(input)
        return input


class MetaModuleList(MetaModule):
    r"""Holds submodules in a list.

    :class:`~MetaModuleList` can be indexed like a regular Python list, but
    modules it contains are properly registered, and will be visible by all
    :class:`~MetaModule` methods.

    Arguments:
        modules (iterable, optional): an iterable of modules to add

    Example::

        class MyModule(MetaModule):
            def __init__(self):
                super(MyModule, self).__init__()
                self.linears = MetaModuleList([MetaLinear(10, 10) for i in range(10)])

            def forward(self, x):
                # ModuleList can act as an iterable, or be indexed using ints
                for i, l in enumerate(self.linears):
                    x = self.linears[i // 2](x) + l(x)
                return x
    """

    def __init__(self, modules=None):
        super(MetaModuleList, self).__init__()
        if modules is not None:
            self += modules

    def _get_abs_string_index(self, idx):
        """Get the absolute index for the list of modules"""
        idx = operator.index(idx)
        if not (-len(self) <= idx < len(self)):
            raise IndexError('index {} is out of range'.format(idx))
        if idx < 0:
            idx += len(self)
        return str(idx)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return self.__class__(list(self._modules.values())[idx])
        else:
            return self._modules[self._get_abs_string_index(idx)]

    def __setitem__(self, idx, module):
        idx = self._get_abs_string_index(idx)
        return setattr(self, str(idx), module)

    def __delitem__(self, idx):
        if isinstance(idx, slice):
            for k in range(len(self._modules))[idx]:
                delattr(self, str(k))
        else:
            delattr(self, self._get_abs_string_index(idx))
        # To preserve numbering, self._modules is being reconstructed with modules after deletion
        str_indices = [str(i) for i in range(len(self._modules))]
        self._modules = OrderedDict(list(zip(str_indices, self._modules.values())))

    def __len__(self):
        return len(self._modules)

    def __iter__(self):
        return iter(self._modules.values())

    def __iadd__(self, modules):
        return self.extend(modules)

    def __dir__(self):
        keys = super(self).__dir__()
        keys = [key for key in keys if not key.isdigit()]
        return keys

    def insert(self, index, module):
        r"""Insert a given module before a given index in the list.

        Arguments:
            index (int): index to insert.
            module (MetaModule): module to insert
        """
        for i in range(len(self._modules), index, -1):
            self._modules[str(i)] = self._modules[str(i - 1)]
        self._modules[str(index)] = module

    def append(self, module):
        r"""Appends a given module to the end of the list.

        Arguments:
            module (MetaModule): module to append
        """
        self.add_module(str(len(self)), module)
        return self

    def extend(self, modules):
        r"""Appends modules from a Python iterable to the end of the list.

        Arguments:
            modules (iterable): iterable of modules to append
        """
        if not isinstance(modules, container_abcs.Iterable):
            raise TypeError("ModuleList.extend should be called with an "
                            "iterable, but got " + type(modules).__name__)
        offset = len(self)
        for i, module in enumerate(modules):
            self.add_module(str(offset + i), module)
        return self


class ModuleDict(MetaModule):
    r"""Holds submodules in a dictionary.

    :class:`~MetaModuleDict` can be indexed like a regular Python dictionary,
    but modules it contains are properly registered, and will be visible by all
    :class:`~MetaModule` methods.

    :class:`~MetaModuleDict` is an **ordered** dictionary that respects

    * the order of insertion, and

    * in :meth:`~MetaModuleDict.update`, the order of the merged ``OrderedDict``
      or another :class:`~MetaModuleDict` (the argument to :meth:`~MetaModuleDict.update`).

    Note that :meth:`~MetaModuleDict.update` with other unordered mapping
    types (e.g., Python's plain ``dict``) does not preserve the order of the
    merged mapping.

    Arguments:
        modules (iterable, optional): a mapping (dictionary) of (string: module)
            or an iterable of key-value pairs of type (string, module)

    Example::

        class MyModule(MetaModule):
            def __init__(self):
                super(MyModule, self).__init__()
                self.choices = MetaModuleDict({
                        'conv': MetaConv2d(10, 10, 3),
                        'pool': nn.MaxPool2d(3)
                })
                self.activations = MetaModuleDict([
                        ['lrelu', nn.LeakyReLU()],
                        ['prelu', nn.PReLU()]
                ])

            def forward(self, x, choice, act):
                x = self.choices[choice](x)
                x = self.activations[act](x)
                return x
    """

    def __init__(self, modules=None):
        super(self).__init__()
        if modules is not None:
            self.update(modules)

    def __getitem__(self, key):
        return self._modules[key]

    def __setitem__(self, key, module):
        self.add_module(key, module)

    def __delitem__(self, key):
        del self._modules[key]

    def __len__(self):
        return len(self._modules)

    def __iter__(self):
        return iter(self._modules)

    def __contains__(self, key):
        return key in self._modules

    def clear(self):
        """Remove all items from the ModuleDict.
        """
        self._modules.clear()

    def pop(self, key):
        r"""Remove key from the ModuleDict and return its module.

        Arguments:
            key (string): key to pop from the ModuleDict
        """
        v = self[key]
        del self[key]
        return v

    def keys(self):
        r"""Return an iterable of the ModuleDict keys.
        """
        return self._modules.keys()

    def items(self):
        r"""Return an iterable of the ModuleDict key/value pairs.
        """
        return self._modules.items()

    def values(self):
        r"""Return an iterable of the ModuleDict values.
        """
        return self._modules.values()

    def update(self, modules):
        r"""Update the :class:`~MetaModuleDict` with the key-value pairs from a
        mapping or an iterable, overwriting existing keys.

        .. note::
            If :attr:`modules` is an ``OrderedDict``, a :class:`~MetaModuleDict`, or
            an iterable of key-value pairs, the order of new elements in it is preserved.

        Arguments:
            modules (iterable): a mapping (dictionary) from string to :class:`~MetaModule`,
                or an iterable of key-value pairs of type (string, :class:`~MetaModule`)
        """
        if not isinstance(modules, container_abcs.Iterable):
            raise TypeError("ModuleDict.update should be called with an "
                            "iterable of key/value pairs, but got " +
                            type(modules).__name__)

        if isinstance(modules, container_abcs.Mapping):
            if isinstance(modules, (OrderedDict, ModuleDict)):
                for key, module in modules.items():
                    self[key] = module
            else:
                for key, module in sorted(modules.items()):
                    self[key] = module
        else:
            for j, m in enumerate(modules):
                if not isinstance(m, container_abcs.Iterable):
                    raise TypeError("ModuleDict update sequence element "
                                    "#" + str(j) + " should be Iterable; is" +
                                    type(m).__name__)
                if not len(m) == 2:
                    raise ValueError("ModuleDict update sequence element "
                                     "#" + str(j) + " has length " + str(len(m)) +
                                     "; 2 is required")
                self[m[0]] = m[1]

    def forward(self):
        raise NotImplementedError()


class LeNet(MetaModule):
    def __init__(self, n_out):
        super(LeNet, self).__init__()

        layers = []
        layers.append(MetaConv2d(1, 6, kernel_size=5))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        layers.append(MetaConv2d(6, 16, kernel_size=5))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        layers.append(MetaConv2d(16, 120, kernel_size=5))
        layers.append(nn.ReLU(inplace=True))

        self.main = nn.Sequential(*layers)

        layers = []
        layers.append(MetaLinear(120, 84))
        layers.append(nn.ReLU(inplace=True))
        layers.append(MetaLinear(84, n_out))

        self.fc_layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.main(x)
        x = x.view(-1, 120)
        return self.fc_layers(x).squeeze()


def gset2npy(id):
    file1 = open(f"./data/maxcut/gset{id}.txt", 'r')
    Lines = file1.readlines()

    count = 0
    for line in Lines:
        count += 1
        s = line.split()
        if count == 1:
            N = int(s[0])
            edge = int(s[1])
            adjacency = th.zeros(N, N)
        else:
            i = int(s[0])
            j = int(s[1])
            w = int(s[2])
            adjacency[i - 1, j - 1] = w
            adjacency[j - 1, i - 1] = w
    sparsity = edge / (N * N)
    np.save(f"./data/gset_G{id}.npy", adjacency)


def detach_var(v, device):
    var = Variable(v.data, requires_grad=True).to(device)
    var.retain_grad()
    return var


def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition('.')
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)


def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)

    return functools.reduce(_getattr, [obj] + attr.split('.'))


def forward_pass(N, opt_net, target, opt_variable, optim_it, device):
    opt_net.eval()

    optimizee = opt_variable(N, device)
    n_params = 0
    for name, p in optimizee.all_named_parameters():
        n_params += int(np.prod(p.size()))
    hidden_states = [Variable(th.zeros(n_params, opt_net.hidden_sz)).to(device) for _ in range(2)]
    cell_states = [Variable(th.zeros(n_params, opt_net.hidden_sz)).to(device) for _ in range(2)]
    all_losses_ever = []
    all_losses = None
    last = 0
    for iteration in range(1, optim_it + 1):

        loss, l_list = optimizee(target)
        if all_losses is None:
            all_losses = loss
        else:
            all_losses += loss

        all_losses_ever = all_losses_ever + l_list
        loss.backward()

        offset = 0
        result_params = {}
        hidden_states2 = [Variable(th.zeros(n_params, opt_net.hidden_sz)).to(device) for _ in range(2)]
        cell_states2 = [Variable(th.zeros(n_params, opt_net.hidden_sz)).to(device) for _ in range(2)]
        for name, p in optimizee.all_named_parameters():
            cur_sz = int(np.prod(p.size()))
            gradients = detach_var(p.grad.view(cur_sz, 1), device)
            try:
                a = result_params[name].detach()
            except Exception as e:
                a = gradients
            updates, new_hidden, new_cell = opt_net(a, [h[offset:offset + cur_sz] for h in hidden_states],
                                                    [c[offset:offset + cur_sz] for c in cell_states])
            for i in range(len(new_hidden)):
                hidden_states2[i][offset:offset + cur_sz] = new_hidden[i]
                cell_states2[i][offset:offset + cur_sz] = new_cell[i]
            temp = p + updates.view(*p.size())
            # print(temp, th.norm(temp))
            result_params[name] = temp
            result_params[name].retain_grad()
            offset += cur_sz
        optimizee = opt_variable(N, device)
        optimizee.load_state_dict(result_params)
        optimizee.zero_grad()
        hidden_states = [detach_var(v, device) for v in hidden_states2]
        cell_states = [detach_var(v, device) for v in cell_states2]
    return all_losses_ever


def get_cwd(folder_name, N):
    N = N
    try:
        os.mkdir(folder_name)
    except:
        pass
    folder_name = folder_name + '/N' + str(N)
    try:
        os.mkdir(folder_name)
    except:
        pass

    file_list = os.listdir('./{}/'.format(folder_name))
    max_exp_id = 0
    for exp_id in file_list:
        if exp_id == '.DS_Store':
            pass
        elif int(exp_id) + 1 > max_exp_id:
            max_exp_id = int(exp_id) + 1
    os.mkdir('./{}/{}/'.format(folder_name, max_exp_id))
    return f"./{folder_name}/{max_exp_id}/", max_exp_id


# choice 0: use Synthetic data with N and sparsity
# choice >= 1: use Gset with the ID choice
def load_test_data(choice, device, N=10, sparsity=0.5):
    sparsity = sparsity
    n = N
    if choice > 0:
        try:
            gset2npy(choice)
            test_data = th.as_tensor(np.load(f"./data/gset_G{choice}.npy")).to(device)
        except Exception as e:
            test_data = th.zeros(n, n, device=device)
            upper_triangle = th.mul(th.ones(n, n).triu(diagonal=1), (th.rand(n, n) < sparsity).int().triu(diagonal=1))
            test_data = upper_triangle + upper_triangle.transpose(-1, -2)
            np.save(f'./data/N{n}Sparsity{sparsity}.npy', test_data.cpu().numpy())
    else:
        test_data = th.zeros(n, n, device=device)
        upper_triangle = th.mul(th.ones(n, n).triu(diagonal=1), (th.rand(n, n) < sparsity).int().triu(diagonal=1))
        test_data = upper_triangle + upper_triangle.transpose(-1, -2)
        np.save(f'./data/N{n}Sparsity{sparsity}.npy', test_data.cpu().numpy())
    return test_data


class Obj_fun():
    def __init__(self, adjacency_matrix, device=th.device("cuda" if th.cuda.is_available() else "cpu")):
        self.adjacency_matrix = adjacency_matrix.to(device)
        self.N = adjacency_matrix.shape[0]

    def get_loss(self, x):
        loss = 0
        x = x.sigmoid()
        loss -= th.mul(th.matmul(x.reshape(self.N, 1), (1 - x.reshape(self.N, 1)).transpose(-1, -2)),
                       self.adjacency_matrix).flatten().sum(dim=-1)
        return loss


class Opt_variable(MetaModule):
    def __init__(self, N, device):
        super().__init__()
        self.N = N
        self.bs = 2
        self.flip_prob = 0.05
        self.loss = []
        self.register_buffer(f'theta', to_var(th.rand(self.bs, self.N, device=device), requires_grad=True))

    def forward(self, target):
        loss = 0
        l_list = []
        for i in range(self.bs):
            l = target.get_loss(self.theta[i])
            l_list.append(l.item())
            loss += l
        return loss, l_list

    def all_named_parameters(self):
        return [('theta', self.theta)]

    def duplicate_parameters(self, id):
        with th.no_grad():
            for i in range(self.bs):
                self.theta[i] = self.theta[id]

    def flip_parameters(self, flip_prob=0.05):
        flip_mat = (th.rand(self.bs, self.N) > flip_prob).int()
        with th.no_grad():
            for i in range(self.bs):
                for j in range(self.N):
                    if (flip_mat[i][j] == 0):
                        self.theta[i][j] = 1 - self.theta[i][j]


class Opt_net2(nn.Module):
    def __init__(self, preproc=False, hidden_sz=20, preproc_factor=10.0,
                 device=th.device("cuda" if th.cuda.is_available() else "cpu")):
        super().__init__()
        self.hidden_sz = hidden_sz
        self.device = device
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
            inp = inp.data
            inp2 = th.zeros(inp.size()[0], 2, device=self.device)
            keep_grads = (th.abs(inp) >= self.preproc_threshold).squeeze()
            inp2[:, 0][keep_grads] = (th.log(th.abs(inp[keep_grads]) + 1e-8) / self.preproc_factor).squeeze()
            inp2[:, 1][keep_grads] = th.sign(inp[keep_grads]).squeeze()

            inp2[:, 0][~keep_grads] = -1
            inp2[:, 1][~keep_grads] = (float(np.exp(self.preproc_factor)) * inp[~keep_grads]).squeeze()
            inp = Variable(inp2, device=self.device)
        hidden0, cell0 = self.recurs(inp, (hidden[0], cell[0]))
        hidden1, cell1 = self.recurs2(hidden0, (hidden[1], cell[1]))
        return self.output(hidden1), (hidden0, hidden1), (cell0, cell1)


class Opt_net(nn.Module):
    def __init__(self, N, hidden_layers):
        super(Opt_net, self).__init__()
        self.N = N
        self.hidden_layers = hidden_layers
        self.lstm = nn.LSTM(self.N, self.hidden_layers, 1, batch_first=True)
        self.output = nn.Linear(hidden_layers, self.N)

    def forward(self, configuration, hidden_state, cell_state):
        x, (h, c) = self.lstm(configuration, (hidden_state, cell_state))
        return self.output(x).sigmoid(), h, c

def gset2npy(file: str, output_file: str):
    file1 = open(file, 'r')
    Lines = file1.readlines()

    count = 0
    for line in Lines:
        count += 1
        s = line.split()
        if count == 1:
            N = int(s[0])
            edge = int(s[1])
            adjacency = th.zeros(N, N)
        else:
            i = int(s[0])
            j = int(s[1])
            w = int(s[2])
            adjacency[i - 1, j - 1] = w
            adjacency[j - 1, i - 1] = w
    sparsity = edge / (N * N)
    print("sparsity: ", sparsity)
    np.save(output_file, adjacency)
    # adjacency = th.as_tensor(np.load("N800Sparsity0.007.npy"))
    # print(adjacency.shape, adjacency.sum())

def run_gset2npy():
    N = 14
    file = f".data/G{N}.txt"
    output_file = f"./data/N{N}.npy"
    gset2npy(file, output_file)

