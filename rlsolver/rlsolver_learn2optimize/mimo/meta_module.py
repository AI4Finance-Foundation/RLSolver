import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
import operator
from itertools import islice
from collections import OrderedDict

def to_var(x, requires_grad=True):
    # if torch.cuda.is_available():
    #     x = x.cuda()
    return Variable(x, requires_grad=requires_grad)

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

    def set_param(self,curr_mod, name, param):
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
        keys = super(Sequential, self).__dir__()
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
        keys = super(ModuleList, self).__dir__()
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
        super(MetaModuleDict, self).__init__()
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
        layers.append(nn.MaxPool2d(kernel_size=2,stride=2))

        layers.append(MetaConv2d(6, 16, kernel_size=5))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.MaxPool2d(kernel_size=2,stride=2))
        
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
