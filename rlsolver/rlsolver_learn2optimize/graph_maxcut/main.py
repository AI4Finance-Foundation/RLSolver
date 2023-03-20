# import glob
import pickle as pkl
import torch as th
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import matplotlib.pyplot as plt
from tqdm import tqdm
import copy
import os
import numpy as np
# import seaborn as sns; sns.set(color_codes=True)
# sns.set_style("white")
from pdb import set_trace as bp
from time import time
# USE_CUDA = th.cuda.is_available()
from meta_module import MetaModule, to_var
# from torchsummary import summary
import numpy.linalg as la
import functools

def w(v):
    if USE_CUDA:
        return v.cuda()
    return v

def detach_var(v):
    var = w(Variable(v.data, requires_grad=True))
    var.retain_grad()
    return var

def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition('.')
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)

# using wonder's beautiful simplification: https://stackoverflow.com/questions/31174295/getattr-and-setattr-on-nested-objects/31174427?noredirect=1#comment86638618_31174427

def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split('.'))

def do_fit(dim, opt_net, meta_opt, objective, optimizee_class, unroll, optim_it, should_train=True):
    if should_train:
        opt_net.train()
    else:
        opt_net.eval()
        unroll = 1

    optimizee = w(optimizee_class(dim))
    n_params = 0
    for name, p in optimizee.all_named_parameters():
        n_params += int(np.prod(p.size()))
    hidden_states = [w(Variable(th.zeros(n_params, opt_net.hidden_sz))) for _ in range(2)]
    cell_states = [w(Variable(th.zeros(n_params, opt_net.hidden_sz))) for _ in range(2)]
    all_losses_ever = []
    if should_train:
        meta_opt.zero_grad()
    all_losses = None
    for iteration in range(1, optim_it + 1):
        loss = optimizee(objective)

        if all_losses is None:
            all_losses = loss
        else:
            all_losses += loss

        all_losses_ever.append(loss.data.cpu().numpy().copy())
        loss.backward(retain_graph=should_train)

        offset = 0
        result_params = {}
        hidden_states2 = [w(Variable(th.zeros(n_params, opt_net.hidden_sz))) for _ in range(2)]
        cell_states2 = [w(Variable(th.zeros(n_params, opt_net.hidden_sz))) for _ in range(2)]
        for name, p in optimizee.all_named_parameters():
            cur_sz = int(np.prod(p.size()))
            # We do this so the gradients are disconnected from the graph but we still get
            # gradients from the rest
            gradients = detach_var(p.grad.view(cur_sz, 1))
            try:
                a = result_params[name].detach()
                #print("no gradients")
            except Exception as e:
                a = gradients
                #print("gradients")
            updates, new_hidden, new_cell = opt_net(
                a, # gradients,
                #result_params[name].detach(),
                [h[offset:offset+cur_sz] for h in hidden_states],
                [c[offset:offset+cur_sz] for c in cell_states]
            )
            for i in range(len(new_hidden)):
                hidden_states2[i][offset:offset+cur_sz] = new_hidden[i]
                cell_states2[i][offset:offset+cur_sz] = new_cell[i]

            temp = p + updates.view(*p.size())
            #print(temp, th.norm(temp))
            #temp = temp / th.norm(temp)
            #temp = temp.clamp(0, 1)
            #print(temp, th.norm(temp))

            result_params[name] = temp#/th.norm(temp)#.clamp(0, 1)#/th.norm(temp)

            result_params[name].retain_grad()

            offset += cur_sz

        if iteration % unroll == 0:
            if should_train:
                meta_opt.zero_grad()
                all_losses.backward()
                meta_opt.step()

            all_losses = None

            optimizee = w(optimizee_class(dim))
            optimizee.load_state_dict(result_params)
            optimizee.zero_grad()
            hidden_states = [detach_var(v) for v in hidden_states2]
            cell_states = [detach_var(v) for v in cell_states2]

        else:
            for name, p in optimizee.all_named_parameters():
                rsetattr(optimizee, name, result_params[name])
            assert len(list(optimizee.all_named_parameters()))
            hidden_states = hidden_states2
            cell_states = cell_states2

    return all_losses_ever

def forward_pass(dim, opt_net, target, optimizee_class, optim_it):
    opt_net.eval()

    optimizee = w(optimizee_class(dim))
    n_params = 0
    for name, p in optimizee.all_named_parameters():
        n_params += int(np.prod(p.size()))

    hidden_states = [w(Variable(th.zeros(n_params, opt_net.hidden_sz))) for _ in range(2)]
    cell_states = [w(Variable(th.zeros(n_params, opt_net.hidden_sz))) for _ in range(2)]
    all_losses_ever = []
    all_losses = None
    last = 0
    for iteration in range(1, optim_it + 1):
        loss = optimizee(target)

        if all_losses is None:
            all_losses = loss
        else:
            all_losses += loss

        all_losses_ever.append(loss.data.cpu().numpy().copy())
        loss.backward()

        offset = 0
        result_params = {}
        hidden_states2 = [w(Variable(th.zeros(n_params, opt_net.hidden_sz))) for _ in range(2)]
        cell_states2 = [w(Variable(th.zeros(n_params, opt_net.hidden_sz))) for _ in range(2)]
        for name, p in optimizee.all_named_parameters():
            cur_sz = int(np.prod(p.size()))
            # We do this so the gradients are disconnected from the graph but we still get
            # gradients from the rest
            gradients = detach_var(p.grad.view(cur_sz, 1))
            try:
                a = result_params[name].detach()
                #print("no gradient")
            except Exception as e:
                a = gradients
                #print("gradients")
            updates, new_hidden, new_cell = opt_net(
                a,
                #result_params[name].detach(),
                [h[offset:offset+cur_sz] for h in hidden_states],
                [c[offset:offset+cur_sz] for c in cell_states]
            )
            for i in range(len(new_hidden)):
                hidden_states2[i][offset:offset+cur_sz] = new_hidden[i]
                cell_states2[i][offset:offset+cur_sz] = new_cell[i]

            temp = p + updates.view(*p.size())
            print(temp, th.norm(temp))
            result_params[name] = temp#/th.norm(temp)#.clamp(0,1)#/th.norm(temp)

            result_params[name].retain_grad()

            offset += cur_sz
        optimizee = w(optimizee_class(dim))
        optimizee.load_state_dict(result_params)
        optimizee.zero_grad()
        hidden_states = [detach_var(v) for v in hidden_states2]
        cell_states = [detach_var(v) for v in cell_states2]

    return all_losses_ever

def do_test(dim, best_loss, opt_net, objective, optimizee_class, optim_it, test_data):
    loss = []
    Ntest = 1
    Hscaled = test_data
    target = objective(Hscaled)
    rnn = forward_pass(dim, opt_net, target, optimizee_class, optim_it)
    loss.append(rnn)
    loss = -np.array(loss)
    loss = loss[0]

    loss_avg_final = loss[-1]
    print(('{:<15}'+'{:<10}').format(*[ 'LSTM', 'LSTM*']))

    print(('{:<15f}'+'{:<10f}').format(*[loss_avg_final, best_loss]))
    return loss_avg_final, loss

def load_test_data():
    sparsity = 0.15
    n = 100

    try:
        test_data = th.as_tensor(np.load(f'./N{n}Samples10Sparsity{sparsity}.npy')).to(device)
    except Exception as e:
        test_data = th.zeros(n, n, device=device)
        #for i in range(10):
        upper_triangle = th.mul(th.ones(n, n).triu(diagonal=1), (th.rand(n, n) < sparsity).int().triu(diagonal=1))
        test_data[0] = upper_triangle + upper_triangle.transpose(-1, -2)
        np.save(f'./N{n}Samples10Sparsity{sparsity}.npy', test_data[0].cpu().numpy())
    print(test_data.shape)
    return test_data

def train_optimizer(dim, run_id, objective, optimizee_class, test_every=20, preproc=False, unroll=10, optim_it=100, lr=0.001, hidden_sz=20, load_net_path=None, Ntest=10, save_path=None, N_train_epochs=1000):
    N = dim
    test_data = load_test_data()

    opt_net = w(Optimizer(preproc=preproc, hidden_sz=hidden_sz))
    # summary(opt_net)
    meta_opt = optim.Adam(opt_net.parameters(), lr=lr)

    best_net = None

    best_loss, _ = do_test(dim, 0, opt_net, objective, optimizee_class=optimizee_class, optim_it=200,  test_data=test_data)

    history = {
        'test_loss':[],
        'train_loss':[]
    }

    history['test_loss'].append(best_loss)

    epoch = 0
    tstart = time()
    for epoch in range(1, N_train_epochs+1):

        # Update learning rate
        try:
            meta_opt.param_groups[0]['lr'] = np.loadtxt(save_path+'lr.txt').item()
        except:
            print('failed to load lr')
            pass

        # generate sample
        # perform one training epoch
        loss = do_fit(dim, opt_net, meta_opt, objective(test_data), optimizee_class,unroll, optim_it, should_train=True)

        history['train_loss'].append(-np.mean(loss))
        if epoch % test_every == 0:
            print('='*60)
            print('epoch',epoch)
            loss, _ = do_test(dim, best_loss, opt_net, objective, optimizee_class=optimizee_class, optim_it=200, test_data=test_data)
            print('lr = {:.2e}'.format(meta_opt.param_groups[0]['lr']))
            print('id', run_id)
            print('N = '+str(N))
            print(round((time()-tstart)/test_every,2), 'seconds per epoch')


            history['test_loss'].append(loss)

            np.save(save_path+'history.npy', history)
            if loss > best_loss:
                savename = 'epoch{}_loss={:.2f}'.format(epoch, loss)
                print('checkpoint:', save_path+savename)
                best_loss = loss
                best_net = copy.deepcopy(opt_net.state_dict())
                th.save(best_net, save_path+savename)
                best_net_path = save_path+savename

            tstart=time()

    return best_loss, best_net_path

class Objective():
    def __init__(self, adjacency_matrix, **kwargs):
        self.adjacency_matrix = adjacency_matrix
        #print(adjacency_matrix)
        self.N = adjacency_matrix.shape[0]
    def get_loss(self, x):
        #print(x)
        x = x.sigmoid()
        #print(x)
        #assert 0
        return -th.mul(th.matmul(x.reshape(self.N, 1), (1 - x.reshape(self.N, 1)).transpose(-1, -2)), self.adjacency_matrix).flatten().sum(dim=-1)

class Optimizee(MetaModule):
    def __init__(self, dim):
        super().__init__()
        self.N = dim
        self.register_buffer('theta', w(to_var(th.rand(self.N,device=device), requires_grad=True)))

    def forward(self, target):
        w = self.theta
        return target.get_loss(w)

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
            inp2 = w(th.zeros(inp.size()[0], 2))
            keep_grads = (th.abs(inp) >= self.preproc_threshold).squeeze()
            inp2[:, 0][keep_grads] = (th.log(th.abs(inp[keep_grads]) + 1e-8) / self.preproc_factor).squeeze()
            inp2[:, 1][keep_grads] = th.sign(inp[keep_grads]).squeeze()

            inp2[:, 0][~keep_grads] = -1
            inp2[:, 1][~keep_grads] = (float(np.exp(self.preproc_factor)) * inp[~keep_grads]).squeeze()
            inp = w(Variable(inp2))
        hidden0, cell0 = self.recurs(inp, (hidden[0], cell[0]))
        hidden1, cell1 = self.recurs2(hidden0, (hidden[1], cell[1]))
        return self.output(hidden1), (hidden0, hidden1), (cell0, cell1)

def get_cwd(env_name,dim):
    N = dim
    try:
        os.mkdir(env_name)
    except:
        pass
    env_name = env_name+'/N'+str(N)
    try:
        os.mkdir(env_name)
    except:
        pass

    file_list = os.listdir('./{}/'.format(env_name))
    max_exp_id = 0
    for exp_id in file_list:
        if exp_id == '.DS_Store':
            pass
        elif int(exp_id) + 1 > max_exp_id:
            max_exp_id = int(exp_id) + 1
    os.mkdir('./{}/{}/'.format(env_name, max_exp_id))
    return f"./{env_name}/{max_exp_id}/", max_exp_id

def test_optimizer(dim, optim_it, Ntest, objective, optimizee_class, preproc=False, hidden_sz=20, load_net_path=None):
    N = dim
    test_data = load_test_data()
    opt_net = w(Optimizer(preproc=preproc, hidden_sz=hidden_sz))
    if load_net_path is not None:
      opt_net.load_state_dict(th.load(load_net_path))

    _,_ = do_test(dim, 0, opt_net, objective, optimizee_class, optim_it, test_data=test_data)
    return


if __name__ == '__main__':
    USE_CUDA = False
    if USE_CUDA:
        device = th.device('cuda:0')
        th.cuda.set_device(device)
    else:
        device=th.device('cpu')

    # dim = (4,4) # (N,K)
    dim = 100 # (N,K)
    # dim = (16,16) # (N,K)
    Ntest = 10
    objective = Objective
    optimizee_class = Optimizee

    env_name = f"nets"
    save_path, run_id = get_cwd(env_name,dim)

    hidden_sz = 40
    lr = 1e-3
    np.savetxt(save_path + 'lr.txt', [lr])

    loss, path = train_optimizer(
        dim=dim,
        run_id=run_id,
        objective=objective,
        optimizee_class=optimizee_class,
        test_every=100,
        hidden_sz=hidden_sz,
        lr=lr,
        load_net_path=None,
        Ntest=10,
        save_path=save_path,
        N_train_epochs=1000)

    test_optimizer(
        dim=dim,
        optim_it=100,
        Ntest=1,
        objective=objective,
        optimizee_class=optimizee_class,
        hidden_sz=hidden_sz,
        load_net_path=path
        )
