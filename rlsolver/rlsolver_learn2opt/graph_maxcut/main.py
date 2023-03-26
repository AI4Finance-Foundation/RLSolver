import torch as th
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import os
import copy
import numpy as np
from utils import *
import sys
from copy import deepcopy
import time
def roll_out(N, opt_net, optimizer, obj_fun, opt_variable_class, look_ahead_K, optim_it, opt_variable):
    opt_variable = opt_variable
    # opt_variable = cpu_to_gpu(opt_variable_class(N, device))
    n_params = 0
    for name, p in opt_variable.all_named_parameters():
        n_params += int(np.prod(p.size()))
    hidden_states = [cpu_to_gpu(Variable(th.randn(n_params, opt_net.hidden_sz))) for _ in range(2)]
    cell_states = [cpu_to_gpu(Variable(th.randn(n_params, opt_net.hidden_sz))) for _ in range(2)]
    loss_H_ever = []
    optimizer.zero_grad()
    loss_H = 0
    for iteration in range(1, optim_it + 1):
        loss, l_list = opt_variable(obj_fun)
        loss_H += loss
        loss_H_ever = loss_H_ever+l_list
        loss.backward(retain_graph=True)
        offset = 0
        result_params = {}
        hidden_states2 = [cpu_to_gpu(Variable(th.zeros(n_params, opt_net.hidden_sz))) for _ in range(2)]
        cell_states2 = [cpu_to_gpu(Variable(th.zeros(n_params, opt_net.hidden_sz))) for _ in range(2)]
        for name, p in opt_variable.all_named_parameters():
            cur_sz = int(np.prod(p.size()))
            gradients = detach_var(p.grad.view(cur_sz, 1))
            updates, new_hidden, new_cell = opt_net(gradients, [h[offset:offset+cur_sz] for h in hidden_states], [c[offset:offset+cur_sz] for c in cell_states])
            for i in range(len(new_hidden)):
                hidden_states2[i][offset:offset+cur_sz] = new_hidden[i]
                cell_states2[i][offset:offset+cur_sz] = new_cell[i]
            temp = p + updates.view(*p.size())
            if iteration==1:
               temp += cpu_to_gpu(th.rand_like(temp)) * 0.1
            result_params[name] = temp
            result_params[name].retain_grad()
            offset += cur_sz
        if iteration % look_ahead_K == 0:
            optimizer.zero_grad()
            loss_H.backward()
            optimizer.step()
            loss_H = 0
            opt_variable = cpu_to_gpu(opt_variable_class(N, device))
            opt_variable.load_state_dict(result_params)
            opt_variable.zero_grad()
            hidden_states = [detach_var(v) for v in hidden_states2]
            cell_states = [detach_var(v) for v in cell_states2]
        else:
            for name, p in opt_variable.all_named_parameters():
                rsetattr(opt_variable, name, result_params[name])
            hidden_states = hidden_states2
            cell_states = cell_states2
    return loss_H_ever, opt_variable

def do_test(N, best_loss, best_train_loss, opt_net, obj_fun, opt_variable_class, optim_it, test_data, look_ahead_K, loss):
    loss = []
    l = loss
    target = obj_fun(test_data)
    test_loss = 0
    test_loss = forward_pass(N, opt_net, target, opt_variable_class, optim_it=1, device=device)
    loss.append(test_loss)
    loss = np.array(loss)[0] * -1
    loss_avg_final = loss[-1]
    print(('{:<25}').format(*[ 'sol*']))
    print(('{:<25.2f}').format(*[max(best_loss, best_train_loss)]))
    wandb.log({"best":round(best_train_loss, 0), "K":look_ahead_K, "training_loss":np.mean(l)})
    return loss_avg_final, loss

def train_opt_net(N, sparsity, opt_net, optimizer, run_id, obj_fun, opt_variable_class, test_every=1, preproc=False, look_ahead_K=10, optim_it=1000, lr=0.001, hidden_sz=20, load_net_path=None, save_path=None, N_train_epochs=1000):
    start_time = time.time()
    test_data = load_test_data(N, sparsity, device)
    opt_variable = cpu_to_gpu(opt_variable_class(N, device))
    best_net = None
    #print(test_data)
    #print(type(test_data))
    #assert 0
    loss = np.array([0])
    best_loss, _ = do_test(N, 0, 0, opt_net, obj_fun, opt_variable_class=opt_variable_class, optim_it=200,  test_data=test_data, look_ahead_K=look_ahead_K, loss=loss)
    history = { 'test_loss':[], 'train_loss':[] }
    epoch = 0
    best_train_loss = 0
    for epoch in range(1, N_train_epochs+1):
        running_time = time.time() - start_time
        print(f"running_time: {running_time: .2f} seconds")
        t = deepcopy(test_data)
        #t[epoch*1000:, epoch*1000:] = 0
        if epoch % 10 == 0:
            look_ahead_K = max(look_ahead_K-1, 1)
        #look_ahead_K = look_ahead_K# max(1, (1 - epoch / N_train_epochs)*look_ahead_K )
        loss, opt_variable = roll_out(N, opt_net, optimizer, obj_fun(t), opt_variable_class,look_ahead_K, optim_it, opt_variable = opt_variable)
        history['train_loss'].append(-np.mean(loss))
        best_train_loss = -np.min(loss) if -np.min(loss) > best_train_loss else best_train_loss

        if epoch % test_every == 0:
            print('='*60)
            print('epoch',epoch)
            loss, _ = do_test(N, best_loss, best_train_loss, opt_net, obj_fun, opt_variable_class=opt_variable_class, optim_it=200, test_data=test_data, look_ahead_K=look_ahead_K, loss=loss)
            history['test_loss'].append(loss)
            np.save(save_path+'history.npy', history)
            if loss > best_loss:
                savename = f'epoch{epoch}_loss={loss:.2f}'
                best_loss = loss
                best_net = copy.deepcopy(opt_net.state_dict())
                with open("best_opt_variable.pkl", 'wb') as f:
                    import pickle as pkl
                    pkl.dump(opt_variable, f)
                th.save(best_net, save_path+f'epoch{epoch}_loss={loss:.2f}')
                best_net_path = save_path+savename
    return best_loss, best_net_path

if __name__ == '__main__':
    N = int(sys.argv[1])
    sparsity= float(sys.argv[2])
    gpu_id = int(sys.argv[3])
    device = th.device(f"cuda:{gpu_id}" if (th.cuda.is_available() and (gpu_id >= 0)) else "cpu")
    look_ahead_K = 5
    N = int(sys.argv[1])
    sparsity= float(sys.argv[2])
    gpu_id = int(sys.argv[3])
    device = th.device(f"cuda:{gpu_id}" if (th.cuda.is_available() and (gpu_id >= 0)) else "cpu")
    look_ahead_K = 20
    obj_fun = Obj_fun
    opt_variable_class = Opt_variable
    folder_name = "opt_nets"
    save_path, run_id = get_cwd(folder_name, N)
    hidden_sz = 40
    lr = 1e-3
    import wandb
    config = {

    }
    if_wandb = True
    if if_wandb:
        wandb.init(
            project=f'graph_maxcut',
            entity="sxun",
            sync_tensorboard=True,
            config=config,
            name=f"G14_N800",
            monitor_gym=True,
            save_code=True,
    )

    preproc=False
    opt_net = cpu_to_gpu(Opt_net(preproc=preproc, hidden_sz=hidden_sz))
    optimizer = optim.Adam(opt_net.parameters(), lr=lr)
    loss, path = train_opt_net(N=N, sparsity=sparsity, opt_net=opt_net, optimizer=optimizer, run_id=run_id, obj_fun=obj_fun, opt_variable_class=opt_variable_class,look_ahead_K=look_ahead_K,
        test_every=1, hidden_sz=hidden_sz, lr=lr, load_net_path=None, save_path=save_path, N_train_epochs=3000)
