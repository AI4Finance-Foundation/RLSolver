import os.path

import torch as th
import torch.nn as nn
from copy import deepcopy
import numpy as np
from torch import Tensor
# from rlsolver.rlsolver_learn2opt.np_complete_problems.envs.maxcut_env import MaxcutEnv
from envs.maxcut_env import MaxcutEnv

from utils import Opt_net
import pickle as pkl
from utils import calc_file_name
graph_node = {"14":800, "15":800, "22":2000, "49":3000, "50":3000, "55":5000, "70":10000  }


def train(num_nodes: int,
          num_envs: int,
          device: th.device,
          opt_net: Opt_net,
          optimizer: th.optim,
          episode_length: int,
          hidden_layer_size: int):
    maxcut_env = MaxcutEnv(num_nodes=num_nodes, num_envs=num_envs, device=device, episode_length=episode_length)

    maxcut_env.load_graph(f"./data/maxcut/gset_{sys.argv[1]}.npy")
    l_num = 1
    h_init = th.zeros(l_num, num_envs, hidden_layer_size).to(device)
    c_init = th.zeros(l_num, num_envs, hidden_layer_size).to(device)
    for epoch in range(100000):
        prev_h, prev_c = h_init.clone(), c_init.clone()
        loss = 0
        if (epoch + 1) % 50 == 0:
            episode_length = max(episode_length - 1, 5)
        loss_list = th.zeros(episode_length * num_envs).to(device)
        action_prev = maxcut_env.reset()
        gamma0 = 0.98
        gamma = gamma0 ** episode_length
        for step in range(episode_length):
            #print(action_prev.shape)
            #print(action_prev.reshape(num_env, N, 1).shape)
            #action, h, c = opt_net(action_prev.reshape(num_env, N, 1), prev_h, prev_c)
            action, h, c = opt_net(action_prev.reshape(num_envs, 1, num_nodes), prev_h, prev_c)

            #action = action.reshape(num_env, N)
            l = maxcut_env.calc_obj_for_one_graph(action.reshape(num_envs, num_nodes))
            loss_list[num_envs * (step):num_envs * (step + 1)] = l.detach()
            loss -= l.sum()
            #print(action_prev.shape, action.shape)
            l = maxcut_env.calc_obj_for_two_graphs_vmap(action_prev.reshape(num_envs, num_nodes), action.reshape(num_envs, num_nodes))
            loss -= 0.2 * l.sum()#max(0.05, (500-epoch) / 500) * l.sum()
            action_prev = action.detach()
            #prev_h, prev_c = h.detach(), c.detach()
            gamma /= gamma0

            if (step + 1) % 4 == 0:
                optimizer.zero_grad()
                #print(loss)
                loss.backward(retain_graph=True)
                optimizer.step()
                loss = 0
                #h, c = h_init.clone(), c_init.clone()
            prev_h, prev_c = h.detach(), c.detach()

        if epoch % 50 == 0:
            print(f"epoch:{epoch} | train:",  loss_list.max().item())
            h, c = h_init, c_init
            # print(h_init.mean(), c_init.mean())
            loss = 0
            #loss_list = []
            loss_list = th.zeros(episode_length * num_envs * 2).to(device)
            action = maxcut_env.reset()
            sol = th.zeros(episode_length * num_envs * 2, num_nodes).to(device)
            for step in range(episode_length * 2):
                action, h, c = opt_net(action.detach().reshape(num_envs, 1, num_nodes), h, c)
                action = action.reshape(num_envs, num_nodes)
                a = action.detach()
                a = (a>0.5).to(th.float32)
                # print(a)
                # assert 0
                l = maxcut_env.calc_obj_for_one_graph(a)
                loss_list[num_envs * (step):num_envs * (step + 1)] = l.detach()
                sol[num_envs * step: num_envs * (step + 1)] = a.detach()
                #if (step + 6) % 2 == 0:
                    #optimizer.zero_grad()
                    #loss.backward()
                    #optimizer.step()
                    #loss = 0
                    #h, c = h_init.clone(), c_init.clone()
            val, ind = loss_list.max(dim=-1)
            dir = "./result"
            front = "gset"
            end = "."
            id2 = sys.argv[1]
            file_name = dir + "/" + calc_file_name(front, id2, int(val.item()), end)
            # print("val: ", int(val.item()))
            # print("file_name: ", file_name)
            if not os.path.exists(dir):
                os.makedirs(dir)
            with open(file_name, 'wb') as f:
                # remove_files_less_equal_new_val(dir, front, end, int(val.item()))
                # print("sol[ind]: ", sol[ind])
                pkl.dump(sol[ind], f)
            print(f"epoch:{epoch} | test :",  loss_list.max().item())




if __name__ == "__main__":
    import sys
    num_nodes = graph_node[sys.argv[1]]
    hidden_layer_size = 3000
    learning_rate = 3e-5
    num_envs = 128
    episode_length = 30
    gpu_id = int(sys.argv[2])
    device = th.device(f"cuda:{gpu_id}" if (th.cuda.is_available() and (gpu_id >= 0)) else "cpu")
    th.manual_seed(10)
    opt_net = Opt_net(num_nodes, hidden_layer_size).to(device)
    optimizer = th.optim.Adam(opt_net.parameters(), lr=learning_rate)

    train(num_nodes, num_envs, device, opt_net, optimizer, episode_length, hidden_layer_size)

