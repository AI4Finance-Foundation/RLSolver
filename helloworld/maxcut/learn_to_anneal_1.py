import os.path

import torch as th
import torch.nn as nn
from copy import deepcopy
import numpy as np
from torch import Tensor
# from rlsolver.rlsolver_learn2opt.np_complete_problems.env.maxcut_env import MCSim
from mcmc_sim.mcmc_sim import MCMCSim

from utils import Opt_net
import pickle as pkl
from utils import calc_file_name
from utils import read_txt_as_networkx_graph
from utils import write_result
# graph_node = {"14":800, "15":800, "22":2000, "49":3000, "50":3000, "55":5000, "70":10000  }


def train(
          filename: str,
          num_nodes: int,
          num_envs: int,
          device: th.device,
          opt_net: Opt_net,
          optimizer: th.optim,
          episode_length: int,
          hidden_layer_size: int):
    mcmc_sim = MCMCSim(filename=filename, num_samples=num_envs, device=device, episode_length=episode_length)


    num_layers = 1
    init_hidden = th.zeros(num_layers, num_envs, hidden_layer_size).to(device)
    init_cell = th.zeros(num_layers, num_envs, hidden_layer_size).to(device)
    for epoch in range(100000):
        prev_hidden, prev_cell = init_hidden.clone(), init_cell.clone()
        loss = 0
        if (epoch + 1) % 500 == 0:
            episode_length = max(episode_length - 1, 5)
        loss_list = th.zeros(episode_length * num_envs).to(device)
        x_prev = mcmc_sim.init(True)
        gamma0 = 0.98
        gamma = gamma0 ** episode_length
        for step in range(episode_length):
            #print(x_prev.shape)
            #print(x_prev.reshape(num_env, N, 1).shape)
            #x, h, c = opt_net(x_prev.reshape(num_env, N, 1), prev_h, prev_c)
            x, hidden, cell = opt_net(x_prev.reshape(num_envs, 1, num_nodes), prev_hidden, prev_cell)

            #x = x.reshape(num_env, N)
            l = mcmc_sim.obj(x.reshape(num_envs, num_nodes))
            loss_list[num_envs * (step):num_envs * (step + 1)] = l.detach()
            loss -= l.sum()
            #print(x_prev.shape, x.shape)
            l = mcmc_sim.calc_obj_for_two_graphs_vmap(x_prev.reshape(num_envs, num_nodes), x.reshape(num_envs, num_nodes))
            loss -= 0.2 * l.sum()#max(0.05, (500-epoch) / 500) * l.sum()
            x_prev = x.detach()
            #prev_h, prev_c = h.detach(), c.detach()
            gamma /= gamma0

            if (step + 1) % 4 == 0:
                optimizer.zero_grad()
                #print(loss)
                loss.backward(retain_graph=True)
                optimizer.step()
                loss = 0
                #h, c = h_init.clone(), c_init.clone()
            prev_hidden, prev_cell = hidden.detach(), cell.detach()

        if epoch % 50 == 0:
            print(f"epoch:{epoch} | train:",  loss_list.max().item())
            hidden, cell = init_hidden, init_cell
            # print(h_init.mean(), c_init.mean())
            loss = 0
            #loss_list = []
            loss_list = th.zeros(episode_length * num_envs * 2).to(device)
            x = mcmc_sim.init(True)
            xs = th.zeros(episode_length * num_envs * 2, num_nodes).to(device)
            for step in range(episode_length * 2):
                x, hidden, cell = opt_net(x.detach().reshape(num_envs, 1, num_nodes), hidden, cell)
                x = x.reshape(num_envs, num_nodes)
                x2 = x.detach()
                x2 = (x2>0.5).to(th.float32)
                # print(a)
                # assert 0
                l = mcmc_sim.obj(x2)
                loss_list[num_envs * (step):num_envs * (step + 1)] = l.detach()
                xs[num_envs * step: num_envs * (step + 1)] = x2.detach()
                #if (step + 6) % 2 == 0:
                    #optimizer.zero_grad()
                    #loss.backward()
                    #optimizer.step()
                    #loss = 0
                    #h, c = h_init.clone(), c_init.clone()
            val, idx = loss_list.max(dim=-1)
            file_name = filename.replace("data", "result")
            file_name = file_name.replace(".txt", "_" + str(int(val.item())) + ".txt")
            write_result(xs[idx], file_name)
            mcmc_sim.best_x = xs[idx]
            print(f"epoch:{epoch} | test :",  loss_list.max().item())



if __name__ == "__main__":
    import sys

    filename = 'data/gset/gset_14.txt'
    gpu_id = 5
    graph = read_txt_as_networkx_graph(filename)
    num_nodes = graph.number_of_nodes()
    hidden_layer_size = 4000
    learning_rate = 2e-5
    num_samples = 20
    episode_length = 30

    device = th.device(f"cuda:{gpu_id}" if (th.cuda.is_available() and (gpu_id >= 0)) else "cpu")
    th.manual_seed(7)
    opt_net = Opt_net(num_nodes, hidden_layer_size).to(device)
    optimizer = th.optim.Adam(opt_net.parameters(), lr=learning_rate)

    train(filename, num_nodes, num_samples, device, opt_net, optimizer, episode_length, hidden_layer_size)

