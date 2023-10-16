from local_search import *
from torch.nn.utils import clip_grad_norm_


class PolicyGNN(nn.Module):
    def __init__(self, inp_dim, mid_dim, out_dim):
        super().__init__()
        self.inp_dim = inp_dim
        self.mid_dim = mid_dim
        self.out_dim = out_dim

        self.inp_enc = nn.Sequential(nn.Linear(inp_dim, mid_dim), nn.ReLU(), nn.LayerNorm(mid_dim),
                                     nn.Linear(mid_dim, mid_dim))

        self.tmp_enc = nn.Sequential(nn.Linear(mid_dim + mid_dim, mid_dim), nn.ReLU(), nn.LayerNorm(mid_dim),
                                     nn.Linear(mid_dim, out_dim), )
        self.soft_max = nn.Softmax(dim=1)

    def forward(self, inp, ids_list):
        num_size, num_nodes, num_dim = inp.shape
        device = inp.device

        tmp1 = th.empty((num_size, num_nodes, self.mid_dim), dtype=th.float32, device=device)
        for node0 in range(num_nodes):
            tmp1[:, node0] = self.inp_enc(inp[:, node0])

        env_i = th.arange(inp.shape[0], device=inp.device)
        tmp2 = th.empty((num_size, num_nodes, self.mid_dim), dtype=th.float32, device=device)
        for node0, node1s in enumerate(ids_list):
            tmp2[:, node0, :] = tmp1[env_i[:, None], node1s[None, :]].mean(dim=1)

        tmp3 = th.cat((tmp1, tmp2), dim=2)
        out = th.empty((num_size, num_nodes, self.out_dim), dtype=th.float32, device=device)
        for node0 in range(num_nodes):
            out[:, node0] = self.tmp_enc(tmp3[:, node0])
        return self.soft_max(out.squeeze(2))


def map_to_power_of_two(x):
    n = 0
    while 2 ** n <= x:
        n += 1
    return 2 ** (n - 1)


def train_embedding_net(adjacency_matrix, num_embed: int, num_epoch: int = 2 ** 10):
    num_nodes = adjacency_matrix.shape[0]
    assert num_nodes == adjacency_matrix.shape[1]
    lr = 4e-3

    '''network'''
    encoder = nn.Sequential(nn.Linear(num_nodes, num_nodes), nn.ReLU(), nn.BatchNorm1d(num_nodes),
                            nn.Linear(num_nodes, num_embed), nn.ReLU(), nn.BatchNorm1d(num_embed), )
    decoder = nn.Sequential(nn.Linear(num_embed, num_nodes), nn.ReLU(), nn.BatchNorm1d(num_nodes),
                            nn.Linear(num_nodes, num_nodes), )
    params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = th.optim.Adam(params, lr=lr)
    criterion = nn.MSELoss()

    '''train loop'''
    device = adjacency_matrix.device
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    inp = adjacency_matrix

    optimizer.param_groups[0]['lr'] = lr
    for i in range(1, num_epoch + 1):
        mid = encoder(inp)
        out = decoder(mid)
        obj = criterion(inp, out)

        optimizer.zero_grad()
        obj.backward()
        clip_grad_norm_(params, 1)
        optimizer.step()

        if not (i % 512):
            print(
                f"|lr {lr:9.3f}  {i:8}  obj*num_nodes {obj * num_nodes:9.3f}  mid {mid.min():9.2f} < {mid.max():9.2f}")
    del decoder

    encoder.eval()
    return encoder


def build_input_tensor(xs, sim: SimulatorGraphMaxCut, inp_dim, feature):
    num_sims, num_nodes = xs.shape

    inp = th.empty((num_sims, num_nodes, inp_dim), dtype=th.float32, device=xs.device)
    inp[:, :, 0] = xs
    inp[:, :, 1] = sim.calculate_obj_values_for_loop(xs, if_sum=False)
    inp[:, :, 2] = sim.n0_num_n1
    inp[:, :, 3:] = feature
    return inp


def check_gnn():
    gpu_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    graph_name, num_nodes = 'gset_14', 800

    inp_dim = map_to_power_of_two(num_nodes) // 2
    mid_dim = 64
    out_dim = 1
    num_embed = inp_dim - 3  # map_to_power_of_two(num_nodes) // 2

    num_sims = 8

    device = th.device(f'cuda:{gpu_id}' if th.cuda.is_available() and gpu_id >= 0 else 'cpu')
    graph = load_graph(graph_name=graph_name)

    sim = SimulatorGraphMaxCut(graph=graph, device=device, if_bidirectional=True)

    '''get adjacency_feature'''
    embed_net_path = f"{graph_name}_embedding_net.pth"
    if os.path.exists(embed_net_path):
        embed_net = th.load(embed_net_path, map_location=device)
        embed_net.eval()
    else:
        embed_net = train_embedding_net(adjacency_matrix=sim.adjacency_matrix, num_embed=num_embed, num_epoch=2 ** 14)
        th.save(embed_net, embed_net_path)
    sim.adjacency_feature = embed_net(sim.adjacency_matrix).detach()

    '''build net'''
    net = PolicyGNN(inp_dim=inp_dim, mid_dim=mid_dim, out_dim=out_dim).to(device)
    print(f"num_nodes {num_nodes}  num_node_features {sim.adjacency_feature.shape[1]}")

    xs = sim.generate_xs_randomly(num_sims=num_sims)
    inp = build_input_tensor(xs=xs, sim=sim, inp_dim=inp_dim, feature=sim.adjacency_feature)
    out = net(inp, sim.adjacency_indies)
    print(out.shape)


def search_and_evaluate_reinforce():
    gpu_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    graph_name, num_nodes = 'gset_14', 800

    inp_dim = map_to_power_of_two(num_nodes) // 2
    mid_dim = 128
    out_dim = 1
    num_embed = inp_dim - 3  # map_to_power_of_two(num_nodes) // 2

    num_sims = 2 ** 9
    num_reset = 2 ** 0
    num_iter1 = 2 ** 6
    num_iter0 = 2 ** 1

    if os.name == 'nt':
        num_sims = 2 ** 3
        num_reset = 2 ** 1
        num_iter1 = 2 ** 4
        num_iter0 = 2 ** 2

    num_skip = 2 ** 0
    gap_print = 2 ** 0

    '''build simulator'''
    device = th.device(f'cuda:{gpu_id}' if th.cuda.is_available() and gpu_id >= 0 else 'cpu')
    graph = load_graph(graph_name=graph_name)

    sim = SimulatorGraphMaxCut(graph=graph, device=device, if_bidirectional=True)

    '''get adjacency_feature'''
    embed_net_path = f"{graph_name}_embedding_net.pth"
    if os.path.exists(embed_net_path):
        embed_net = th.load(embed_net_path, map_location=device)
        embed_net.eval()
    else:
        embed_net = train_embedding_net(adjacency_matrix=sim.adjacency_matrix, num_embed=num_embed, num_epoch=2 ** 14)
        th.save(embed_net, embed_net_path)
    sim.adjacency_feature = embed_net(sim.adjacency_matrix).detach()

    '''build net'''
    net = PolicyGNN(inp_dim=inp_dim, mid_dim=mid_dim, out_dim=out_dim).to(device)
    net_params = list(net.parameters())
    print(f"num_nodes {num_nodes}  num_node_features {sim.adjacency_feature.shape[1]}")

    solver = SolverLocalSearch(simulator=sim, num_nodes=num_nodes)
    optimizer = th.optim.Adam(net_params, lr=2e-3, maximize=True)

    '''evaluator'''
    temp_xs = sim.generate_xs_randomly(num_sims=1)
    temp_vs = sim.calculate_obj_values(temp_xs)
    evaluator = Evaluator(save_dir=f"{graph_name}_{gpu_id}", num_nodes=num_nodes, x=temp_xs[0], v=temp_vs[0].item())
    del temp_xs, temp_vs

    print("start searching")
    sim_ids = th.arange(num_sims, device=device)
    for j2 in range(1, num_reset + 1):
        prev_xs = sim.generate_xs_randomly(num_sims)
        prev_vs = sim.calculate_obj_values(prev_xs)

        for j1 in range(1, num_iter1 + 1):
            prev_i = prev_vs.argmax()
            xs = prev_xs[prev_i:prev_i + 1].repeat(num_sims, 1)
            vs = prev_vs[prev_i:prev_i + 1].repeat(num_sims)

            '''update xs via probability, obtain logprobs for VPG'''
            logprobs = th.empty((num_sims, num_iter0), dtype=th.float32, device=device)
            for i0 in range(num_iter0):
                if i0 == 0:
                    output_tensor = th.ones((num_sims, num_nodes), dtype=th.float32, device=device) / num_nodes
                else:
                    input_tensor = build_input_tensor(xs=xs.clone().detach(), sim=sim, inp_dim=inp_dim,
                                                      feature=sim.adjacency_feature.detach())
                    output_tensor = net(input_tensor, sim.adjacency_indies)
                dist = Categorical(probs=output_tensor)
                sample = dist.sample(th.Size((1,)))[0]
                xs[sim_ids, sample] = th.logical_not(xs[sim_ids, sample])

                logprobs[:, i0] = dist.log_prob(sample)
            logprobs = logprobs.mean(dim=1)
            logprobs = logprobs - logprobs.mean()

            '''update xs via max local search'''
            solver.reset(xs)
            solver.random_search(num_iters=2 ** 6, num_spin=8, noise_std=0.2)
            advantage_value = (solver.good_vs - vs).detach()

            objective = (logprobs.exp() * advantage_value).mean()

            optimizer.zero_grad()
            objective.backward()
            clip_grad_norm_(net.parameters(), 1)
            optimizer.step()

            prev_xs = solver.good_xs.clone()
            prev_vs = solver.good_vs.clone()

            if j1 > num_skip and j1 % gap_print == 0:
                good_i = solver.good_vs.argmax()
                i = j2 * num_iter1 + j1
                x = solver.good_xs[good_i]
                v = solver.good_vs[good_i].item()

                evaluator.record2(i=i, v=v, x=x)
                evaluator.logging_print(v=v)


if __name__ == '__main__':
    search_and_evaluate_reinforce()
    # check_gnn()
