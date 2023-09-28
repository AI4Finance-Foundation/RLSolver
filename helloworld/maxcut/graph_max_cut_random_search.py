import os
import sys
import torch as th
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.distributions import Categorical

from graph_max_cut_simulator import load_graph, SimulatorGraphMaxCut
from graph_max_cut_evaluator import Evaluator, EncoderBase64

TEN = th.Tensor


class SolverRandomLocalSearch:
    def __init__(self, simulator: SimulatorGraphMaxCut, num_nodes: int, spin_thresh: float):
        self.simulator = simulator
        self.num_nodes = num_nodes
        self.spin_thresh = spin_thresh

        self.num_sims = 0
        self.good_xs = th.tensor([])  # solution x
        self.good_vs = th.tensor([])  # objective value

    def reset(self, xs: TEN):
        self.good_xs = xs
        self.good_vs = self.simulator.calculate_obj_values(xs=xs)
        self.num_sims = xs.shape[0]

    def reset_search(self, num_sims):
        xs = th.empty((num_sims, self.num_nodes), dtype=th.bool, device=self.simulator.device)
        for sim_id in range(num_sims):
            _xs = self.simulator.generate_xs_randomly(num_sims=num_sims)
            _vs = self.simulator.calculate_obj_values(_xs)
            xs[sim_id] = _xs[_vs.argmax()]
        return xs

    def random_search(self, num_iters):
        device = self.good_xs.device

        xss = self.good_xs.unsqueeze(0).repeat(num_iters, 1, 1)
        vss = th.empty((num_iters, self.num_sims), dtype=th.long, device=device)

        spin_masks = th.rand_like(xss, dtype=th.float32).lt(self.spin_thresh)
        spin_masks[:, :, 0] = False
        xss[spin_masks] = th.logical_not(xss[spin_masks])

        xss[0] = self.good_xs
        vss[0] = self.good_vs
        for i in range(1, num_iters):
            xs = xss[i]
            vs = self.simulator.calculate_obj_values(xs)
            vss[i] = vs

        good_is = vss.argmax(dim=0)
        sim_id = th.arange(self.num_sims, device=device)
        self.good_xs = xss[good_is, sim_id]
        self.good_vs = vss[good_is, sim_id]

        return self.good_xs, self.good_vs


X_G14 = """
11Re2ycMx2zCiEhQl5ey$HyYnkUhDVE6KkPnuuhcWXwUO9Rn1fxrt_cn_g6iZFQex1YpwjD_j7KzbNN71qVekltv3QscNQJjrnrqHfsnOKWJzg9nJhZ$qh69
$X_BvBQirx$i3F
"""  # 3064, SOTA=3064
"""
11Re2ydMx2zCiEhQl5ey$PyYnkUhDVE6KkQnuuhc0XwUO9RnXfxrt_dn_g6aZFQ8x1YpwbD_j7KzaNN71qVuklpv3Q_cNQJjnnrrHjsnOKWIzg9nJxZ$qh69
$n_BHBRirx$i3F
"""  # 3064, SOTA=3064
"""
2_aNz3Of4z2pJnKaGwN30k3TEHXKoWnvhHaE77KPlU5XdsaE_UCA81PE1LvJSmbN4_Ti5Qo1IOh2Aeeu_BWNHGC6yb1GebiIAEAAkI9EdhVj2LsEiKS2BKvs
0E1qkqaJ840Jym
"""  # 3064, SOTA=3064

X_G15 = """
hzvKByHMl4xek23GZucTFBM0f530k4DymcJ5QIcqJyrAoJBkI3g5OaCIpvGsf$l4cLezTm6YOtuDvHtp38hIwUQc3tdTBWocjZj5dX$u1DEA_XX6vESoZz2W
NZpaM3tN$bzhE
"""  # 3050, SOTA=3050
"""
3K26hq3kfGx4N1zylS7HYmqf$Mwy$Hxo3FPiubjPBiBgrDirHj_LwZRpjC6l8I0GxPgN2YBvTb87oMkeCytKj5pbPy8OYyPDPIS2wOS27_onr1UUP6pZDCAV
VeSCVfyCe2Q0Kn
"""  # 3050, SOTA=3050
"""
3K26hq3kfGx4N1zylS7PYmqf$Mwy$nxo3FPiwbjPBi3ArDirHjyLwdRpjC6l8M0mxPgN2YFvTb87o4k8CStKj5fbPyCOYqRDPISIwOUA7_onr1UUn6vZDCAV
VeSCRfy8eAQ3Sn
"""  # 3050, SOTA=3050

X_G49 = """
LLLLLLLLLLLLLLLLMggggggggggggggggbLLLLLLLLLLLLLLLLggggggggggggggggfLLLLLLLLLLLLLLLLQggggggggggggggggLLLLLLLLLLLLLLLLMggg
gggggggggggggbLLLLLLLLLLLLLLLLggggggggggggggggfLLLLLLLLLLLLLLLLQggggggggggggggggLLLLLLLLLLLLLLLLMggggggggggggggggbLLLLLL
LLLLLLLLLLggggggggggggggggfLLLLLLLLLLLLLLLLQggggggggggggggggLLLLLLLLLLLLLLLLMggggggggggggggggbLLLLLLLLLLLLLLLLgggggggggg
ggggggfLLLLLLLLLLLLLLLLQggggggggggggggggLLLLLLLLLLLLLLLLMggggggggggggggggbLLLLLLLLLLLLLLLLggggggggggggggggfLLLLLLLLLLLLL
LLLQgggggggggggggggg
"""  # 6000, SOTA=6000
"""
ggggggggggggggggfLLLLLLLLLLLLLLLLQggggggggggggggggLLLLLLLLLLLLLLLLMggggggggggggggggbLLLLLLLLLLLLLLLLggggggggggggggggfLLL
LLLLLLLLLLLLLQggggggggggggggggLLLLLLLLLLLLLLLLMggggggggggggggggbLLLLLLLLLLLLLLLLggggggggggggggggfLLLLLLLLLLLLLLLLQgggggg
ggggggggggLLLLLLLLLLLLLLLLMggggggggggggggggbLLLLLLLLLLLLLLLLggggggggggggggggfLLLLLLLLLLLLLLLLQggggggggggggggggLLLLLLLLLL
LLLLLLMggggggggggggggggbLLLLLLLLLLLLLLLLggggggggggggggggfLLLLLLLLLLLLLLLLQggggggggggggggggLLLLLLLLLLLLLLLLMggggggggggggg
gggbLLLLLLLLLLLLLLLL
"""  # 6000, SOTA=6000

X_G50 = """
LLLLLLLLLLLLLLLLLLLLggggggggggggggggggggLLLLLLLLLLLLLLLLLLLLggggggggggggggggggggLLLLLLLLLLLLLLLLLLLLgggggggggggggggggggg
LLLLLLLLLLLLLLLLLLLLggggggggggggggggggggLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLggggggggggggggggggggLLLLLLLLLLLLLLLLLLLL
ggggggggggggggggggggLLLLLLLLLLLLLLLLLLLLggggggggggggggggggggLLLLLLLLLLLLLLLLLLLLggggggggggggggggggggLLLLLLLLLLLLLLLLLLLL
ggggggggggggggggggggLLLLLLLLLLLLLLLLLLLLggggggggggggggggggggLLLLLLLLLLLLLLLLLLLLggggggggggggggggggggLLLLLLLLLLLLLLLLLLLL
gggggggggggggggggggg
"""  # 5880, SOTA=5880
"""
LLLLLLLLLLLLLLLLLLLLggggggggggggggggggggLLLLLLLLLLLLLLLLLLLLggggggggggggggggggggLLLLLLLLLLLLLLLLLLLLgggggggggggggggggggg
LLLLLLLLLLLLLLLLLLLLggggggggggggggggggggLLLLLLLLLLLLLLLLLLLLggggggggggggggggggggLLLLLLLLLLLLLLLLLLLLgggggggggggggggggggg
LLLLLLLLLLLLLLLLLLLLggggggggggggggggggggLLLLLLLLLLLLLLLLLLLLggggggggggggggggggggggggggggggggggggggggLLLLLLLLLLLLLLLLLLLL
ggggggggggggggggggggLLLLLLLLLLLLLLLLLLLLggggggggggggggggggggLLLLLLLLLLLLLLLLLLLLggggggggggggggggggggLLLLLLLLLLLLLLLLLLLL
gggggggggggggggggggg
"""  # 5880, SOTA=5880

X_G22 = """
1fQjeO55DWYGVijk0yQIkpciRhwxq8h5KGiApTNoziAj3hNK3fr0s7CM6buP5hx1nB4zyDjRWzWvkXvJAU5_PAxjtr26vsA6HgN7siPrkPWuo82zsi3Pa0a7
sbdAyZoMzRQvcVUN4OOjBMclYRBPcIyq6E4OeVyVU1q8Mescg0OGQk9M$bTAfASwKbmqsQxLLJzTCUH75Y0HPsO_S3BOsJXrMcK$o$4_EsfmTq9ye2NZvOxZ
UXg36w6DeWfBgMQBtpQilUAN9ZHnUQihVviFEqa8aQUxSAU04$u5e$Y3aRmCCqidF2a9MEtFAfpqiDY1V24j2yb$$zv5ah
"""  # 13358, SOTA=13359

X_G55 = '''
2VTaG16Jc4tj7oaVSD03sSCB6m5PSNJ8UxVzXdCBqpc7Tut1q55NV9g$mLtF7XO7qqKOHZpywPOpynWtkPosv9Kwrw9m2u9JdGCYgONFuworfD4$sIrDyUBp
wXNqalEWd7ygkwISznPscKYvsWc3GdpXig4YOeq91BFIGh2C9uDx9bKJt9uG6s2Iad9WcDMxcOEWrNu8uNJh7Jbkzw_p8pCZRFa16BJBLwQ2iqyB1W1YEqE3
qUoRf1tVuyk5D9NNDtvIVE_ScKm7FQqLJgAy97PiHxAkbB34o$Io67rOnOo5KmUbrL$EamIEk4VXV2ItMf74gc1gU7JukPofUWfbo_ihFDGSbcepjrk8tcMw
9Go_XsC169PLqugqzaK6A5eISQ0xBmzCDpzy00EP4dETLEJqcGqiU5TMQh1FB0FKgA31Llc4aag1DmG6ZDbkIyZtoOd6Khf3vPsxrBCtL$XptsYZhWwOPzxZ
oqXZbbkMFZ7CEb$YWt$K1UUvoGfYt2zQUki7iZ1x4IbU1nkwtSyPGigkLTic08Hep57encZ56VMjpnoiRo_JWhYQyq_j9yf8DF4zVuK0hwRQ8EtRnOTaRlzg
OuH235tBxiv87rIqmW_pWvhhLY1Jzqtm3FRWXl1WQ_EiOfl9qF8nWEjLl33UXvO5MByzxno6LFJdXmOZnWszaHoA3fznrvQssFHjB59K_icrHvp3Ytac2gKo
ONhb0xSZRpn$Hoz0jvQx_23IFRWIlGhWIqRdvudD3ikaEMzgqAlZFEPMUYe6X$2EAxhqZupAkCCeF4eHGJ6GrMnqj$DMme6d9220sT86G1VbRzK$iz
'''  # 10297, SOTA=10296

X_G70 = """
FBtWcG5X07seVYDuz2jQ72Kzrw3Oq5n1b9BfnKBzT0KO2PXMz6E$kGArGaiNwSXw3wrMNHXvy8y5FS$nkXVzzYhuLd6M6qZnHJxNJKWuIaiOCUD_KzLIe0aR
8qgkoKC72f2kOQVHvLEb34byr2nN3xkw6aL2QkpUg0sbI3Lbzu$wBWL2P_N2FTnGfXiLIOdSeEBtQSiMvmJhA3Bh5A4hO0wE13PltnYr6BHb6MfySrZpqViR
LKF7jnVlEXUP2HQsA0KPquFn3q5tSE4kFoBDNcGYuc0LvSNANGMudSQ1oTlRGt$V1RNVFzeBR52dsRiofnBxbfFUn4TW1glAqHXZFgAdwpuqcJC7M52m06Kr
AvFPo$x_sIjPIe8AVZeSin09YFZpXHTlozHxnuqR1vyz$DZdo9jXTLRO4AHmjZ8eg5ux3GdwpTRzPYCh$DbL7hO6qLd01AS1vkm2kkJZBI4tuBWIvYIkrxjC
T5xuFFAvAHTa7$jw_shWWHaCK0ls1tM4blCfxWHwW0WGCi8Lm7VstrVxF4IciO1gazimp0dS$XyyJ8O$2VCX5w5dv4qxQ7vmOws$sqfUosEV8MURKs2HIEqK
Qou63jNkOwUKI0_tMlBfGIR65uUTTlPSGMAKQefzNFMhaSuXUFin_Lq4d987P_lv17vQ46HuWcPIZKbrQ9KX0WdG_1nHoZuhrlpMVQTVQsi4Hu07lTrJ85Kc
sJEPu_EiQ15P9M5pDq5qb4uOW7UBEK555vYyfJ6CnuJGQWDDzeXN1Wqb2VBPeW5g$qiTs5jf2flMSkyHXekznUFQd0f9f4F_xxlqRtwSwSGPYUWZx$_wC4X8
IKRNNOphRCLy3hsUege4QBtkCXn9KPbA6VP9qEtRm2bp2p0jtTRXyyHmLL_nePcA0Yfzloh2NhW7D4X4K_7ELw45fnxLfb7PvsOAEvKPIl_XKpmK2uw373Of
XigZL8ql13e9T8NdfIREuJV87LZMovQv3kVnxZZMuV1_xxTP$0L7$O57N3W8sAtHhFpYX91UcTQww2bIGe6qKnii2_l0ytTwA0vNM59U5A32Fu_x$wNj4UGM
HorT1OofXde5kTO_$A3bKagyku$VT_L_boNivYFraZ9bY5Krzzkeb28LexhCpvGtdxKqvHSS2whEwBTnNQzmTJqLj$v0opnpUPfiMvz1iRfd9oyMUvZ_XUxX
AKDL38_nuXY5jJLXEFM7RUgHNLVEv3UtbtiP2E6b$GoHI4xT21b5MsfJcaQ9jXN522zUfS4xEgsiMQ8LYBInwP6Of69izJVskmki4pMK52mOi6rx5gBlvLuB
jAaImcY2CQXXvHwCPKTFgAbn_UEbEVyn5jcTN2ed_bA2CcHtw_a8iskfqbAq4RHx6BQNHeUoiiCAHY2mgXtxOFPgz$oEVO7tbJ21Nqt4L0bJaH0rfBf8V572
3ogtqSHkWz$RcJW5igq$A9CZ8u1KQyUMcFPr5JOY7Tg2feuFTM3uouaDf4VFs21HOHqZinUjG20adiUyqIPQCUx1NJJqDUauEOjD2NENiGuwDNxBnjDkqBO1
isopF5$xuJgqYdhYu0jIh9tWeN_QTLB7aUHArPGkqRqAz4X$RHLijCvwWUcA6MicAjuha0Ec7BZLQnL5HZWMSTzKSSNs8I0sISExFxu67L4
"""  # 9568, SOTA=9595


def check_x():
    num_nodes, graph_id = 128, 1
    graph_name = f'powerlaw_{num_nodes}_ID{graph_id}'
    # x_str = "32tbLf8nvhrD7VqvobBftE"  # 363
    x_str = "0vCQYstA7KhmvWB6DQ6M8H"  # 363

    graph = load_graph(graph_name=graph_name)

    simulator = SimulatorGraphMaxCut(graph=graph)
    encoder_base64 = EncoderBase64(num_nodes=num_nodes)

    x = encoder_base64.str_to_bool(x_str)
    v = simulator.calculate_obj_values(xs=x.unsqueeze(0))[0].item()
    print(f"best_obj_value {v:8}  {x_str}")


class PolicyGNN(nn.Module):
    def __init__(self, inp_dim, mid_dim, out_dim, num_nodes):
        super().__init__()
        self.inp_dim = inp_dim
        self.mid_dim = mid_dim
        self.out_dim = out_dim
        self.num_nodes = num_nodes

        self.mat_dim = mat_dim = int(num_nodes ** 0.25)

        self.mat_enc = nn.Sequential(nn.Linear(num_nodes, mid_dim), nn.ReLU(),
                                     nn.Linear(mid_dim, mat_dim))

        self.inp_enc = nn.Sequential(nn.Linear(inp_dim + mat_dim, mid_dim), nn.ReLU(),
                                     nn.Linear(mid_dim, mid_dim))

        self.tmp_enc = nn.Sequential(nn.Linear(mid_dim + mid_dim, mid_dim), nn.ReLU(),
                                     nn.Linear(mid_dim, out_dim), nn.Softmax(dim=1))

    def forward(self, inp, mat, ids_list):
        size = inp.shape[0]
        device = inp.device
        mat = self.mat_enc(mat)  # (num_nodes, mid_dim)

        tmp0 = th.cat((inp, mat.repeat(size, 1, 1)), dim=2)
        tmp1 = self.inp_enc(tmp0)  # (size, num_nodes, inp_dim)

        env_i = th.arange(size, device=device)
        tmp2 = th.stack([tmp1[env_i[:, None], ids[None, :]].sum(dim=1) for ids in ids_list], dim=1)

        tmp3 = th.cat((tmp1, tmp2), dim=2)
        tmp4 = self.tmp_enc(tmp3)[:, :, 0]  # (size, num_nodes)
        return tmp4


def search_and_evaluate_random_search():
    gpu_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0

    graph_name, num_nodes = 'gset_14', 800
    num_sims = 2 ** 9
    num_reset = 2 ** 6

    num_iter0 = 2 ** 1
    num_iter1 = 2 ** 8
    spin_thresh = 8 / num_nodes

    num_skip = 2 ** 2  # 2 ** 4
    gap_print = 2 ** 1

    # graph_name, num_nodes = 'gset_70', 10000
    # num_sims = 2 ** 8

    if os.name == 'nt':
        num_sims = 2 ** 6
        num_iter0 = 2 ** 2
        num_iter1 = 2 ** 4
        num_reset = 2 ** 1

    device = th.device(f'cuda:{gpu_id}' if th.cuda.is_available() and gpu_id >= 0 else 'cpu')
    graph = load_graph(graph_name=graph_name)

    simulator = SimulatorGraphMaxCut(graph=graph, device=device, if_bidirectional=False)
    solver = SolverRandomLocalSearch(simulator=simulator, num_nodes=num_nodes, spin_thresh=spin_thresh)

    temp_xs = simulator.generate_xs_randomly(num_sims=1)
    temp_vs = simulator.calculate_obj_values(xs=temp_xs)
    evaluator = Evaluator(save_dir=f"{graph_name}_{gpu_id}", num_nodes=num_nodes, x=temp_xs[0], v=temp_vs[0].item())

    print("start searching")
    sim_ids = th.arange(num_sims, device=device)
    for j2 in range(1, num_reset + 1):
        prev_xs = simulator.generate_xs_randomly(num_sims)
        prev_vs = simulator.calculate_obj_values(prev_xs)

        for j1 in range(1, num_iter1 + 1):
            prev_i = prev_vs.argmax()
            prev_xs[:] = prev_xs[prev_i]
            prev_vs[:] = prev_vs[prev_i]

            '''update xs via probability, obtain logprobs for VPG'''
            xs = prev_xs.clone()
            output_tensor = th.ones_like(xs) * (1 / num_nodes)
            for _ in range(num_iter0):
                dist = Categorical(probs=output_tensor)
                sample = dist.sample(th.Size((num_sims,)))[:, 0]
                xs[sim_ids, sample] = th.logical_not(xs[sim_ids, sample])

            '''update xs via max local search'''
            solver.reset(xs)

            good_vs_list = []
            for j0 in range(2 ** 5):
                solver.random_search(2 ** 8)
                good_vs_list.append(solver.good_vs.clone())

            prev_xs = solver.good_xs
            prev_vs = solver.good_vs
            th.set_grad_enabled(True)

            if j1 > num_skip and j1 % gap_print == 0:
                good_i = solver.good_vs.argmax()
                i = j2 * num_iter1 + j1
                x = solver.good_xs[good_i]
                v = solver.good_vs[good_i].item()

                evaluator.record2(i=i, v=v, x=x)
                evaluator.logging_print(v=v)
        evaluator.plot_record()


def search_and_evaluate_reinforce():
    gpu_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0

    graph_name, num_nodes = 'gset_14', 800
    num_sims = 2 ** 9
    num_reset = 2 ** 6

    num_iter0 = 2 ** 1
    num_iter1 = 2 ** 8
    spin_thresh = 8 / num_nodes

    inp_dim = 2
    mid_dim = 32
    out_dim = 1
    learning_rate = 1e-3
    clip_grad_norm = 3.0
    if_bidirectional = False  # GPU 4567
    # if_bidirectional = True  # GPU 0123
    temperature = (gpu_id + 1) * 4

    num_skip = 2 ** 2  # 2 ** 4
    gap_plot = 2 ** 1

    if os.name == 'nt':
        num_sims = 2 ** 6
        num_iter = 2 ** 2
        num_update = 2 ** 4
        num_reset = 2 ** 1
        record_gap = 2 ** 0

    device = th.device(f'cuda:{gpu_id}' if th.cuda.is_available() and gpu_id >= 0 else 'cpu')
    graph = load_graph(graph_name=graph_name)

    simulator = SimulatorGraphMaxCut(graph=graph, device=device, if_bidirectional=if_bidirectional)
    matrix_tensor = simulator.adjacency_matrix
    indies_tensor = simulator.adjacency_indies

    solver = SolverRandomLocalSearch(simulator=simulator, num_nodes=num_nodes, spin_thresh=spin_thresh)
    opt_opti = PolicyGNN(inp_dim=inp_dim, mid_dim=mid_dim, out_dim=out_dim, num_nodes=num_nodes).to(device)
    opt_base = th.optim.Adam(opt_opti.parameters(), lr=learning_rate, maximize=True)  # todo

    temp_xs = simulator.generate_xs_randomly(num_sims=1)
    temp_vs = simulator.calculate_obj_values(xs=temp_xs)
    evaluator = Evaluator(save_dir=f"{graph_name}_{gpu_id}", num_nodes=num_nodes, x=temp_xs[0], v=temp_vs[0].item())

    sim_ids = th.arange(num_sims, device=device)
    output_tensor = None

    print("start searching")
    for j2 in range(1, num_reset + 1):
        prev_xs = simulator.generate_xs_randomly(num_sims)
        prev_vs = simulator.calculate_obj_values(prev_xs)

        for j1 in range(1, num_iter1 + 1):
            prev_i = prev_vs.argmax()
            prev_xs[:] = prev_xs[prev_i]
            prev_vs[:] = prev_vs[prev_i]

            '''update xs via probability, obtain logprobs for VPG'''
            xs = prev_xs.clone()
            logprobs = []
            for _ in range(num_iter0):
                input_tensor = build_input_tensor(xs, simulator, num_sims, num_nodes, device)
                output_tensor = opt_opti(input_tensor, matrix_tensor, indies_tensor) + temperature / num_nodes
                dist = Categorical(probs=output_tensor / output_tensor.sum(dim=1, keepdim=True))
                sample = dist.sample(th.Size((num_sims,)))[:, 0]
                xs[sim_ids, sample] = th.logical_not(xs[sim_ids, sample])

                logprob = dist.log_prob(sample)

                logprobs.append(logprob)
            logprobs = th.stack(logprobs, dim=0).mean(dim=0)

            '''update xs via max local search, obtain good_vs for VPG'''
            th.set_grad_enabled(False)
            for _ in range(num_iter0):
                input_tensor = build_input_tensor(xs, simulator, num_sims, num_nodes, device)
                output_tensor = opt_opti(input_tensor, matrix_tensor, indies_tensor)
                sample = output_tensor.argmax(dim=1)
                xs[sim_ids, sample] = th.logical_not(xs[sim_ids, sample])

            solver.reset(xs)

            good_vs_list = []
            for j0 in range(2 ** 5):
                solver.random_search(2 ** 8)
                good_vs_list.append(solver.good_vs.clone())
            scaled_vs0 = th.stack(good_vs_list, dim=0).float().mean(dim=0)
            # scaled_vs0 = solver.good_vs.float() # todo

            scaled_vs = (scaled_vs0 - scaled_vs0.mean()) / (scaled_vs0.std() + 1e-4)
            prev_xs = solver.good_xs
            prev_vs = solver.good_vs
            th.set_grad_enabled(True)

            '''update objective of optimizer'''
            obj_logprobs = ((logprobs - logprobs.mean().detach()) / (logprobs.std().detach() + 1e-6)).exp()
            obj_opti = (obj_logprobs * scaled_vs).mean()

            opt_base.zero_grad()
            obj_opti.backward()
            clip_grad_norm_(parameters=opt_base.param_groups[0]["params"], max_norm=clip_grad_norm)
            opt_base.step()

            if j1 > num_skip and j1 % gap_plot == 0:
                i = j2 * num_iter1 + j1
                v = solver.good_vs.max().item()
                evaluator.record1(i=i, v=v)
                out = output_tensor.detach().cpu().numpy()
                print(f"| i {j2:6} {j1:6}  "
                      f"output  {out.min():8.1e} {out.mean():8.1e}  {out.std():8.1e}  {out.max():8.1e}  |  "
                      f"avg_value {prev_vs.float().mean().item():8.2f}  "
                      f"max_value {prev_vs.max().item():8.2f}")

        i = j2 * num_iter1 + num_iter1
        good_i = solver.good_vs.argmax()
        x = solver.good_xs[good_i]
        v = solver.good_vs[good_i].item()
        evaluator.record2(i=i, v=v, x=x)
        evaluator.plot_record()
        evaluator.logging_print(v=v)


def build_input_tensor(xs, simulator, num_sims, num_nodes, device):
    input_tensor = th.empty((num_sims, num_nodes, 2), dtype=th.float32, device=device)
    input_tensor[:, :, 0] = xs
    _vs = simulator.calculate_obj_values(xs, if_sum=False)
    for i in range(num_sims):
        input_tensor[i, :, 1] = th.bincount(simulator.n0_ids[i], weights=_vs[i], minlength=num_nodes)
    return input_tensor


def check_solution_x():
    graph_name = 'gset_14'

    graph = load_graph(graph_name=graph_name)
    simulator = SimulatorGraphMaxCut(graph=graph)

    x_str = X_G14
    num_nodes = simulator.num_nodes
    encoder = EncoderBase64(num_nodes=num_nodes)

    x = encoder.str_to_bool(x_str)
    vs = simulator.calculate_obj_values(xs=x[None, :])
    print(f"objective value  {vs[0].item():8.2f}  solution {x_str}")


if __name__ == '__main__':
    # search_and_evaluate_random_search()
    # search_and_evaluate_reinforce()
    check_solution_x()
