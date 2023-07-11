import sys
import time
import torch as th
import torch.nn as nn

TEN = th.Tensor
ThetaGset_14 = """
BYazUs1eWYN9rkM72XZVmSlFWkztsLbW1oQvepFN1ncCSwepfwZqfPkX94Wj72s83xu3ogvwW3OqwCDhlUBqQIVr5PWKK@m9oxFFj4ggJ2Iwz3cQQCWYYhsS
pl!bxDGo7kLtG3
"""  # 3025
"""
BHak9ztUcX9XcYhD!w61dHdiNl8SPEc969fwLIK3NChqc1J6uDC5CSuQSgzcifjw6JjyENQ!G0UMBGbVqefGnzhtbe4P3ShJXTB!Zi1Y6c2S@qLv4RTkm7bI
3Xay!wxOG70GyL
"""  # 3017

ThetaGset_70 = """
Mk@CGftMyn1hBJtZJ8JboOSf@DJDxG2sSqM68H75c!8TqFWAHT4TaxU2wU8RMCTsD9aP3nUO8EN6Q6nZsiooW!N7iRQvYzw6co8iAdmbDBthJRehiqXj1gzZ
SwDSCnO4aQcEJtNOAuWNUu1Vjw8B3xP7egjg08!5TtD5A4L@mJmzU@JOU5JXWt5MqRJw8NqRavCN8aJxQXvsaeOKs9MloHShxQc5MR0D4dB9nXE8@By3FwkJ
HTOu5du!gvbRGQhOjjw@W59zsAc0l7fxApcKU8NRaCBrOYuzoD8qebH0cvSWWE9vc9JJCTv6VrJTXUdLcKKrxJpQEbMdIGcZeQoDG07Wjsz!3tmM3PIpDx5C
vJcyuu03bPfoqHrTv82HNIxrp9b4NeIKl!rBvkaBN!4ZNgXkoqc7BeSTar0nglBCBsFh6L1HdW@nk6eOQNdTQuOwXw2kquu1vlywGiQxNaGwUZ@6xExrZzyd
5wnnBgIbnlJstOjmX17Dn0n74D!hFAVcBelJKbnSUz0KwxCkfDDobKG1tb9QbF8@W57HI@NSStS4Ie2N3T9zVjOSm5Z3@aGrhQM7mYF@oe5H5JnKjcCbNPPd
X0x33QOJjOhdZyE@JdFi3Fhm0uIX92cj0ACAiNO8Ji7LiDpT9bQi5p1Cqjan@8rGQ@PVKTWmwR6NOJLGU@cOdbOqhqnpEEX6HIC4AtEmd2wwiLp6rY5f4xCn
lOBgzoOzlGjPhib2HyVGZTTxCi3wdA!dBbDeWjBMyB9Ph@VrkdceNabC1MJuoFmB6KfkixB7TZY5D@fxUlkC1KsyNp1NDhv9ogW@PlhRFP6kACHmTQbNH@Y@
0LTzBSUaF7votcNavjP4cqh4bD5iHR7IaZpXY4as91Bb3usfxCAjSK4KmyFbLW8dtxcaLHynnsw@RkKylb8BUnqtp0ifyJ7BB1!qUPuTxak2BV9Bb0DBtzpG
YxLiUKiaG4qO7D88eD34HXSKJYAYzmSSYo@TosuLc1EXao1MqupvsKAih4PaiWytP@LsQ6piJ3i9AMMhoOc0i72wWHGl!ttbD55emBXC3G0bd0qaASmj3P9H
qKJ9nNe2fnRCkTPwEODN8IDvjPQzEDUD6FfJqgSjzUmjJqIbug9N7R5Hukc9aTZVCV9b0t!JYny9yNeLaRyRgalVvpQ8AOo3Sl17X@BQ2c3d8R4MMNEbh8Cm
WwTkNp1PpEXjn7REugoJkhaDdTFwHnSTGUvxwN4RZom9H88PdaQZ!8GS27rr0MQmPezJLMQOq5EtYPTH8YI@@JzcaGG@MqxC1nHREtQkshM@gw5lX9MCTYXy
6CyoAHKjeOfu3n2J@ZFKAXEAbk8AiTk5JeRv6js0CRYIANMFQr0rE3T!6Pb51TREuoDZx6tH6LLeR2z94ki!Twiif!eRMsmfoLe6YiQsGmk5agRlysPUILjC
OdiFbtRRd!8VniBOE56X2SUlZIBwl6L9qFFEqeGIAEVvA0Ls81aHDoP!7Fs5PsSpYN2VuF2n7wl!LbO0Jpw1SPuXU@ntGCZeEfSL1uxwALYR6pfHqqEDZ@jt
G3WiENwEEjBv3ROzleajhLLaGDz3hSI5FUWco8po3YDaN3Pv9QUiXY!8G7fci7HlvnV8m1pGt8UGdeyzt85YEjE0wyWQl2oIkjr9a@Jp5NH
"""  # 5124

"""Graph Max Cut Env"""

from env.maxcut_env2 import GraphMaxCutEnv





def check_env():
    th.manual_seed(0)
    num_envs = 6

    # env = GraphMaxCutEnv(graph_key='gset_14')
    env = GraphMaxCutEnv(graph_key='gset_70')

    probs = env.get_rand_probs(num_envs=num_envs)
    print(env.get_objective(probs))
    print(env.get_objectives(probs))

    for thresh in th.linspace(0, 1, 8):
        objs = env.get_objectives(p0s=env.convert_prob_to_bool(probs, thresh))
        print(f"{thresh.item():6.3f}  {objs.numpy()}")

    best_score = -th.inf
    best_theta = None
    for _ in range(8):
        probs = env.get_rand_probs(num_envs=num_envs)
        thetas = env.convert_prob_to_bool(probs)
        scores = env.get_scores(thetas)
        max_score, max_id = th.max(scores, dim=0)
        if max_score > best_score:
            best_score = max_score
            best_theta = thetas[max_id]
            print(f"best_score {best_score}  best_theta \n{env.node_prob_bool_to_str(best_theta)}")

    print()
    print(f"best_score {best_score}  best_theta \n{env.node_prob_bool_to_str(best_theta)}")


def check_theta():
    env = GraphMaxCutEnv(graph_key='gset_14')
    best_theta = env.node_prob_str_to_bool(ThetaGset_14)
    # env = GraphMaxCutEnv(graph_key='gset_70')
    # best_theta = env.node_prob_str_to_bool(ThetaGset_70)

    print(best_theta.shape)
    best_score = env.get_scores(best_theta.unsqueeze(0)).squeeze(0)
    print(f"score {best_score}  theta \n{env.node_prob_bool_to_str(best_theta)}")




def convert_between_str_and_bool():
    gpu_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    num_envs = 1
    graph_key = 'gset_14'
    # graph_key = 'gset_70'

    env = GraphMaxCutEnv(graph_key=graph_key, gpu_id=gpu_id)

    x_prob = env.get_rand_probs(num_envs=num_envs)[0]
    x_bool = env.convert_prob_to_bool(x_prob)

    x_str = env.node_prob_bool_to_str(x_bool)
    print(x_str)
    x_bool = env.node_prob_str_to_bool(x_str)

    assert all(x_bool == env.convert_prob_to_bool(x_prob))


"""Optimize with sequences"""


class OptimizerOpti(nn.Module):
    def __init__(self, inp_dim, mid_dim, out_dim, num_layers):
        super().__init__()
        self.inp_dim = inp_dim
        self.mid_dim = mid_dim
        self.out_dim = out_dim
        self.num_layers = num_layers

        self.rnn0 = nn.LSTM(inp_dim, mid_dim, num_layers=num_layers)
        self.mlp0 = nn.Linear(mid_dim, out_dim)
        self.rnn1 = nn.LSTM(1, 8, num_layers=num_layers)
        self.mlp1 = nn.Linear(8, 1)

    def forward(self, inp, hid0=None, hid1=None):
        tmp0, hid0 = self.rnn0(inp, hid0)
        out0 = self.mlp0(tmp0)

        d0, d1, d2 = inp.shape
        inp1 = inp.reshape(d0, d1 * d2, 1)
        tmp1, hid1 = self.rnn1(inp1, hid1)
        out1 = self.mlp1(tmp1).reshape(d0, d1, d2)

        out = out0 + out1
        return out, hid0, hid1


def train_optimizer_level1_update_theta_by_grad():
    gpu_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    num_envs = 2 ** 6
    graph_key = 'gset_14'
    # graph_key = 'gset_70'

    '''hyper-parameters'''
    lr = 1e-3
    opt_num = 2 ** 12
    eval_gap = 2 ** 6

    '''init task'''
    env = GraphMaxCutEnv(graph_key=graph_key, gpu_id=gpu_id)
    probs = env.get_rand_probs(num_envs=num_envs)
    probs.requires_grad_(True)

    '''loop'''
    best_score = -th.inf
    best_theta = probs
    start_time = time.time()
    for i in range(1, opt_num + 1):
        obj = env.get_objectives(probs).mean()
        obj.backward()

        grads = probs.grad.data
        probs.data.add_(-lr * grads).clip_(0, 1)

        if i % eval_gap == 0:
            thetas = env.convert_prob_to_bool(probs)
            scores = env.get_scores(thetas)
            used_time = time.time() - start_time
            print(f"|{used_time:9.0f}  {i:6}  {obj.item():9.3f}  {scores.max().item():9.0f}")

            max_score, max_id = th.max(scores, dim=0)
            if max_score > best_score:
                best_score = max_score
                best_theta = thetas[max_id]
                print(f"best_score {best_score}  best_theta \n{env.node_prob_bool_to_str(best_theta)}")

    print()
    print(f"best_score {best_score}  best_theta \n{env.node_prob_bool_to_str(best_theta)}")


def train_optimizer_level2_update_theta_by_adam():
    gpu_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    num_envs = 2 ** 4
    graph_key = 'gset_14'
    # graph_key = 'gset_70'

    '''hyper-parameters'''
    lr = 1e-3
    opt_num = 2 ** 12
    eval_gap = 2 ** 6

    '''init task'''
    env = GraphMaxCutEnv(graph_key=graph_key, gpu_id=gpu_id)
    probs = env.get_rand_probs(num_envs=num_envs)
    probs.requires_grad_(True)

    '''init opti'''
    opt_base = th.optim.Adam([probs, ], lr=lr)

    '''loop'''
    best_score = -th.inf
    best_theta = probs
    start_time = time.time()
    for i in range(1, opt_num + 1):
        obj = env.get_objectives(probs).mean()
        opt_base.zero_grad()
        obj.backward()
        opt_base.step()

        probs.data.clip_(0, 1)

        if i % eval_gap == 0:
            thetas = env.convert_prob_to_bool(probs)
            scores = env.get_scores(thetas)
            used_time = time.time() - start_time
            print(f"|{used_time:9.0f}  {i:6}  {obj.item():9.3f}  {scores.max().item():9.0f}")

            max_score, max_id = th.max(scores, dim=0)
            if max_score > best_score:
                best_score = max_score
                best_theta = thetas[max_id]
                print(f"best_score {best_score}  best_theta \n{env.node_prob_bool_to_str(best_theta)}")

    print()
    print(f"best_score {best_score}  best_theta \n{env.node_prob_bool_to_str(best_theta)}")


def train_optimizer_level3_update_theta_by_opti():
    gpu_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    num_envs = 2 ** 4
    graph_key = 'gset_14'
    # graph_key = 'gset_70'

    '''hyper-parameters'''
    lr = 1e-3
    mid_dim = 2 ** 8
    num_layers = 1
    seq_len = 2 ** 4

    reset_gap = 2 ** 6
    opt_num = int(2 ** 16 / num_envs)
    eval_gap = 2 ** 1

    '''init task'''
    env = GraphMaxCutEnv(graph_key=graph_key, gpu_id=gpu_id)
    dim = env.num_nodes
    probs = env.get_rand_probs(num_envs=num_envs)
    probs.requires_grad_(True)
    obj = None
    hidden0 = None
    hidden1 = None

    '''init opti'''
    device = th.device(f'cuda:{gpu_id}' if th.cuda.is_available() and gpu_id >= 0 else 'cpu')
    opt_opti = OptimizerOpti(inp_dim=dim, mid_dim=mid_dim, out_dim=dim, num_layers=num_layers).to(device)
    opt_base = th.optim.Adam(opt_opti.parameters(), lr=lr)

    '''loop'''
    best_score = -th.inf
    best_theta = probs
    start_time = time.time()
    for i in range(1, opt_num + 1):
        if i % reset_gap == 0:
            probs = env.get_rand_probs(num_envs=num_envs)
            probs.requires_grad_(True)
            obj = None
            hidden0 = None
            hidden1 = None

        prob_ = probs.clone()
        updates = []

        for j in range(seq_len):
            obj = env.get_objectives(probs).mean()
            obj.backward()

            grad_s = probs.grad.data
            update, hidden0, hidden1 = opt_opti(grad_s.unsqueeze(0), hidden0, hidden1)
            update = (update.squeeze_(0) - grad_s) * lr
            updates.append(update)
            probs.data.add_(update).clip_(0, 1)
        hidden0 = [h.detach() for h in hidden0]
        hidden1 = [h.detach() for h in hidden1]

        updates = th.stack(updates, dim=0)
        prob_ = (prob_ + updates.mean(0)).clip(0, 1)
        obj_ = env.get_objectives(prob_).mean()

        opt_base.zero_grad()
        obj_.backward()
        opt_base.step()

        probs.data[:] = prob_

        if i % eval_gap == 0:
            thetas = env.convert_prob_to_bool(probs)
            scores = env.get_scores(thetas)
            used_time = time.time() - start_time
            print(f"|{used_time:9.0f}  {i:6}  {obj.item():9.3f}  {scores.max().item():9.0f}  best_score {best_score:9.0f}")
            

            max_score, max_id = th.max(scores, dim=0)
            if max_score > best_score:
                best_score = max_score
                best_theta = thetas[max_id]
                print(f"best_score {best_score}  best_theta \n{env.node_prob_bool_to_str(best_theta)}")

    print()
    print(f"best_score {best_score}  best_theta \n{env.node_prob_bool_to_str(best_theta)}")


if __name__ == '__main__':
    # check_env()
    # check_theta()
    # train_optimizer_level1_update_theta_by_grad()
    # train_optimizer_level2_update_theta_by_adam()
    train_optimizer_level3_update_theta_by_opti()
