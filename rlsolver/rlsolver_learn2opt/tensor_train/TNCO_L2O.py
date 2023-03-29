import torch
from L2O_H_term import *
from TNCO_env import TensorNetworkEnv, NodesSycamoreN53M12


def build_mlp(dims: [int], activation: nn = None, if_raw_out: bool = True) -> nn.Sequential:
    """
    build MLP (MultiLayer Perceptron)

    dims: the middle dimension, `dims[-1]` is the output dimension of this network
    activation: the activation function
    if_remove_out_layer: if remove the activation function of the output layer.
    """
    if activation is None:
        activation = nn.ReLU
    net_list = []
    for i in range(len(dims) - 1):
        net_list.extend([nn.Linear(dims[i], dims[i + 1]), activation()])
    if if_raw_out:
        del net_list[-1]  # delete the activation function of the output layer to keep raw output
    return nn.Sequential(*net_list)


def layer_init_with_orthogonal(layer, std=1.0, bias_const=1e-6):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)


class MLP(nn.Module):
    def __init__(self, inp_dim, out_dim, dims=(256, 256, 256)):
        super().__init__()
        self.net = build_mlp(dims=[inp_dim, *dims, out_dim], activation=nn.Tanh)
        layer_init_with_orthogonal(self.net[-1], std=0.1)

    def forward(self, x):
        return self.net(x)


class ObjectiveTNCO(ObjectiveTask):
    def __init__(self, dim, device):
        super(ObjectiveTNCO, self).__init__()
        self.dim = dim
        self.device = device

        self.args = ()

        self.env = TensorNetworkEnv(nodes_list=NodesSycamoreN53M12)
        self.dim = self.env.num_edges  # todo not elegant

        self.theta_list = []  # todo 设置成全局变量，建议每次训练后，独立地保存并收集这些数据，
        self.score_list = []  # todo 设置成全局变量，建议每次训练后，独立地保存并收集这些数据，
        self.obj_model = MLP(inp_dim=self.env.num_edges, out_dim=1, dims=(256, 256, 256)).to(device)
        self.optimizer = th.optim.SGD(self.obj_model.parameters(), lr=1e-4)
        self.criterion = nn.MSELoss()
        self.batch_size = 64
        self.min_train_size = self.batch_size * 2 ** 8

    def get_objective(self, theta, *args) -> TEN:
        score = self.get_objective_without_grad(theta)
        self.theta_list.append(theta)
        self.score_list.append(score)

        theta_tensor = torch.stack(self.theta_list).to(self.device)
        score_tensor = torch.tensor(self.score_list).to(self.device)
        train_size = score_tensor.shape[0]

        if train_size > self.min_train_size:
            for epoch in range(128):  # fast_fit_it
                # indices = torch.randint(train_size, size=(batch_size,), device=self.device)
                indices = torch.randperm(train_size, device=self.device)[:self.batch_size]

                inputs = theta_tensor[indices]
                labels = score_tensor[indices]

                outputs = self.obj_model(inputs)
                loss = self.criterion(outputs, labels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        objective = self.obj_model(theta)
        return objective

    def get_objective_without_grad(self, theta, *_args) -> float:
        # assert theta.shape[0] == self.env.num_edges
        log10_multiple_times = self.env.get_log10_multiple_times(edge_argsort=theta.argsort())
        return log10_multiple_times


"""trainable objective function"""


def train_optimizer():
    gpu_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0

    '''train'''
    train_times = 2 ** 9
    lr = 1e-4  # 要更小一些，甚至前期先不训练。
    unroll = 16
    num_opt = 64  # 要更大一些
    hid_dim = 40

    '''eval'''
    eval_gap = 128

    print('start training')
    device = th.device(f'cuda:{gpu_id}' if th.cuda.is_available() and gpu_id >= 0 else 'cpu')

    dim = None  # set by env.num_edges
    obj_task = ObjectiveTNCO(dim=dim, device=device)
    dim = obj_task.env.num_edges

    opt_task = OptimizerTask(dim=dim, device=device)
    opt_opti = OptimizerOpti(hid_dim=hid_dim).to(device)
    opt_base = th.optim.Adam(opt_opti.parameters(), lr=lr)

    start_time = time.time()
    '''loop'''
    for i in range(train_times + 1):
        opt_train(obj_task=obj_task, opt_task=opt_task, opt_opti=opt_opti,
                  num_opt=num_opt, device=device, unroll=unroll, opt_base=opt_base)

        if i % eval_gap == 0:
            best_result, min_loss = opt_eval(
                obj_task=obj_task, opt_opti=opt_opti, opt_task=opt_task,
                num_opt=num_opt * 2, device=device
            )
            time_used = time.time() - start_time

            log10_multiple_times = obj_task.env.get_log10_multiple_times(edge_argsort=best_result.squeeze(0).argsort())
            print(f"{i:>9}    {log10_multiple_times:9.3f}    {min_loss.item():9.3e}    TimeUsed {time_used:9.0f}")

    # todo 训练结束后，一定要时刻保存 我们的 score_list 以及 theta_list，比避免保存的时候因为重名覆盖掉上一次训练的，积攒数据
    # torch.save(torch.stack(theta_list), './theta_list.pth')
    # torch.save(torch.stack(score_list), './score_list.pth')


if __name__ == '__main__':
    train_optimizer()
