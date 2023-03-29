import os.path

from TNCO_L2O import *
from TNCO_env import *


class ReplayBuffer:  # for off-policy
    def __init__(self, max_size: int, state_dim: int, gpu_id: int = 0):
        self.p = 0  # pointer
        self.if_full = False
        self.cur_size = 0
        self.max_size = max_size
        self.device = torch.device(f"cuda:{gpu_id}" if (torch.cuda.is_available() and (gpu_id >= 0)) else "cpu")

        self.states = torch.empty((max_size, state_dim), dtype=torch.float32, device=self.device)
        self.scores = torch.empty((max_size, 1), dtype=torch.float32, device=self.device)

    def update(self, items: [TEN]):
        states, rewards = items
        p = self.p + rewards.shape[0]  # pointer
        if p > self.max_size:
            self.if_full = True
            p0 = self.p
            p1 = self.max_size
            p2 = self.max_size - self.p
            p = p - self.max_size

            self.states[p0:p1], self.states[0:p] = states[:p2], states[-p:]
            self.scores[p0:p1], self.scores[0:p] = rewards[:p2], rewards[-p:]
        else:
            self.states[self.p:p] = states
            self.scores[self.p:p] = rewards
        self.p = p
        self.cur_size = self.max_size if self.if_full else self.p

    def sample(self, batch_size: int) -> [TEN]:
        ids = torch.randint(self.cur_size - 1, size=(batch_size,), requires_grad=False)
        return self.states[ids], self.scores[ids]

    def save_or_load_history(self, cwd: str, if_save: bool):
        item_names = (
            (self.states, "states"),
            (self.scores, "scores"),
        )

        if if_save:
            for item, name in item_names:
                if self.cur_size == self.p:
                    buf_item = item[:self.cur_size]
                else:
                    buf_item = torch.vstack((item[self.p:self.cur_size], item[0:self.p]))
                file_path = f"{cwd}/replay_buffer_{name}.pth"
                print(f"| buffer.save_or_load_history(): Save {file_path}")
                torch.save(buf_item, file_path)

        elif all([os.path.isfile(f"{cwd}/replay_buffer_{name}.pth") for item, name in item_names]):
            max_sizes = []
            for item, name in item_names:
                file_path = f"{cwd}/replay_buffer_{name}.pth"
                print(f"| buffer.save_or_load_history(): Load {file_path}")
                buf_item = torch.load(file_path)

                max_size = buf_item.shape[0]
                item[:max_size] = buf_item
                max_sizes.append(max_size)
            assert all([max_size == max_sizes[0] for max_size in max_sizes])
            self.cur_size = max_sizes[0]


class ObjectiveTNCO(ObjectiveTask):
    def __init__(self, dim, device):
        super(ObjectiveTNCO, self).__init__()
        self.device = device
        self.args = ()

        self.env = TensorNetworkEnv(nodes_list=get_nodes_list(len_list=100))
        self.dim = self.env.num_edges
        assert self.dim == dim

        self.obj_model = MLP(inp_dim=self.env.num_edges, out_dim=1, dims=(256, 256, 256)).to(device)
        self.optimizer = th.optim.SGD(self.obj_model.parameters(), lr=1e-4)
        self.criterion = nn.MSELoss()
        self.batch_size = 2 ** 8
        self.min_train_size = self.batch_size * 2 ** 8
        self.train_thresh = 2 ** -5  # 0.03125
        self.ema_loss = 0.0

        self.save_path = './task_TNCO'
        os.makedirs(self.save_path, exist_ok=True)

        '''warm up'''
        gpu_id = device.index  # not elegant
        warm_up_size = 2 ** 14
        self.buffer = ReplayBuffer(max_size=2 ** 16, state_dim=self.dim, gpu_id=gpu_id)
        self.buffer.save_or_load_history(cwd=self.save_path, if_save=False)
        if self.buffer.cur_size < warm_up_size:
            thetas, scores = self.random_generate_input_output(warm_up_size=warm_up_size, if_tqdm=True)
            self.buffer.update(items=(thetas, scores))
        self.save_and_check_buffer()

        self.fast_train_obj_model()

    def get_objective(self, theta, *args) -> TEN:
        num_repeats = 8

        thetas = theta.repeat(num_repeats, 1)
        thetas[1:] += th.rand_like(thetas[1:])

        scores = th.tensor([self.get_objective_without_grad(theta) for theta in thetas],
                           dtype=th.float32, device=self.device)

        self.buffer.update(items=(thetas, scores))
        self.fast_train_obj_model()

        objective = self.obj_model(theta)
        return objective

    def get_objective_without_grad(self, theta, *_args) -> float:
        # assert theta.shape[0] == self.env.num_edges
        log10_multiple_times = self.env.get_log10_multiple_times(edge_argsort=theta.argsort())
        return log10_multiple_times

    def random_generate_input_output(self, warm_up_size: int = 1024, if_tqdm: bool = False):
        print(f"TNCO | random_generate_input_output: num_warm_up={warm_up_size}")
        thetas = th.rand((warm_up_size, self.dim), dtype=th.float32, device=self.device)
        thetas = thetas / thetas.norm(dim=1, keepdim=True)

        if if_tqdm:
            from tqdm import tqdm
            scores = th.tensor([self.get_objective_without_grad(theta) for theta in tqdm(thetas, ascii=True)],
                               dtype=th.float32, device=self.device)
        else:
            scores = th.tensor([self.get_objective_without_grad(theta) for theta in thetas],
                               dtype=th.float32, device=self.device)

        return thetas, scores

    def fast_train_obj_model(self):
        # print(f"TNCO | fast_train_obj_model    batch_size {batch_size}    train_thresh={train_thresh}")
        train_thresh = self.train_thresh

        ema_loss = 100  # Exponential Moving Average (EMA) loss value
        gamma = 0.98
        while ema_loss > train_thresh:
            inputs, labels = self.buffer.sample(self.batch_size)

            outputs = self.obj_model(inputs).squeeze(1)
            loss = self.criterion(outputs, labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            ema_loss = gamma * ema_loss + (1 - gamma) * loss.item()
        self.ema_loss = ema_loss

    def save_and_check_buffer(self):
        self.buffer.save_or_load_history(cwd=self.save_path, if_save=True)

        scores = self.buffer.scores
        print(f"num_train: {scores.shape[0]}")
        print(f"min_score: {scores.min().item():9.3f}")
        print(f"avg_score: {scores.mean().item():9.3f} ± {scores.std(dim=0).item():9.3f}")
        print(f"max_score: {scores.max().item():9.3f}")


def run():
    gpu_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    device = th.device(f'cuda:{gpu_id}' if th.cuda.is_available() and gpu_id >= 0 else 'cpu')

    obj_task = ObjectiveTNCO(dim=-1, device=device)

    obj_task.get_objective(theta=th.rand(obj_task.dim, dtype=th.float32, device=obj_task.device))
    obj_task.save_and_check_buffer()


if __name__ == '__main__':
    run()
