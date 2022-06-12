import os
from typing import List, Tuple
import numpy as np
import numpy.random as rd
import torch
from torch import Tensor


class ReplayBuffer:  # for off-policy
    def __init__(self, max_capacity: int, state_dim: int, action_dim: int, gpu_id=0, if_use_per=False):
        self.prev_p = 0  # previous pointer
        self.next_p = 0  # next pointer
        self.if_full = False
        self.cur_capacity = 0  # current capacity
        self.max_capacity = int(max_capacity)
        self.add_capacity = 0  # update in self.update_buffer

        self.device = torch.device(f"cuda:{gpu_id}" if (torch.cuda.is_available() and (gpu_id >= 0)) else "cpu")

        self.buf_action = torch.empty((self.max_capacity, int(action_dim)), dtype=torch.float32, device=self.device)
        self.buf_reward = torch.empty((self.max_capacity, 1), dtype=torch.float32, device=self.device)
        self.buf_done = torch.empty((self.max_capacity, 1), dtype=torch.float32, device=self.device)

        buf_state_size = (self.max_capacity, state_dim) if isinstance(state_dim, int) else (max_capacity, *state_dim)
        self.buf_state = torch.empty(buf_state_size, dtype=torch.float32, device=self.device)
        self.buf_next_state = torch.empty(buf_state_size, dtype=torch.float32, device=self.device)

        dir(if_use_per)

    def update_buffer(self, trajectory: Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]):
        states, actions, rewards, next_states, dones = trajectory
        self.add_capacity = rewards.shape[0]
        p = self.next_p + self.add_capacity  # update pointer

        if p > self.max_capacity:
            self.buf_state[self.next_p:self.max_capacity] = states[:self.max_capacity - self.next_p]
            self.buf_action[self.next_p:self.max_capacity] = actions[:self.max_capacity - self.next_p]
            self.buf_reward[self.next_p:self.max_capacity] = rewards[:self.max_capacity - self.next_p]
            self.buf_next_state[self.next_p:self.max_capacity] = next_states[:self.max_capacity - self.next_p]
            self.buf_done[self.next_p:self.max_capacity] = dones[:self.max_capacity - self.next_p]
            self.if_full = True

            p = p - self.max_capacity
            self.buf_state[0:p] = states[-p:]
            self.buf_action[0:p] = actions[-p:]
            self.buf_reward[0:p] = rewards[-p:]
            self.buf_next_state[0:p] = next_states[-p:]
            self.buf_done[0:p] = dones[-p:]
            
        else:
            self.buf_state[self.next_p:p] = states
            self.buf_action[self.next_p:p] = actions
            self.buf_reward[self.next_p:p] = rewards
            self.buf_next_state[self.next_p:p] = next_states
            self.buf_done[self.next_p:p] = dones

        self.next_p = p  # update pointer
        self.cur_capacity = self.max_capacity if self.if_full else self.next_p

    def sample_batch(self, batch_size: int) -> (Tensor, Tensor, Tensor, Tensor):
        indices = torch.randint(self.cur_capacity, size=(batch_size,), device=self.device)
        
        '''replace indices using the latest sample'''
        i1 = self.next_p
        i0 = self.next_p - self.add_capacity
        num_new_indices = 1
        new_indices = torch.randint(i0, i1, size=(num_new_indices,)) % (self.max_capacity - 1)
        indices[0:num_new_indices] = new_indices

        return (
            self.buf_state[indices],
            self.buf_action[indices],
            self.buf_reward[indices],
            self.buf_next_state[indices],
            self.buf_done[indices]  
        )

    def save_or_load_history(self, cwd: str, if_save: bool):
        obj_names = (
            (self.buf_reward, "reward"),
            (self.buf_done, "mask"),
            (self.buf_action, "action"),
            (self.buf_state, "state"),
        )

        if if_save:
            print(f"| {self.__class__.__name__}: Saving in cwd {cwd}")
            for obj, name in obj_names:
                if self.cur_capacity == self.next_p:
                    buf_tensor = obj[:self.cur_capacity]
                else:
                    buf_tensor = torch.vstack((obj[self.next_p:self.cur_capacity], obj[0:self.next_p]))

                torch.save(buf_tensor, f"{cwd}/replay_buffer_{name}.pt")

            print(f"| {self.__class__.__name__}: Saved in cwd {cwd}")

        elif os.path.isfile(f"{cwd}/replay_buffer_state.pt"):
            print(f"| {self.__class__.__name__}: Loading from cwd {cwd}")
            buf_capacity = 0
            for obj, name in obj_names:
                buf_tensor = torch.load(f"{cwd}/replay_buffer_{name}.pt")
                buf_capacity = buf_tensor.shape[0]

                obj[:buf_capacity] = buf_tensor
            self.cur_capacity = buf_capacity

            print(f"| {self.__class__.__name__}: Loaded from cwd {cwd}")

    def get_state_norm(self, cwd: str = '.',
                       neg_state_avg: [float, Tensor] = 0.0,
                       div_state_std: [float, Tensor] = 1.0):
        state_avg, state_std = get_state_avg_std(
            buf_state=self.buf_state, batch_size=2 ** 10,
            neg_state_avg=neg_state_avg, div_state_std=div_state_std,
        )

        torch.save(state_avg, f"{cwd}/state_norm_avg.pt")
        print(f"| {self.__class__.__name__}: state_avg = {state_avg}")
        torch.save(state_std, f"{cwd}/state_norm_std.pt")
        print(f"| {self.__class__.__name__}: state_std = {state_std}")

    def concatenate_state(self) -> Tensor:
        if self.prev_p <= self.next_p:
            buf_state = self.buf_state[self.prev_p:self.next_p]
        else:
            buf_state = torch.vstack((self.buf_state[self.prev_p:], self.buf_state[:self.next_p],))
        self.prev_p = self.next_p
        return buf_state

    def concatenate_buffer(self) -> (Tensor, Tensor, Tensor, Tensor):
        if self.prev_p <= self.next_p:
            buf_state = self.buf_state[self.prev_p:self.next_p]
            buf_action = self.buf_action[self.prev_p:self.next_p]
            buf_reward = self.buf_reward[self.prev_p:self.next_p]
            buf_done = self.buf_done[self.prev_p:self.next_p]
        else:
            buf_state = torch.vstack((self.buf_state[self.prev_p:], self.buf_state[:self.next_p],))
            buf_action = torch.vstack((self.buf_action[self.prev_p:], self.buf_action[:self.next_p],))
            buf_reward = torch.vstack((self.buf_reward[self.prev_p:], self.buf_reward[:self.next_p],))
            buf_done = torch.vstack((self.buf_done[self.prev_p:], self.buf_done[:self.next_p],))
        self.prev_p = self.next_p
        return buf_state, buf_action, buf_reward, buf_done


class ReplayBufferList(list):  # for on-policy
    def __init__(self):
        list.__init__(self)  # (buf_state, buf_next_state, buf_reward, buf_done, buf_action, buf_noise) = self[:]

    def update_buffer(self, traj_list: List[List]) -> (int, float):
        cur_items = list(map(list, zip(*traj_list)))
        self[:] = [torch.cat(item, dim=0) for item in cur_items]

        steps = self[2].shape[0]

        return steps

    def get_state_norm(self, cwd='.', neg_state_avg=0, div_state_std=1):
        state_avg, state_std = get_state_avg_std(
            buf_state=self[0], batch_size=2 ** 10,
            neg_state_avg=neg_state_avg, div_state_std=div_state_std,
        )

        torch.save(state_avg, f"{cwd}/state_norm_avg.pt")
        print(f"| {self.__class__.__name__}: state_avg = {state_avg}")
        torch.save(state_std, f"{cwd}/state_norm_std.pt")
        print(f"| {self.__class__.__name__}: state_std = {state_std}")


def get_state_avg_std(
        buf_state: Tensor, batch_size=2 ** 10,
        neg_state_avg: [float, Tensor] = 0.0,
        div_state_std: [float, Tensor] = 1.0,
) -> (Tensor, Tensor):
    state_len = buf_state.shape[0]
    state_avg = torch.zeros_like(buf_state[0])
    state_std = torch.zeros_like(buf_state[0])

    from tqdm import trange
    for i in trange(0, state_len, batch_size):
        state_part = buf_state[i:i + batch_size]
        state_avg += state_part.mean(axis=0)
        state_std += state_part.std(axis=0)

    num = max(1, state_len // batch_size)
    state_avg /= num
    state_std /= num

    state_avg = state_avg / div_state_std - neg_state_avg
    state_std = state_std / div_state_std - neg_state_avg
    return state_avg.cpu(), state_std.cpu()
