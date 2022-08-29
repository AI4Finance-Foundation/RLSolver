import os
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch import Tensor
from elegantrl.net import ActorPPO, ActorSAC, CriticPPO, CriticTwin
from typing import List, Tuple
from collections import deque


class AgentBase:
    def __init__(self, net_dim: int, state_dim: int, action_dim: int, gpu_id: int = 0, args=None):
        """initialize
        replace by different DRL algorithms

        :param net_dim: the dimension of networks (the width of neural networks)
        :param state_dim: the dimension of state (the number of state vector)
        :param action_dim: the dimension of action (the number of discrete action)
        :param gpu_id: the gpu_id of the training device. Use CPU when cuda is not available.
        :param args: the arguments for agent training. `args = Arguments()`
        """

        self.gamma = getattr(args, 'gamma', 0.99)
        self.env_num = getattr(args, 'env_num', 1)
        self.num_layer = getattr(args, 'num_layer', 3)
        self.batch_size = getattr(args, 'batch_size', 128)
        self.action_dim = getattr(args, 'action_dim', 3)
        self.state_dim = getattr(args, 'state_dim', 10)
        self.repeat_times = getattr(args, 'repeat_times', 1.)
        self.reward_scale = getattr(args, 'reward_scale', 1.)
        self.lambda_critic = getattr(args, 'lambda_critic', 1.)
        self.learning_rate = getattr(args, 'learning_rate', 2 ** -15)
        self.clip_grad_norm = getattr(args, 'clip_grad_norm', 3.0)
        self.soft_update_tau = getattr(args, 'soft_update_tau', 2 ** -8)

        self.if_use_per = getattr(args, 'if_use_per', None)
        self.if_off_policy = getattr(args, 'if_off_policy', None)
        self.if_use_old_traj = getattr(args, 'if_use_old_traj', False)

        self.state = None  # assert self.state == (env_num, state_dim)
        self.device = torch.device(f"cuda:{gpu_id}" if (torch.cuda.is_available() and (gpu_id >= 0)) else "cpu")
        # self.traj_list = [[torch.tensor((), dtype=torch.float32, device=self.device)
                           # for _ in range(4 if self.if_off_policy else 5)]
                          # for _ in range(self.env_num)]  # for `self.explore_vec_env()`

        '''network'''
        act_class = getattr(self, "act_class", None)
        cri_class = getattr(self, "cri_class", None)
        self.act = act_class(net_dim, self.num_layer, state_dim, action_dim).to(self.device)
        self.cri = cri_class(net_dim, self.num_layer, state_dim, action_dim).to(self.device) \
            if cri_class else self.act

        '''optimizer'''
        self.act_optimizer = torch.optim.Adam(self.act.parameters(), self.learning_rate)
        self.cri_optimizer = torch.optim.Adam(self.cri.parameters(), self.learning_rate) \
            if cri_class else self.act_optimizer

        '''target network'''
        from copy import deepcopy
        self.if_act_target = args.if_act_target if hasattr(args, 'if_act_target') else \
            getattr(self, 'if_act_target', None)
        self.if_cri_target = args.if_cri_target if hasattr(args, 'if_cri_target') else \
            getattr(self, 'if_cri_target', None)
        self.act_target = deepcopy(self.act) if self.if_act_target else self.act
        self.cri_target = deepcopy(self.cri) if self.if_cri_target else self.cri

        """attribute"""
        if self.env_num == 1:
            self.explore_env = self.explore_one_env
        else:
            self.explore_env = self.explore_vec_env

        self.criterion = torch.nn.SmoothL1Loss(reduction="mean")

        """tracker"""
        self.reward_tracker = Tracker(args.tracker_len)
        self.step_tracker = Tracker(args.tracker_len)
        self.current_rewards = torch.zeros(self.env_num, dtype=torch.float32, device=self.device)
        self.current_lengths = torch.zeros(self.env_num, dtype=torch.float32, device=self.device)

    def explore_one_env(self, env, horizon_len: int) -> list:
        """
        Collect trajectories through the actor-environment interaction for a **single** environment instance.

        :param env: RL training environment. env.reset() env.step(). It should be a vector env.
        :param horizon_len: explored horizon_len number of step in env
        :return: `[traj, ]`
        `traj = [(state, reward, done, action, noise), ...]` for on-policy
        `traj = [(state, reward, done, action), ...]` for off-policy
        """
        traj_list = []
        last_dones = [0, ]
        state = self.state[0]

        i = 0
        done = False
        while i < horizon_len or not done:
            tensor_state = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
            tensor_action = self.act.get_action(tensor_state.to(self.device)).detach().cpu()  # different
            next_state, reward, done, _ = env.step(tensor_action[0].numpy())  # different

            traj_list.append((tensor_state, reward, done, tensor_action))  # different

            i += 1
            state = env.reset() if done else next_state

        self.state[0] = state
        last_dones[0] = i
        return self.convert_trajectory(traj_list, last_dones)  # traj_list

    def explore_vec_env(self, env, horizon_len: int) -> list:
        """
        Collect trajectories through the actor-environment interaction for a **vectorized** environment instance.

        :param env: RL training environment. env.reset() env.step(). It should be a vector env.
        :param horizon_len: explored horizon_len number of step in env
        """
        obs = torch.zeros((horizon_len * self.env_num, self.state_dim)).to(self.device)
        actions = torch.zeros((horizon_len * self.env_num, self.action_dim)).to(self.device)
        rewards = torch.zeros((horizon_len * self.env_num)).to(self.device)
        next_obs = torch.zeros((horizon_len * self.env_num, self.state_dim)).to(self.device)
        dones = torch.zeros((horizon_len * self.env_num)).to(self.device)

        state = self.state if self.if_use_old_traj else env.reset()
        done = torch.zeros(self.env_num).to(self.device)

        for i in range(horizon_len):
            start = i * self.env_num
            end = (i + 1) * self.env_num
            obs[start:end] = state
            dones[start:end] = done

            action = self.act.get_action(state).detach()  # different
            next_state, reward, done, _ = env.step(action)  # different
            state = next_state

            actions[start:end] = action
            rewards[start:end] = reward
            next_obs[start:end] = next_state

            self.current_rewards += reward
            self.current_lengths += 1
            env_done_indices = torch.where(done == 1)
            self.reward_tracker.update(self.current_rewards[env_done_indices])
            self.step_tracker.update(self.current_lengths[env_done_indices])
            not_dones = 1.0 - done.float()
            self.current_rewards = self.current_rewards * not_dones
            self.current_lengths = self.current_lengths * not_dones

        self.state = state
        return (obs, actions, self.reward_scale *  rewards.reshape(-1, 1), next_obs, dones.reshape(-1, 1)), horizon_len * self.env_num

    def update_net(self, buffer) -> tuple:
        return 0.0, 0.0

    def get_obj_critic(self, buffer, batch_size: int) -> (Tensor, Tensor):
        """
        Calculate the loss of networks with **uniform sampling**.

        :param buffer: the ReplayBuffer instance that stores the trajectories.
        :param batch_size: the size of batch data for Stochastic Gradient Descent (SGD).
        :return: the loss of the network and state.
        """
        with torch.no_grad():
            reward, done, action, state, next_state = buffer.sample_batch(batch_size)
            next_a = self.act_target(next_state)
            critic_targets: torch.Tensor = self.cri_target(next_state, next_a)
            (next_q, min_indices) = torch.min(critic_targets, dim=1, keepdim=True)
            q_label = reward + done * next_q

        q = self.cri(state, action)
        obj_critic = self.criterion(q, q_label)

        return obj_critic, state

    def optimizer_update(self, optimizer, objective):  # [ElegantRL 2021.11.11]
        """minimize the optimization objective via update the network parameters

        :param optimizer: `optimizer = torch.optim.SGD(net.parameters(), learning_rate)`
        :param objective: `objective = net(...)` the optimization objective, sometimes is a loss function.
        """
        optimizer.zero_grad()
        objective.backward()
        clip_grad_norm_(parameters=optimizer.param_groups[0]["params"], max_norm=self.clip_grad_norm)
        optimizer.step()

    def optimizer_update_amp(self, optimizer, objective):  # automatic mixed precision
        """minimize the optimization objective via update the network parameters

        amp: Automatic Mixed Precision

        :param optimizer: `optimizer = torch.optim.SGD(net.parameters(), learning_rate)`
        :param objective: `objective = net(...)` the optimization objective, sometimes is a loss function.
        """
        amp_scale = torch.cuda.amp.GradScaler()  # write in __init__()

        optimizer.zero_grad()
        amp_scale.scale(objective).backward()  # loss.backward()
        amp_scale.unscale_(optimizer)  # amp

        # from torch.nn.utils import clip_grad_norm_
        clip_grad_norm_(parameters=optimizer.param_groups[0]["params"], max_norm=self.clip_grad_norm)
        amp_scale.step(optimizer)  # optimizer.step()
        amp_scale.update()  # optimizer.step()

    @staticmethod
    def soft_update(target_net, current_net, tau: float):
        """soft update target network via current network

        :param target_net: update target network via current network to make training more stable.
        :param current_net: current network update via an optimizer
        :param tau: tau of soft target update: `target_net = target_net * (1-tau) + current_net * tau`
        """
        for tar, cur in zip(target_net.parameters(), current_net.parameters()):
            tar.data.copy_(cur.data * tau + tar.data * (1.0 - tau))

    def save_or_load_agent(self, cwd: str, if_save: bool):
        """save or load training files for Agent

        :param cwd: Current Working Directory. ElegantRL save training files in CWD.
        :param if_save: True: save files. False: load files.
        """

        def load_torch_file(model, path: str):
            state_dict = torch.load(path, map_location=lambda storage, loc: storage)
            model.load_state_dict(state_dict)

        name_obj_list = [
            ("actor", self.act),
            ("act_target", self.act_target),
            ("act_optimizer", self.act_optimizer),
            ("critic", self.cri),
            ("cri_target", self.cri_target),
            ("cri_optimizer", self.cri_optimizer),
        ]
        name_obj_list = [(name, obj) for name, obj in name_obj_list if obj is not None]

        if if_save:
            for name, obj in name_obj_list:
                save_path = f"{cwd}/{name}.pth"
                torch.save(obj.state_dict(), save_path)
        else:
            for name, obj in name_obj_list:
                save_path = f"{cwd}/{name}.pth"
                load_torch_file(obj, save_path) if os.path.isfile(save_path) else None

    # def convert_trajectory(self,
    #         traj_list: List[Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]]) -> List[Tensor]:
    #     traj_list = list(map(list, zip(*traj_list)))  # state, next_state, reward, done, action, noise

    #     '''stack items'''  # todo
    #     traj_list[0] = torch.stack(traj_list[0])
    #     traj_list[1] = torch.stack(traj_list[1])  # todo
    #     traj_list[4:] = [torch.stack(item) for item in traj_list[4:]]

    #     if len(traj_list[4].shape) == 2:
    #         traj_list[4] = traj_list[4].unsqueeze(2)

    #     if self.env_num > 1:
    #         traj_list[2] = (torch.stack(traj_list[2]) * self.reward_scale).unsqueeze(2)
    #         traj_list[3] = ((1 - torch.stack(traj_list[3])) * self.gamma).unsqueeze(2)
    #     else:
    #         traj_list[2] = (torch.tensor(traj_list[2], dtype=torch.float32) * self.reward_scale
    #                         ).unsqueeze(1).unsqueeze(2)
    #         traj_list[3] = ((1 - torch.tensor(traj_list[3], dtype=torch.float32)) * self.gamma
    #                         ).unsqueeze(1).unsqueeze(2)
    #     # assert all([buf_item.shape[:2] == (step, self.env_num) for buf_item in buf_items])

    #     '''splice items'''  # todo
    #     for j in range(len(traj_list)):
    #         buf_item = traj_list[j]

    #         traj_list[j] = torch.vstack([buf_item[:, env_i]
    #                                      for env_i in range(self.env_num)])

    #     # on-policy:  buf_item = [state, rewards, dones, actions, noises]
    #     # off-policy: buf_item = [state, rewards, dones, actions]
    #     # buf_items = [buf_item, ...]
    #     return traj_list


class AgentPPO(AgentBase):
    def __init__(self, net_dim: int, state_dim: int, action_dim: int, gpu_id: int = 0, args = None):
        self.if_off_policy = False
        self.act_class = getattr(self, 'act_class', ActorPPO)
        self.cri_class = getattr(self, 'cri_class', CriticPPO)
        args.if_act_target = getattr(args, 'if_act_target', False)
        args.if_cri_target = getattr(args, 'if_cri_target', False)
        AgentBase.__init__(self, net_dim, state_dim, action_dim, gpu_id, args)

        if getattr(args, 'if_use_gae', False):
            self.get_reward_sum = self.get_reward_sum_gae
        else:
            self.get_reward_sum = self.get_reward_sum_raw

        self.ratio_clip = getattr(args, 'ratio_clip', 0.25)  # could be 0.00 ~ 0.50 `ratio.clamp(1 - clip, 1 + clip)`
        self.lambda_entropy = getattr(args, 'lambda_entropy', 0.02)  # could be 0.00~0.10
        self.lambda_gae_adv = getattr(args, 'lambda_gae_adv', 0.95)  # could be 0.50~0.99, GAE (ICLR.2016.)
        self.act_update_gap = getattr(args, 'act_update_gap', 1)

    def explore_one_env(self, env, horizon_len: int) -> list:
        traj_list = list()
        last_dones = [0, ]
        state = self.state[0]

        i = 0
        done = False
        get_action = self.act.get_action
        convert = self.act.convert_action_for_env
        while i < horizon_len or not done:
            state = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
            actions, noises = [item.cpu() for item in get_action(state.to(self.device))]  # different
            next_state, reward, done, _ = env.step(convert(actions)[0].numpy())

            traj_list.append((state, reward, done, actions, noises))  # different

            i += 1
            state = env.reset() if done else next_state
        self.state[0] = state
        last_dones[0] = i
        return self.convert_trajectory(traj_list, last_dones)  # traj_list

    def explore_vec_env(self, env, horizon_len: int) -> list:
        obs = torch.zeros((horizon_len, self.env_num) + (self.state_dim,)).to(self.device)
        actions = torch.zeros((horizon_len, self.env_num) + (self.action_dim,)).to(self.device)
        noises = torch.zeros((horizon_len, self.env_num) + (self.action_dim,)).to(self.device)
        rewards = torch.zeros((horizon_len, self.env_num)).to(self.device)
        dones = torch.zeros((horizon_len, self.env_num)).to(self.device)

        state = self.state if self.if_use_old_traj else env.reset()
        done = torch.zeros(self.env_num).to(self.device)

        get_action = self.act.get_action
        convert = self.act.convert_action_for_env
        for i in range(horizon_len):
            obs[i] = state
            dones[i] = done

            action, noise = get_action(state)
            next_state, reward, done, _ = env.step(convert(action))
            state = next_state

            actions[i] = action
            noises[i] = noise
            rewards[i] = reward

            self.current_rewards += reward
            self.current_lengths += 1
            env_done_indices = torch.where(done == 1)
            self.reward_tracker.update(self.current_rewards[env_done_indices])
            self.step_tracker.update(self.current_lengths[env_done_indices])
            not_dones = 1.0 - done.float()
            self.current_rewards = self.current_rewards * not_dones
            self.current_lengths = self.current_lengths * not_dones

        self.state = state

        return (obs, actions, noises, self.reward_scale * rewards, dones, state, done), horizon_len * self.env_num

    def update_net(self, buffer):
        buf_state, buf_action, buf_logprob, buf_adv, buf_r_sum = self.get_reward_sum(buffer)
        buffer_size = buf_state.size()[0]
        assert buffer_size >= self.batch_size

        '''update network'''
        obj_critic_list = list()
        obj_actor_list = list()
        indices = np.arange(buffer_size)
        for epoch in range(self.repeat_times):
            np.random.shuffle(indices)

            for i in range(0, buffer_size, self.batch_size):
                minibatch_indices = indices[i:i + self.batch_size]
                state = buf_state[minibatch_indices]
                r_sum = buf_r_sum[minibatch_indices]
                adv_v = buf_adv[minibatch_indices]
                adv_v = (adv_v - adv_v.mean()) / (adv_v.std() + 1e-8)
                action = buf_action[minibatch_indices]
                logprob = buf_logprob[minibatch_indices]
        
                value = self.cri(state).squeeze(1)  # critic network predicts the reward_sum (Q value) of state
                obj_critic = self.criterion(value, r_sum) * self.lambda_critic
                self.optimizer_update(self.cri_optimizer, obj_critic)
                if self.if_cri_target:
                    self.soft_update(self.cri_target, self.cri, self.soft_update_tau)

                '''PPO: Surrogate objective of Trust Region'''
                new_logprob, obj_entropy = self.act.get_logprob_entropy(state, action)
                ratio = (new_logprob - logprob.detach()).exp()
                surrogate1 = adv_v * ratio
                surrogate2 = adv_v * ratio.clamp(1 - self.ratio_clip, 1 + self.ratio_clip)
                obj_surrogate = -torch.min(surrogate1, surrogate2).mean()
                obj_actor = obj_surrogate + obj_entropy * self.lambda_entropy
                self.optimizer_update(self.act_optimizer, obj_actor)

                obj_critic_list.append(obj_critic.item())
                obj_actor_list.append(-obj_actor.item())

        action_std_log = getattr(self.act, 'action_std_log', torch.zeros(1)).mean()
        return np.array(obj_critic_list).mean(), np.array(obj_actor_list).mean(), action_std_log.item()  # logging_tuple

    def get_reward_sum_raw(
            self, buffer_len: int, rewards: Tensor, masks: Tensor, values: Tensor, buf_next_state: Tensor,
    ) -> (Tensor, Tensor):
        """
        Calculate the **reward-to-go** and **advantage estimation**.
        """
        buf_r_sum = rewards + masks * self.cri(buf_next_state)
        buf_adv = buf_r_sum - values[:, 0]
        return buf_r_sum, buf_adv

    def get_reward_sum_gae(self, buffer) -> (Tensor, Tensor, Tensor, Tensor, Tensor):
        """
        Calculate the **reward-to-go** and **advantage estimation** using GAE.
        """
        with torch.no_grad():
            buf_state, buf_action, buf_noise, buf_reward, buf_done, next_state, next_done = buffer
            next_state_value = self.cri(next_state)

            buf_adv = torch.zeros_like(buf_reward).to(self.device)
            values = torch.zeros_like(buf_reward).to(self.device)
            
            lastgaelam = 0
            horizon_len = buf_state.size()[0]
            for t in reversed(range(horizon_len)):
                values[t] = self.cri(buf_state[t]).reshape(-1,)
                if t == horizon_len - 1:
                    nextnonterminal = 1.0 - next_done
                    next_values = next_state_value
                else:
                    nextnonterminal = 1.0 - buf_done[t + 1]
                    next_values = values[t + 1]
                    delta = buf_reward[t] + self.gamma * next_values * nextnonterminal - values[t]
                    buf_adv[t] = lastgaelam = delta + self.gamma * self.lambda_gae_adv * nextnonterminal * lastgaelam
            buf_r_sum = buf_adv + values
            
            buf_state = buf_state.reshape((-1,) + (self.state_dim,))
            buf_action = buf_action.reshape((-1,) + (self.action_dim,))
            buf_logprob = self.act.get_old_logprob(buf_action, buf_noise.reshape((-1,) + (self.action_dim,)))
            buf_logprob = buf_logprob.reshape(-1,)
            buf_adv = buf_adv.reshape(-1,)
            buf_r_sum = buf_r_sum.reshape(-1,)
            
        return buf_state, buf_action, buf_logprob, buf_adv, buf_r_sum


class AgentSAC(AgentBase):  # [ElegantRL.2022.03.03]
    def __init__(self, net_dim: int, state_dim: int, action_dim: int, gpu_id: int = 0, args=None):
        self.if_off_policy = True
        self.act_class = getattr(self, 'act_class', ActorSAC)
        self.cri_class = getattr(self, 'cri_class', CriticTwin)
        args.if_act_target = getattr(args, 'if_act_target', False)
        args.if_cri_target = getattr(args, 'if_cri_target', True)
        super().__init__(net_dim, state_dim, action_dim, gpu_id, args)

        self.alpha_log = torch.tensor(
            (-np.log(action_dim),), dtype=torch.float32, requires_grad=True, device=self.device
        )  # trainable parameter
        self.alpha_optim = torch.optim.Adam((self.alpha_log,), lr=0.005)
        self.target_entropy = getattr(args, 'target_entropy', np.log(action_dim))

    def update_net(self, buffer):
        obj_critic = torch.zeros(1)
        obj_actor = torch.zeros(1)

        for _ in range(self.repeat_times):
            '''objective of critic (loss function of critic)'''
            obj_critic, state = self.get_obj_critic(buffer, self.batch_size)
            self.optimizer_update(self.cri_optimizer, obj_critic)
            self.soft_update(self.cri_target, self.cri, self.soft_update_tau)

            '''objective of alpha (temperature parameter automatic adjustment)'''
            a_noise_pg, log_prob = self.act.get_action_logprob(state)  # policy gradient
            obj_alpha = (self.alpha_log * (log_prob - self.target_entropy).detach()).mean()
            self.optimizer_update(self.alpha_optim, obj_alpha)

            '''objective of actor'''
            alpha = self.alpha_log.exp().detach()
            with torch.no_grad():
                self.alpha_log[:] = self.alpha_log.clamp(-20, 2)

            q_value_pg = self.cri(state, a_noise_pg)
            obj_actor = -(q_value_pg + log_prob * alpha).mean()
            self.optimizer_update(self.act_optimizer, obj_actor)
            if self.if_act_target:
                self.soft_update(self.act_target, self.act, self.soft_update_tau)

        return obj_critic.item(), -obj_actor.item(), self.alpha_log.exp().detach().item()

    def get_obj_critic(self, buffer, batch_size: int) -> (Tensor, Tensor):
        with torch.no_grad():
            state, action, reward, next_state, done = buffer.sample_batch(batch_size)
            mask = (1 - done) * self.gamma

            next_action, next_logprob = self.act_target.get_action_logprob(next_state)  # stochastic policy
            next_q = self.cri_target.get_q_min(next_state, next_action)

            alpha = self.alpha_log.exp().detach()
            q_label = reward + mask * (next_q + next_logprob * alpha)
        q1, q2 = self.cri.get_q1_q2(state, action)
        obj_critic = (self.criterion(q1, q_label) + self.criterion(q2, q_label)) / 2
        return obj_critic, state

    def get_obj_critic_per(self, buffer, batch_size: int) -> (Tensor, Tensor):
        with torch.no_grad():
            state, action, reward, next_state, done, is_weights = buffer.sample_batch(batch_size)
            mask = (1 - done) * self.gamma

            next_action, next_logprob = self.act_target.get_action_logprob(next_state)  # stochastic policy
            next_q = self.cri_target.get_q_min(next_state, next_action)

            alpha = self.alpha_log.exp().detach()
            q_label = reward + mask * (next_q + next_logprob * alpha)
        q1, q2 = self.cri.get_q1_q2(state, action)
        td_error = (self.criterion(q1, q_label) + self.criterion(q2, q_label)) / 2.
        obj_critic = (td_error * is_weights).mean()

        buffer.td_error_update(td_error.detach())
        return obj_critic, state


class Tracker:
    def __init__(self, max_len):
        self.moving_average = deque([0 for _ in range(max_len)], maxlen=max_len)
        self.max_len = max_len

    def __repr__(self):
        return self.moving_average.__repr__()

    def update(self, values):
        self.moving_average.extend(values.tolist())

    def mean(self):
        return sum(self.moving_average) / self.max_len
