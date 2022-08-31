import torch as th
import numpy as np
class MIMO():
    def __init__(self, K=4, N=4, P=10, noise_power=1, episode_length=6, num_env=4, device=th.device("cuda:0")):
        self.N = N
        self.K = K
        self.P = P
        self.noise_power = noise_power
        self.basis_vectors, _ = th.linalg.qr(th.rand(2 * self.K * self.N, 2 * self.K * self.N, dtype=th.float))
        self.subspace_dim = 1
        self.num_env = num_env
        self.device = device
        self.episode_length = episode_length
        
    def reset(self,):
        if self.subspace_dim <= 2 * self.K * self.N:
            self.vec_H = self.generate_channel_batch(self.N, self.K, self.num_env, self.subspace_dim, self.basis_vectors).to(self.device)
        else:
            self.vec_H = th.randn(self.num_env, 2 * self.K * self.N, dtype=th.cfloat).to(self.device)
        self.mat_H = (self.vec_H[:, :self.K * self.N] + self.vec_H[:, self.K * self.N:] * 1.j).reshape(-1, self.K, self.N)
        self.mat_W = self.calc_mmse(self.mat_H).to(self.device)
        self.num_steps = 0
        return (self.mat_H, self.mat_W)

    def step(self, action):
        self.reward = self.calc_sum_rate(self.mat_H, action)
        self.mat_W = action.detach()
        self.num_steps += 1
        self.done = True if self.num_steps >= self.episode_length else False
        return self.get_state(), self.reward, self.done
    
    def get_state(self,):
        return (self.mat_H, self.mat_W)

    def generate_channel_batch(self, N, K, batch_size, subspace_dim, basis_vectors):
        coordinates = th.randn(batch_size, subspace_dim, 1)
        basis_vectors_batch = basis_vectors[:subspace_dim].T.repeat(batch_size, 1).reshape(-1, 2 * K * N, subspace_dim)
        vec_channel = th.bmm(basis_vectors_batch, coordinates).reshape(-1 ,2 * K * N) * (( 2 * K * N / subspace_dim) ** 0.5)
        return  (N * K) ** 0.5 * (vec_channel / vec_channel.norm(dim=1, keepdim = True))
    
    def calc_mmse(self, channel):
        channel = channel.to(self.device)
        lambda_ = th.ones(self.K).repeat((channel.shape[0], 1)) * self.P / self.K
        p = th.ones(self.K).repeat((channel.shape[0], 1)).to(self.device) * np.sqrt(self.P / self.K)
        effective_channel = channel.conj().transpose(1,2).to(th.cfloat).to(self.device)
        eye_N = (th.zeros(lambda_.shape[0], self.N) + 1).to(self.device)
        eye_N = th.diag_embed(eye_N)
        lambda_ = th.diag_embed(lambda_)
        channel = th.bmm(lambda_.to(th.cfloat), channel.type(th.cfloat))
        denominator = th.inverse(eye_N + th.bmm(effective_channel,channel))
        wslnr_max = th.zeros((lambda_.shape[0], self.N, self.K), dtype=th.cfloat).to(self.device)
        wslnr_max = th.bmm(denominator, effective_channel)
        wslnr_max = wslnr_max.transpose(1,2)
        wslnr_max = wslnr_max / wslnr_max.norm(dim=2, keepdim=True)
        p = th.diag_embed(p)
        W = th.bmm(p.to(th.cfloat), wslnr_max)
        return W

    def calc_sum_rate(self,channel, precoder):
        HTF = channel
        HTFGW = th.bmm(HTF.to(th.cfloat), precoder.to(th.cfloat).transpose(-1, -2))
        S = th.abs(th.diagonal(HTFGW, dim1=-2, dim2=-1))**2
        I = th.sum(th.abs(HTFGW)**2, dim=-1) - th.abs(th.diagonal(HTFGW, dim1=-2, dim2=-1))**2
        N = 1
        SINR = S/(I+N)
        return th.log2(1+SINR).sum(dim=-1).unsqueeze(-1)