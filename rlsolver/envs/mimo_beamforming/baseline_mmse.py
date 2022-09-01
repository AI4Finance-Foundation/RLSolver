import torch as th
import numpy as np


def compute_mmse_beamformer(mat_H, K=4, N=4, P=10, device=th.device("cuda:0")):
    mat_H = mat_H.to(device)
    lambda_ = th.ones(K).repeat((mat_H.shape[0], 1)) * P / K
    p = th.ones(K).repeat((mat_H.shape[0], 1)).to(device) * np.sqrt(P / K)
    effective_mat_H = mat_H.conj().transpose(1,2).to(th.cfloat).to(device)
    eye_N = (th.zeros(lambda_.shape[0], N) + 1).to(device)
    eye_N = th.diag_embed(eye_N)
    lambda_ = th.diag_embed(lambda_)
    mat_H = th.bmm(lambda_.to(th.cfloat), mat_H.type(th.cfloat))
    denominator = th.inverse(eye_N + th.bmm(effective_mat_H,mat_H))
    wslnr_max = th.zeros((lambda_.shape[0], N, K), dtype=th.cfloat).to(device)
    wslnr_max = th.bmm(denominator, effective_mat_H)
    wslnr_max = wslnr_max.transpose(1,2)
    wslnr_max = wslnr_max / wslnr_max.norm(dim=2, keepdim=True)
    p = th.diag_embed(p)
    mat_W = th.bmm(p.to(th.cfloat), wslnr_max)
    
    HW = th.bmm(mat_H, mat_W.transpose(-1, -2))
    S = th.abs(th.diagonal(HW, dim1=-2, dim2=-1))**2
    I = th.sum(th.abs(HW)**2, dim=-1) - th.abs(th.diagonal(HW, dim1=-2, dim2=-1))**2
    N = 1
    SINR = S/(I+N)
    sum_rate =  th.log2(1+SINR).sum(dim=-1).unsqueeze(-1)
    return mat_W, sum_rate

def compute_mmse_beamformer_relay(mat_H, mat_F, mat_G, K=4, N=4, P=10, noise_power=1, device=th.device("cuda:0")):
    mat_H = mat_H.to(device)
    lambda_ = th.ones(K).repeat((mat_H.shape[0], 1)) * P / K
    p = th.ones(K).repeat((mat_H.shape[0], 1)).to(device) * np.sqrt(P / K)
    effective_mat_H = mat_H.conj().transpose(1,2).to(th.cfloat).to(device)
    eye_N = (th.zeros(lambda_.shape[0], N) + 1).to(device)
    eye_N = eye_N + mat_F.flatten(start_dim=1).norm(dim=1, keepdim=True)
    eye_N = th.diag_embed(eye_N).to(device)
    lambda_ = th.diag_embed(lambda_).to(device)
    mat_H = th.bmm(lambda_.to(th.cfloat), mat_H.type(th.cfloat))
    denominator = th.inverse(eye_N + th.bmm(effective_mat_H,mat_H))
    wslnr_max = th.zeros((lambda_.shape[0], N, K), dtype=th.cfloat).to(device)
    wslnr_max = th.bmm(denominator, effective_mat_H)
    wslnr_max = wslnr_max.transpose(1,2)
    wslnr_max = wslnr_max / wslnr_max.norm(dim=2, keepdim=True)
    p = th.diag_embed(p)
    mat_W = th.bmm(p.to(th.cfloat), wslnr_max)
    HTF = th.bmm(mat_H.conj().transpose(-1,-2), mat_F)
    HTFGW = th.bmm(th.bmm(HTF.to(th.cfloat), mat_G), mat_W.to(th.cfloat).transpose(-1, -2))
    S = th.abs(th.diagonal(HTFGW, dim1=-2, dim2=-1))**2
    I = th.sum(th.abs(HTFGW)**2, dim=-1) - th.abs(th.diagonal(HTFGW, dim1=-2, dim2=-1))**2
    N = th.norm(HTF, dim=-1)**2 * 1 + noise_power
    SINR = S/(I+N)
    reward = th.log2(1+SINR).sum(dim=-1).unsqueeze(-1)
    return mat_W, reward
