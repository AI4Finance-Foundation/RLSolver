import torch as th
import numpy as np
import os
class beamforming_env:
    def __init__(self, num_antennas = 2, num_users = 2, total_power = 10, noise_power = 1, path_loss_option=False, path_loss_range=[-5, 5], H_transition=False):
        self.state = None
        self.N = num_antennas
        self.K =  num_users
        self.total_power = 10
        self.noise_power = 1
        self.path_loss_option = path_loss_option
        self.path_loss_range = path_loss_range
        self.max_step = 50
        self.encode_dim = 32
        self.user_weights = np.ones(self.K)
        self.wsr = 0
        self.selected_users = [i for i in range(self.K)]
        self.H_test = th.zeros((10,2,2), dtype=th.cfloat)
    
        if os.path.exists("./test_H_K2N2.npy"):
            print("True")
            with open("./test_H_K2N2.npy", "rb") as f:
                tmp = np.load(f)
            self.H_test = th.as_tensor(tmp)

        else:
            with open("./test_H_K2N2.npy", "wb") as f:
                np.save(f, self.H_test.numpy())
    
    
        self.H_mu = 0
        self.H_sigma = 1   
        self.device=th.device("cuda:0")
        self.step_ = 0
    
    def reset(self, H = -1):
    
    
        if H == -1:
            
            _, _, H, _ = self.compute_channel(self.N, self.K, self.total_power,np.array([self.H_[:, :, 0]]), self.path_loss_option, self.path_loss_range, std = 1.0)
            H = th.diag(th.randn(2, dtype=th.cfloat))
        
        else:
    
            H = self.H_test[H]
            
        
    
        self.H = th.as_tensor(H, dtype=th.cfloat).to(self.device)
    
        self.W = th.randn(self.K, self.N, dtype=th.cfloat).to(self.device)
        self.step_ = 0
    
        self.wsr = self.calc_wsr(self.user_weights, self.H.unsqueeze(0), self.W.unsqueeze(0), self.noise_power, self.selected_users)
    
        return self.get_state()
    
    def get_state(self,):
        
        mmse_net_input = self.H  
        mmse_net_target = self.W.to(th.cfloat)
        tmp = mmse_net_input.to(th.cfloat)
        
        mmse_net_input= th.cat((th.as_tensor(mmse_net_input.real).reshape(self.K * self.N), th.as_tensor(mmse_net_input.imag).reshape(self.K * self.N)), 0)
        h_w_input = th.matmul(tmp, mmse_net_target.transpose(0,1).conj())
        h = h_w_input
        mmse_net_target= th.cat((th.as_tensor(mmse_net_target.real).reshape(self.K * self.N), th.as_tensor(mmse_net_target.imag).reshape(self.K * self.N)), 0)
        h_w_input= th.cat((th.as_tensor(h_w_input.real).reshape(self.K * self.N), th.as_tensor(h_w_input.imag).reshape(self.K * self.N)), 0)
        
        self.state = th.cat((mmse_net_input.reshape(self.N * 2*self.K), 
                            mmse_net_target.reshape(self.N * 2* self.K), 
                            h_w_input.reshape(self.N * 2* self.K), tmp.reshape(self.N* self.K), 
                            h.reshape(self.K* self.K)), dim=0)
        return self.state  
  
    def step(self, action):
    
    
        precoder = action.reshape(2, self.K, self.N)
    
        precoder = precoder[0] + precoder[1] * 1.j
    
        precoder = precoder.reshape(self.K,self.N)
    
        self.W = precoder #+ self.W
    
        if self.H_transition:
            self.H = self.H + th.normal(self.H_mu, self.H_sigma, (self.K, self.N))
       
    
        wsr = self.calc_wsr(self.user_weights, self.H.unsqueeze(0), self.W.unsqueeze(0), self.noise_power, self.selected_users)
    
        reward = wsr - self.wsr
    
        self.wsr = wsr
    
        self.step_ += 1
        done = False
        if self.step_ > self.max_step:
            done = True
        
        return self.get_state(), reward, done, {}
  

    def calc_mmse(self, channel):

        K = self.K
        N = self.N

        lambda_ = th.ones(self.K).repeat((channel.shape[0], 1)).to(self.device) * self.total_power / self.K
        p = th.ones(self.K).repeat((channel.shape[0], 1)).to(self.device) * np.sqrt(self.total_power / self.K)
        effective_channel = channel.conj().transpose(1,2).to(th.cfloat).to(self.device)
        channel = channel.to(self.device)

        eye_N = (th.zeros(lambda_.shape[0], self.N) + 1).to(self.device)
        eye_N = th.diag_embed(eye_N)
        lambda_ = th.diag_embed(lambda_)

        channel = th.bmm(lambda_.to(th.cfloat), channel.type(th.cfloat))

        denominator = th.inverse(eye_N + th.bmm(effective_channel,channel))

        wslnr_max = th.zeros((lambda_.shape[0], N, K), dtype=th.cfloat).to(self.device)
        wslnr_max = th.bmm(denominator, effective_channel)
        wslnr_max = wslnr_max.transpose(1,2)
        wslnr_max = wslnr_max / wslnr_max.norm(dim=2, keepdim=True)

        p = th.diag_embed(p)

        W = th.bmm(p.to(th.cfloat), wslnr_max)
        return W

  
    def calc_wsr(self, user_weights, channel, precoder, noise_power, selected_users):
        result = 0
        num_users = self.K
        #print(th.matmul(channel[0].to(th.cfloat), precoder[0].T.to(th.cfloat)))
        for user_index in range(num_users):
            if user_index in selected_users:
                user_sinr = self.compute_sinr_th(channel, precoder, noise_power, user_index, selected_users)    
            result = result + user_weights[user_index]*th.log2(1 + user_sinr)
        return result
 
  # Computes a channel realization and returns it in two formats, one for the WMMSE and one for the deep unfolded WMMSE.
  # It also returns the initialization value of the transmitter precoder, which is used as input in the computation graph of the deep unfolded WMMSE.
    def compute_channel(self,num_antennas, num_users, total_power, H = None,):
        
        channel = np.zeros((num_users, num_antennas)) + 1j*np.zeros((num_users, num_antennas))
        
        for i in range(num_users):
            path_loss = 0 # path loss is 0 dB by default, otherwise it is drawn randomly from a uniform distribution (N.B. it is different for each user)
            
            if H.shape[0] != 1:
                result_real = H.real[i,:].reshape(-1, 1)
                result_imag  = H.imag[i, :].reshape(-1, 1) 
            else:
                result_real = np.sqrt(10**(path_loss/10))*np.sqrt(0.5)*np.random.normal(size = (num_antennas,1))
                result_imag = np.sqrt(10**(path_loss/10))*np.sqrt(0.5)*np.random.normal(size = (num_antennas,1))

            channel[i,:] = np.reshape(result_real,(1,num_antennas)) + 1j*np.reshape(result_imag, (1,num_antennas))    
        return channel

    def compute_sinr_th(self, channel, precoder, noise_power, user_id, selected_users):
        
        num_users = self.K
        numerator = (th.absolute(th.bmm(th.as_tensor(channel[:,user_id,:], dtype=th.complex64).view(channel.shape[0], 1, self.N),th.as_tensor(precoder[:, user_id,:].view(channel.shape[0], self.N, 1), dtype=th.complex64)).view(channel.shape[0], 1)))**2
        inter_user_interference = 0

        for user_index in range(num_users):
            if user_index != user_id and user_index in selected_users:
                inter_user_interference = inter_user_interference + (th.absolute(th.bmm(th.as_tensor(channel[:,user_id,:], dtype=th.complex64).view(channel.shape[0], 1, self.N),th.as_tensor(precoder[:,user_index,:], dtype=th.complex64).view(channel.shape[0], self.N, 1))).view(channel.shape[0], 1))**2
        denominator = noise_power + inter_user_interference
        result = numerator/denominator
        return result