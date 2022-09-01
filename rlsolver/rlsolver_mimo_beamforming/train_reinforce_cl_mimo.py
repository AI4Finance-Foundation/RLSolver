import os
import torch as th
from rlsolver.envs.mimo_beamforming.net_mimo import Policy_Net_MIMO
from rlsolver.envs.mimo_beamforming.env_mimo import MIMO

def train_curriculum_learning(policy_net_mimo, optimizer, save_path, device, K=4, N=4, P=10, noise_power=1, num_epochs=40000,
                num_epochs_per_subspace=400, num_epochs_to_save_model=1000):
    env_mimo = MIMO(K=K, N=N, P=P, noise_power=noise_power, device=device)
    for epoch in range(num_epochs):
        obj_value = 0
        while(True):
            action = policy_net_mimo(state)
            next_state, reward, done = env_mimo.step(action)
            obj_value -= reward.mean()
            state = next_state
            if done:
                break
        optimizer.zero_grad()
        obj_value.backward()
        optimizer.step()
        print(f" training_loss: {obj_value.item():.3f} | gpu memory: {th.cuda.memory_allocated():3d}")
        if epoch % num_epochs_to_save_model == 0:
            th.save(policy_net_mimo.state_dict(), save_path + f"{epoch}.pth")    
        if (epoch + 1) % num_epochs_per_subspace == 0 and env_mimosubspace_dim <= 2 * K * N:
            env_mimo.subspace_dim +=1
            
def get_cwd(env_name):
    file_list = os.listdir()
    if env_name not in file_list:
        os.mkdir(env_name)
    file_list = os.listdir('./{}/'.format(env_name))
    max_exp_id = 0
    for exp_id in file_list:
        if int(exp_id) + 1 > max_exp_id:
            max_exp_id = int(exp_id) + 1
    os.mkdir('./{}/{}/'.format(env_name, max_exp_id))
    return f"./{env_name}/{max_exp_id}/"

if __name__  == "__main__":
    N = 4   # number of antennas
    K = 4   # number of users
    P = 10  # power constraint
    noise_power = 1
    learning_rate = 5e-5
    
    env_name = "mimo_beamforming"
    save_path = get_cwd(env_name) # cwd (current work directory): folder to save the trained policy net
    device=th.device("cuda:0" if th.cuda.is_available() else "cpu")
    policy_net_mimo = Policy_Net_MIMO().to(device)
    optimizer = th.optim.Adam(policy_net_mimo.parameters(), lr=learning_rate)
    
    try:
        train_curriculum_learning(policy_net_mimo, optimizer, K=K, N=N, save_path=save_path, device=device, P=P, noise_power=noise_power)
        th.save(policy_net_mimo.state_dict(), save_path + "policy_net_mimo_1.pth")  # number your result policy net
    except KeyboardInterrupt:
        th.save(policy_net_mimo.state_dict(), save_path + "policy_net_mimo_1.pth")  # number your result policy net
        exit()
