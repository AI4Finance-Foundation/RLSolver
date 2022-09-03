import os
import torch as th
from maxcut import MaxcutEnv
from net_maxcut import Policy_Net_Maxcut
from test_maxcut import test

def train_curriculum_learning(policy_net_maxcut, optimizer, save_path, device, N=4, num_epochs=40000,
                num_epochs_per_subspace=400, num_epochs_to_save_model=1000, num_epochs_to_test=100):
    env_maxcut = MaxcutEnv(N=N, device=device)
    for epoch in range(num_epochs):
        obj_value = 0
        while(True):
            action = policy_net_maxcut(state)
            next_state, reward, done = env_maxcut.step(action)
            obj_value -= reward.mean()
            state = next_state
            if done:
                break
        optimizer.zero_grad()
        obj_value.backward()
        optimizer.step()
        print(f" training_loss: {obj_value.item():.3f} | gpu memory: {th.cuda.memory_allocated():3d}")
        if epoch % num_epochs_to_save_model == 0:
            th.save(policy_net_maxcut.state_dict(), save_path + f"{epoch}.pth")    
        if (epoch + 1) % num_epochs_per_subspace == 0 and env_maxcut.subspace_dim <= 2 * N * N:
            env_maxcut.subspace_dim +=1
        if (epoch + 1) % num_epochs_to_test == 0:
            test(policy_net_maxcut, device, N=N)
            
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
    N = 4   # number of nodes
    learning_rate = 5e-5
    
    env_name = "maxcut"
    save_path = get_cwd(env_name) # cwd (current work directory): folder to save the trained policy net
    device=th.device("cuda:0" if th.cuda.is_available() else "cpu")
    policy_net_maxcut = Policy_Net_Maxcut().to(device)
    optimizer = th.optim.Adam(policy_net_maxcut.parameters(), lr=learning_rate)
    
    try:
        train_curriculum_learning(policy_net_maxcut, optimizer, N=N, save_path=save_path, device=device)
        th.save(policy_net_maxcut.state_dict(), save_path + "policy_net_maxcut_1.pth")  # number your result policy net
    except KeyboardInterrupt:
        th.save(policy_net_maxcut.state_dict(), save_path + "policy_net_maxcut_1.pth")  # number your result policy net
        exit()
