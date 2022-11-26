import os
import torch
from env_maxcut import MaxcutEnv
from net_maxcut import Policy_Net_Maxcut
from evaluate_maxcut import evaluator


def train_curriculum_learning(policy_net_maxcut, optimizer, save_path, device, N=4, num_epochs=400000,
                num_epochs_per_subspace=400, num_epochs_to_save_model=1000, num_epochs_to_evaluate=5):
    env_maxcut = MaxcutEnv(N=N, device=device)
    evaluate_cut_value = 0
    for epoch in range(num_epochs):
        obj_value = 0
        state = env_maxcut.reset()
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
        if (epoch + 1) % num_epochs_per_subspace == 0 and env_maxcut.subspace_dim <= N * N and env_maxcut.sparsity <= 0.145:
            env_maxcut.subspace_dim +=1
            env_maxcut.sparsity += 0.005
        if (epoch + 1) % num_epochs_to_evaluate == 0:
            evaluate_cut_value = evaluator(policy_net_maxcut, device=device, N=N)
        print(f" training_loss: {obj_value.item() / env_maxcut.episode_length:.3f} | evaluate_cut_value: {evaluate_cut_value} |gpu memory: {torch.cuda.memory_allocated():3d}")
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
    N = 15   # number of nodes
    learning_rate = 5e-5
    
    env_name = "maxcut"
    save_path = get_cwd(env_name) # cwd (current work directory): folder to save the trained policy net
    save_path = None
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    policy_net_maxcut = Policy_Net_Maxcut(N=N).to(device)
    optimizer = torch.optim.Adam(policy_net_maxcut.parameters(), lr=learning_rate)
    
    train_curriculum_learning(policy_net_maxcut, optimizer, N=N, save_path=save_path, device=device)
