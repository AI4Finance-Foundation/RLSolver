import os
import torch
from net_tsp import Policy_Net_TSP
from env_tsp import TSPEnv
from tqdm import tqdm
def train_curriculum_learning(policy_net_tsp, optimizer, save_path, device, N=4, num_epochs=400000,
                    num_epochs_per_subspace=400, num_epochs_to_save_model=1000, num_epochs_to_evaluate = 5):
    env_tsp = TSPEnv(N=N, device=device, num_env=16)
    pbar = tqdm(range(num_epochs))
    for epoch in pbar:
        state = env_tsp.reset()
        loss = 0
        while(1):
            action = policy_net_tsp(state)
            next_state, reward, done = env_tsp.step(action)
            loss -= reward.mean()
            state = next_state
            if done:
                break
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % num_epochs_to_save_model == 0:
            torch.save(policy_net_tsp.state_dict(), save_path + f"{epoch}.pth")
        if (epoch + 1) % num_epochs_per_subspace == 0 and env_tsp.subspace_dim <= 2 * K * N:
            env_tsp.subspace_dim +=1
        if epoch % num_epochs_to_evaluate == 0:
            with torch.no_grad():
                sum_rate = torch.zeros(100, env_tsp.episode_length, 1)
                for i_p in range(1):
                    state = env_tsp.reset(test=True) 
                    while(1):
                        action = policy_net_tsp(state)
                        next_state, reward, done = env_tsp.step(action)
                        sum_rate[:, env_tsp.num_steps-1, i_p] = reward.squeeze()
                        state = next_state
                        if done:
                            break
            pbar.set_description(f"id: {epoch} | test_sum_rate_SNR=5: {sum_rate[:, :, 0].max(dim=1)[0].mean()} | test_sum_rate_SNR=10: {sum_rate[:, :, 1].max(dim=1)[0].mean()} || test_sum_rate_SNR=15: {sum_rate[:, :, 2].max(dim=1)[0].mean()} | test_sum_rate_SNR=20: {sum_rate[:, :, 3].max(dim=1)[0].mean()} | test_sum_rate_SNR=25: {sum_rate[:, :, 4].max(dim=1)[0].mean()} | training_loss: {loss.mean().item() / env_tsp.episode_length:.3f} | gpu memory: {torch.cuda.memory_allocated():3d}")

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
    N = 15   # number of cities
    noise_power = 1
    learning_rate = 5e-5
    env_name = f"N{N}_TSP"
    save_path = get_cwd(env_name) 
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    policy_net_tsp = Policy_Net_TSP(mid_dim= 1024, N=N).to(device)
    optimizer = torch.optim.Adam(policy_net_tsp.parameters(), lr=learning_rate)

    try:
        train_curriculum_learning(policy_net_tsp, optimizer, N=N, save_path=save_path, device=device,)
        torch.save(policy_net_tsp.state_dict(), save_path + "policy_net_tsp_1.pth")  # number your result policy net
        print(f"saved at " + save_path)
    except KeyboardInterrupt:
        torch.save(policy_net_tsp.state_dict(), save_path + "policy_net_tsp_1.pth")  # number your result policy net
        print(f"saved at " + save_path)
        exit()

