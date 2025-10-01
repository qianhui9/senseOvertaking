import time
import torch
import argparse
import numpy as np
import torch.optim as optim
from model import Actor, Critic
from utils import get_action
from utils import discrete_action
from utils import discrete_action_user
from collections import deque
from hparams import HyperParams as hp
import matplotlib.pyplot as plt
import Env_UAV_network
import os
import pandas as pd
import seaborn as sns
import time
from tqdm import tqdm
from tqdm._tqdm import trange
import random
import warnings
warnings.filterwarnings("ignore", category=Warning)

sns.set(style='whitegrid', color_codes=True)
parser = argparse.ArgumentParser()
parser.add_argument('--algorithm', type=str, default='PPO',
                    help='select one of algorithms among Vanilla_PG, NPG, TPRO, PPO')
parser.add_argument('--env', type=str, default="Humanoid-v2",
                    help='name of Mujoco environement')
parser.add_argument('--render', default=False)
args = parser.parse_args()

if args.algorithm == "PG":
    from vanila_pg import train_model
elif args.algorithm == "NPG":
    from npg import train_model
elif args.algorithm == "TRPO":
    from trpo import train_model
elif args.algorithm == "PPO":
    from ppo_clip_change import train_model

if __name__ == "__main__":

    N_UAV = 4
    N_USER = 20

    position_record = np.zeros((4, 80, 80))

    num_input = 48
    num_output = 3  # 动作A：V，theta
    num_output1 = N_USER*2 # a

    actors = []
    critics = []
    actor_optims = []
    critic_optims = []
    std_change = 0.3
    for i in range(N_UAV):
        print("初始化agentUAV:", i)

        actor = Actor(num_input, num_output)
        critic = Critic(num_input)
        actor_optim = optim.Adam(actor.parameters(), lr=hp.actor_lr1)
        critic_optim = optim.Adam(critic.parameters(), lr=hp.critic_lr1, weight_decay=hp.l2_rate)

        actors.append(actor)
        critics.append(critic)
        actor_optims.append(actor_optim)
        critic_optims.append(critic_optim)

    actor1 = Actor(num_input, num_output1)
    critic1 = Critic(num_input)
    actor_optim1 = optim.Adam(actor1.parameters(), lr=hp.actor_lr)
    critic_optim1 = optim.Adam(critic1.parameters(), lr=hp.critic_lr, weight_decay=hp.l2_rate)

    actors.append(actor1)
    critics.append(critic1)
    actor_optims.append(actor_optim1)
    critic_optims.append(critic_optim1)

    episodes = 0
    xar = []
    yar = []
    best_score = 0
    for iter in tqdm(range(1000)):

        # if iter == 9999:
        #     sns.heatmap(position_record[0])
        #     plt.plot(xar, yar, linewidth=3)
        #     plt.title("movement of UAV", fontsize=19)
        #     plt.xlabel("X(m)", fontsize=10)
        #     plt.ylabel("Y(m)", fontsize=10)
        #     plt.tick_params(axis='both', labelsize=9)
        #     plt.savefig('img/heatmap_clip_change.pdf',  format='pdf')

        memorys = []
        for i in range(N_UAV + 1):
            actors[i].eval(), critics[i].eval()
            memory = deque()
            memorys.append(memory)

        steps = 0
        scores = []

        for j in range(10):
            score = 0
            episodes += 1
            np.random.seed(1234)
            env = Env_UAV_network.Environment()
            for i_step in range(1000000):
                action_all = np.zeros([N_UAV, 3])
                action_all_idx = 0
                s_all = []
                a_all = []
                for i in range(N_UAV):
                    state = env.observe_uav(i)
                    #print(state.shape)
                    s_all.append(state)
                    mu, std, _ = actors[i](torch.Tensor(state).unsqueeze(0))
                    action = get_action(mu, std)[0]

                    action_discrete = discrete_action(action)

                    a_all.append(action)
                    action_all[action_all_idx, 0] = action_discrete[0]
                    action_all[action_all_idx, 1] = action_discrete[1]
                    action_all[action_all_idx, 2] = action_discrete[2]

                    action_all_idx += 1

                state = env.observe_uav(1)

                s_all.append(state)

                mu, std, _ = actors[N_UAV](torch.Tensor(state).unsqueeze(0))

                action_user = get_action(mu, std)[0]
                action_discrete_user = discrete_action_user(action_user)

                a_all.append(action_user)

                actions_user_temp = action_discrete_user.copy()
                actions_temp = action_all.copy()
                env.update_position(actions_temp)
                # for i in range(env.n_uav):
                #     position_x, position_y = env.cal_position_record(env.bs_pos[i+1][0], env.bs_pos[i+1][1])
                #     position_record[i][position_x][position_y] += 1
                reward = env.calc_reward(actions_temp, actions_user_temp)
                next_state = env.observe_uav(0)

                done = 0

                if i_step == env.T:
                    done = 1
                if done:
                    mask = 0
                else:
                    mask = 1
                for i in range(N_UAV + 1):
                    s = s_all[i]
                    a = a_all[i]
                    memorys[i].append([s, a, reward, mask])

                score += reward
                if done == 1:  # latency 退出
                    break

            scores.append(score)

        score_avg = np.mean(scores)
        if iter % 100 == 0:
            print('iter{}, {} episode score is {:.2f}'.format(iter, episodes, score_avg))

        #if best_score < score_avg:
        if iter == 9000:
            best_score = score_avg
            checkpoint_dir = 'agent_UAV_clip_change'
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            for j in range(N_UAV + 1):
                model_path = 'agent_UAV_clip_change/actor%d.pt' % j
                torch.save(actors[j].state_dict(), model_path)
                model_path = 'agent_UAV_clip_change/critic%d.pt' % j
                torch.save(critics[j].state_dict(), model_path)

  
        clip_param = random.normalvariate(0,std_change)

        if clip_param >1:
            clip_param = 1
        elif clip_param <0.15 or iter > 3000:
            clip_param = 0.15

        
        for i in range(N_UAV + 1):
            actors[i].train(), critics[i].train()
            train_model(actors[i], critics[i], memorys[i], actor_optims[i], critic_optims[i],clip_param)
        std = std * 0.99975
        
        f = "agent_UAV_clip_change.txt"
        with open(f,"a") as file:
            file.write(str(score_avg)+"\n")


