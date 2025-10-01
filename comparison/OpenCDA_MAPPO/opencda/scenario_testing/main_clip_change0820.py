import time
import torch
import argparse
import numpy as np
import torch.optim as optim
import traci

from model import Actor, Critic
from my_Env import LaneChangePredict
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

    sumo_binary = "sumo-gui"  # SUMO的可执行文件路径，如果没有设置环境变量，需要指定完整路径
    sumocfg_file = "data/StraightRoad.sumocfg"  # SUMO配置文件路径

    sumo_cmd = [sumo_binary, "-c", sumocfg_file, "--start", "--delay", "10", "--scale", "1"]
    traci.start(sumo_cmd)

    N_UAV = 1

    num_input = 18
    num_output = 2  # 动作A：V，theta

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

    actor1 = Actor(num_input, num_output)
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
    max_episodes = 8000
    max_steps = 550
    # 创建一个 UAV 环境
    env = LaneChangePredict()
    for iter in range(max_episodes):

        memorys = []
        for i in range(N_UAV):
            actors[i].eval(), critics[i].eval()
            memory = deque()
            memorys.append(memory)

        scores = []

        score = 0
        episodes += 1
        np.random.seed(1234)
        state = env.reset()
        state = np.array(state, dtype=np.float32, copy=False)  # 获取 UAV 的状态
        for i_step in range(max_steps):
            action_all = np.zeros([N_UAV, 3])
            action_all_idx = 0
            s_all = []
            a_all = []
            for i in range(N_UAV):
                s_all.append(state)
                mu, std, _ = actors[i](torch.Tensor(state).unsqueeze(0))
                action = get_action(mu, std)[0]

                action_discrete = discrete_action(action)

                a_all.append(action)
                action_all[action_all_idx, 0] = action_discrete[0]
                action_all[action_all_idx, 1] = action_discrete[1]

                # 用于直接跟自己的环境更好地搭配
                action1 = action_discrete[0]
                action2 = action_discrete[1]

            s_all.append(state)

            next_state, reward, terminal = env.step(action1, action2)

            if iter == max_episodes - 1:
                df1 = pd.DataFrame({'speedControl': action2}, index=[i_step])
                ttc = env.getFinaTCC()
                df2 = pd.DataFrame({'TCC': ttc}, index=[i_step])
                df1.to_csv('result/ppo/speedControl0820.csv', index=False, mode='a', header=False)
                df2.to_csv('result/ppo/TTC0820.csv', index=False, mode='a', header=False)

            # 标记本轮训练是否结束（done）
            done = 1
            if i_step == max_steps - 1:
                env.writer(0)

            if terminal:
                done = 0
            for i in range(N_UAV):
                s = s_all[i]
                a = a_all[i]
                # 将状态、动作、奖励等信息存储到对应 UAV 的经验回放缓存中
                memorys[i].append([s, a, reward, done])

            # 更新状态
            state = next_state

            score += reward
            if done == 0:  # latency 退出
                break

        scores.append(score/i_step)

        print(iter, "average_episode_reward：", scores)

  
        clip_param = random.normalvariate(0,std_change)

        if clip_param >1:
            clip_param = 1
        elif clip_param <0.15 or iter > 3000:
            clip_param = 0.15

        
        for i in range(N_UAV):
            actors[i].train(), critics[i].train()
            train_model(actors[i], critics[i], memorys[i], actor_optims[i], critic_optims[i],clip_param)
        std = std * 0.99975

        df = pd.DataFrame({'Reward': scores})
        # 将数据保存为CSV文件，并去除标题行
        df.to_csv('result/ppo/reward0820.csv', index=False, mode='a', header=False)


