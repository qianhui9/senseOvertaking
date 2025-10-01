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
    from ppo import train_model

if __name__ == "__main__":

    N_UAV = 4  # 设置无人机（UAV）代理的数量为 4
    N_USER = 20  # 设置用户数量为 20

    # 创建一个形状为 (4, 80, 80) 的多维数组，用于记录无人机在网格上的位置
    position_record = np.zeros((4, 80, 80))

    num_input = 48
    num_output = 3  # 动作A：V，theta
    num_output1 = N_USER*2 # a  # 设置输出动作的维度为用户数量的两倍。这里表示无人机与用户之间的通信动作

    # 初始化用于存储多个 UAV 代理的 Actor、Critic 模型以及它们对应的优化器
    actors = []
    critics = []
    actor_optims = []
    critic_optims = []

    # 循环初始化多个 UAV 代理及其相关模型和优化器
    for i in range(N_UAV):
        print("初始化agentUAV:", i)

        # 创建 Actor 和 Critic 模型对象
        actor = Actor(num_input, num_output) # 输入为num_input，输出为num_output
        critic = Critic(num_input)  # 输入为num_input

        # 创建用于 Actor 和 Critic 模型的优化器，采用 Adam 优化算法
        actor_optim = optim.Adam(actor.parameters(), lr=hp.actor_lr1)
        critic_optim = optim.Adam(critic.parameters(), lr=hp.critic_lr1, weight_decay=hp.l2_rate)

        # 将 Actor、Critic 模型和对应的优化器分别添加到...
        actors.append(actor)
        critics.append(critic)
        actor_optims.append(actor_optim)
        critic_optims.append(critic_optim)

    # 单独初始化一个用于通信动作的 Actor-Critic 模型和优化器
    # 创建 Actor 和 Critic 模型对象
    actor1 = Actor(num_input, num_output1)
    critic1 = Critic(num_input)
    # 创建用于 Actor 和 Critic 模型的优化器
    actor_optim1 = optim.Adam(actor1.parameters(), lr=hp.actor_lr)
    critic_optim1 = optim.Adam(critic1.parameters(), lr=hp.critic_lr, weight_decay=hp.l2_rate)

    # 同样表示添加
    actors.append(actor1)
    critics.append(critic1)
    actor_optims.append(actor_optim1)
    critic_optims.append(critic_optim1)

    episodes = 0
    xar = []  # 存储训练轮数（episodes）的数据
    yar = []  # 存储每轮训练的平均分数（score_avg）
    best_score = 0  # 记录目前为止的最佳分数，初始化为0
    for iter in tqdm(range(100)):  # 10000   # 使用 tqdm 来显示训练进度

        if iter == 99:  # 9999  # 在第99轮时，绘制一个 UAV 移动热图，保存为 PDF 文件
            sns.heatmap(position_record[0])
            plt.title("movement of UAV", fontsize=19)
            plt.xlabel("X(m)", fontsize=10)
            plt.ylabel("Y(m)", fontsize=10)
            plt.tick_params(axis='both', labelsize=9)
            plt.savefig('img/heatmap.pdf',  format='pdf')

        # 存储每个 UAV 的经验回放缓存
        memorys = []
        for i in range(N_UAV + 1):
            actors[i].eval(), critics[i].eval()
            memory = deque()
            memorys.append(memory)

        steps = 0
        scores = []

        for j in range(10):
            # 初始化计分器 score 为0，表示本轮的分数
            score = 0
            episodes += 1
            # 通过随机种子初始化 UAV 环境
            np.random.seed(1234)
            # 创建一个 UAV 环境
            env = Env_UAV_network.Environment()
            for i_step in range(1000000):  # 对每个 UAV 进行动作选择和执行
                action_all = np.zeros([N_UAV, 3])
                action_all_idx = 0
                s_all = []
                a_all = []
                for i in range(N_UAV):
                    state = env.observe_uav(i)  # 获取 UAV 的状态
                    #print(state.shape)
                    s_all.append(state)  # 通过训练好的 Actor 神经网络模型计算动作的均值（mu）和标准差（std）
                    mu, std, _ = actors[i](torch.Tensor(state).unsqueeze(0))
                    # 使用动作的均值和标准差来选择动作，并得到离散动作
                    action = get_action(mu, std)[0]
                    action_discrete = discrete_action(action)

                    # 将选择的动作添加到 action_all 数组中
                    a_all.append(action)
                    action_all[action_all_idx, 0] = action_discrete[0]
                    action_all[action_all_idx, 1] = action_discrete[1]
                    # action_all[action_all_idx, 2] = action_discrete[2]

                    action_all_idx += 1

                state = env.observe_uav(1)

                s_all.append(state)

                mu, std, _ = actors[N_UAV](torch.Tensor(state).unsqueeze(0))

                action_user = get_action(mu, std)[0]
                action_discrete_user = discrete_action_user(action_user)

                a_all.append(action_user)

                actions_user_temp = action_discrete_user.copy()
                actions_temp = action_all.copy()
                # 更新 UAV 的位置和状态
                env.update_position(actions_temp)
                # for i in range(env.n_uav):
                #     position_x, position_y = env.cal_position_record(env.bs_pos[i+1][0], env.bs_pos[i+1][1])
                #     position_record[i][position_x][position_y] += 1
                # 计算奖励（reward）和下一个状态（next_state）
                reward = env.calc_reward(actions_temp, actions_user_temp)
                next_state = env.observe_uav(0)

                # 标记本轮训练是否结束（done）
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
                    # 将状态、动作、奖励等信息存储到对应 UAV 的经验回放缓存中
                    memorys[i].append([s, a, reward, mask])

                # 累积分数 score
                score += reward
                if done == 1:  # latency 退出
                    break

            scores.append(score)

        # # 计算本轮训练的平均分数 score_avg
        score_avg = np.mean(scores)
        if iter % 10 == 0:  # 1000  # 每10轮输出一次训练进度
            print('iter{}, {} episode score is {:.2f}'.format(iter, episodes, score_avg))

        #if best_score < score_avg:
        # 在第90轮时，如果当前平均分数 score_avg 超过之前的最佳分数 best_score，则更新 best_score，并将 Actor 和 Critic 模型保存到文件
        if iter == 90:  # 9000
            best_score = score_avg
            checkpoint_dir = 'agent_UAV'
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            for j in range(N_UAV + 1):
                model_path = 'agent_UAV/actor%d.pt' % j
                torch.save(actors[j].state_dict(), model_path)
                model_path = 'agent_UAV/critic%d.pt' % j
                torch.save(critics[j].state_dict(), model_path)

        # 将当前训练轮数和平均分数分别添加到 xar 和 yar 列表中
        xar.append(int(episodes))
        yar.append(score_avg)
        start_train = time.time()
        # 对每个 UAV 的 Actor 和 Critic 模型进行训练
        for i in range(N_UAV + 1):
            # 将模型设为训练模式
            actors[i].train(), critics[i].train()
            train_model(actors[i], critics[i], memorys[i], actor_optims[i], critic_optims[i])

    # 将平均分数 yar 保存为文本文件 agent_UAV.txt
    np.savetxt('agent_UAV.txt', np.array(yar))



