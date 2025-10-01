import time

import carla
import torch
import argparse
import numpy as np
import torch.optim as optim
from collections import deque
import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns
import time
from hparams import HyperParams as hp
from tqdm import tqdm
from tqdm._tqdm import trange
import warnings

from MARL.my_EnvCluster import LaneChangePredict
from MARL.utils import get_action, discrete_action
from opencda.scenario_testing.model import Actor, Critic
from opencda.scenario_testing.single_2lanefree_carla_MyTest import mainEnv, readSenseResult, physicsCorrection

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



    N_Vehicle = 3  # 设置无人机（UAV）代理的数量为 4

    num_input = 29
    num_output = 3  # 动作A：V，theta

    # 初始化用于存储多个 UAV 代理的 Actor、Critic 模型以及它们对应的优化器
    actors = []
    critics = []
    actor_optims = []
    critic_optims = []

    # 循环初始化多个 UAV 代理及其相关模型和优化器
    for i in range(N_Vehicle):
        print("初始化agent:", i)

        # 创建 Actor 和 Critic 模型对象
        actor = Actor(num_input, num_output)  # 输入为num_input，输出为num_output
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
    actor1 = Actor(num_input, num_output)
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
    max_episodes = 2000
    max_steps = 70
    # 创建一个 UAV 环境
    env = LaneChangePredict()

    single_cav_list, scenario_manager, spectator, bg_veh_list = mainEnv()

    for iter in range(max_episodes):  # 使用 tqdm 来显示训练进度

        readFlag = 0

        # 存储每个 UAV 的经验回放缓存
        memorys = []
        # 在循环初始化多个 UAV 代理时，对于每个 UAV，都创建了一个经验回放缓存
        for i in range(N_Vehicle):
            actors[i].eval(), critics[i].eval()
            memory = deque()
            # memorys则是包含所有UAV的经验回放缓存的列表
            memorys.append(memory)

        step = 0
        scores = []

        # 初始化计分器 score 为0，表示本轮的分数
        score = 0
        episodes += 1
        # 通过随机种子初始化 UAV 环境
        np.random.seed(1234)

        single_cav_list, spectator, bg_veh_list, cav_ids = env.reset(single_cav_list, scenario_manager, bg_veh_list)

        scenario_manager.tick()

        # 来确定目标车辆的ID
        deterTarIDVeh = 0

        state_all = np.zeros((N_Vehicle, env.n_features), dtype=np.float32)

        current_episode_data = {i: {'speedControl': [], 'ttc': []} for i in range(N_Vehicle)}

        for i_step in range(max_steps):  # 每轮最大步数

            readFlag += 1

            action_all = np.zeros([N_Vehicle, 3])
            action_all_idx = 0
            s_all = []
            a_all = []

            for i in range(N_Vehicle):

                single_cav_list[i].update_info()
                single_cav_list[i].savedData_dumper()

                # 物理修正
                physicsCorrection(i, single_cav_list, scenario_manager)

                # 获取环境信息
                information = readSenseResult(cav_ids[i], readFlag)
                if deterTarIDVeh == 0:
                    env.deterTarIDVehID(information)
                    deterTarIDVeh += 1


                state = env._findstate(i, information)
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
                action_all[action_all_idx, 2] = action_discrete[2]
                # print(action1, action2)

                action_all_idx += 1

            if single_cav_list[0].vehicle.is_alive:
                transform = single_cav_list[0].vehicle.get_transform()
                spectator.set_transform(carla.Transform(
                    transform.location + carla.Location(z=70),
                    carla.Rotation(pitch=-90)))
            else:
                print("[Warning] Vehicle actor is destroyed. Skipping this frame.")
                continue

            # s_all.append(state)
            all_next_state = np.array([])
            all_reward = np.array([])
            all_terminal = np.array([])

            scenario_manager.tick()



            for i in range(N_Vehicle):

                # 物理修正
                physicsCorrection(i, single_cav_list, scenario_manager)

                information = readSenseResult(cav_ids[i], readFlag)

                action1 = action_all[i, 0]
                action2 = action_all[i, 1]
                action3 = action_all[i, 2]
                env._findCurrentState(i, information)
                next_state, reward, terminal = env.step(single_cav_list[i], action1, action2, action3, i, readFlag + 1)



                ttc = env.getFinaTCC()
                speed = env.getFinaSpeed()
                current_episode_data[i]['speedControl'].append(speed)
                current_episode_data[i]['ttc'].append(ttc)



                all_next_state = np.append(all_next_state, next_state)

                state_all[i, :] = next_state
                all_reward = np.append(all_reward, reward)
                all_terminal = np.append(all_terminal, terminal)

            sumReward = sum(all_reward)

            # 使用all函数检查数组中的所有元素是否都为1
            terminal = int(any(element == 1 for element in all_terminal))

            if terminal:
                break
            for i in range(N_Vehicle):
                s = s_all[i]
                a = a_all[i]
                # 将状态、动作、奖励等信息存储到对应 UAV 的经验回放缓存中
                memorys[i].append([s, a, reward, terminal])




            # 累积分数 score
            score += sumReward

        # 最后，将最大平均回报轮次的数据写入文件
        for i in range(N_Vehicle):
            df1 = pd.DataFrame({'speedControl': current_episode_data[i]['speedControl']})
            df2 = pd.DataFrame({'ttc': current_episode_data[i]['ttc']})
            df1.to_csv(f'E:/postgraduate/V2X/senseOvertaking/comparison/OpenCDA_MAPPO/opencda/scenario_testing/result/normal/agent{i + 1}_speedControl.csv', index=False,
                       header=False)
            df2.to_csv(f'E:/postgraduate/V2X/senseOvertaking/comparison/OpenCDA_MAPPO/opencda/scenario_testing/result/normal/agent{i + 1}_ttc.csv', index=False,
                       header=False)


        scores.append(score/i_step)  # average reward
        avg_reward = score/i_step





        print(iter, "average_episode_reward：", scores)

        # 将当前训练轮数和平均分数分别添加到 xar 和 yar 列表中
        xar.append(int(episodes))
        yar.append(scores)
        start_train = time.time()
        # 对每个 UAV 的 Actor 和 Critic 模型进行训练
        for i in range(N_Vehicle):
            # 将模型设为训练模式
            actors[i].train(), critics[i].train()
            train_model(actors[i], critics[i], memorys[i], actor_optims[i], critic_optims[i])

        df = pd.DataFrame({'Reward': scores})
        # 将数据保存为CSV文件，并去除标题行
        df.to_csv('result/ppo/reward0820.csv', index=False, mode='a', header=False)



