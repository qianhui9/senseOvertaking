# -*- coding: utf-8 -*-
"""
Scenario testing: Single vehicle dring in the customized 2 lane highway map.
"""
import argparse
import importlib
import os
import sys

import time
from pathlib import Path

import carla
# from opencda.core.plan.behavior_agent import BehaviorAgent
import click
import numpy as np
import pandas as pd
import torch
import yaml
from omegaconf import OmegaConf

import opencda.scenario_testing.utils.sim_api as sim_api
import opencda.scenario_testing.utils.customized_map_api as map_api
from MARL.agents.nstep_pdqn import PDQNNStepAgent
from MARL.my_EnvCluster import LaneChangePredict

from opencda.core.common.cav_world import CavWorld
from opencda.scenario_testing.evaluations.evaluate_manager import \
    EvaluationManager
from opencda.scenario_testing.utils.yaml_utils import \
    add_current_time

# 获取项目根路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent  # 视 opencda.py 所在层级而定

def arg_parse():
    # 判断是否通过命令行运行（比如你之后想还支持命令行）
    if len(sys.argv) > 1:
        # create an argument parser   定义参数解析函数，创建参数解析器，设置描述为"OpenCDA场景运行器"
        parser = argparse.ArgumentParser(description="OpenCDA scenario runner.")
        # add arguments to the parser  添加必需参数test_scenario
        # 短选项-t和长选项--test_scenario
        # 类型为字符串
        # 帮助信息说明要测试的场景名称必须与测试脚本和YAML文件匹配
        parser.add_argument('-t', "--test_scenario", required=True, type=str,
                            help='Define the name of the scenario you want to test. The given name must'
                                 'match one of the testing scripts(e.g. single_2lanefree_carla) in '
                                 'opencda/scenario_testing/ folder'
                                 ' as well as the corresponding yaml file in opencda/scenario_testing/config_yaml.')
        # 添加可选参数--record：
        # 是一个标志参数(action='store_true')
        # 帮助信息说明是否记录仿真过程到.log文件
        parser.add_argument("--record", action='store_true',
                            help='whether to record and save the simulation process to .log file')
        # 添加可选参数--apply_ml：
        # 是一个标志参数
        # 帮助信息说明是否需要机器学习框架
        parser.add_argument("--apply_ml",
                            action='store_true',
                            help='whether ml/dl framework such as sklearn/pytorch is needed in the testing. '
                                 'Set it to true only when you have installed the pytorch/sklearn package.')
        # 添加可选参数version：
        # 短选项-v和长选项--version
        # 类型为字符串
        # 默认值'0.9.11',下面修改成自己所使用的0.9.12了
        # 帮助信息说明指定CARLA模拟器版本
        parser.add_argument('-v', "--version", type=str, default='0.9.12',
                            help='Specify the CARLA simulator version, default'
                                 'is 0.9.11, 0.9.12 is also supported.')
        # 解析参数并返回结果
        opt = parser.parse_args()
    else:
        class Args:
            test_scenario = "single_2lanefree_carla_MyTest"   # single_2lanefree_carla_MyTest
            record = False
            apply_ml = False
            version = "0.9.12"

        opt = Args()
    return opt

def mainEnv():
    # 首先解析命令行参数
    opt = arg_parse()
    # print the version of OpenCDA
    # print("OpenCDA Version: %s" % __version__)
    # set the default yaml file
    default_yaml = config_yaml = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        'config_yaml/default.yaml')
    # set the yaml file for the specific testing scenario
    config_yaml = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                               'config_yaml/%s.yaml' % opt.test_scenario)
    # load the default yaml file and the scenario yaml file as dictionaries
    default_dict = OmegaConf.load(default_yaml)
    scene_dict = OmegaConf.load(config_yaml)
    # merge the dictionaries
    scene_dict = OmegaConf.merge(default_dict, scene_dict)

    # 动态导入测试场景对应的Python模块
    testing_scenario = importlib.import_module(
        "opencda.scenario_testing.%s" % opt.test_scenario)
    # check if the yaml file for the specific testing scenario exists
    if not os.path.isfile(config_yaml):
        sys.exit(
            "opencda/scenario_testing/config_yaml/%s.yaml not found!" % opt.test_cenario)

    # 从测试模块中获取run_scenario函数
    scenario_runner = getattr(testing_scenario, 'run_scenario')
    # run the scenario testing  运行场景测试，传入参数和配置
    single_cav_list, scenario_manager, spectator, bg_veh_list = scenario_runner(opt, scene_dict)
    return single_cav_list, scenario_manager, spectator, bg_veh_list

def run_scenario(opt, scenario_params):
    scenario_params = add_current_time(scenario_params)
    current_path = os.path.dirname(os.path.realpath(__file__))
    xodr_path = os.path.join(
        current_path,
        '../assets/testMy_3lane_freeway/saveSource.xodr')

    # create CAV world
    cav_world = CavWorld(opt.apply_ml)
    # create scenario manager
    scenario_manager = sim_api.ScenarioManager(scenario_params,
                                               opt.apply_ml,
                                               opt.version,
                                               xodr_path=xodr_path,
                                               cav_world=cav_world)

    if opt.record:
        scenario_manager.client. \
            start_recorder("single_2lanefree_carla.log", True)

    single_cav_list = \
        scenario_manager.create_vehicle_manager(application=['single'],
                                                map_helper=map_api.
                                                spawn_helper_2lanefree)  # 删除 , default_model='vehicle.ford.mustang'

    # create background traffic in carla
    traffic_manager, bg_veh_list = \
        scenario_manager.create_traffic_carla()

    # create evaluation manager
    # eval_manager = \
    #     EvaluationManager(scenario_manager.cav_world,
    #                       script_name='single_2lanefree_carla',
    #                       current_time=scenario_params['current_time'])

    spectator = scenario_manager.world.get_spectator()

    return single_cav_list, scenario_manager, spectator, bg_veh_list


@click.command()
@click.option('--seed', default=0, help='Random seed.', type=int)
@click.option('--episodes', default=1, help='Number of epsiodes.', type=int)
@click.option('--evaluation-episodes', default=1000, help='Episodes over which to evaluate after training.', type=int)
@click.option('--batch-size', default=256, help='Minibatch size.', type=int)
@click.option('--gamma', default=0.99, help='Discount factor.', type=float)
@click.option('--update-ratio', default=0.1, help='Ratio of updates to samples.', type=float)
@click.option('--inverting-gradients', default=True,
              help='Use inverting gradients scheme instead of squashing function.', type=bool)
@click.option('--initial-memory-threshold', default=1000, help='Number of transitions required to start learning.',
              type=int)
@click.option('--replay-memory-size', default=5000, help='Replay memory size in transitions.', type=int)  # 500000
@click.option('--epsilon-start', default=0.95, help='Initial epsilon value.', type=float)
@click.option('--epsilon-steps', default=1000, help='Number of episodes over which to linearly anneal epsilon.',
              type=int)
@click.option('--epsilon-final', default=0.02, help='Final epsilon value.', type=float)
@click.option('--learning-rate-actor', default=0.00001, help="Actor network learning rate.", type=float)
@click.option('--learning-rate-actor-param', default=0.00001, help="Critic network learning rate.", type=float)
@click.option('--clip-grad', default=1., help="Gradient clipping.", type=float)  # 1 better than 10.
@click.option('--beta', default=0.2, help='Averaging factor for on-policy and off-policy targets.', type=float)  # 0.5
@click.option('--scale-actions', default=True, help="Scale actions.", type=bool)
# @click.option('--layers', default=(256,128,64), help='Duplicate action-parameter inputs.')
#               # cls=ClickPythonLiteralOption)
# 保存模型的频率，单位为 episode 数量，0 表示不保存，默认为 0
@click.option('--save-freq', default=1, help='How often to save models (0 = never).', type=int)
@click.option('--save-dir', default="opencda/scenario_testing/result/normal", help='Output directory.', type=str)
@click.option('--title', default="PDQN", help="Prefix of output files", type=str)

def runMARL(seed, episodes, batch_size, gamma, inverting_gradients, initial_memory_threshold, replay_memory_size,
        epsilon_steps, learning_rate_actor, learning_rate_actor_param, title, epsilon_start, epsilon_final, clip_grad,
        beta,
        scale_actions, evaluation_episodes, update_ratio, save_freq, save_dir):
    N_Vehicle = 3

    # os.makedirs('result/normal', exist_ok=True)  # 自动创建目录，若已存在则跳过

    # 判断是否生成模型保存路径
    if save_freq > 0:
        save_dir = os.path.join(save_dir, f"{title}_{seed}")
        os.makedirs(save_dir, exist_ok=True)

    model_path = 'opencda/scenario_testing/result/HMAP-DQN/PDQN_0/PDQN.pkl'
    # 加载模型参数
    checkpoint = torch.load(model_path)
    actor_state_dict = checkpoint['actor']
    actor_param_state_dict = checkpoint['actor_param']
    actor_target_state_dict = checkpoint['actor_target']
    actor_param_target_state_dict = checkpoint['actor_param_target']

    env = LaneChangePredict()
    # 设置随机种子，获取环境实例
    np.random.seed(seed)
    # 初始化 agent
    agent_class = PDQNNStepAgent
    agents = [agent_class(
        env.n_features, env.n_actions,
        actor_kwargs={
            'activation': "relu", },
        actor_param_kwargs={
            'activation': "relu", },
        batch_size=batch_size,
        learning_rate_actor=learning_rate_actor,
        learning_rate_actor_param=learning_rate_actor_param,
        epsilon_initial=epsilon_start,
        epsilon_steps=epsilon_steps,
        epsilon_final=epsilon_final,
        gamma=gamma,  # 0.99
        clip_grad=clip_grad,
        beta=beta,
        initial_memory_threshold=initial_memory_threshold,
        replay_memory_size=replay_memory_size,
        inverting_gradients=inverting_gradients,
        seed=seed + i) for i in range(N_Vehicle)]

    # Load the pre-trained weights into the agents
    for i, agent in enumerate(agents):
        agent.actor.load_state_dict(actor_state_dict)
        agent.actor_param.load_state_dict(actor_param_state_dict)
        agent.actor_target.load_state_dict(actor_target_state_dict)
        agent.actor_param_target.load_state_dict(actor_param_target_state_dict)

    # 训练智能体
    max_steps = 70

    # 用于保存最大平均回报轮次的数据
    max_episode_data = None
    max_avg_reward = -np.inf  # 初始化为负无穷大

    single_cav_list, scenario_manager, spectator, bg_veh_list = mainEnv()

    for i_eps in range(episodes):
        readFlag = 0
        single_cav_list, spectator, bg_veh_list, cav_ids = env.reset(single_cav_list, scenario_manager, bg_veh_list)

        all_reward = np.array([])
        transitions = [np.array([], dtype=np.float32).tolist() for _ in range(N_Vehicle)]
        action_all = np.zeros([N_Vehicle, env.n_actions], dtype=np.float32)   # [离散, 连续1, 连续2]
        act_all = np.zeros((N_Vehicle, 1), dtype=np.float32)
        state_all = np.zeros((N_Vehicle, env.n_features), dtype=np.float32)
        action_all_idx = 0
        all_action_par = np.zeros((N_Vehicle, env.n_actions * 2), dtype=np.float32)

        scenario_manager.tick()

        for i, agent in enumerate(agents):
            single_cav_list[i].update_info()
            single_cav_list[i].savedData_dumper()

            # 获取环境信息
            information = readSenseResult(cav_ids[i], readFlag + 1)
            state = np.array(env._findstate(i, information), dtype=np.float32)

            state = add_noise_to_state(state)
            state = np.array(state, dtype=np.float32, copy=False)


            act, act_param, all_action_parameters = agent.act(state)
            action = env.find_action(act)

            action_all[action_all_idx, 0] = action
            action_all[action_all_idx, 1] = act_param[0]
            action_all[action_all_idx, 2] = act_param[1]
            act_all[action_all_idx, 0] = act

            all_action_par[action_all_idx, :] = all_action_parameters

            state_all[action_all_idx, :] = state

            action_all_idx += 1

        current_episode_data = {i: {'speedControl': [], 'ttc': []} for i in range(N_Vehicle)}

        if single_cav_list[0].vehicle.is_alive:
            transform = single_cav_list[0].vehicle.get_transform()
            spectator.set_transform(carla.Transform(
                transform.location + carla.Location(z=70),
                carla.Rotation(pitch=-90)))
        else:
            print("[Warning] Vehicle actor is destroyed. Skipping this frame.")
            continue

        # 开始一个 episode 的循环
        for i_step in range(max_steps):
            readFlag += 1
            next_state_all = np.zeros((N_Vehicle, 29), dtype=np.float32)
            reward_all = np.zeros((N_Vehicle, 1), dtype=np.float32)
            next_action_all = np.zeros((N_Vehicle, 3), dtype=np.float32)   # [离散, 连续1, 连续2]
            next_act_all = np.zeros((N_Vehicle, 1), dtype=np.float32)
            all_terminal = np.array([], dtype=np.float32)
            next_all_action_par = np.zeros((N_Vehicle, env.n_actions * 2), dtype=np.float32)

            scenario_manager.tick()

            for i, agent in enumerate(agents):

                # 物理修正
                physicsCorrection(i, single_cav_list, scenario_manager)

                information = readSenseResult(cav_ids[i], readFlag)
                action = action_all[i, 0]
                act_param1 = action_all[i, 1]
                act_param2 = action_all[i, 2]
                env._findCurrentState(i, information)
                next_state, reward, terminal = env.step(single_cav_list[i], action, act_param1, act_param2, i, readFlag + 1)

                if i_eps == episodes - 1:
                    ttc = env.getFinaTCC()
                    speed = env.getFinaSpeed()
                    current_episode_data[i]['speedControl'].append(speed)
                    current_episode_data[i]['ttc'].append(ttc)

                next_state = np.array(next_state, dtype=np.float32, copy=False)
                next_act, next_act_param, next_all_action_parameters = agent.act(next_state)
                next_action = env.find_action(next_act)

                next_act_all[i, 0] = next_act
                next_state_all[i, :] = next_state
                next_action_all[i, 0] = next_action
                next_action_all[i, 1] = next_act_param[0]
                next_action_all[i, 2] = next_act_param[1]
                next_all_action_par[i, :] = next_all_action_parameters
                reward_all[i, 0] = reward
                all_reward = np.append(all_reward, reward)
                all_terminal = np.append(all_terminal, terminal)
                transitions[i].append(
                    [state_all[i, :], np.concatenate(([act_all[i, 0]], all_action_par[i, :])).ravel(),
                     reward_all[i, 0], next_state_all[i, :], np.concatenate(([next_act_all[i, 0]],
                                                                             next_all_action_par[i, :])).ravel(),
                     terminal])

            for i in range(N_Vehicle):
                act_all[i, 0], action_all[i, 1], all_action_par[i, :] = next_act_all[i, 0], next_action_all[
                    i, 1], next_all_action_par[i, :]
                action_all[i, 0] = next_action_all[i, 0]
                state_all[i, :] = next_state_all[i, :]

            # 使用all函数检查数组中的所有元素是否都为1
            terminal = int(any(element == 1 for element in all_terminal))
            if terminal:
                print("完成！！")
                break

        # scenario_manager.close()

        # 更新最大值和数据
        if (i_eps == episodes - 1):
            max_episode_data = current_episode_data


        agent.end_episode()
        # 平均每步的回报

    # 最后，将最大平均回报轮次的数据写入文件
    for i in range(N_Vehicle):
        df1 = pd.DataFrame({'speedControl': max_episode_data[i]['speedControl']})
        df2 = pd.DataFrame({'ttc': max_episode_data[i]['ttc']})
        df1.to_csv(f'opencda/scenario_testing/result/inference/agent{i + 1}_speedControl.csv', index=False, header=False)
        df2.to_csv(f'opencda/scenario_testing/result/inference/agent{i + 1}_ttc.csv', index=False, header=False)

def physicsCorrection(i, single_cav_list, scenario_manager):
    # 检查并纠正车辆姿态
    transform = single_cav_list[i].vehicle.get_transform()
    rotation = transform.rotation

    # 设定 roll / pitch 阈值，例如 ±10度
    if abs(rotation.roll) > 10 or abs(rotation.pitch) > 10:
        # 找到当前车道中心
        waypoint = scenario_manager.world.get_map().get_waypoint(transform.location)

        # 重置车辆姿态（位置相同，角度归正）
        safe_transform = carla.Transform(
            location=waypoint.transform.location,
            rotation=carla.Rotation(pitch=0.0, yaw=rotation.yaw, roll=0.0)
        )
        single_cav_list[i].vehicle.set_transform(safe_transform)  # 物理强制归正

# 计算每个转换的 n 步回报。第一个表示本轮训练中转换事件的转换集合、第二个表示回报衰减因子
def compute_n_step_returns(episode_transitions, gamma):
    n = len(episode_transitions)
    n_step_returns = np.zeros((n,))
    n_step_returns[n - 1] = episode_transitions[n - 1][2]  # Q-value is just the final reward
    for i in range(n - 2, 0, -1):
        reward = episode_transitions[i][2]
        target = n_step_returns[i + 1]
        n_step_returns[i] = reward + gamma * target
    return n_step_returns

def readSenseResult(folder_id, t):
    # 生成文件名（六位数字补零）
    filename = f"{t:06d}.yaml"
    base_path = Path("E:/postgraduate/V2X/senseOvertaking/OpenCDA_MyTest/data_dumping")
    subdirs = sorted([
        d for d in base_path.iterdir()
        if d.is_dir()
    ])
    # 取最后一个子文件夹（例如 2025_05_24_12_14_19）
    first_subdir = subdirs[-1]

    # 拼接到具体的 ID 子目录下（例如 103），并加上文件名
    file_path = first_subdir / str(folder_id) / filename

    with open(file_path, 'r') as f:
        data = yaml.safe_load(f)

    # 提取自车预测位置
    ego_data = {
        "predicted_ego": {
            "x": data["predicted_ego_pos"][0],
            "y": data["predicted_ego_pos"][1],
            "v": data["ego_speed"],
            "yaw": data["predicted_ego_pos"][4],
        }
    }

    # 提取周围车辆信息
    vehicles_info = []
    for vehicle_id, vehicle in data.get("vehicles", {}).items():
        vehicle_info = {
            "id": int(vehicle_id),
            "heading": vehicle["angle"][2],  # 取Z轴旋转作为航向角
            "x": vehicle["location"][0],
            "y": vehicle["location"][1],
            "speed": vehicle["speed"]
        }
        vehicles_info.append(vehicle_info)

    return {
        "timestamp": t,
        "ego": ego_data,
        "surrounding_vehicles": vehicles_info
    }

def getID(single_cav_list):
    cav_ids = [cav.vehicle.id for cav in single_cav_list]
    return cav_ids


def add_noise_to_state(state, noise_level=0.07):
    """
    在每个维度上根据该维度的值，添加相关的噪声。
    :param state: 车辆的状态（包含多个维度，元组类型）
    :param noise_level: 噪声大小（基于每个维度的值）
    :return: 添加噪声后的状态
    """
    # 将状态转为列表，便于修改
    noisy_state = list(state)  # 直接将元组转换为列表

    # 对每个维度，生成与其值相关的噪声
    for i in range(len(state)):
        value = state[i]  # 获取当前维度的值
        noise_std = noise_level * np.abs(value)  # 噪声的标准差和维度值的绝对值成正比
        noise = np.random.normal(0, noise_std)  # 生成正态分布噪声
        noisy_state[i] += noise  # 将噪声加到该维度上

    return noisy_state