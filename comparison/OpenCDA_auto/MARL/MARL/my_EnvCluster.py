from __future__ import absolute_import
from __future__ import print_function

import csv
import logging
import math
import time

import gym
import numpy as np
import pandas as pd
from gym import spaces
import random as rn
import os
import sys
import traci
import traci.constants as tc

from mpmath import norm
from scipy import stats
from scipy.stats import norm
import torch.nn.functional as F

# we need to import python modules from the $SUMO_HOME/tools directory
from driverStyleCluster import Point
from driverStyleCluster import driverStyleCluster

try:
    sys.path.append(os.path.join(os.path.dirname(
        __file__), '../..', '..', '..', '..', "tools"))  # tutorial in tests
    sys.path.append(os.path.join(os.environ.get("SUMO_HOME", os.path.join(
        os.path.dirname(__file__), "../..", "..", "..")), "tools"))  # tutorial in docs
    from sumolib import checkBinary
except ImportError:
    sys.exit(
        "please declare environment variable 'SUMO_HOME' as the root directory of your sumo installation (it should contain folders 'bin', 'tools' and 'docs')")

# 是否打开GUI界面
gui = False
if gui:
    sumoBinary = checkBinary('sumo-gui')
else:
    sumoBinary = checkBinary('sumo')

config_path = "data/Lane3/StraightRoad.sumocfg"

class LaneChangePredict(gym.Env):
    # 定义了 gym 环境的 metadata，指定了可视化模式为 human（即显示在屏幕上）
    metadata = {'render.modes': ['human']}
    def __init__(self):

        self.minVelocity = 0
        self.maxVelocity = 30

        self.minDistanceFrontVeh = 0
        self.maxDistanceFrontVeh = 150

        self.minDistanceRearVeh = 0
        self.maxDistanceRearVeh = 150

        self.maxRoadLength = 3000

        self.minLaneNumber = 0
        self.maxLaneNumber = 2 # 车道数为3，但由于获取车道index时，index从0开始计算

        self.CommRange = 150  # 联合感知范围为150米

        self.delta_t = 0.1    # 0.1秒为一个时隙
        self.AutoCarIDAll = ['Car24', 'Car5','Car8']  # 目标车辆
        self.PrevSpeed = 0
        self.PrevVehDistance = 0
        self.VehicleIds = 0
        self.traFlowNumber = 0   # 前方范围内车辆数
        self.finaTCC = 0
        self.opi = [0, 0, 0]

        self.velbefore = 0

        self.punish = 0

        self.laneFlag = [0, 0, 0]   # 判断车辆是否发生了变道
        self.lanChangeFlag = [0, 0, 0]   # 判断车辆当前是否发生了变道
        self.triggerFlag = [0, 0, 0]  # 当完成超车后也触发优先级检测

        self.csvfile = 'data_train5.csv'

        self.overpassFlag = [0, 0, 0]
        self.AutoCarFrontID = ["", "", ""]  # 超车结束标志：前方车辆
        self.tempAutoCarFrontID = ''
        self.ttc_safe = 3  # 最低的安全TTC的值

        self.dFront = [0, 0, 0]
        self.vFront = [0, 0, 0]

        # 离散动作：左变道、保持车道、右变道
        self.action_space_vehicle =[-1, 0, 1]   # 0为不变道，-1为左变道，1为右变道
        self.n_actions = len(self.action_space_vehicle)
        self.n_actions = int(self.n_actions)
        # 连续动作：速度变化
        self.param_velocity = [0, 30]
        self.n_features = 16  # 状态的维度

        # self.actions = np.zeros((int(self.n_actions), 1 + 1))  # 第一个1表示索引，第二个1表示变道动作
        self.actions = np.array([[0, -1], [1, 0], [2, 1]])



    def reset(self):
        self.TotalReward = 0
        self.numberOfLaneChanges = 0  # 车辆变道的次数
        self.numberOfOvertakes = 0  # 完成超车的次数
        self.currentTrackingVehId = 'None'
        self.overpassFlag = [0, 0, 0]  # 设置完成标志,即是否达到终止条件
        self.laneFlag = [0, 0, 0]
        self.countOPI = [0, 0, 0]  # 用来记录优先级中opi是否需要更新

        data = pd.read_csv('modified_result2.csv', header=None)
        # 与前车的相对间距
        self.myData = data.iloc[:, 4]

        # 计算 Z 分数
        z_scores = stats.zscore(self.myData)

        # 定义阈值，通常选择 Z 分数绝对值小于某个值（例如3）
        threshold = 3

        # 剔除异常值
        filtered_data = self.myData[(np.abs(z_scores) < threshold)]

        self.filteredAbsMin = abs(min(filtered_data))

        transformed_data = np.log(filtered_data - self.filteredAbsMin)

        # 计算均值和标准差
        masked_A = transformed_data[~np.isinf(transformed_data)]  # 通过掩码将 -inf 值排除
        mean = np.mean(masked_A)  # 计算排除 -inf 值后的均值
        std_dev = np.std(masked_A)  # 计算标准差

        standard_error = std_dev / (len(masked_A) ** 0.5)

        # 计算95%置信区间
        self.confidence_interval = stats.norm.interval(0.95, loc=mean, scale=standard_error)

        # 与前车的相对速度
        self.myData2 = data.iloc[:, 5]

        # 计算 Z 分数
        z_scores2 = stats.zscore(self.myData2)

        # 剔除异常值
        filtered_data2 = self.myData2[(np.abs(z_scores2) < threshold)]

        self.filteredAbsMin2 = abs(min(filtered_data2))

        transformed_data2 = np.log(filtered_data + self.filteredAbsMin2)

        # 计算均值和标准差
        masked_A2 = transformed_data2[~np.isinf(transformed_data2)]  # 通过掩码将 -inf 值排除
        mean2 = np.mean(masked_A2)  # 计算排除 -inf 值后的均值
        std_dev2 = np.std(masked_A2)  # 计算标准差

        standard_error2 = std_dev2 / (len(masked_A2) ** 0.5)

        # 计算95%置信区间
        self.confidence_interval2 = stats.norm.interval(0.95, loc=mean2, scale=standard_error2)

        traci.close()

        # sumo_binary = "sumo-gui"  # SUMO的可执行文件路径，如果没有设置环境变量，需要指定完整路径
        # sumocfg_file = "../data/Lane3/StraightRoad.sumocfg"  # SUMO配置文件路径

        sumo_binary = "sumo"  # SUMO的可执行文件路径，如果没有设置环境变量，需要指定完整路径
        sumocfg_file = "../data/Lane3/StraightRoad.sumocfg"  # SUMO配置文件路径

        sumo_cmd = [sumo_binary, "-c", sumocfg_file, "--delay", "1", "--scale", "1"]
        # sumo_cmd = [sumo_binary, "-c", sumocfg_file, "--start", "--delay", "1", "--scale", "1"]
        traci.start(sumo_cmd)

        print('Resetting the layout')
        # 执行一次仿真步长，模拟 SUMO 执行初始化
        traci.simulationStep()

        # 驾驶风格定义相关
        if os.path.exists(self.csvfile):
            os.remove(self.csvfile)

        self.VehicleIds = traci.vehicle.getIDList()  # 获取模拟环境中所有车辆 ID 集合

        # 为每辆车都订阅指定的变量
        for veh_id in self.VehicleIds:
            traci.vehicle.subscribe(veh_id, [tc.VAR_LANE_INDEX, tc.VAR_LANEPOSITION, tc.VAR_SPEED, tc.VAR_ACCELERATION])

    # 接受一个 index 参数，在 actions 矩阵中查找并返回对应行的动作数组
    def find_action(self, index):
        return self.actions[index][1]

    def step(self, action, action_param, i):
        x = action  # 变道
        v_n = (np.tanh(action_param) + 1) * 15
        desired_speed = float(v_n.item())  # 将张量转换成Python标量
        # print("变道结果：", x,";速度结果：", v_n)
        Vehicle_Params = traci.vehicle.getAllSubscriptionResults()

        self.punish = 0  # 将上一步的惩罚先归0

        # 获取执行前车辆的速度
        self.PrevSpeed = Vehicle_Params[self.AutoCarID][tc.VAR_SPEED]
        # 获取执行前车辆所在车道的纵向位置
        self.PrevVehDistance = Vehicle_Params[self.AutoCarID][tc.VAR_LANEPOSITION]

        # 设置车辆速度为期望速度
        traci.vehicle.setSpeed(self.AutoCarID, desired_speed)

        # 存储每个时刻的车辆信息
        vehicles = traci.vehicle.getIDList()

        for vehicle_id in vehicles:
            # 获取车辆信息
            time = traci.simulation.getTime()
            vehicle_speed = traci.vehicle.getSpeed(vehicle_id)
            vehicle_accler = traci.vehicle.getAcceleration(vehicle_id)

            with open(self.csvfile, 'a', newline='') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow([time, vehicle_id, vehicle_speed, vehicle_accler])

        time = traci.simulation.getTime()
        self.lanChangeFlag[i] = 0 # 判断当前是否变道之前先将变道设置为0
        # 右变道
        if x == 1:
            # 锁定超车的目标车辆
            if self.laneFlag[i] == 0:
                self.AutoCarFrontID[i] = self.tempAutoCarFrontID
            self.laneFlag[i] = self.laneFlag[i] + 1
            self.lanChangeFlag[i] = 1
            # 处理聚类数据
            self.selectClusterData(time, i)
            laneindex = traci.vehicle.getSubscriptionResults(self.AutoCarID)[tc.VAR_LANE_INDEX]
            # 记录变道前的速度
            self.velbefore = self.state[12]
            if laneindex != 0:
                traci.vehicle.changeLane(self.AutoCarID, laneindex - 1, 100)
                self.numberOfLaneChanges += 1
                self.traFlowNumber = self.trafficFlowCal(self.state[1])[laneindex - 1]
                # 惩罚不必要的变道，如果车辆前面150米内都没有车辆，那么这种变道是无意义的，所以给出惩罚
                if self.state[3] == -1:
                    self.punish = self.punish - 1
            else:
                self.punish = self.punish - 1
        # 左变道
        elif x == -1:
            # 锁定超车的目标车辆
            if self.laneFlag[i] == 0:
                self.AutoCarFrontID[i] = self.tempAutoCarFrontID

            self.laneFlag[i] = self.laneFlag[i] + 1

            self.lanChangeFlag[i] = 1
            # 处理聚类数据
            self.selectClusterData(time, i)

            laneindex = traci.vehicle.getSubscriptionResults(self.AutoCarID)[tc.VAR_LANE_INDEX]
            # 记录变道前的速度
            self.velbefore = self.state[8]
            if laneindex != self.maxLaneNumber:
                traci.vehicle.changeLane(self.AutoCarID, laneindex + 1, 100)
                self.numberOfLaneChanges += 1
                self.traFlowNumber = self.trafficFlowCal(self.state[1])[laneindex + 1]
                # 惩罚不必要的变道，如果车辆前面150米内都没有车辆，那么这种变道是无意义的，所以给出惩罚
                if self.state[3] == -1:
                    self.punish = self.punish - 1
            else:
                self.punish = self.punish - 1
        else:
            self.selectClusterData(time, i)
            laneindex = traci.vehicle.getSubscriptionResults(self.AutoCarID)[tc.VAR_LANE_INDEX]
            self.traFlowNumber = self.trafficFlowCal(self.state[1])[laneindex]

        traci.simulationStep()

        # 更新状态
        self.state = self._findstate(i)

        # 结束标志
        self.end = self.is_overtake_complete(self.state, i)

        # 计算奖励
        reward = self.updateReward(action, self.state, i)

        return self.state, reward, self.end

    def selectClusterData(self, time, i):
        vehicle_ids = self.id_list
        laneFlag = self.laneFlag[i]
        # 创建一个新的空 DataFrame A
        self.historyData = pd.DataFrame()
        csvFirst = 'data_trainFirst.csv'
        dataFirst = pd.read_csv(csvFirst, header=None)
        data = pd.read_csv(self.csvfile, header=None)
        self.selected_data = {}
        if time < 6:
            if laneFlag == 1:
                # 将列表扁平化为单个 ID 的列表
                flat_car_ids = [id[0] for id in vehicle_ids if id]

                # 遍历每个车辆 ID，并找到在 t-1 时刻到 t 时刻的数据
                for car_id in flat_car_ids:
                    # 选取t-1到t时刻的数据
                    if time > 1:
                        self.selected_data[car_id] = data[(data[1] == car_id) & (data[0] >= time - 1) & (data[0] <= time)]
                    else:
                        self.selected_data[car_id] = data[(data[1] == car_id) & (data[0] >= time - len(data)) & (data[0] <= time)]
                    # 获取 t-6 到 t-1 的数据
                    selected_data_part = dataFirst[(dataFirst[1] == car_id) & (dataFirst[0] >= 0) & (dataFirst[0] <= 6)].iloc[:,
                                             2:4]
                    self.historyData = self.historyData.append(selected_data_part, ignore_index=True)
            elif laneFlag > 1:
                # 将列表扁平化为单个 ID 的列表
                flat_car_ids = [id[0] for id in vehicle_ids if id]

                # 遍历每个车辆 ID，并找到在 t-1 时刻到 t 时刻的数据
                for car_id in flat_car_ids:
                    # 选取t-1到t时刻的数据
                    if time > 1:
                        self.selected_data[car_id] = data[(data[1] == car_id) & (data[0] >= time - 1) & (data[0] <= time)]
                    else:
                        self.selected_data[car_id] = data[(data[1] == car_id) & (data[0] >= time - len(data)) & (data[0] <= time)]
        else:
            if laneFlag == 1:
                # 将列表扁平化为单个 ID 的列表
                flat_car_ids = [id[0] for id in vehicle_ids if id]

                # 遍历每个车辆 ID，并找到在 t-1 时刻到 t 时刻的数据
                for car_id in flat_car_ids:
                    # 选取t-1到t时刻的数据
                    self.selected_data[car_id] = data[(data[1] == car_id) & (data[0] >= time - 1) & (data[0] <= time)]
                    if self.historyData.empty:
                        # 获取 t-6 到 t-1 的数据
                        self.historyData = data[(data[1] == car_id) & (data[0] >= time - 6) & (data[0] <= time - 1)].iloc[:, 2:4]
                    else:
                        selected_data_part = data[(data[1] == car_id) & (data[0] >= time - 6) & (data[0] <= time - 1)].iloc[:,
                                             2:4]
                        self.historyData = self.historyData.append(selected_data_part, ignore_index=True)
            else:
                # 将列表扁平化为单个 ID 的列表
                flat_car_ids = [id[0] for id in vehicle_ids if id]

                # 遍历每个车辆 ID，并找到在 t-1 时刻到 t 时刻的数据
                for car_id in flat_car_ids:
                    # 选取t-1到t时刻的数据
                    self.selected_data[car_id] = data[(data[1] == car_id) & (data[0] >= time - 1) & (data[0] <= time)]

    # 计算最近1s内各车辆数据的平均值
    def calculateAverage(self):
        data = self.selected_data
        # 创建一个空字典，用于存储每个键对应的第四列和第五列的平均值
        mean_values = {}

        # 遍历字典中每个键值对
        for key, value in data.items():
            # 获取每个值（对应的 DataFrame）的第四列和第五列，然后计算平均值
            col_4 = value[2]  # 第四列的数据
            col_5 = value[3]  # 第五列的数据

            # 计算第四列和第五列的平均值
            avg_col_4 = col_4.mean()  # 速度的平均值
            avg_col_5 = col_5.mean()  # 加速度的平均值

            # 存储每个键对应的第四列和第五列的平均值
            mean_values[key] = {'averageVel': avg_col_4, 'averageAcc': avg_col_5}

        return mean_values

    def close(self):
        traci.close()

    # 计算车辆之间的距离
    def _findRearVehDistance(self, vehicleparameters):
        # 二维数组parameters，用于存储每辆车的相关信息
        parameters = [[0 for x in range(5)] for x in range(len(vehicleparameters))]
        i = 0
        d1 = -1
        d2 = -1
        d3 = -1
        d4 = -1
        d5 = -1
        d6 = -1
        v1 = -1
        v2 = -1
        v3 = -1
        v4 = -1
        v5 = -1
        v6 = -1
        # 遍历全部车辆的ID

        self.id_list = [[] for _ in range(6)]  # 创建一个包含六个空列表的二维列表，用于存储不同列的车辆 ID

        for VehID in self.VehicleIds:
            parameters[i][0] = VehID
            parameters[i][1] = vehicleparameters[VehID][tc.VAR_LANEPOSITION]  # X position
            parameters[i][2] = vehicleparameters[VehID][tc.VAR_LANE_INDEX]  # lane Index
            parameters[i][3] = vehicleparameters[VehID][tc.VAR_LANE_INDEX]  # v
            parameters[i][4] = vehicleparameters[VehID][tc.VAR_LANE_INDEX]  # a
            i = i + 1

        # 通过 X 方向的坐标值升序排序存储在二维数组 parameters 中的车辆列表
        parameters = sorted(parameters, key=lambda x: x[1])  # Sorted in ascending order based on x distance
        # Find Row with Auto Car
        # 找出目标车辆并将记录其在列表中的位置，以及RowIDAuto 变量用于存储下标，值为目标车辆所在行的位置
        index = [x for x in parameters if self.AutoCarID in x][0]
        RowIDAuto = parameters.index(index)

        # 用于计算汽车周围车辆的状态信息，包括各个方向的车辆距离 d、速度 v 等参数，并更新超车次数
        # if there are no vehicles in front
        if RowIDAuto == len(self.VehicleIds) - 1:
            d1 = -1
            v1 = -1
            d3 = -1
            v3 = -1
            d5 = -1
            v5 = -1
            self.CurrFrontVehID = 'None'
            self.CurrFrontVehDistance = 150
            # Check if an overtake has happend
            if (self.currentTrackingVehId != 'None' and (
                    vehicleparameters[self.currentTrackingVehId][tc.VAR_LANEPOSITION] <
                    vehicleparameters[self.AutoCarID][tc.VAR_LANEPOSITION])):
                self.numberOfOvertakes += 1
            # 当前超车的车辆ID也设置为 None
            self.currentTrackingVehId = 'None'
        else:
            # If vehicle is in the lowest lane（最右侧车道）, then d5,d6,v5,v6 do not exist
            if parameters[RowIDAuto][2] == 0:
                d5 = -1
                v5 = -1
                d6 = -1
                v6 = -1
            # if the vehicle is in the maximum lane index（最左侧车道）, then d3.d4.v3.v4 do not exist
            elif parameters[RowIDAuto][2] == (self.maxLaneNumber - 1):
                d3 = -1
                v3 = -1
                d4 = -1
                v4 = -1
            # find d1 and v1  从当前行向下搜索车辆，以查找前方车辆的状态参数
            index = RowIDAuto + 1
            # 如果存在同一车道上的前方车辆，则计算前方车辆与当前车辆之间的距离 d1和速度 v1
            while index != len(self.VehicleIds):
                if parameters[index][2] == parameters[RowIDAuto][2]:
                    d1 = parameters[index][1] - parameters[RowIDAuto][1]
                    v1 = vehicleparameters[parameters[index][0]][tc.VAR_SPEED]
                    # 聚类模块
                    veh_id_d1 = parameters[index][0]
                    self.tempAutoCarFrontID = veh_id_d1
                    self.id_list[0].append(veh_id_d1)
                    break
                index += 1
            # there is no vehicle in front
            if index == len(self.VehicleIds):
                d1 = -1
                v1 = -1
                self.CurrFrontVehID = 'None'
                self.CurrFrontVehDistance = 150
            # find d3 and v3  从当前行向下搜索车辆，以查找右侧车道的前方车辆的状态参数
            index = RowIDAuto + 1
            # 如果左侧车道存在前方车辆，则计算其于当前车辆之间的距离 d3 和速度 v3
            while index != len(self.VehicleIds):
                if parameters[index][2] == (parameters[RowIDAuto][2] + 1):
                    d3 = parameters[index][1] - parameters[RowIDAuto][1]
                    v3 = vehicleparameters[parameters[index][0]][tc.VAR_SPEED]
                    # 聚类模块
                    veh_id_d3 = parameters[index][0]
                    self.id_list[2].append(veh_id_d3)
                    break
                index += 1
            # there is no vehicle in front
            if index == len(self.VehicleIds):
                d3 = -1
                v3 = -1
            # find d5 and v5
            index = RowIDAuto + 1
            # 如果右侧车道存在前方车辆，则计算其于当前车辆之间的距离 d5 和速度 v5
            while index != len(self.VehicleIds):
                if parameters[index][2] == (parameters[RowIDAuto][2] - 1):
                    d5 = parameters[index][1] - parameters[RowIDAuto][1]
                    v5 = vehicleparameters[parameters[index][0]][tc.VAR_SPEED]
                    # 聚类模块
                    veh_id_d5 = parameters[index][0]
                    self.id_list[4].append(veh_id_d5)
                    break
                index += 1
            # there is no vehicle in front
            if index == len(self.VehicleIds):
                d5 = -1
                v5 = -1
            # find d2 and v2
            index = RowIDAuto - 1
            # 如果存在同一车道上的后方车辆，则计算后方车辆与当前车辆之间的距离 d2 速度 v2
            while index >= 0:
                if parameters[index][2] == parameters[RowIDAuto][2]:
                    d2 = parameters[RowIDAuto][1] - parameters[index][1]
                    v2 = vehicleparameters[parameters[index][0]][tc.VAR_SPEED]
                    # 聚类模块
                    veh_id_d2 = parameters[index][0]
                    self.id_list[1].append(veh_id_d2)
                    break
                index -= 1
            # 如果同一车道上没有后方车辆
            if index < 0:
                d2 = -1
                v2 = -1
            # find d4 and v4
            # 类似地，计算右侧和左侧车道的后方车辆状态参数d4、v4、d6 和 v6
            index = RowIDAuto - 1
            while index >= 0:
                if parameters[index][2] == (parameters[RowIDAuto][2] + 1):
                    d4 = parameters[RowIDAuto][1] - parameters[index][1]
                    v4 = vehicleparameters[parameters[index][0]][tc.VAR_SPEED]
                    # 聚类模块
                    veh_id_d4 = parameters[index][0]
                    self.id_list[3].append(veh_id_d4)
                    break
                index -= 1
            if index < 0:
                d4 = -1
                v4 = -1
            # find d6 and v6
            index = RowIDAuto - 1
            while index >= 0:
                if parameters[index][2] == (parameters[RowIDAuto][2] - 1):
                    d6 = parameters[RowIDAuto][1] - parameters[index][1]
                    v6 = vehicleparameters[parameters[index][0]][tc.VAR_SPEED]
                    # 聚类模块
                    veh_id_d6 = parameters[index][0]
                    self.id_list[5].append(veh_id_d6)
                    break
                index -= 1
            if index < 0:
                d6 = -1
                v6 = -1
            # Find if any overtakes has happend
            if (self.currentTrackingVehId != 'None' and (
                    vehicleparameters[self.currentTrackingVehId][tc.VAR_LANEPOSITION] <
                    vehicleparameters[self.AutoCarID][tc.VAR_LANEPOSITION])):
                self.numberOfOvertakes += 1
            # 将当前正在追踪的前方车辆ID设置为当前车道上的下一辆车辆的ID  这个ID存储在列表 parameters 中的第 RowIDAuto + 1 行第一个元素中，即该车辆的ID。这是一个用于跟踪当前车辆前方的车辆的ID。
            self.currentTrackingVehId = parameters[RowIDAuto + 1][0]
        if RowIDAuto == 0:  # This means that there is no car behind  没有后方车辆
            RearDist = -1
        else:  # There is a car behind return the distance between them
            RearDist = (parameters[RowIDAuto][1] - parameters[RowIDAuto - 1][
                1])  # 如果当前存在后方的车辆，计算当前车辆和后方车辆之间的距离，即当前车辆的位置减去上一行车道上的车辆的位置
        # Return car in front distance
        if RowIDAuto == len(self.VehicleIds) - 1:  # 没有前方车辆
            FrontDist = -1
            # Save the current front vehicle Features
            self.CurrFrontVehID = 'None'
            self.CurrFrontVehDistance = 150
        else:
            FrontDist = (parameters[RowIDAuto + 1][1] - parameters[RowIDAuto][
                1])  # 计算当前车辆和前方车辆之间的距离，即下一行车道上的车辆的位置减去当前车辆的位置，这是计算前方车辆间距的方法
            # Save the current front vehicle Features
            self.CurrFrontVehID = parameters[RowIDAuto + 1][0]
            self.CurrFrontVehDistance = FrontDist
        # return RearDist, FrontDist
        return d1, v1, d2, v2, d3, v3, d4, v4, d5, v5, d6, v6

    def _findCurrentState(self,i):
        self.AutoCarID = self.AutoCarIDAll[i]
        self.state = self._findstate(i)

    def _findCurrentOtherState(self, j, i):
        self.AutoCarID = self.AutoCarIDAll[j]
        state = self._findstate(j)
        self.AutoCarID = self.AutoCarIDAll[i]
        return state

    def _findstate(self, i):
        self.AutoCarID = self.AutoCarIDAll[i]
        # 使用getAllSubscriptionResults()方法获取已订阅车辆的状态列表
        VehicleParameters = traci.vehicle.getAllSubscriptionResults()
        # find d1,v1,d2,v2,d3,v3,d4,v4, d5, v5, d6, v6  调用该函数来查找后方的车辆的距离和速度，并将它们分配给相应的变量。
        d1, v1, d2, v2, d3, v3, d4, v4, d5, v5, d6, v6 = self._findRearVehDistance(VehicleParameters)
        # 检查前方车辆距离 d1是否小于通信范围，如果在通信范围之外，则将其设置为最大可能距离。如果前方没有车辆，则将其设置为最大距离。
        if ((d1 > self.CommRange)):
            d1 = self.maxDistanceFrontVeh
            v1 = -1
        elif d1 < 0:  # if there is no vehicle ahead in L0
            d1 = self.maxDistanceFrontVeh  # as this can be considered as vehicle is far away
        # 检查前方车速 v1 是否为负数，如果为负数，则将其设置为零。这通常会出现在没有前方车辆或者前方车辆被超车时
        if ((v1 < 0) and (d1 <= self.CommRange)):
            # there is no vehicle ahead in L0 or there is a communication error: # there is no vehicle ahead in L0
            v1 = 0

        # 检查后方车辆距离 d2 是否大于通信范围，如果是，则将其设置为最大可能距离。如果后方没有车辆，则将其设置为零，以避免出现负回报
        if ((d2 > self.CommRange)):
            d2 = self.maxDistanceRearVeh
            v2 = -1
        elif d2 < 0:  # There is no vehicle behind in L0
            d2 = 0  # to avoid negetive reward
        # 检查后方车速 v2 是否为负数，如果为负数，则将其设置为零。这通常会出现在没有后方车辆或者后方车辆被超车时
        if ((v2 < 0) and (d2 <= self.CommRange)):
            # there is no vehicle behind in L0 or there is a communication error
            v2 = 0
        if ((d3 > self.CommRange)):
            d3 = self.maxDistanceFrontVeh
            v3 = -1
        elif d3 < 0: # no vehicle ahead in L1
            d3 = self.maxDistanceFrontVeh # as this can be considered as vehicle is far away
        if ((v3 < 0) and (d3 <= self.CommRange)) : # there is no vehicle ahead in L1 or there is a communication error: # there is no vehicle ahead in L1
            v3 = 0

        if ((d4 > self.CommRange)):
            d4 = self.maxDistanceRearVeh
            v4 = -1
        elif d4 < 0: #There is no vehicle behind in L1
            d4 = self.maxDistanceRearVeh # so that oue vehicle can go to the overtaking lane
        if ((v4 < 0) and (d4 <= self.CommRange)) : # there is no vehicle behind in L1 or there is a communication error: # there is no vehicle behind in L1
            v4 = 0

        if ((d5 > self.CommRange)):
            d5 = self.maxDistanceFrontVeh
            v5 = -1
        elif d5 < 0: # no vehicle ahead in L1
            d5 = self.maxDistanceFrontVeh # as this can be considered as vehicle is far away
        if ((v5 < 0) and (d5 <= self.CommRange)) : # there is no vehicle ahead in L1 or there is a communication error: # there is no vehicle ahead in L1
            v5 = 0

        if ((d6 > self.CommRange)):
            d6 = self.maxDistanceRearVeh
            v6 = -1
        elif d6 < 0: #There is no vehicle behind in L1
            d6 = self.maxDistanceRearVeh # so that oue vehicle can go to the overtaking lane
        if ((v6 < 0) and (d6 <= self.CommRange)): # there is no vehicle behind in L1 or there is a communication error: # there is no vehicle behind in L1
            v6 = 0

        # 获取当前车速 va
        va = VehicleParameters[self.AutoCarID][tc.VAR_SPEED]
        # 获取执行前车辆所在车道的纵向位置
        da = VehicleParameters[self.AutoCarID][tc.VAR_LANEPOSITION]
        # 获取执行前车辆前方车辆的纵向位置
        # 锁定超车的目标车辆
        if self.laneFlag[i] != 0:
            id = self.AutoCarFrontID[i]
            self.dFront[i] = VehicleParameters[id][tc.VAR_LANEPOSITION]
            self.vFront[i] = VehicleParameters[id][tc.VAR_SPEED]
        # Vehicle acceleration rate 计算速度加速度 vacc。由于时间步长为 1 秒，所以可以用当前速度和上一个时间步长的速度差来计算速将度加速度
        vacc = (va - self.PrevSpeed)/self.delta_t  # as the time step is 1sec long
        # print("d1, v1, d2, v2, d3, v3, d4, v4, d5, v5, d6, v6:", d1, v1, d2, v2, d3, v3, d4, v4, d5, v5, d6, v6)
        # 这些参数是用于车辆行驶过程中的决策和控制，例如加速和转向
        return va, da, v1, d1, v2, d2, v3, d3, v4, d4, v5, d5, v6, d6, VehicleParameters[self.AutoCarID][tc.VAR_LANE_INDEX], vacc

    # 获取联网车辆的状态信息
    def getCVInformation(self):
        distance = {}
        lane_id = {}
        vehicle_speed = {}
        for index, vehicle_id in enumerate(self.AutoCarIDAll):
            # 获取车辆信息
            distance[index] = traci.vehicle.getLanePosition(vehicle_id)
            lane_id[index] = traci.vehicle.getLaneID(vehicle_id).split('_')[-1]
            vehicle_speed[index] = traci.vehicle.getSpeed(vehicle_id)

        return distance, lane_id, vehicle_speed


    # 程序终止条件
    def is_overtake_complete(self, state, i):
        if state[1] >= 400:
            self.overpassFlag[i] = 1
        return self.overpassFlag[i]

    # 超车完成标志
    def is_laneOvertakeComp(self, state, i):
        delta_v = abs(state[0] - self.vFront)
        overtake_distance = self.ttc_safe * delta_v
        if (state[1] - self.dFront[i] - 5) >= overtake_distance:
            self.laneFlag[i] = 0
            self.triggerFlag[i] = 0
            # 对于完成了超车动作的车辆设置奖励
            self.punish = self.punish + 1
            self.countOPI[i] = 0

        # 可能会由于决策等因素导致完成对目标车辆的超车，此时需要更换新的目标车辆
        if self.dFront[i] - state[1] > 100:
            self.laneFlag[i] = 0
            # 未完成超车动作，必须给出惩罚
            self.punish = self.punish - 1

    # 车流量计算,这里state就是当前车辆行驶的距离
    def trafficFlowCal(self, state):
        # 目标车辆前方范围
        front_distance_min = -500
        front_distance_max = 500
        front_position_y_min = state + front_distance_min
        front_position_y_max = state + front_distance_max
        if front_position_y_max > self.maxRoadLength:
            front_position_y_max = self.maxRoadLength
        if front_position_y_min < 0:
            front_position_y_min = 0
        L = front_position_y_max - front_position_y_min
        # 获取目标车道的车流量
        target_lane0 = 'Lane_0'
        target_lane1 = 'Lane_1'
        target_lane2 = 'Lane_2'
        target_lane0_vehicles = traci.lane.getLastStepVehicleIDs(target_lane0)
        target_lane1_vehicles = traci.lane.getLastStepVehicleIDs(target_lane1)
        target_lane2_vehicles = traci.lane.getLastStepVehicleIDs(target_lane2)
        # 初始化各车道的车流量字典，并设置初始值为0
        lane_traffic = {0: 0, 1: 0, 2: 0}  # key：0车道，1车道，2车道
        VehicleParameters = traci.vehicle.getAllSubscriptionResults()
        for veh_id in target_lane0_vehicles:
            y = VehicleParameters[veh_id][tc.VAR_LANEPOSITION]
            if y >= front_position_y_min and y <= front_position_y_max:
                lane_traffic[0] += 1
        for veh_id in target_lane1_vehicles:
            y = VehicleParameters[veh_id][tc.VAR_LANEPOSITION]
            if y >= front_position_y_min and y <= front_position_y_max:
                lane_traffic[1] += 1
        for veh_id in target_lane2_vehicles:
            y = VehicleParameters[veh_id][tc.VAR_LANEPOSITION]
            if y >= front_position_y_min and y <= front_position_y_max:
                lane_traffic[2] += 1

        # print("各车道车流量：", lane_traffic)
        return lane_traffic

    def calTTCDri(self, action, state, i):
        x = action  # 变道

        w_front = 0.5

        # 结合驾驶风格
        d_a = -0.1
        d_n = 0
        d_d = 0.2
        dsID = self.id_list
        u = self.driverStyleReward(i)

        leftFrontDsId = 0
        leftBehDsId = 0
        rightFrontDsId = 0
        rightBehDsId = 0
        middleFrontDsId = 0

        u_lf = []
        u_lb = []
        u_rf = []
        u_rb = []
        u_mb = []


        if x == -1:
            # 左变道计算 TCC
            if state[6] != -1:
                if self.laneFlag[i] != 0:
                    if len(dsID[2]) > 0:  # 检查索引位置是否存在且不为空列表
                        leftFrontDsId = dsID[2][0]
                    for car, values in u.items():
                        if car == leftFrontDsId:
                            u_lf = values
                    if leftFrontDsId != 0 & len(u_lf) != 0:
                        dr = u_lf[0] * d_a + u_lf[1] * d_n + u_lf[2] * d_d
                    else:
                        dr = 1
                else:
                    dr = 1
                delta_V1 = state[0] - state[6]
                delta_D1 = state[7]
                if delta_V1 <= 0:  # 车辆A速度比车辆B快
                    TCC_front = 10  # 设置一个较大的值，表示车辆A和车辆B之间的时间间隔很大
                else:
                    TCC_front = (delta_D1 / delta_V1) * dr
            else:
                TCC_front = 10     # 默认TCC很大，默认为5
            if state[8] != -1:
                if self.laneFlag[i] != 0:
                    if len(dsID[3]) > 0:  # 检查索引位置是否存在且不为空列表
                        leftBehDsId = dsID[3][0]
                    for car, values in u.items():
                        if car == leftBehDsId:
                            u_lb = values
                    if leftBehDsId != 0 & len(u_lb) != 0:
                        dr = u_lb[0] * d_a + u_lb[1] * d_n + u_lb[2] * d_d
                    else:
                        dr = 1
                else:
                    dr = 1
                delta_V2 = state[0] - state[8]
                delta_D2 = state[9]
                if delta_V2 >= 0:
                    TCC_back = 10  # 设置一个较大的值，表示车辆A和车辆B之间的时间间隔很大
                else:
                    TCC_back = (delta_D2 / delta_V2) * dr
            else:
                TCC_back = 10     # 默认TCC很大，默认为5
            if abs(TCC_front) > 10:
                TCC_front = 10
            if abs(TCC_back) > 10:
                TCC_back = 10
            TCC_surround = w_front * TCC_front + (1 - w_front) * TCC_back  # 前后车的 TCC 是综合计算的

        elif x == 1:
            if state[10] != -1:
                if self.laneFlag[i] != 0:
                    if len(dsID[4]) > 0:  # 检查索引位置是否存在且不为空列表
                        rightFrontDsId = dsID[4][0]
                    for car, values in u.items():
                        if car == rightFrontDsId:
                            u_rf = values
                    if rightFrontDsId != 0 & len(u_rf) != 0:
                        dr = u_rf[0] * d_a + u_rf[1] * d_n + u_rf[2] * d_d
                    else:
                        dr = 1
                else:
                    dr = 1
                delta_V1 = state[0] - state[10]
                delta_D1 = state[11]
                if delta_V1 <= 0:
                    TCC_front = 10  # 设置一个较大的值，表示车辆A和车辆B之间的时间间隔很大
                else:
                    TCC_front = (delta_D1 / delta_V1) * dr
            else:
                TCC_front = 10     # 默认TCC很大，默认为5
            if state[12] != -1:
                if self.laneFlag[i] != 0:
                    if len(dsID[5]) > 0:  # 检查索引位置是否存在且不为空列表
                        rightBehDsId = dsID[5][0]
                    for car, values in u.items():
                        if car == rightBehDsId:
                            u_rb = values
                    if rightFrontDsId != 0 & len(u_rb) != 0:
                        dr = u_rb[0] * d_a + u_rb[1] * d_n + u_rb[2] * d_d
                    else:
                        dr = 1
                else:
                    dr = 1
                delta_V2 = state[0] - state[12]
                delta_D2 = state[13]
                if delta_V2 >= 0:
                    TCC_back = 10  # 设置一个较大的值，表示车辆A和车辆B之间的时间间隔很大
                else:
                    TCC_back = (delta_D2 / delta_V2) * dr
            else:
                TCC_back = 10
            if abs(TCC_front) > 10:
                TCC_front = 10
            if abs(TCC_back) > 10:
                TCC_back = 10
            TCC_surround = w_front * TCC_front + (1 - w_front) * TCC_back  # 前后车的 TCC 是综合计算的

        else:
            if state[2] != -1:
                if self.laneFlag[i] != 0:
                    if len(dsID[0]) > 0:  # 检查索引位置是否存在且不为空列表
                        middleFrontDsId = dsID[0][0]
                    for car, values in u.items():
                        if car == middleFrontDsId:
                            u_mb = values
                    if middleFrontDsId != 0 & len(u_mb) != 0:
                        dr = u_mb[0] * d_a + u_mb[1] * d_n + u_mb[2] * d_d
                    else:
                        dr = 1
                else:
                    dr = 1
                delta_V = state[0] - state[2]
                delta_D = state[3]
                if delta_V <= 0:  # 车辆A速度比车辆B快
                    TCC_front = 10  # 设置一个较大的值，表示车辆A和车辆B之间的时间间隔很大
                else:
                    TCC_front = (delta_D / delta_V) * dr
            else:
                TCC_front = 10
            if abs(TCC_front) > 10:
                TCC_front = 10
            TCC_surround = TCC_front

        finaTCC = TCC_surround

        return finaTCC

    # 优先级触发机制
    def priority_trigger_mechanism(self, state, i):
        if state[3] != -1 and (state[3] > self.filteredAbsMin) and (
                (state[0] - state[2] + self.filteredAbsMin2) > 0):
            transformed_new_x = np.log(state[3] - self.filteredAbsMin)

            transformed_new_x2 = np.log((state[0] - state[2]) + self.filteredAbsMin2)

            if (self.confidence_interval[0] <= transformed_new_x <= self.confidence_interval[1]) and (
                    self.confidence_interval2[0] <= transformed_new_x2 <= self.confidence_interval2[1]):
                self.triggerFlag[i] = 1

    # 车辆优先级设置
    def priority_settings(self, state, i):
        th = 1.2  # 预定义的车头时距时间阈值
        w1 = 1
        w2 = 1
        w3 = 1
        w4 = 1

        # 车速差异
        if state[0] > state[2]:
            fv = self.min_max_normalize(state[2] - state[0], 0, 30)
        else:
            fv = 0

        # 车辆间距
        if state[0] > state[2]:
            fdTemp = -math.log((state[3] - 5) / (th * (state[0] - state[2])))
            # 将对数值限制在[-1, 1]范围内
            fd = np.clip(fdTemp, -1, 1)
        else:
            fd = 0

        # 交通流量
        laneTraffic = self.trafficFlowCal(self.state[1])
        laneindex = state[14]
        if laneindex < self.maxLaneNumber and laneindex > self.minLaneNumber:
            laneTraLeft = laneTraffic[laneindex + 1]
            laneTraRight = laneTraffic[laneindex - 1]
        elif laneindex == self.maxLaneNumber:
            laneTraLeft = -1
            laneTraRight = laneTraffic[laneindex - 1]
        elif laneindex == self.minLaneNumber:
            laneTraLeft = -1
            laneTraRight = laneTraffic[laneindex + 1]

        if laneTraLeft != 0 and laneTraRight != 0:
            ft = max(self.min_max_normalize(laneTraffic[laneindex] / laneTraLeft, 0, 30), self.min_max_normalize(laneTraffic[laneindex] / laneTraRight, 0, 30))
        else:
            ft = 1

        # 安全条件
        ttc = max(self.calTTCDri(-1, state, i), self.calTTCDri(1, state, i))
        if ttc <= self.ttc_safe-1:
            fs = self.min_max_normalize(ttc, 0, self.ttc_safe-1)
        else:
            fs = 1

        # 生成服从 N(0, 0.001) 分布的小型随机变量
        sigma_t = np.random.normal(0, 0.001)

        # 超车优先级指数opi
        opi = w1 * fv + w2 * fd + w3 * ft + w4 * fs + sigma_t

        print("Priority detection activated!!")
        # 除以4是为了让opi在1的范围内或者左右
        return opi / 3

    # 变道冲突-优先级问题
    def check_lane_change_conflict(self, lineEgo, lineOther, disEgo, disOther, laneChangeFlag, i):
        if laneChangeFlag[i] == 1:
            distance_threshold = 20  # 冲突距离阈值
            for lc in range(len(laneChangeFlag)):
                if lc != i and laneChangeFlag[lc] == 1 and lineEgo == lineOther:
                    position_diff = abs(disEgo - disOther)
                    if position_diff < distance_threshold:
                        return True  # 存在冲突
        return False  # 无冲突

    # 变道冲突-速度问题
    def check_velocity_conflict(self, preVel, currVel, i, disEgo, disOther, laneChangeFlag):
        # 检查是否有车辆在减速让道，并且让道的对象是优先级更高的车辆
        distance_threshold = 20  # 假定的距离阈值，可以根据实际情况调整
        position_diff = abs(disEgo - disOther)

        if position_diff < distance_threshold:
            # 两车距离足够近，让道可能有意义
            for lc in range(len(laneChangeFlag)):
                if lc != i and laneChangeFlag[lc] == 1 and currVel < preVel:
                    return True  # ego车辆让速
        return False

    # 处理奖励函数，使其各个参数范围相似
    def min_max_normalize(self, value, min_value, max_value):
        return (value - min_value) / (max_value - min_value)

   # 计算聚类的奖励
    def driverStyleReward(self, i):
        clusterCenterNumber = 3
        weight = 2
        ds = driverStyleCluster()
        new_memberships = {}
        mean_value = self.calculateAverage()
        if self.laneFlag[i] == 1:
            data = self.historyData.values.tolist()

            # 创建数据点列表
            points = [Point(clusterCenterNumber) for _ in range(len(data))]

            # 使用CSV数据来初始化数据点
            for i, point in enumerate(points):
                x, y = float(data[i][0]), float(data[i][1])
                point.x = x
                point.y = y

            filtered_data = ds.remove_outliers(points)
            self.clusterCenterGroup, clusterCenterTrace = ds.fuzzyCMeansClustering(filtered_data, clusterCenterNumber, weight)
            categories = ds.categorize_clusters(filtered_data, self.clusterCenterGroup)
            # ds.showClusterAnalysisResults(filtered_data, clusterCenterTrace, categories)

            for car, values in mean_value.items():
                new_data_point = Point(clusterCenterNumber, x=values['averageVel'], y=values['averageAcc'])
                new_memberships[car] = ds.calculateMembership(new_data_point, self.clusterCenterGroup, weight)
        else:
            # 根据 mean_value 中的键值对替换 new_data_point 的 x 和 y 值
            for car, values in mean_value.items():
                new_data_point = Point(clusterCenterNumber, x=values['averageVel'], y=values['averageAcc'])
                ds.updateClusterCenter(new_data_point, self.clusterCenterGroup, weight)
                # 将 'car' 作为键，'new_data_point' 作为对应的值添加到新字典中
                new_memberships[car] = ds.calculateMembership(new_data_point, self.clusterCenterGroup, weight)

        return new_memberships

    # 奖励
    def updateReward(self, action, state, i):
        a_max = 5  # 一般设置在 3 - 5 m/s2

        w_sd = 1
        w_comfVel = 1
        w_ef = 1
        w_pr = 1
        w_no = 1

        # 安全性
        TCC_surround = self.calTTCDri(action, state, i)
        self.finaTCC = TCC_surround

        if (TCC_surround <= self.ttc_safe - 1) and TCC_surround > 0:
            # 控制在0-1之间（负奖励）
            r_dis = -1 * self.min_max_normalize(TCC_surround, 0, self.ttc_safe - 1)
        elif TCC_surround < 0:
            r_dis = -1
        else:
            # 控制在0-1之间
            r_dis = 1

        # # 如果未完成超车，TTC即使很大，奖励也应该很小
        # # 设置调节系数
        # adjustment_coefficient = 0.1
        # # 判断是否超过了前方目标车辆
        # if state[1] > self.dFront[i]:
        #     # 超过前方目标车辆，正常计算TCC值
        #     r_safe = r_dis
        # else:
        #     r_safe = r_dis * adjustment_coefficient

        r_sd = r_dis

        # reward related to efficiency
        if action == -1:
            velafter = state[8]
        elif action == 1:
            velafter = state[12]
        else:
            velafter = -1
        if velafter != -1:
            r_ef = (velafter - self.velbefore) / self.maxVelocity
        else:
            r_ef = 1

        # reward related to 舒适度
        r_comf = - self.min_max_normalize(abs(state[15]), 0, a_max)
        # 速度差异越大，奖励越高，鼓励追赶前方车辆
        if state[0] > self.vFront[i]:
            r_v = 2 * self.min_max_normalize(state[0] - self.vFront[i], 0, 15)  # 15=30/2
        else:
            r_v = -2 * self.min_max_normalize(self.vFront[i] - state[0], 0, 15)

        r_comfVel = r_comf + r_v

        # reward related to priority
        self.priority_trigger_mechanism(state, i)
        r_pr = 1

        if self.triggerFlag.count(1) > 0:
            if self.countOPI[i] == 0:
                self.countOPI[i] += 1
                self.opi[i] = self.priority_settings(state, i)
            distance, lane_id, vehicle_speed = self.getCVInformation()
            # 关于冲突的优先级
            egoVehicle = self.AutoCarIDAll[i]
            for veh, _ in enumerate(self.AutoCarIDAll):
                r_pp = np.array([])
                if egoVehicle != self.AutoCarIDAll[veh]:
                    r_pr1 = 0
                    r_pr2 = 0
                    # 优先级-让道相关
                    flagGiveLine = self.check_lane_change_conflict(lane_id[i], lane_id[veh], distance[i],
                                                              distance[veh], self.lanChangeFlag, i)

                    currentOtherState = self._findCurrentOtherState(veh, i)
                    currentOtherPri = self.priority_settings(currentOtherState, i)
                    # 获取相对其他车辆的优先级
                    priority_diff = self.opi[i] - currentOtherPri
                    # 检查是否存在变道冲突
                    if flagGiveLine:
                        if priority_diff < 0:
                            # 自己的优先级较低，且尝试变道，受到惩罚
                            r_pr1 = priority_diff * self.opi[i]
                        else:
                            r_pr1 = (1 + priority_diff) * self.opi[i]

                    # 优先级-让速相关
                    flagGiveVel = self.check_velocity_conflict(self.PrevSpeed, vehicle_speed[i], i, distance[i], distance[veh], self.lanChangeFlag)
                    # 检查是否存在让速
                    if flagGiveVel:
                        if priority_diff > 0:
                            # 自己的优先级较低，且尝试变道，受到惩罚
                            r_pr2 = priority_diff * self.opi[i]
                        else:
                            r_pr2 = (1 + priority_diff) * self.opi[i]
                    r_p = 0.5 * r_pr1 + 0.5 * r_pr2
                    r_pp = np.concatenate((r_pp, [r_p]))
                r_pr = np.mean(r_pp)
        else:
            r_pr = 0

        # 对不可执行动作进行惩罚
        r_no = self.punish   # 非法变道惩罚值为-1，无效变道惩罚为-1


        # total reward
        r_total = w_sd * r_sd + w_comfVel * r_comfVel + w_ef * r_ef + w_pr * r_pr + w_no * r_no

        return r_total

    def getFinaTCC(self):

        return self.finaTCC


# if __name__ == '__main__':
#     state = (10, 100, 8, 20, 8, 30, 15, 100, 8, 50, -1, -1, 12, 100, 1, 1, 120, 10)
#     laneCP = LaneChangePredict()
#     result = laneCP.updateReward(1, state)
#     print(result)


