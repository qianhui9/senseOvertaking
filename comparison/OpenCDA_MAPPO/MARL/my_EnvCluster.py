from __future__ import absolute_import
from __future__ import print_function

import csv
import logging
import math
import shutil
import time
from pathlib import Path

import carla
import numpy as np
import pandas as pd
import random as rn
import os
import sys

import opencda.scenario_testing.utils.customized_map_api as map_api
import yaml
from mpmath import norm
from scipy import stats
from scipy.stats import norm
import torch.nn.functional as F


class LaneChangePredict():
    # 定义了 gym 环境的 metadata，指定了可视化模式为 human（即显示在屏幕上）
    metadata = {'render.modes': ['human']}
    def __init__(self):

        self.minVelocity = 0
        self.maxVelocity = 35

        self.minDistanceFrontVeh = 0
        self.maxXDistanceFrontVeh = 100
        self.maxYDistanceFrontVeh = 14   # 3.5*4

        self.minDistanceRearVeh = 0
        self.maxXDistanceRearVeh = 100
        self.maxYDistanceRearVeh = 14

        self.maxRoadLength = 2000

        self.roadWidth = 3.5

        self.ittc_safe = 0

        self.minLaneNumber = 0
        self.maxLaneNumber = 3 # 车道数为4，但由于获取车道index时，index从0开始计算

        self.CommRange = 100  # 联合感知范围为100米

        self.delta_t = 0.1    # 0.1秒为一个时隙
        self.AutoCarIDAll = []  # 目标车辆

        self.velbefore = 0

        self.traFlowNumber = 0   # 前方范围内车辆数
        self.finaTCC = 0
        self.speed = 0

        self.prevx = 0
        self.prevy = 0

        self.punish = 0
        self.rewardplus = -20 # 防止车辆为了避免碰撞得到大奖励而导致车辆不运动

        self.laneFlag = [0, 0, 0]   # 判断车辆是否发生了变道


        self.overpassFlag = [0, 0, 0]
        self.AutoCarFrontID = ["", "", ""]  # 超车结束标志：前方车辆
        self.tempAutoCarFrontID = ''

        self.dFront = [0, 0, 0]

        # 离散动作：左变道、保持车道、右变道
        self.action_space_vehicle = [-1, 0, 1]   # 0为不变道，-1为左变道，1为右变道
        self.n_actions = len(self.action_space_vehicle)
        self.n_actions = int(self.n_actions)
        # 连续动作：速度变化
        self.param_velocity = [0, 35]
        # 转向角
        self.param_theta = [-40, 40]
        self.n_features = 29  # 状态的维度

        # self.actions = np.zeros((int(self.n_actions), 1 + 1))  # 第一个1表示索引，第二个1表示变道动作
        self.actions = np.array([[0, -1], [1, 0], [2, 1]])


    def deterTarIDVehID(self, information):
        # 获取车辆A的ID（目标车辆 A）
        target_vehicle_a_id = None
        # 在周围车辆信息中查找目标车辆A（x, y 为 -940, 8.75）
        for vehicle in information["surrounding_vehicles"]:
            # 读取每辆车的 x 和 y 信息
            x, y = vehicle["x"], vehicle["y"]

            # 使用容差值来匹配目标车辆的位置
            if abs(x - (-940)) < 3 and abs(y - 8.75) < 3:
                target_vehicle_a_id = vehicle["id"]  # 获取目标车辆的ID
                break

        # 确保目标车辆A的ID已获取并设置为所有车辆的前方车辆ID
        if target_vehicle_a_id is not None:
            self.AutoCarFrontID = [target_vehicle_a_id] * 3  # 设置所有车辆的前方车辆为目标车辆 A
        else:
            print("[ERROR] 未找到目标车辆 A，位置匹配失败。")

    def reset(self, single_cav_list, scenario_manager, bg_veh_list):
        self.TotalReward = 0
        self.overpassFlag = [0] * 3
        self.speed = 0
        self.laneFlag = [0] * 3
        self.rewardplus = -20
        self.punish = 0
        self.currentTrackingVehId = 'None'

        # 1. 清理上一轮车辆的感知数据文件夹
        base_path = Path("E:/postgraduate/V2X/senseOvertaking/comparison/OpenCDA_MAPPO/data_dumping")
        subdirs = sorted([d for d in base_path.iterdir() if d.is_dir()])
        if subdirs:
            latest_dir = subdirs[-1]  # 最新的时间戳文件夹
            print(f"[INFO] Cleaning perception data folder: {latest_dir}")
            # 删除最新时间戳目录下所有车辆ID文件夹及其内容
            for vehicle_dir in latest_dir.iterdir():
                if vehicle_dir.is_dir():
                    try:
                        shutil.rmtree(vehicle_dir)
                        print(f"[INFO] Deleted folder: {vehicle_dir}")
                    except Exception as e:
                        print(f"[WARN] Failed to delete {vehicle_dir}: {e}")

        # scenario_manager.close()

        # 1. 销毁上一轮的所有车辆（包括 background）
        for v in single_cav_list + bg_veh_list:
            try:
                v.destroy()
            except Exception:
                pass

        # 2. 同时销毁 world 里残留的任何 vehicle actor
        all_vehicles = scenario_manager.world.get_actors().filter('vehicle.*')
        for actor in all_vehicles:
            try:
                actor.destroy()
            except Exception:
                pass

        # 3. 等一帧／短暂 sleep，确保 CARLA 更新完成
        scenario_manager.world.tick()
        time.sleep(0.1)

        from opencda.scenario_testing.single_2lanefree_carla_MyTest import getID
        single_cav_list = \
            scenario_manager.create_vehicle_manager(application=['single'],
                                                    map_helper=map_api.
                                                    spawn_helper_2lanefree)  # 删除 , default_model='vehicle.ford.mustang'

        # create background traffic in carla
        traffic_manager, bg_veh_list = \
            scenario_manager.create_traffic_carla()

        spectator = scenario_manager.world.get_spectator()
        self.AutoCarIDAll = getID(single_cav_list)

        print("Environment reset successfully.")
        return single_cav_list, spectator, bg_veh_list, self.AutoCarIDAll

    # 接受一个 index 参数，在 actions 矩阵中查找并返回对应行的动作数组
    def find_action(self, index):
        return self.actions[index][1]

    def readSenseResult(self, folder_id, t):
        # 生成文件名（六位数字补零）
        filename = f"{t:06d}.yaml"
        base_path = Path("E:/postgraduate/V2X/senseOvertaking/comparison/OpenCDA_MAPPO/data_dumping")
        subdirs = sorted([
            d for d in base_path.iterdir()
            if d.is_dir()
        ])
        # 取第一个子文件夹（例如 2025_05_24_12_14_19）
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

    def step(self, single_cav, action, action_param1, action_param2, i, t):
        th = np.tanh(action_param1) * 40
        v_n = (np.tanh(action_param2) + 1) * 17.5

        # 记录变道前速度
        self.prevx = self.state[2]
        self.prevy = self.state[3]

        yaw = self.state[4]

        transform = single_cav.vehicle.get_transform()
        rotation = transform.rotation

        # 限制航向角
        rotation.yaw = max(min(rotation.yaw, 70), -70)

        # 侧翻复位检测
        if abs(rotation.roll) > 30 or abs(rotation.pitch) > 30:
            rotation.roll = 0
            rotation.pitch = 0
            transform.rotation = rotation
            single_cav.vehicle.set_transform(transform)
            single_cav.vehicle.set_target_velocity(carla.Vector3D(0, 0, 0))
        else:
            transform.rotation = rotation
            single_cav.vehicle.set_transform(transform)

        # 计算目标速度向量
        if -70 <= yaw + th <= 70:
            yaw_rad = math.radians(yaw + th)   # 转换为弧度
            v_x = math.cos(yaw_rad) * v_n
            v_y = math.sin(yaw_rad) * v_n
        elif yaw + th < -70:
            yaw_rad = math.radians(-70)  # 转换为弧度
            v_x = math.cos(yaw_rad) * v_n
            v_y = math.sin(yaw_rad) * v_n
        elif yaw + th > 70:
            yaw_rad = math.radians(70)  # 转换为弧度
            v_x = math.cos(yaw_rad) * v_n
            v_y = math.sin(yaw_rad) * v_n

        self.speed = v_x
        vel = carla.Vector3D(x=v_x, y=v_y, z=0.0)
        single_cav.vehicle.set_target_velocity(vel)

        # print(f"Vehicle {i} action: {x}, param1: {yaw + th}, param2: {v_x}")

        # 限制车辆Y坐标边界
        road_left_boundary = 0
        road_right_boundary = self.roadWidth * (self.maxLaneNumber + 1)

        location = transform.location
        changed = False
        if location.y < road_left_boundary:
            location.y = road_left_boundary
            changed = True
        elif location.y > road_right_boundary:
            location.y = road_right_boundary
            changed = True
        if changed:
            transform.location = location
            single_cav.vehicle.set_transform(transform)
            vel = single_cav.vehicle.get_velocity()
            vel.y = 0
            single_cav.vehicle.set_target_velocity(vel)

        # 变道时锁定超车目标（左右变道合并逻辑）
        # if x in (-1, 1):
        #     if self.laneFlag[i] == 0 and self.tempAutoCarFrontID not in self.AutoCarIDAll:
        #         self.AutoCarFrontID[i] = self.tempAutoCarFrontID
        #         self.laneFlag[i] += 1

        # 更新状态
        single_cav.update_info()
        single_cav.savedData_dumper()
        information = self.readSenseResult(self.AutoCarID, t)
        self.state = self._findstate(i, information)

        self.end = self.is_overtake_complete(self.state, i)
        reward = self.updateReward(action, action_param1, self.state)

        return self.state, reward, self.end

    def close(self):
        pass
        # traci.close()

    # 检查两个车辆是否在同一个车道上,四舍五入
    def checkSameLane(self, y1, y2):
        lane_id1 = int(round(y1 / self.roadWidth))
        lane_id2 = int(round(y2 / self.roadWidth))
        return lane_id1 == lane_id2

    def is_left_lane(self, y_self, y_other):
        """
        判断另一辆车是否在自车左侧车道上（基于车道ID比较）。
        """
        lane_id_self = int(round(y_self / self.roadWidth))
        lane_id_other = int(round(y_other / self.roadWidth))
        return lane_id_other == lane_id_self - 1  # 左侧车道的ID比自车小1

    def is_right_lane(self, y_self, y_other):
        """
        判断另一辆车是否在自车右侧车道上（基于车道ID比较）。
        """
        lane_id_self = int(round(y_self / self.roadWidth))
        lane_id_other = int(round(y_other / self.roadWidth))
        return lane_id_other == lane_id_self + 1  # 右侧车道的ID比自车大1

    # 计算车辆之间的距离
    def _findRearVehDistance(self, flag_i, information):
        # 二维数组parameters，用于存储每辆车的相关信息
        parameters = [[0 for x in range(6)] for x in range(len(information['surrounding_vehicles'])+1)]
        dx1 = -1
        dx2 = -1
        dx3 = -1
        dx4 = -1
        dx5 = -1
        dx6 = -1
        dy1 = -1
        dy2 = -1
        dy3 = -1
        dy4 = -1
        dy5 = -1
        dy6 = -1
        v1 = -1
        v2 = -1
        v3 = -1
        v4 = -1
        v5 = -1
        v6 = -1
        ittc1 = -1
        ittc2 = -1
        ittc3 = -1
        ittc4 = -1
        ittc5 = -1
        ittc6 = -1
        # 遍历全部车辆的ID

        for i, VehID in enumerate(information['surrounding_vehicles']):
            parameters[i][0] = VehID['id']
            parameters[i][1] = VehID['x']  # X position
            parameters[i][2] = VehID['y']  # y
            parameters[i][3] = VehID['speed'] * math.cos(VehID['heading'])  # vx
            parameters[i][4] = VehID['speed'] * math.sin(VehID['heading'])  # vy
            parameters[i][5] = VehID['heading']

        # 自己车辆信息
        parameters[len(information['surrounding_vehicles'])][0] = self.AutoCarID
        parameters[len(information['surrounding_vehicles'])][1] = information['ego']['predicted_ego']['x']  # X position
        parameters[len(information['surrounding_vehicles'])][2] = information['ego']['predicted_ego']['y']  # y
        parameters[len(information['surrounding_vehicles'])][3] = information['ego']['predicted_ego']['v'] * math.cos(information['ego']['predicted_ego']['yaw'])  # vx
        parameters[len(information['surrounding_vehicles'])][4] = information['ego']['predicted_ego']['v'] * math.sin(information['ego']['predicted_ego']['yaw'])  # vy
        parameters[len(information['surrounding_vehicles'])][5] = information['ego']['predicted_ego']['yaw']

        # 通过 X 方向的坐标值升序排序存储在二维数组 parameters 中的车辆列表
        parameters = sorted(parameters, key=lambda x: x[1])  # Sorted in ascending order based on x distance
        # Find Row with Auto Car
        # 找出目标车辆并将记录其在列表中的位置，以及RowIDAuto 变量用于存储下标，值为目标车辆所在行的位置
        index = [x for x in parameters if self.AutoCarID in x][0]
        RowIDAuto = parameters.index(index)

        # 用于计算汽车周围车辆的状态信息，包括各个方向的车辆距离 d、速度 v 等参数，并更新超车次数
        # if there are no vehicles in front;下面这个长度是全局所有的车辆数
        if RowIDAuto == len(information['surrounding_vehicles']):
            dx1 = -1
            dy1 = -1
            v1 = -1
            ittc1 = -1
            dx3 = -1
            dy3 = -1
            v3 = -1
            ittc3 = -1
            dx5 = -1
            dy5 = -1
            v5 = -1
            ittc5 = -1
            self.CurrFrontVehID = 'None'
            self.CurrFrontVehDistance = 100
            # 当前超车的车辆ID也设置为 None
            self.currentTrackingVehId = 'None'
        else:
            # If vehicle is in the lowest lane（最右侧车道）, then d5,d6,v5,v6 do not exist
            if 0 <= parameters[RowIDAuto][2] <= self.roadWidth:
                dx5 = -1
                dy5 = -1
                v5 = -1
                ittc5 = -1
                dx6 = -1
                dy6 = -1
                v6 = -1
                ittc6 = -1
            # if the vehicle is in the maximum lane index（最左侧车道）, then d3.d4.v3.v4 do not exist
            elif 3 * self.roadWidth <= parameters[RowIDAuto][2] <= 4 * self.roadWidth:
                dx3 = -1
                dy3 = -1
                v3 = -1
                ittc3 = -1
                dx4 = -1
                dy4 = -1
                v4 = -1
                ittc4 = -1
            # find d1 and v1  从当前行向下搜索车辆，以查找前方车辆的状态参数
            index = RowIDAuto + 1
            # 如果存在同一车道上的前方车辆，则计算前方车辆与当前车辆之间的距离 d1和速度 v1
            while index != len(information['surrounding_vehicles']) + 1:
                if self.checkSameLane(parameters[index][2], parameters[RowIDAuto][2]):
                    dx1 = parameters[index][1] - parameters[RowIDAuto][1]
                    dy1 = parameters[index][2] - parameters[RowIDAuto][2]
                    v1 = parameters[index][3] / math.cos(parameters[index][5])
                    ittc1 = self.caliTTC(parameters[index][1], parameters[RowIDAuto][1], parameters[index][2], parameters[RowIDAuto][2],
                                        parameters[index][3] / math.cos(parameters[index][5]), parameters[RowIDAuto][3] / math.cos(parameters[RowIDAuto][5]),
                                        parameters[index][5], parameters[RowIDAuto][5])
                    self.tempAutoCarFrontID = parameters[index][0]
                    break
                index += 1
            # there is no vehicle in front
            if index == len(information['surrounding_vehicles']) + 1:
                dx1 = -1
                dy1 = -1
                v1 = -1
                ittc1 = -1
                self.CurrFrontVehID = 'None'
                self.CurrFrontVehDistance = 100
            # find d3 and v3  从当前行向下搜索车辆，以查找右侧车道的前方车辆的状态参数
            index = RowIDAuto + 1
            # 如果左侧车道存在前方车辆，则计算其于当前车辆之间的距离 d3 和速度 v3
            while index != len(information['surrounding_vehicles']) + 1:
                if self.is_left_lane(parameters[RowIDAuto][2], parameters[index][2]):
                    dx3 = parameters[index][1] - parameters[RowIDAuto][1]
                    dy3 = -(parameters[index][2] - parameters[RowIDAuto][2])
                    v3 = parameters[index][3] / math.cos(parameters[index][5])
                    ittc3 = self.caliTTC(parameters[index][1], parameters[RowIDAuto][1], parameters[index][2],
                                        parameters[RowIDAuto][2],
                                        parameters[index][3] / math.cos(parameters[index][5]), parameters[RowIDAuto][3] / math.cos(parameters[RowIDAuto][5]),
                                        parameters[index][5], parameters[RowIDAuto][5])
                    break
                index += 1
            # there is no vehicle in front
            if index == len(information['surrounding_vehicles']) + 1:
                dx3 = -1
                dy3 = -1
                v3 = -1
                ittc3 = -1
            # find d5 and v5
            index = RowIDAuto + 1
            # 如果右侧车道存在前方车辆，则计算其于当前车辆之间的距离 d5 和速度 v5
            while index != len(information['surrounding_vehicles']) + 1:
                if self.is_right_lane(parameters[RowIDAuto][2], parameters[index][2]):
                    dx5 = parameters[RowIDAuto][1] - parameters[index][1]
                    dy5 = parameters[RowIDAuto][2] - parameters[index][2]
                    v5 = parameters[index][3] / math.cos(parameters[index][5])
                    ittc5 = self.caliTTC(parameters[index][1], parameters[RowIDAuto][1], parameters[index][2],
                                        parameters[RowIDAuto][2],
                                        parameters[index][3] / math.cos(parameters[index][5]), parameters[RowIDAuto][3] / math.cos(parameters[RowIDAuto][5]),
                                        parameters[index][5], parameters[RowIDAuto][5])
                    break
                index += 1
            # there is no vehicle in front
            if index == len(information['surrounding_vehicles']) + 1:
                dx5 = -1
                dy5 = -1
                v5 = -1
                ittc5 = -1
            # find d2 and v2
            index = RowIDAuto - 1
            # 如果存在同一车道上的后方车辆，则计算后方车辆与当前车辆之间的距离 d2 速度 v2
            while index >= 0:
                if self.checkSameLane(parameters[index][2], parameters[RowIDAuto][2]):
                    dx2 = parameters[RowIDAuto][1] - parameters[index][1]
                    dy2 = parameters[RowIDAuto][2] - parameters[index][2]
                    v2 = parameters[index][3] / math.cos(parameters[index][5])
                    ittc2 = self.caliTTC(parameters[RowIDAuto][1], parameters[index][1], parameters[RowIDAuto][2],
                                        parameters[index][2],
                                        parameters[RowIDAuto][3] / math.cos(parameters[RowIDAuto][5]),
                                        parameters[index][3] / math.cos(parameters[index][5]),
                                        parameters[index][5], parameters[RowIDAuto][5])
                    break
                index -= 1
            # 如果同一车道上没有后方车辆
            if index < 0:
                dx2 = -1
                dy2 = -1
                v2 = -1
                ittc2 = -1
            # find d4 and v4
            # 类似地，计算右侧和左侧车道的后方车辆状态参数d4、v4、d6 和 v6
            index = RowIDAuto - 1
            while index >= 0:
                if self.is_right_lane(parameters[RowIDAuto][2], parameters[index][2]):
                    dx4 = parameters[RowIDAuto][1] - parameters[index][1]
                    dy4 = -(parameters[RowIDAuto][2] - parameters[index][2])
                    v4 = parameters[index][3] / math.cos(parameters[index][5])
                    ittc4 = self.caliTTC(parameters[RowIDAuto][1], parameters[index][1], parameters[RowIDAuto][2],
                                        parameters[index][2],
                                        parameters[RowIDAuto][3] / math.cos(parameters[RowIDAuto][5]),
                                        parameters[index][3] / math.cos(parameters[index][5]),
                                        parameters[index][5], parameters[RowIDAuto][5])
                    break
                index -= 1
            if index < 0:
                dx4 = -1
                dy4 = -1
                v4 = -1
                ittc4 = -1
            # find d6 and v6
            index = RowIDAuto - 1
            while index >= 0:
                if self.is_left_lane(parameters[RowIDAuto][2], parameters[index][2]):
                    dx6 = parameters[RowIDAuto][1] - parameters[index][1]
                    dy6 = parameters[RowIDAuto][2] - parameters[index][2]
                    v6 = parameters[index][3] / math.cos(parameters[index][5])
                    ittc6 = self.caliTTC(parameters[RowIDAuto][1], parameters[index][1], parameters[RowIDAuto][2],
                                        parameters[index][2],
                                        parameters[RowIDAuto][3] / math.cos(parameters[RowIDAuto][5]),
                                        parameters[index][3] / math.cos(parameters[index][5]),
                                        parameters[RowIDAuto][5], parameters[index][5])
                    break
                index -= 1
            if index < 0:
                dx6 = -1
                dy6 = -1
                v6 = -1
                ittc6 = -1
            # 将当前正在追踪的前方车辆ID设置为当前车道上的下一辆车辆的ID  这个ID存储在列表 parameters 中的第 RowIDAuto + 1 行第一个元素中，即该车辆的ID。这是一个用于跟踪当前车辆前方的车辆的ID。
            self.currentTrackingVehId = parameters[RowIDAuto + 1][0]
        if RowIDAuto == 0:  # This means that there is no car behind  没有后方车辆
            RearDist = -1
        else:  # There is a car behind return the distance between them
            RearDist = (parameters[RowIDAuto][1] - parameters[RowIDAuto - 1][
                1])  # 如果当前存在后方的车辆，计算当前车辆和后方车辆之间的距离，即当前车辆的位置减去上一行车道上的车辆的位置
        # Return car in front distance
        if RowIDAuto == len(information['surrounding_vehicles']):  # 没有前方车辆
            FrontDist = -1
            # Save the current front vehicle Features
            self.CurrFrontVehID = 'None'
            self.CurrFrontVehDistance = 100
        else:
            FrontDist = (parameters[RowIDAuto + 1][1] - parameters[RowIDAuto][
                1])  # 计算当前车辆和前方车辆之间的距离，即下一行车道上的车辆的位置减去当前车辆的位置，这是计算前方车辆间距的方法
            # Save the current front vehicle Features
            self.CurrFrontVehID = parameters[RowIDAuto + 1][0]
            self.CurrFrontVehDistance = FrontDist

        # 获取执行前车辆前方车辆的纵向位置
        # 锁定超车的目标车辆
        # if self.laneFlag[flag_i] != 0:
        id = self.AutoCarFrontID[flag_i]
        for vehicle in information['surrounding_vehicles']:
            if vehicle['id'] == id:
                self.dFront[flag_i] = vehicle['x']

        # return RearDist, FrontDist
        return dx1, dy1, v1, ittc1, dx2, dy2, v2, ittc2, dx3, dy3, v3, ittc3, dx4, dy4, v4, ittc4, dx5, dy5, v5, ittc5, \
               dx6, dy6, v6, ittc6

    def _findCurrentState(self,i, information):
        self.AutoCarID = self.AutoCarIDAll[i]
        self.state = self._findstate(i, information)

    def _findCurrentOtherState(self, j, i):
        self.AutoCarID = self.AutoCarIDAll[j]
        state = self._findstate(j)
        self.AutoCarID = self.AutoCarIDAll[i]
        return state

    def _findstate(self, i, information):
        self.AutoCarID = self.AutoCarIDAll[i]
        # 使用getAllSubscriptionResults()方法获取已订阅车辆的状态列表
        # VehicleParameters = traci.vehicle.getAllSubscriptionResults()
        # find d1,v1,d2,v2,d3,v3,d4,v4, d5, v5, d6, v6  调用该函数来查找后方的车辆的距离和速度，并将它们分配给相应的变量。
        dx1, dy1, v1, ittc1, dx2, dy2, v2, ittc2, dx3, dy3, v3, ittc3, dx4, dy4, v4, ittc4, dx5, dy5, v5, ittc5, \
        dx6, dy6, v6, ittc6 = self._findRearVehDistance(i, information)
        # 检查前方车辆距离 d1是否小于通信范围，如果在通信范围之外，则将其设置为最大可能距离。如果前方没有车辆，则将其设置为最大距离。
        if ((math.sqrt(dx1**2 + dy1**2) > self.CommRange)):
            dx1 = self.maxXDistanceFrontVeh
            dy1 = self.maxYDistanceFrontVeh
            v1 = -1
            ittc1 = -1
        elif dx1 < 0:  # if there is no vehicle ahead in L0
            dx1 = self.maxXDistanceFrontVeh  # as this can be considered as vehicle is far away
            dy1 = self.maxYDistanceFrontVeh
        # 检查前方车速 v1 是否为负数，如果为负数，则将其设置为零。这通常会出现在没有前方车辆或者前方车辆被超车时
        if ((v1 < 0) and (math.sqrt(dx1**2 + dy1**2) <= self.CommRange)):
            # there is no vehicle ahead in L0 or there is a communication error: # there is no vehicle ahead in L0
            v1 = 0
            ittc1 = 0

        # 检查后方车辆距离 d2 是否大于通信范围，如果是，则将其设置为最大可能距离。如果后方没有车辆，则将其设置为零，以避免出现负回报
        if ((math.sqrt(dx2**2 + dy2**2) > self.CommRange)):
            dx2 = self.maxXDistanceRearVeh
            dy2 = self.maxYDistanceRearVeh
            v2 = -1
            ittc2 = -1
        elif dx2 < 0:  # There is no vehicle behind in L0
            dx2 = 0  # to avoid negetive reward
            dy2 = 0
        # 检查后方车速 v2 是否为负数，如果为负数，则将其设置为零。这通常会出现在没有后方车辆或者后方车辆被超车时
        if ((v2 < 0) and (math.sqrt(dx2**2 + dy2**2) <= self.CommRange)):
            # there is no vehicle behind in L0 or there is a communication error
            v2 = 0
            ittc2 = 0
        if ((math.sqrt(dx3**2 + dy3**2) > self.CommRange)):
            dx3 = self.maxDistanceXFrontVeh
            dy3 = self.maxDistanceYFrontVeh
            v3 = -1
            ittc3 = -1
        elif dx3 < 0: # no vehicle ahead in L1
            dx3 = self.maxXDistanceFrontVeh # as this can be considered as vehicle is far away
            dy3 = self.maxYDistanceFrontVeh
        if ((v3 < 0) and (math.sqrt(dx3**2 + dy3**2) <= self.CommRange)) : # there is no vehicle ahead in L1 or there is a communication error: # there is no vehicle ahead in L1
            v3 = 0
            ittc3 = 0

        if ((math.sqrt(dx4**2 + dy4**2) > self.CommRange)):
            dx4 = self.maxXDistanceRearVeh
            dy4 = self.maxYDistanceRearVeh
            v4 = -1
            ittc4 = -1
        elif dx4 < 0: #There is no vehicle behind in L1
            dx4 = self.maxXDistanceRearVeh # so that oue vehicle can go to the overtaking lane
            dy4 = self.maxYDistanceRearVeh
        if ((v4 < 0) and (math.sqrt(dx4**2 + dy4**2) <= self.CommRange)) : # there is no vehicle behind in L1 or there is a communication error: # there is no vehicle behind in L1
            v4 = 0
            ittc4 = 0

        if ((math.sqrt(dx5**2 + dy5**2) > self.CommRange)):
            dx5 = self.maxXDistanceFrontVeh
            dy5 = self.maxYDistanceFrontVeh
            v5 = -1
            ittc5 = -1
        elif dx5 < 0: # no vehicle ahead in L1
            dx5 = self.maxXDistanceFrontVeh # as this can be considered as vehicle is far away
            dy5 = self.maxYDistanceFrontVeh
        if ((v5 < 0) and (math.sqrt(dx5**2 + dy5**2) <= self.CommRange)) : # there is no vehicle ahead in L1 or there is a communication error: # there is no vehicle ahead in L1
            v5 = 0
            ittc5 = 0

        if ((math.sqrt(dx6**2 + dy6**2) > self.CommRange)):
            dx6 = self.maxXDistanceRearVeh
            dy6 = self.maxYDistanceRearVeh
            v6 = -1
            ittc6 = -1
        elif dx6 < 0: #There is no vehicle behind in L1
            dx6 = self.maxXDistanceRearVeh # so that oue vehicle can go to the overtaking lane
            dy6 = self.maxYDistanceRearVeh
        if ((v6 < 0) and (math.sqrt(dx6**2 + dy6**2) <= self.CommRange)): # there is no vehicle behind in L1 or there is a communication error: # there is no vehicle behind in L1
            v6 = 0
            ittc6 = 0

        # 行进方向 va
        va = information['ego']['predicted_ego']['v']
        # 航向角 yaw
        yaw = information['ego']['predicted_ego']['yaw']
        vx = va * math.cos(yaw)
        vy = va * math.sin(yaw)

        # 坐标
        x = information['ego']['predicted_ego']['x']
        y = information['ego']['predicted_ego']['y']

        # print("d1, v1, d2, v2, d3, v3, d4, v4, d5, v5, d6, v6:", d1, v1, d2, v2, d3, v3, d4, v4, d5, v5, d6, v6)
        # 这些参数是用于车辆行驶过程中的决策和控制，例如加速和转向
        return x, y, vx, vy, yaw, v1, ittc1, dx1, dy1, v2, ittc2, dx2, dy2, v3, ittc3, dx3, dy3, v4, ittc4, dx4, dy4, v5, ittc5, \
               dx5, dy5, v6, ittc6, dx6, dy6

    # 程序终止条件
    def is_overtake_complete(self, state, i):
        if state[0] - self.dFront[i] >= 10:
            self.overpassFlag[i] = 1
            self.rewardplus = 1000
        else:  # 若没有完成超车，则给出与目标车辆相对距离的惩罚
            self.punish = -30 * (self.dFront[i] - state[0])
        return self.overpassFlag[i]

    # x1是前车 x2是后车
    def caliTTC(self, x1, x2, y1, y2, v1, v2, yaw1, yaw2):
        lh = 5  # 车长
        d_lead = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        phi = np.arccos((x1 - x2) / d_lead) if d_lead != 0 else 0
        numerator = v2 * np.cos(phi - yaw2) - v1 * np.cos(phi - yaw1)
        denominator = d_lead - lh
        ittc = max(0, numerator / denominator) if denominator != 0 else 0

        return ittc

    def calTTCDri(self, action, state):
        x = action  # 变道

        w_front = 0.5

        if x == -1:
            # 左变道计算 TCC
            if state[13] != -1:
                if state[2] * math.cos(state[4]) - state[13] <= 0:
                    TCC_front = self.ittc_safe
                else:
                    TCC_front = state[14]
            else:
                TCC_front = self.ittc_safe     # 默认TCC很大，默认为5
            if state[17] != -1:
                if state[2] * math.cos(state[4]) - state[17] >= 0:
                    TCC_back = self.ittc_safe  # 设置一个较大的值，表示车辆A和车辆B之间的时间间隔很大
                else:
                    TCC_back = state[18]
            else:
                TCC_back = self.ittc_safe     # 默认TCC很大，默认为5
            if abs(TCC_front) < self.ittc_safe:
                TCC_front = self.ittc_safe
            if abs(TCC_back) < self.ittc_safe:
                TCC_back = self.ittc_safe
            TCC_surround = w_front * TCC_front + (1 - w_front) * TCC_back  # 前后车的 TCC 是综合计算的

        elif x == 1:
            if state[21] != -1:
                if state[2] * math.cos(state[4]) - state[21] <= 0:
                    TCC_front = self.ittc_safe  # 设置一个较大的值，表示车辆A和车辆B之间的时间间隔很大
                else:
                    TCC_front = state[22]
            else:
                TCC_front = self.ittc_safe     # 默认TCC很大，默认为5
            if state[25] != -1:
                if state[2] * math.cos(state[4]) - state[25] >= 0:
                    TCC_back = self.ittc_safe  # 设置一个较大的值，表示车辆A和车辆B之间的时间间隔很大
                else:
                    TCC_back = state[26]
            else:
                TCC_back = self.ittc_safe
            if abs(TCC_front) < self.ittc_safe:
                TCC_front = self.ittc_safe
            if abs(TCC_back) < self.ittc_safe:
                TCC_back = self.ittc_safe
            TCC_surround = w_front * TCC_front + (1 - w_front) * TCC_back  # 前后车的 TCC 是综合计算的

        else:
            if state[5] != -1:
                if state[2] * math.cos(state[4]) - state[5] <= 0:  # 车辆A速度比车辆B快
                    TCC_front = self.ittc_safe  # 设置一个较大的值，表示车辆A和车辆B之间的时间间隔很大
                else:
                    TCC_front = state[6]
            else:
                TCC_front = self.ittc_safe
            if abs(TCC_front) < self.ittc_safe:
                TCC_front = self.ittc_safe
            TCC_surround = TCC_front

        finaTCC = TCC_surround

        return finaTCC


    # 处理奖励函数，使其各个参数范围相似
    def min_max_normalize(self, value, min_value, max_value):
        return (value - min_value) / (max_value - min_value)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # 奖励
    def updateReward(self, action_lc, action_th, state):
        a_x_max = 4
        a_y_max = 1

        v_max = 35

        omega_e = 2
        omega_s = 2
        omega_c = 1
        omega_p = 1
        lw = 1.88 # 车辆宽度

        # ===== 1. Efficiency reward =====
        r_speed = np.sqrt(state[2]**2 + state[3]**2) / v_max

        a_x = (state[2] - self.prevx) / self.delta_t
        a_y = (state[3] - self.prevy) / self.delta_t

        acc_mag = np.sqrt(a_x ** 2 + a_y ** 2)
        if abs(a_x) <= a_x_max and abs(a_y) <= a_y_max:
            r_comf = acc_mag / (a_x_max + a_y_max)
        else:
            r_comf = -acc_mag / (a_x_max + a_y_max)

        r_eff = r_speed + r_comf

        # ===== 2. Safety reward =====
        ittc_true = self.calTTCDri(action_lc, state)
        ittc_max = 0.5
        r_safe = max(min(-ittc_true / ittc_max, 0), -2)

        if ittc_true != 0:
            self.finaTCC = 1 / ittc_true
            if self.finaTCC > 10:
                self.finaTCC = 10
        else:
            self.finaTCC = 10

        # ===== 3. Cooperation penalty =====

        # 计算当前车道编号（四舍五入）
        lane_id = int(round(state[1] / self.roadWidth))
        # 当前车道的中心线坐标
        y_c_target = (lane_id + 0.5) * self.roadWidth

        # 左侧车道的中心线坐标（如果存在）
        y_l_target = None
        if lane_id > 0:  # 如果不是最左侧车道
            y_l_target = (lane_id - 1 + 0.5) * self.roadWidth

        # 右侧车道的中心线坐标（如果存在）
        y_r_target = None
        if lane_id < self.maxLaneNumber:  # 如果不是最右侧车道
            y_r_target = (lane_id + 1 + 0.5) * self.roadWidth

        k = 0.2   # 奖励陡峭程度
        epsilon = 10  # 直行角度容忍范围

        if action_lc == -1:
            if y_l_target is None:  # 已经在最左车道，禁止左变道
                r_rule = -5  # 大惩罚
            else:
                r_rule = -abs(state[1] - y_l_target)
        elif action_lc == 0:
            r_rule = -abs(state[1] - y_c_target)
        elif action_lc == 1:
            if y_r_target is None:  # 已经在最右车道，禁止右变道
                r_rule = -5  # 大惩罚
            else:
                r_rule = -abs(state[1] - y_r_target)
        else:
            r_rule = 0

        if action_lc == -1:
            r_align = self.sigmoid(-k * (action_th + epsilon))
        elif action_lc == 0:
            r_align = self.sigmoid(-k * abs(action_th) + k * epsilon)
        elif action_lc == 1:
            r_align = self.sigmoid(k * (action_th - epsilon))
        else:
            r_align = 0

        r_cp = r_rule + r_align

        # ===== 4. Illegal behavior penalty =====

        # 道路边界信息，后期设置
        l_lm = 0   # 最左侧车道距离
        l_rm = 14   # 最右侧车道距离
        xi = 0.5   # 可容忍的道路安全边界

        # Intent-level
        r_pun_intent = 0
        if state[1] - l_lm - lw/2 < xi and action_lc == -1:
            r_pun_intent = -1
        elif l_rm - state[1] + lw/2 < xi and action_lc == 1:
            r_pun_intent = -1

        # Execution-level
        r_pun_exec = 0
        if state[1] - l_lm - lw/2 < xi and action_th < 0:
            r_pun_exec = -1
        elif l_rm - state[1] + lw/2 < xi and action_th > 0:
            r_pun_exec = -1

        r_pun = r_pun_intent + r_pun_exec

        # ===== Final reward =====
        r_total = omega_e * r_eff + omega_s * r_safe + omega_c * r_cp + omega_p * r_pun + self.rewardplus + self.punish
        # print("r_eff: ", r_eff, "; r_safe: ", r_safe, "; r_cp: ", r_cp, "; r_pun:", r_pun, "; self.rewardplus:", self.rewardplus, "; self.punish:", self.punish)

        return r_total

    def getFinaTCC(self):

        return self.finaTCC

    def getFinaSpeed(self):

        return self.speed


if __name__ == '__main__':
    state = (10, 100, 8, 20, 0, 30, 0.15, 100, 8, 50, -1, -1, 12, 100, -1, -1, 120, 10, -1, -1, 10, 100, -1, -1, 20, 10, -1, -1, 30, 10)
    laneCP = LaneChangePredict()
    result = laneCP.updateReward(0, 0, state)
    print(result)
