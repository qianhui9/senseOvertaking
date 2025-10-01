# -*- coding: utf-8 -*-
"""
Dumping sensor data.
"""

# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib

import os

import cv2
import open3d as o3d
import numpy as np

from opencda.core.common.misc import get_speed
from opencda.core.sensing.perception import sensor_transformation as st
from opencda.scenario_testing.utils.yaml_utils import save_yaml

# 定义数据转储类，用于将数据保存到本地磁盘
class DataDumper(object):
    """
    Data dumper class to save data in local disk.

    Parameters
    ----------
    perception_manager : opencda object
        The perception manager contains rgb camera data and lidar data.

    vehicle_id : int
        The carla.Vehicle id.

    save_time : str
        The timestamp at the beginning of the simulation.

    Attributes
    ----------
    rgb_camera : list
        A list of opencda.CameraSensor that containing all rgb sensor data
        of the managed vehicle.

    lidar ; opencda object
        The lidar manager from perception manager.

    save_parent_folder : str
        The parent folder to save all data related to a specific vehicle.

    count : int
        Used to count how many steps have been executed. We dump data
        every 10 steps.

    """
    # 初始化方法，接收感知管理器、车辆ID和保存时间
    def __init__(self,
                 perception_manager,
                 vehicle_id,
                 save_time):

        # 从感知管理器获取RGB摄像头和激光雷达数据
        # self.rgb_camera = perception_manager.rgb_camera
        self.rgb_camera = {}

        self.lidar = perception_manager.lidar

        # 保存传入的时间戳和车辆ID
        self.save_time = save_time
        self.vehicle_id = vehicle_id

        # 获取当前脚本路径
        current_path = os.path.dirname(os.path.realpath(__file__))
        # 构建数据保存的父文件夹路径，格式为：.../data_dumping/保存时间/车辆ID/
        self.save_parent_folder = \
            os.path.join(current_path,
                         '../../../data_dumping',
                         save_time,
                         str(self.vehicle_id))

        # 如果父文件夹不存在则创建
        if not os.path.exists(self.save_parent_folder):
            os.makedirs(self.save_parent_folder)

        # 初始化计数器为0，用于记录执行步数
        self.count = 0

    # 运行步骤方法，接收感知、定位和行为管理器
    def run_step(self,
                 perception_manager,
                 localization_manager,
                 behavior_agent):
        """
        Dump data at running time.

        Parameters
        ----------
        perception_manager : opencda object
            OpenCDA perception manager.

        localization_manager : opencda object
            OpenCDA localization manager.

        behavior_agent : opencda object
            Open
        """
        self.count += 1
        # # 忽略前60步，不保存数据
        # if self.count < 60:
        #     return
        #
        # # 10hz,每2步保存一次数据
        # if self.count % 2 != 0:
        #     return

        # self.save_rgb_image(self.count)   # 保存RGB图像

        # self.save_lidar_points()     # 保存激光雷达点云
        self.save_lidar_points(self.count)  # 保存激光雷达点云
        # 保存YAML配置文件(包含感知、定位和行为数据)
        self.save_yaml_file(perception_manager,
                            localization_manager,
                            behavior_agent,
                            self.count)

    # 保存RGB图像方法，接收当前计数
    def save_rgb_image(self, count):
        """
        Save camera rgb images to disk.
        """
        # 遍历所有RGB摄像头
        for (i, camera) in enumerate(self.rgb_camera):

            # 获取摄像头帧号和图像数据
            frame = camera.frame
            image = camera.image

            # 生成图像文件名，格式：000001_camera0.png
            image_name = '%06d' % count + '_' + 'camera%d' % i + '.png'

            # 使用OpenCV将图像写入文件
            cv2.imwrite(os.path.join(self.save_parent_folder, image_name),
                        image)

    # 保存激光雷达点云方法
    def save_lidar_points(self, count):
        """
        Save 3D lidar points to disk.
        """
        # 获取点云数据和帧号
        point_cloud = self.lidar.data
        # frame = self.lidar.frame
        frame = count

        # 分离点云坐标(x,y,z)和强度值
        point_xyz = point_cloud[:, :-1]
        point_intensity = point_cloud[:, -1]
        # 将强度值转换为RGB格式(强度值在R通道，G和B通道为0)
        point_intensity = np.c_[
            point_intensity,
            np.zeros_like(point_intensity),
            np.zeros_like(point_intensity)
        ]

        # 创建Open3D点云对象
        o3d_pcd = o3d.geometry.PointCloud()
        # 设置点云坐标和颜色
        o3d_pcd.points = o3d.utility.Vector3dVector(point_xyz)
        o3d_pcd.colors = o3d.utility.Vector3dVector(point_intensity)

        # 生成点云文件名，格式：000001.pcd
        pcd_name = '%06d' % frame + '.pcd'
        # 使用Open3D将点云写入PCD文件(ASCII格式)
        o3d.io.write_point_cloud(os.path.join(self.save_parent_folder,
                                              pcd_name),
                                 pointcloud=o3d_pcd,
                                 write_ascii=True)

    # 保存YAML文件的方法;perception_manager: 感知管理器对象; localization_manager: 定位管理器对象;behavior_agent: 行为代理对象;count: 当前计数/帧号
    def save_yaml_file(self,
                       perception_manager,
                       localization_manager,
                       behavior_agent,
                       count):
        """
        Save objects positions/spped, true ego position,
        predicted ego position, sensor transformations.

        Parameters
        ----------
        perception_manager : opencda object
            OpenCDA perception manager.

        localization_manager : opencda object
            OpenCDA localization manager.

        behavior_agent : opencda object
            OpenCDA behavior agent.
        """
        # 设置当前帧号
        frame = count

        # 初始化要转储的YAML字典和车辆字典
        dump_yml = {}
        vehicle_dict = {}

        # 从感知管理器获取所有检测到的物体
        objects = perception_manager.objects
        # 提取其中的车辆列表
        vehicle_list = objects['vehicles']

        # 遍历每个检测到的车辆
        for veh in vehicle_list:
            # 获取车辆ID、位置变换、边界框和速度
            veh_carla_id = veh.carla_id
            veh_pos = veh.get_transform()
            veh_bbx = veh.bounding_box
            veh_speed = get_speed(veh)

            # 确保车辆有有效ID(非-1)，否则抛出异常
            assert veh_carla_id != -1, "Please turn off perception active" \
                                       "mode if you are dumping data"

            # 将车辆信息存入字典，包括： 蓝图ID、颜色、 位置、边界框信息、旋转角度、速度
            vehicle_dict.update({veh_carla_id: {
                'bp_id': veh.type_id,
                'color': veh.color,
                "location": [veh_pos.location.x,
                             veh_pos.location.y,
                             veh_pos.location.z],
                "center": [veh_bbx.location.x,
                           veh_bbx.location.y,
                           veh_bbx.location.z],
                "angle": [veh_pos.rotation.roll,
                          veh_pos.rotation.yaw,
                          veh_pos.rotation.pitch],
                "extent": [veh_bbx.extent.x,
                           veh_bbx.extent.y,
                           veh_bbx.extent.z],
                "speed": veh_speed
            }})

        # 将车辆字典添加到总字典
        dump_yml.update({'vehicles': vehicle_dict})

        # dump ego pose and speed, if vehicle does not exist, then it is
        # a rsu(road side unit).
        # 获取预测的自车位置
        predicted_ego_pos = localization_manager.get_ego_pos()
        # 获取真实自车位置(如果是RSU则使用存储的位置)
        true_ego_pos = localization_manager.vehicle.get_transform() \
            if hasattr(localization_manager, 'vehicle') \
            else localization_manager.true_ego_pos

        # 保存预测位置、真实位置和速度
        dump_yml.update({'predicted_ego_pos': [
            predicted_ego_pos.location.x,
            predicted_ego_pos.location.y,
            predicted_ego_pos.location.z,
            predicted_ego_pos.rotation.roll,
            predicted_ego_pos.rotation.yaw,
            predicted_ego_pos.rotation.pitch]})
        dump_yml.update({'true_ego_pos': [
            true_ego_pos.location.x,
            true_ego_pos.location.y,
            true_ego_pos.location.z,
            true_ego_pos.rotation.roll,
            true_ego_pos.rotation.yaw,
            true_ego_pos.rotation.pitch]})
        dump_yml.update({'ego_speed':
                        float(localization_manager.get_ego_spd())})

        # dump lidar sensor coordinates under world coordinate system
        # 获取激光雷达传感器变换并保存
        lidar_transformation = self.lidar.sensor.get_transform()
        dump_yml.update({'lidar_pose': [
            lidar_transformation.location.x,
            lidar_transformation.location.y,
            lidar_transformation.location.z,
            lidar_transformation.rotation.roll,
            lidar_transformation.rotation.yaw,
            lidar_transformation.rotation.pitch]})

        # dump camera sensor coordinates under world coordinate system
        # 遍历每个RGB相机，保存其变换信息
        for (i, camera) in enumerate(self.rgb_camera):
            camera_param = {}
            camera_transformation = camera.sensor.get_transform()
            camera_param.update({'cords': [
                camera_transformation.location.x,
                camera_transformation.location.y,
                camera_transformation.location.z,
                camera_transformation.rotation.roll,
                camera_transformation.rotation.yaw,
                camera_transformation.rotation.pitch
            ]})

            # dump intrinsic matrix
            # 获取并保存相机内参矩阵(转换为列表)
            camera_intrinsic = st.get_camera_intrinsic(camera.sensor)
            camera_intrinsic = self.matrix2list(camera_intrinsic)
            camera_param.update({'intrinsic': camera_intrinsic})


            # dump extrinsic matrix lidar2camera
            # 计算激光雷达到相机的变换矩阵(外参)
            lidar2world = \
                st.x_to_world_transformation(self.lidar.sensor.get_transform())
            camera2world = \
                st.x_to_world_transformation(camera.sensor.get_transform())

            # 转换为列表并保存
            world2camera = np.linalg.inv(camera2world)
            lidar2camera = np.dot(world2camera, lidar2world)
            lidar2camera = self.matrix2list(lidar2camera)
            camera_param.update({'extrinsic': lidar2camera})
            dump_yml.update({'camera%d' % i: camera_param})

        # 默认标记为RSU(路侧单元)
        dump_yml.update({'RSU': True})
        # dump the planned trajectory if it exisit.
        # 如果有行为代理，获取规划轨迹
        if behavior_agent is not None:
            trajectory_deque = \
                behavior_agent.get_local_planner().get_trajectory()
            trajectory_list = []

            # 将轨迹点(x,y)和速度存入列表
            for i in range(len(trajectory_deque)):
                tmp_buffer = trajectory_deque.popleft()
                x = tmp_buffer[0].location.x
                y = tmp_buffer[0].location.y
                spd = tmp_buffer[1]

                trajectory_list.append([x, y, spd])

            # 更新轨迹信息和RSU标记(False表示是车辆)
            dump_yml.update({'plan_trajectory': trajectory_list})
            dump_yml.update({'RSU': False})

        yml_name = '%06d' % frame + '.yaml'
        save_path = os.path.join(self.save_parent_folder,
                                 yml_name)

        # 调用save_yaml函数保存数据
        save_yaml(dump_yml, save_path)

    # 静态方法，将numpy矩阵转换为列表
    @staticmethod
    def matrix2list(matrix):
        """
        To generate readable yaml file, we need to convert the matrix
        to list format.

        Parameters
        ----------
        matrix : np.ndarray
            The extrinsic/intrinsic matrix.

        Returns
        -------
        matrix_list : list
            The matrix represents in list format.
        """

        # 确保是二维矩阵后调用tolist()转换
        assert len(matrix.shape) == 2
        return matrix.tolist()