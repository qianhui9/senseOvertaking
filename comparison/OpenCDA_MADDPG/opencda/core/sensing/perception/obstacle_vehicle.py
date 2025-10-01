# -*- coding: utf-8 -*-
"""
Obstacle vehicle class to save object detection.
"""

import sys

import carla
import numpy as np
import open3d as o3d

import opencda.core.sensing.perception.sensor_transformation as st
from opencda.core.common.misc import get_speed_sumo

# 判断一个检测标签是否属于车辆类别(基于COCO数据集)
def is_vehicle_cococlass(label):
    """
    Check whether the label belongs to the vehicle class
    according to coco dataset.
    Args:
        -label(int): The lable of the detecting object.
    检查标签是否在预定义的车辆类别数组[1,2,3,5,7]中：1:自行车；2:汽车；3:摩托车；5:公交车；7:卡车
    Returns:
        -is_vehicle(bool): Whether this label belongs to the vehicle class
    """
    vehicle_class_array = np.array([1, 2, 3, 5, 7], dtype=np.int)
    return True if 0 in (label - vehicle_class_array) else False


class BoundingBox(object):
    """
    障碍物车辆的边界框类

    Params:
    -corners : nd.nparray
        Eight corners of the bounding box. (shape:(8, 3))
        - 边界框的8个角点(形状:(8,3))
    Attributes:
    -location : carla.location
        The location of the object.  - 对象中心位置
    -extent : carla.vector3D
        The extent of  the object.    - 对象尺寸(长宽高的一半)
    """

    def __init__(self, corners):
        # 计算中心点坐标(三个维度的平均值)
        center_x = np.mean(corners[:, 0])
        center_y = np.mean(corners[:, 1])
        center_z = np.mean(corners[:, 2])

        # 计算尺寸(每个维度最大值减最小值的一半)
        extent_x = (np.max(corners[:, 0]) - np.min(corners[:, 0])) / 2
        extent_y = (np.max(corners[:, 1]) - np.min(corners[:, 1])) / 2
        extent_z = (np.max(corners[:, 2]) - np.min(corners[:, 2])) / 2

        # 创建CARLA位置和尺寸对象
        self.location = carla.Location(x=center_x, y=center_y, z=center_z)
        self.extent = carla.Vector3D(x=extent_x, y=extent_y, z=extent_z)


class ObstacleVehicle(object):
    """
    障碍物车辆类，属性设计为与carla.Vehicle类匹配

    Parameters
    ----------
    corners : nd.nparray
        Eight corners of the bounding box. shape:(8, 3).

    o3d_bbx : pen3d.AlignedBoundingBox
        The bounding box object in Open3d. This is mainly used for
        visualization

    vehicle : carla.Vehicle
        The carla.Vehicle object.

    lidar : carla.sensor.lidar
        The lidar sensor.

    sumo2carla_ids : dict
        Sumo to carla mapping dictionary, this is used only when co-simulation
        is activated. We need this since the speed info of  vehicles that
        are controlled by sumo can not be read from carla server. We will
        need this dict to read vehicle speed from sumo api--traci.

    Attributes
    ----------
    bounding_box : BoundingBox
        Bounding box of the osbject vehicle.

    location : carla.location
        Location of the object.

    velocity : carla.Vector3D
        Velocity of the object vehicle.

    carla_id : int
        The obstacle vehicle's id. It should be the same with the
        corresponding carla.Vehicle's id. If no carla vehicle is
        matched with the obstacle vehicle, it should be -1.  SUMO到CARLA的ID映射字典(用于协同仿真)
    """

    def __init__(self, corners, o3d_bbx,
                 vehicle=None, lidar=None, sumo2carla_ids=None):

        if not vehicle:
            self.bounding_box = BoundingBox(corners)
            self.location = self.bounding_box.location
            # todo: next version will add rotation estimation
            self.transform = None
            self.o3d_bbx = o3d_bbx
            self.carla_id = -1
            self.velocity = carla.Vector3D(0.0, 0.0, 0.0)
        else:
            if sumo2carla_ids is None:
                sumo2carla_ids = dict()
            self.set_vehicle(vehicle, lidar, sumo2carla_ids)

    def get_transform(self):
        """
        Return the transform of the object vehicle.
        """
        return self.transform

    def get_location(self):
        """
        Return the location of the object vehicle.
        """
        return self.location

    def get_velocity(self):
        """
        Return the velocity of the object vehicle.
        """
        return self.velocity

    def set_carla_id(self, id):
        """
        Set carla id according to the carla.vehicle.

        Parameters
        ----------
        id : int
            The id from the carla.vehicle.
        """
        self.carla_id = id

    def set_velocity(self, velocity):
        """
        Set the velocity of the vehicle.

        Parameters
        ----------
        velocity : carla.Vector3D
            The target velocity in 3d vector format.

        """
        self.velocity = velocity

    def set_vehicle(self, vehicle, lidar, sumo2carla_ids):
        """
        从CARLA车辆对象复制属性到障碍物车辆

        Parameters
        ----------
        vehicle : carla.Vehicle
            The carla.Vehicle object.

        lidar : carla.sensor.lidar
            The lidar sensor, it is used to project world coordinates to
             sensor coordinates.

        sumo2carla_ids : dict
            Sumo to carla mapping dictionary, this is used only when
            co-simulation is activated. We need this since the speed info of
            vehicles that are controlled by sumo can not be read from carla
            server. We will need this dict to read vehicle speed
            from sumo api--traci.
        """
        self.location = vehicle.get_location()
        self.transform = vehicle.get_transform()
        self.bounding_box = vehicle.bounding_box
        self.carla_id = vehicle.id
        self.type_id = vehicle.type_id
        self.color = vehicle.attributes["color"] \
            if hasattr(vehicle, "attributes") \
               and "color" in vehicle.attributes else None

        # 设置速度
        self.set_velocity(vehicle.get_velocity())
        # the vehicle controlled by sumo has speed 0 in carla,
        # thus we need to retrieve the correct number from sumo
        # 处理SUMO控制的车辆速度
        if len(sumo2carla_ids) > 0:
            sumo_speed = get_speed_sumo(sumo2carla_ids, self.carla_id)
            if sumo_speed > 0:
                # todo: consider the yaw angle in the future
                speed_vector = carla.Vector3D(sumo_speed, 0, 0)   # 暂时不考虑偏航角
                self.set_velocity(speed_vector)

        # find the min and max boundary
        min_boundary = np.array([self.location.x - self.bounding_box.extent.x,
                                 self.location.y - self.bounding_box.extent.y,
                                 self.location.z + self.bounding_box.location.z
                                 - self.bounding_box.extent.z,    # # 齐次坐标
                                 1])
        max_boundary = np.array([self.location.x + self.bounding_box.extent.x,
                                 self.location.y + self.bounding_box.extent.y,
                                 self.location.z + self.bounding_box.location.z
                                 + self.bounding_box.extent.z,
                                 1])

        # 转换为列向量并水平堆叠
        min_boundary = min_boundary.reshape((4, 1))
        max_boundary = max_boundary.reshape((4, 1))
        stack_boundary = np.hstack((min_boundary, max_boundary))

        if lidar is None:
            return
        # the boundary coord at the lidar sensor space
        # 将边界坐标转换到激光雷达传感器空间
        stack_boundary_sensor_cords = st.world_to_sensor(stack_boundary,
                                                         lidar.get_transform())
        # convert unreal space to o3d space
        # 转换坐标系(Unreal到Open3D)
        stack_boundary_sensor_cords[:1, :] = - \
            stack_boundary_sensor_cords[:1, :]
        # (4,2) -> (3, 2)
        stack_boundary_sensor_cords = stack_boundary_sensor_cords[:-1, :]  # 移除齐次坐标

        # 计算传感器空间中的最小和最大边界
        min_boundary_sensor = np.min(stack_boundary_sensor_cords, axis=1)
        max_boundary_sensor = np.max(stack_boundary_sensor_cords, axis=1)

        # 创建Open3D轴对齐边界框
        aabb = \
            o3d.geometry.AxisAlignedBoundingBox(min_bound=min_boundary_sensor,
                                                max_bound=max_boundary_sensor)
        aabb.color = (1, 0, 0)      # 红色
        self.o3d_bbx = aabb   # 保存边界框
