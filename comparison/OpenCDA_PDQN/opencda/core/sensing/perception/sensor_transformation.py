# -*- coding: utf-8 -*-
"""
This script contains the transformations between world and different sensors.
"""

import numpy as np
from matplotlib import cm

from opencda.opencda_carla import Transform

# 获取viridis颜色映射的RGB颜色数组
VIRIDIS = np.array(cm.get_cmap('viridis').colors)
# 生成0.0到1.0的等间距数组，长度与VIRIDIS相同
VID_RANGE = np.linspace(0.0, 1.0, VIRIDIS.shape[0])


def get_camera_intrinsic(sensor):
    """
    Retrieve the camera intrinsic matrix.
            获取相机内参矩阵

    Parameters
    ----------
    sensor : carla.sensor
        Carla rgb camera object.

    Returns
    -------
    matrix_x : np.ndarray
        The 2d intrinsic matrix.

    """
    VIEW_WIDTH = int(sensor.attributes['image_size_x'])
    VIEW_HEIGHT = int(sensor.attributes['image_size_y'])
    VIEW_FOV = int(float(sensor.attributes['fov']))

    # 初始化单位矩阵
    matrix_k = np.identity(3)
    matrix_k[0, 2] = VIEW_WIDTH / 2.0   # 设置主点x坐标
    matrix_k[1, 2] = VIEW_HEIGHT / 2.0   # 设置主点y坐标
    # 计算焦距
    matrix_k[0, 0] = matrix_k[1, 1] = VIEW_WIDTH / \
        (2.0 * np.tan(VIEW_FOV * np.pi / 360.0))

    return matrix_k


def create_bb_points(vehicle):
    """
    Extract the eight vertices of the bounding box from the vehicle.
    从车辆对象提取边界框的8个顶点
    Parameters
    ----------
    vehicle : opencda object
        Opencda ObstacleVehicle that has attributes.

    Returns
    -------
    bbx : np.ndarray
        3d bounding box, shape:(8, 4).   从车辆对象提取边界框的8个顶点

    """
    bbx = np.zeros((8, 4))
    extent = vehicle.bounding_box.extent    # 获取边界框尺寸

    # 设置8个顶点的坐标(相对于中心点)
    bbx[0, :] = np.array([extent.x, extent.y, -extent.z, 1])   # 前右上
    bbx[1, :] = np.array([-extent.x, extent.y, -extent.z, 1])   # 前左上
    bbx[2, :] = np.array([-extent.x, -extent.y, -extent.z, 1])  # 前左下
    bbx[3, :] = np.array([extent.x, -extent.y, -extent.z, 1])   # 前右下
    bbx[4, :] = np.array([extent.x, extent.y, extent.z, 1])   # 后右上
    bbx[5, :] = np.array([-extent.x, extent.y, extent.z, 1])     # 后左上
    bbx[6, :] = np.array([-extent.x, -extent.y, extent.z, 1])   # 后左下
    bbx[7, :] = np.array([extent.x, -extent.y, extent.z, 1])    # 后右下

    return bbx


def x_to_world_transformation(transform):
    """
    Get the transformation matrix from x(it can be vehicle or sensor)
    coordinates to world coordinate.

    Parameters
    ----------
    transform : carla.Transform
        The transform that contains location and rotation

    Returns
    -------
    matrix : np.ndarray
        The transformation matrx.

    """
    rotation = transform.rotation
    location = transform.location

    #  计算旋转角度的三角函数值
    c_y = np.cos(np.radians(rotation.yaw))   # 偏航角
    s_y = np.sin(np.radians(rotation.yaw))
    c_r = np.cos(np.radians(rotation.roll))   # 翻滚角
    s_r = np.sin(np.radians(rotation.roll))
    c_p = np.cos(np.radians(rotation.pitch))  # 俯仰角
    s_p = np.sin(np.radians(rotation.pitch))

    matrix = np.identity(4)   # 初始化单位矩阵
    # 设置平移部分
    matrix[0, 3] = location.x
    matrix[1, 3] = location.y
    matrix[2, 3] = location.z

    # 设置旋转部分(3x3旋转矩阵)
    matrix[0, 0] = c_p * c_y
    matrix[0, 1] = c_y * s_p * s_r - s_y * c_r
    matrix[0, 2] = -c_y * s_p * c_r - s_y * s_r
    matrix[1, 0] = s_y * c_p
    matrix[1, 1] = s_y * s_p * s_r + c_y * c_r
    matrix[1, 2] = -s_y * s_p * c_r + c_y * s_r
    matrix[2, 0] = s_p
    matrix[2, 1] = -c_p * s_r
    matrix[2, 2] = c_p * c_r

    return matrix


def bbx_to_world(cords, vehicle):
    """
    Convert bounding box coordinate at vehicle reference to world reference.
    将边界框坐标从车辆坐标系转换到世界坐标系
    Parameters
    ----------
    cords : np.ndarray
        Bounding box coordinates with 8 vertices, shape (8, 4)
    vehicle : opencda object
        Opencda ObstacleVehicle.

    Returns
    -------
    bb_world_cords : np.ndarray
        Bounding box coordinates under world reference.

    """

    # 获取边界框到车辆的变换矩阵
    bb_transform = Transform(vehicle.bounding_box.location)

    # 获取车辆到世界的变换矩阵
    bb_vehicle_matrix = x_to_world_transformation(bb_transform)

    # 计算边界框到世界的变换矩阵
    vehicle_world_matrix = x_to_world_transformation(vehicle.get_transform())
    # bounding box to world transformation matrix
    bb_world_matrix = np.dot(vehicle_world_matrix, bb_vehicle_matrix)

    # 8 vertices are relative to bbx center, thus multiply with bbx_2_world to
    # get the world coords.
    # 应用变换矩阵
    bb_world_cords = np.dot(bb_world_matrix, np.transpose(cords))

    return bb_world_cords


def world_to_sensor(cords, sensor_transform):
    """
    Transform coordinates from world reference to sensor reference.
    将坐标从世界参考系转换到传感器参考系
    Parameters
    ----------
    cords : np.ndarray
        Coordinates under world reference, shape: (4, n).

    sensor_transform : carla.Transform
        Sensor position in the world.

    Returns
    -------
    sensor_cords : np.ndarray
        Coordinates in the sensor reference.

    """
    # 获取传感器到世界的变换矩阵
    sensor_world_matrix = x_to_world_transformation(sensor_transform)
    # 计算世界到传感器的逆变换
    world_sensor_matrix = np.linalg.inv(sensor_world_matrix)
    # 应用变换
    sensor_cords = np.dot(world_sensor_matrix, cords)

    return sensor_cords


def sensor_to_world(cords, sensor_transform):
    """
    Project coordinates in sensor to world reference.
    将坐标从传感器参考系投影到世界参考系
    Parameters
    ----------
    cords : np.ndarray
        Coordinates under sensor reference.

    sensor_transform : carla.Transform
        Sensor position in the world.

    Returns
    -------
    world_cords : np.ndarray
        Coordinates projected to world space.

    """
    # 获取传感器到世界的变换矩阵
    sensor_world_matrix = x_to_world_transformation(sensor_transform)
    # 应用变换
    world_cords = np.dot(sensor_world_matrix, cords)

    return world_cords


def vehicle_to_sensor(cords, vehicle, sensor_transform):
    """
    Transform coordinates from vehicle reference to sensor reference.
    将坐标从车辆参考系转换到传感器参考系
    Parameters
    ----------
    cords : np.ndarray
         Coordinates under vehicle reference, shape (n, 4).

    vehicle : opencda object
        Carla ObstacleVehicle.

    sensor_transform : carla.Transform
        Sensor position in the world.

    Returns
    -------
    sensor_cord : np.ndarray
        Coordinates in the sensor reference, shape(4, n)

    """
    # 先转换到世界坐标，再转换到传感器坐标
    world_cord = bbx_to_world(cords, vehicle)
    sensor_cord = world_to_sensor(world_cord, sensor_transform)

    return sensor_cord


def get_bounding_box(vehicle, camera, sensor_transform):
    """
    Get vehicle bounding box and project to sensor image.
    获取车辆边界框并投影到传感器图像
    Parameters
    ----------
    vehicle : carla.Vehicle
        Ego vehicle.

    camera : carla.sensor
        Carla rgb camera spawned at the vehicles.

    sensor_transform : carla.Transform
        Sensor position in the world.

    Returns
    -------
    camera_bbx : np.ndarray
        Bounding box coordinates in sensor image.

    """
    # 获取相机内参
    camera_k_matrix = get_camera_intrinsic(camera)
    # bb_cords is relative to bbx center(approximate the vehicle center)
    # 创建边界框点
    bb_cords = create_bb_points(vehicle)

    # bbx coordinates in sensor coordinate system. shape: (3, 8)
    # 转换到传感器坐标系
    cords_x_y_z = vehicle_to_sensor(bb_cords, vehicle, sensor_transform)[:3, :]
    # refer to https://github.com/carla-simulator/carla/issues/553
    # 调整坐标系顺序(y, -z, x)
    cords_y_minus_z_x = np.concatenate([cords_x_y_z[1, :].reshape(1, 8),
                                        -cords_x_y_z[2, :].reshape(1, 8),
                                        cords_x_y_z[0, :].reshape(1, 8)])
    # bounding box in sensor image. Shape:(8, 3)
    # 投影到图像平面
    bbox = np.transpose(np.dot(camera_k_matrix, cords_y_minus_z_x))

    # 归一化坐标
    new_x = (bbox[:, 0] / bbox[:, 2]).reshape(8, 1)
    new_y = (bbox[:, 1] / bbox[:, 2]).reshape(8, 1)
    new_z = bbox[:, 2].reshape(8, 1)
    camera_bbox = np.concatenate([new_x, new_y, new_z], axis=1)

    return camera_bbox


def p3d_to_p2d_bb(p3d_bb):
    """
    Draw 2d bounding box(4 vertices) from 3d bounding box(8 vertices). 2D
    bounding box is represented by two corner points.
    从3D边界框(8个顶点)绘制2D边界框(用两个角点表示)
    Parameters
    ----------
    p3d_bb : np.ndarray
        The 3d bounding box is going to project to 2d.

    Returns
    -------
    p2d_bb : np.ndarray
        Projected 2d bounding box.

    """
    # 计算x,y的最小最大值
    min_x = np.amin(p3d_bb[:, 0])
    min_y = np.amin(p3d_bb[:, 1])
    max_x = np.amax(p3d_bb[:, 0])
    max_y = np.amax(p3d_bb[:, 1])
    # 返回两个角点
    p2d_bb = np.array([[min_x, min_y], [max_x, max_y]])
    return p2d_bb


def get_2d_bb(vehicle, sensor, senosr_transform):
    """
    Summarize 2D bounding box creation.

    Parameters
    ----------
    vehicle : carla.Vehicle
        Ego vehicle.

    sensor : carla.sensor
        Carla sensor.

    senosr_transform : carla.Transform
        Sensor position.

    Returns
    -------
    p2d_bb : np.ndarray
        2D bounding box.

    """
    p3d_bb = get_bounding_box(vehicle, sensor, senosr_transform)
    p2d_bb = p3d_to_p2d_bb(p3d_bb)
    return p2d_bb


def project_lidar_to_camera(lidar, camera, point_cloud, rgb_image):
    """
    Project lidar to camera space.
    将激光雷达点云投影到相机空间
    Parameters
    ----------
    lidar : carla.sensor
        Lidar sensor.

    camera : carla.sensor
        RGB camera.

    point_cloud : np.ndarray
        Cloud points, shape: (n, 4).

    rgb_image : np.ndarray
        RGB image from camera.

    Returns
    -------
    rgb_image : np.ndarray
        New rgb image with lidar points projected.

    points_2d : np.ndarrya
        Point cloud projected to camera space.

    """

    # Lidar intensity array of shape (p_cloud_size,) but, for now, let's
    # focus on the 3D points.
    # 提取强度值
    intensity = np.array(point_cloud[:, 3])

    # Point cloud in lidar sensor space array of shape (3, p_cloud_size).
    # 获取点云坐标并转换为齐次坐标
    local_lidar_points = np.array(point_cloud[:, :3]).T
    # Add an extra 1.0 at the end of each 3d point so it becomes of
    # shape (4, p_cloud_size) and it can be multiplied by a (4, 4) matrix.
    local_lidar_points = np.r_[
        local_lidar_points, [np.ones(local_lidar_points.shape[1])]]

    # This (4, 4) matrix transforms the points from lidar space to world space.
    # 计算激光雷达到世界的变换矩阵
    lidar_2_world = x_to_world_transformation(lidar.get_transform())

    # transform lidar points from lidar space to world space
    # 转换到世界坐标
    world_points = np.dot(lidar_2_world, local_lidar_points)

    # project lidar world points to camera space
    # 转换到相机坐标
    sensor_points = world_to_sensor(world_points, camera.get_transform())

    # New we must change from UE4's coordinate system to an "standard"
    # camera coordinate system (the same used by OpenCV):

    # ^ z                       . z
    # |                        /
    # |              to:      +-------> x
    # | . x                   |
    # |/                      |
    # +-------> y             v y

    # (x, y ,z) -> (y, -z, x)
    # 调整坐标系顺序(y, -z, x)
    point_in_camera_coords = np.array([
        sensor_points[1],
        sensor_points[2] * -1,
        sensor_points[0]])

    # retrieve camera intrinsic
    # 获取相机内参并投影到图像平面
    K = get_camera_intrinsic(camera)
    # project the 3d points in camera space to image space
    points_2d = np.dot(K, point_in_camera_coords)

    # normalize x,y,z
    # 归一化坐标
    points_2d = np.array([
        points_2d[0, :] / points_2d[2, :],
        points_2d[1, :] / points_2d[2, :],
        points_2d[2, :]])

    # 获取图像尺寸
    image_w = int(camera.attributes['image_size_x'])
    image_h = int(camera.attributes['image_size_y'])

    # remove points out the camera scope
    # 过滤超出图像范围的点
    points_2d = points_2d.T
    intensity = intensity.T
    points_in_canvas_mask = \
        (points_2d[:, 0] > 0.0) & (points_2d[:, 0] < image_w) & \
        (points_2d[:, 1] > 0.0) & (points_2d[:, 1] < image_h) & \
        (points_2d[:, 2] > 0.0)
    new_points_2d = points_2d[points_in_canvas_mask]
    new_intensity = intensity[points_in_canvas_mask]

    # Extract the screen coords (uv) as integers.
    # 获取整数像素坐标
    u_coord = new_points_2d[:, 0].astype(np.int)
    v_coord = new_points_2d[:, 1].astype(np.int)

    # Since at the time of the creation of this script, the intensity function
    # is returning high values, these are adjusted to be nicely visualized.
    # 调整强度值范围并映射到颜色
    new_intensity = 4 * new_intensity - 3
    color_map = np.array([
        np.interp(new_intensity, VID_RANGE, VIRIDIS[:, 0]) * 255.0,
        np.interp(new_intensity, VID_RANGE, VIRIDIS[:, 1]) * 255.0,
        np.interp(new_intensity, VID_RANGE, VIRIDIS[:, 2]) * 255.0]).\
        astype(np.int).T

    # 在图像上绘制点
    for i in range(len(new_points_2d)):
        rgb_image[v_coord[i] - 1: v_coord[i] + 1,
                  u_coord[i] - 1: u_coord[i] + 1] = color_map[i]

    return rgb_image, points_2d
