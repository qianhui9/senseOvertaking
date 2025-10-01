# -*- coding: utf-8 -*-
"""
Utility functions for 3d lidar visualization
and processing by utilizing open3d.
"""

import time

import open3d as o3d
import numpy as np

from matplotlib import cm
from scipy.stats import mode

import opencda.core.sensing.perception.sensor_transformation as st
from opencda.core.sensing.perception.obstacle_vehicle import \
    is_vehicle_cococlass, ObstacleVehicle
from opencda.core.sensing.perception.static_obstacle import StaticObstacle

# VIRIDIS: 获取plasma颜色映射的RGB颜色数组
# VID_RANGE: 生成0.0到1.0的等间距数组，长度与VIRIDIS相同
VIRIDIS = np.array(cm.get_cmap('plasma').colors)
VID_RANGE = np.linspace(0.0, 1.0, VIRIDIS.shape[0])
# 定义不同类别的颜色编码(COCO数据集标准)；包含20+种物体类别(建筑、行人、车辆等)的RGB颜色；除以255将值归一化到[0,1]范围(Open3D使用的格式)
LABEL_COLORS = np.array([
    (255, 255, 255),  # None
    (70, 70, 70),  # Building
    (100, 40, 40),  # Fences
    (55, 90, 80),  # Other
    (220, 20, 60),  # Pedestrian
    (153, 153, 153),  # Pole
    (157, 234, 50),  # RoadLines
    (128, 64, 128),  # Road
    (244, 35, 232),  # Sidewalk
    (107, 142, 35),  # Vegetation
    (0, 0, 142),  # Vehicle
    (102, 102, 156),  # Wall
    (220, 220, 0),  # TrafficSign
    (70, 130, 180),  # Sky
    (81, 0, 81),  # Ground
    (150, 100, 100),  # Bridge
    (230, 150, 140),  # RailTrack
    (180, 165, 180),  # GuardRail
    (250, 170, 30),  # TrafficLight
    (110, 190, 160),  # Static
    (170, 120, 50),  # Dynamic
    (45, 60, 150),  # Water
    (145, 170, 100),  # Terrain
]) / 255.0  # normalize each channel [0-1] since is what Open3D uses


# 将原始点云数据(np数组)编码为Open3D点云对象
def o3d_pointcloud_encode(raw_data, point_cloud):
    """
    Encode the raw point cloud(np.array) to Open3d PointCloud object.

    Parameters
    ----------
    raw_data : np.ndarray
        Raw lidar points, (N, 4).

    point_cloud : o3d.PointCloud
        Open3d PointCloud.

    """

    # Isolate the intensity and compute a color for it
    # 提取强度值(最后一列)
    intensity = raw_data[:, -1]
    # 计算强度颜色映射值(对数变换)
    intensity_col = 1.0 - np.log(intensity) / np.log(np.exp(-0.004 * 100))
    # 使用插值将强度值映射到VIRIDIS颜色空间
    int_color = np.c_[
        np.interp(intensity_col, VID_RANGE, VIRIDIS[:, 0]),
        np.interp(intensity_col, VID_RANGE, VIRIDIS[:, 1]),
        np.interp(intensity_col, VID_RANGE, VIRIDIS[:, 2])]

    # Isolate the 3D data；提取3D坐标点(x,y,z)
    points = np.array(raw_data[:, :-1], copy=True)
    # We're negating the y to correclty visualize a world that matches
    # what we see in Unreal since Open3D uses a right-handed coordinate system
    # 取反y坐标以匹配Unreal引擎的坐标系
    points[:, :1] = -points[:, :1]

    # 设置点云的坐标和颜色属性
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.colors = o3d.utility.Vector3dVector(int_color)

# 初始化Open3D可视化窗口
def o3d_visualizer_init(actor_id):
    """
    Initialize the visualizer.

    Parameters
    ----------
    actor_id : int
        Ego vehicle's id.

    Returns
    -------
    vis : o3d.visualizer
        Initialize open3d visualizer.

    """
    # 创建可视化窗口，设置窗口名称(使用actor_id)、大小和位置
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=str(actor_id),
                      width=480,
                      height=320,
                      left=480,
                      top=270)
    # 设置渲染选项：背景色、点大小、显示坐标系
    vis.get_render_option().background_color = [0.05, 0.05, 0.05]
    vis.get_render_option().point_size = 1
    vis.get_render_option().show_coordinate_frame = True

    return vis

# 实时可视化点云和物体
def o3d_visualizer_show(vis, count, point_cloud, objects):
    """
    Visualize the point cloud at runtime.

    Parameters
    ----------
    vis : o3d.Visualizer
        Visualization interface.

    count : int
        Current step since simulation started.

    point_cloud : o3d.PointCloud
        Open3d point cloud.

    objects : dict
        The dictionary containing objects.

    Returns
    -------

    """

    # 首次运行时添加点云几何体
    if count == 2:
        vis.add_geometry(point_cloud)

    # 更新点云数据
    vis.update_geometry(point_cloud)

    # 只绘制车辆类别
    for key, object_list in objects.items():
        # we only draw vehicles for now
        if key != 'vehicles':
            continue
        # 添加每个车辆的3D边界框
        for object_ in object_list:
            aabb = object_.o3d_bbx
            vis.add_geometry(aabb)

    # 处理事件和更新渲染
    vis.poll_events()
    vis.update_renderer()
    # # This can fix Open3D jittering issues:
    # 短暂睡眠防止界面抖动
    time.sleep(0.001)

    # 移除临时绘制的边界框
    for key, object_list in objects.items():
        if key != 'vehicles':
            continue
        for object_ in object_list:
            aabb = object_.o3d_bbx
            vis.remove_geometry(aabb)

# 融合相机2D检测和激光雷达3D点云，生成3D边界框
def o3d_camera_lidar_fusion(objects,
                            yolo_bbx,
                            lidar_3d,
                            projected_lidar,
                            lidar_sensor):
    """
    Utilize the 3D lidar points to extend the 2D bounding box
    from camera to 3D bounding box under world coordinates.

    Parameters
    ----------
    objects : dict
        The dictionary contains all object detection results.

    yolo_bbx : torch.Tensor
        Object detection bounding box at current photo from yolov5,
        shape (n, 5)->(n, [x1, y1, x2, y2, label])

    lidar_3d : np.ndarray
        Raw 3D lidar points in lidar coordinate system.

    projected_lidar : np.ndarray
        3D lidar points projected to the camera space.

    lidar_sensor : carla.sensor
        The lidar sensor.

    Returns
    -------
    objects : dict
        The update object dictionary that contains 3d bounding boxes.
    """

    # convert torch tensor to numpy array first
    # 将YOLO检测结果从GPU转到CPU并转为numpy数组
    if yolo_bbx.is_cuda:
        yolo_bbx = yolo_bbx.cpu().detach().numpy()
    else:
        yolo_bbx = yolo_bbx.detach().numpy()

    for i in range(yolo_bbx.shape[0]):
        detection = yolo_bbx[i]
        # 2d bbx coordinates
        x1, y1, x2, y2 = int(detection[0]), int(detection[1]),\
            int(detection[2]), int(detection[3])
        label = int(detection[5])

        # choose the lidar points in the 2d yolo bounding box
        # 选择位于2D边界框内的激光雷达点
        points_in_bbx = \
            (projected_lidar[:, 0] > x1) & (projected_lidar[:, 0] < x2) & \
            (projected_lidar[:, 1] > y1) & (projected_lidar[:, 1] < y2) & \
            (projected_lidar[:, 2] > 0.0)
        # ignore intensity channel
        # 去除强度值，只保留坐标
        select_points = lidar_3d[points_in_bbx][:, :-1]

        if select_points.shape[0] == 0:
            continue

        # filter out the outlier
        # 使用众数滤波去除离群点
        x_common = mode(np.array(np.abs(select_points[:, 0]),
                                 dtype=np.int), axis=0)[0][0]
        y_common = mode(np.array(np.abs(select_points[:, 1]),
                                 dtype=np.int), axis=0)[0][0]
        # 保留在常见值±3范围内的点
        points_inlier = (np.abs(select_points[:, 0]) > x_common - 3) & \
                        (np.abs(select_points[:, 0]) < x_common + 3) & \
                        (np.abs(select_points[:, 1]) > y_common - 3) & \
                        (np.abs(select_points[:, 1]) < y_common + 3)
        select_points = select_points[points_inlier]

        if select_points.shape[0] < 2:
            continue

        # to visualize 3d lidar points in o3d visualizer, we need to
        # revert the x coordinates
        select_points[:, :1] = -select_points[:, :1]

        # create o3d.PointCloud object
        # 创建Open3D点云并计算轴对齐边界框(AABB)
        o3d_pointcloud = o3d.geometry.PointCloud()
        o3d_pointcloud.points = o3d.utility.Vector3dVector(select_points)
        # add o3d bounding box
        aabb = o3d_pointcloud.get_axis_aligned_bounding_box()
        aabb.color = (0, 1, 0)

        # get the eight corner of the bounding boxes.
        # 获取边界框8个角点
        corner = np.asarray(aabb.get_box_points())
        # covert back to unreal coordinate
        # 转换到世界坐标系
        corner[:, :1] = -corner[:, :1]
        corner = corner.transpose()
        # extend (3, 8) to (4, 8) for homogenous transformation
        corner = np.r_[corner, [np.ones(corner.shape[1])]]
        # project to world reference
        corner = st.sensor_to_world(corner, lidar_sensor.get_transform())
        corner = corner.transpose()[:, :3]

        # 根据类别创建车辆或静态障碍物对象，添加到objects字典中返回
        if is_vehicle_cococlass(label):
            obstacle_vehicle = ObstacleVehicle(corner, aabb)
            if 'vehicles' in objects:
                objects['vehicles'].append(obstacle_vehicle)
            else:
                objects['vehicles'] = [obstacle_vehicle]
        # todo: refine the category
        # we regard or other obstacle rather than vehicle as static class
        else:
            static_obstacle = StaticObstacle(corner, aabb)
            if 'static' in objects:
                objects['static'].append(static_obstacle)
            else:
                objects['static'] = [static_obstacle]

    return objects
