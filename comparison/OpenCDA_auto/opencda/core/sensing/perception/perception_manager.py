# -*- coding: utf-8 -*-
"""
Perception module base.
"""

import weakref
import sys
import time

import carla
import cv2
import numpy as np
import open3d as o3d

import opencda.core.sensing.perception.sensor_transformation as st
from opencda.core.common.misc import \
    cal_distance_angle, get_speed, get_speed_sumo
from opencda.core.sensing.perception.obstacle_vehicle import \
    ObstacleVehicle
from opencda.core.sensing.perception.static_obstacle import TrafficLight
from opencda.core.sensing.perception.o3d_lidar_libs import \
    o3d_visualizer_init, o3d_pointcloud_encode, o3d_visualizer_show, \
    o3d_camera_lidar_fusion

# 用于车辆或基础设施的相机管理器
class CameraSensor:
    """
    Camera manager for vehicle or infrastructure.
    参数：
    vehicle : carla.Vehicle - 车辆对象(用于CAV)
    world : carla.World - 世界对象(用于RSU)
    global_position : list - 基础设施的全局位置[x,y,z]
    relative_position : str - 传感器位置(前/左/右)
    Parameters
    ----------
    vehicle : carla.Vehicle
        The carla.Vehicle, this is for cav.

    world : carla.World
        The carla world object, this is for rsu.

    global_position : list
        Global position of the infrastructure, [x, y, z]

    relative_position : str
        Indicates the sensor is a front or rear camera. option:
        front, left, right.

    Attributes
    ----------
    image : np.ndarray
        Current received rgb image.
    sensor : carla.sensor
        The carla sensor that mounts at the vehicle.

    """

    def __init__(self, vehicle, world, relative_position, global_position):
        if vehicle is not None:
            world = vehicle.get_world()   # 从车辆获取世界对象

        # 获取RGB相机蓝图并设置视场角(FOV)为100度
        blueprint = world.get_blueprint_library().find('sensor.camera.rgb')
        blueprint.set_attribute('fov', '100')

        # 计算生成点位置
        spawn_point = self.spawn_point_estimation(relative_position,
                                                  global_position)

        # 生成传感器actor
        if vehicle is not None:
            self.sensor = world.spawn_actor(
                blueprint, spawn_point, attach_to=vehicle)
        else:
            self.sensor = world.spawn_actor(blueprint, spawn_point)

        # 初始化图像数据和时间戳
        self.image = None
        self.timstamp = None
        self.frame = 0
        # 设置弱引用和监听回调
        weak_self = weakref.ref(self)
        self.sensor.listen(
            lambda event: CameraSensor._on_rgb_image_event(
                weak_self, event))

        #  获取相机分辨率属性
        self.image_width = int(self.sensor.attributes['image_size_x'])
        self.image_height = int(self.sensor.attributes['image_size_y'])

    # 生成点估计方法
    @staticmethod
    def spawn_point_estimation(relative_position, global_position):

        pitch = 0  # 默认俯仰角
        carla_location = carla.Location(x=0, y=0, z=0)
        x, y, z, yaw = relative_position  # 解构相对位置

        # this is for rsu. It utilizes global position instead of relative
        # position to the vehicle
        if global_position is not None:  # RSU使用全局位置
            carla_location = carla.Location(
                x=global_position[0],
                y=global_position[1],
                z=global_position[2])
            pitch = -35   # RSU相机俯角35度

        # 计算最终位置
        carla_location = carla.Location(x=carla_location.x + x,
                                        y=carla_location.y + y,
                                        z=carla_location.z + z)

        carla_rotation = carla.Rotation(roll=0, yaw=yaw, pitch=pitch)
        spawn_point = carla.Transform(carla_location, carla_rotation)

        return spawn_point

    # 图像回调方法
    @staticmethod
    def _on_rgb_image_event(weak_self, event):
        """CAMERA  method"""
        self = weak_self()
        if not self:
            return
        image = np.array(event.raw_data)  # 原始数据转numpy
        image = image.reshape((self.image_height, self.image_width, 4))    # 重塑为H×W×4
        # we need to remove the alpha channel
        image = image[:, :, :3]

        self.image = image  # 更新当前图像
        self.frame = event.frame   # 更新帧号
        self.timestamp = event.timestamp   # 更新时间戳


# 激光雷达传感器管理
class LidarSensor:
    """
    Lidar sensor manager.

    Parameters
    ----------
    vehicle : carla.Vehicle
        The carla.Vehicle, this is for cav.

    world : carla.World
        The carla world object, this is for rsu.

    config_yaml : dict
        Configuration dictionary for lidar.

    global_position : list
        Global position of the infrastructure, [x, y, z]

    Attributes
    ----------
    o3d_pointcloud : 03d object
        Received point cloud, saved in o3d.Pointcloud format.

    sensor : carla.sensor
        Lidar sensor that will be attached to the vehicle.

    """

    def __init__(self, vehicle, world, config_yaml, global_position):
        if vehicle is not None:
            world = vehicle.get_world()
        blueprint = world.get_blueprint_library().find('sensor.lidar.ray_cast')

        # 根据配置设置雷达属性
        blueprint.set_attribute('upper_fov', str(config_yaml['upper_fov']))
        blueprint.set_attribute('lower_fov', str(config_yaml['lower_fov']))
        blueprint.set_attribute('channels', str(config_yaml['channels']))
        blueprint.set_attribute('range', str(config_yaml['range']))
        blueprint.set_attribute(
            'points_per_second', str(
                config_yaml['points_per_second']))
        blueprint.set_attribute(
            'rotation_frequency', str(
                config_yaml['rotation_frequency']))
        blueprint.set_attribute(
            'dropoff_general_rate', str(
                config_yaml['dropoff_general_rate']))
        blueprint.set_attribute(
            'dropoff_intensity_limit', str(
                config_yaml['dropoff_intensity_limit']))
        blueprint.set_attribute(
            'dropoff_zero_intensity', str(
                config_yaml['dropoff_zero_intensity']))
        blueprint.set_attribute(
            'noise_stddev', str(
                config_yaml['noise_stddev']))

        # 生成传感器
        if global_position is None:  # 车辆安装位置
            spawn_point = carla.Transform(carla.Location(x=-0.5, z=1.9))  # 车头前0.5米，高1.9米
        else:   # RSU安装位置
            spawn_point = carla.Transform(carla.Location(x=global_position[0],
                                                         y=global_position[1],
                                                         z=global_position[2]))
        if vehicle is not None:
            self.sensor = world.spawn_actor(
                blueprint, spawn_point, attach_to=vehicle)
        else:
            self.sensor = world.spawn_actor(blueprint, spawn_point)

        # 初始化数据存储
        self.data = None
        self.timestamp = None
        self.frame = 0
        # Open3D点云对象
        self.o3d_pointcloud = o3d.geometry.PointCloud()

        weak_self = weakref.ref(self)
        self.sensor.listen(
            lambda event: LidarSensor._on_data_event(
                weak_self, event))

    @staticmethod
    def _on_data_event(weak_self, event):
        """Lidar  method"""
        self = weak_self()
        if not self:
            return

        # retrieve the raw lidar data and reshape to (N, 4)
        data = np.copy(np.frombuffer(event.raw_data, dtype=np.dtype('f4')))
        # (x, y, z, intensity)
        data = np.reshape(data, (int(data.shape[0] / 4), 4))

        self.data = data
        self.frame = event.frame
        self.timestamp = event.timestamp

# 语义激光雷达管理
class SemanticLidarSensor:
    """
    Semantic lidar sensor manager. This class is used when data dumping
    is needed.

    Parameters
    ----------
    vehicle : carla.Vehicle
        The carla.Vehicle, this is for cav.

    world : carla.World
        The carla world object, this is for rsu.

    config_yaml : dict
        Configuration dictionary for lidar.

    global_position : list
        Global position of the infrastructure, [x, y, z]

    Attributes
    ----------
    o3d_pointcloud : 03d object
        Received point cloud, saved in o3d.Pointcloud format.

    sensor : carla.sensor
        Lidar sensor that will be attached to the vehicle.


    """

    def __init__(self, vehicle, world, config_yaml, global_position):
        if vehicle is not None:
            world = vehicle.get_world()

        blueprint = \
            world.get_blueprint_library(). \
                find('sensor.lidar.ray_cast_semantic')

        # 设置语义雷达属性(配置项比普通雷达少)
        blueprint.set_attribute('upper_fov', str(config_yaml['upper_fov']))
        blueprint.set_attribute('lower_fov', str(config_yaml['lower_fov']))
        blueprint.set_attribute('channels', str(config_yaml['channels']))
        blueprint.set_attribute('range', str(config_yaml['range']))
        blueprint.set_attribute(
            'points_per_second', str(
                config_yaml['points_per_second']))
        blueprint.set_attribute(
            'rotation_frequency', str(
                config_yaml['rotation_frequency']))

        # 生成逻辑与普通雷达相同
        if global_position is None:
            spawn_point = carla.Transform(carla.Location(x=-0.5, z=1.9))
        else:
            spawn_point = carla.Transform(carla.Location(x=global_position[0],
                                                         y=global_position[1],
                                                         z=global_position[2]))

        if vehicle is not None:
            self.sensor = world.spawn_actor(
                blueprint, spawn_point, attach_to=vehicle)
        else:
            self.sensor = world.spawn_actor(blueprint, spawn_point)

        # 语义特有数据
        self.points = None  # 点坐标
        self.obj_idx = None  # 物体索引
        self.obj_tag = None  # 物体标签

        self.timestamp = None
        self.frame = 0
        # open3d point cloud object
        self.o3d_pointcloud = o3d.geometry.PointCloud()

        weak_self = weakref.ref(self)
        self.sensor.listen(
            lambda event: SemanticLidarSensor._on_data_event(
                weak_self, event))

    # 数据回调方法
    @staticmethod
    def _on_data_event(weak_self, event):
        """Semantic Lidar  method"""
        self = weak_self()
        if not self:
            return

        # 解析带语义的雷达数据(每个点6个属性)
        data = np.frombuffer(event.raw_data, dtype=np.dtype([
            ('x', np.float32), ('y', np.float32), ('z', np.float32),
            ('CosAngle', np.float32), ('ObjIdx', np.uint32),
            ('ObjTag', np.uint32)]))

        # 分别存储不同属性
        self.points = np.array([data['x'], data['y'], data['z']]).T   # 坐标
        self.obj_tag = np.array(data['ObjTag'])    # 语义标签
        self.obj_idx = np.array(data['ObjIdx'])   # 物体标签

        self.data = data
        self.frame = event.frame
        self.timestamp = event.timestamp


class PerceptionManager:
    """
    Default perception module. Currenly only used to detect vehicles.
     默认感知模块，当前仅用于检测车辆

    参数：
    vehicle : carla.Vehicle - 用于生成传感器的车辆
    config_yaml : dict - 感知配置字典
    cav_world : opencda对象 - 保存所有CAV信息的共享世界
    data_dump : bool - 是否转储数据(转储时会生成语义激光雷达)
    carla_world : carla.World - CARLA世界对象(用于RSU)
    infra_id : int - 基础设施ID(用于RSU)

    Parameters
    ----------
    vehicle : carla.Vehicle
        carla Vehicle, we need this to spawn sensors.

    config_yaml : dict
        Configuration dictionary for perception.

    cav_world : opencda object
        CAV World object that saves all cav information, shared ML model,
         and sumo2carla id mapping dictionary.

    data_dump : bool
        Whether dumping data, if true, semantic lidar will be spawned.

    carla_world : carla.world
        CARLA world, used for rsu.

    Attributes
    ----------
    lidar : opencda object
        Lidar sensor manager.

    rgb_camera : opencda object
        RGB camera manager.

    o3d_vis : o3d object
        Open3d point cloud visualizer.
    """

    def __init__(self, vehicle, config_yaml, cav_world,
                 data_dump=False, carla_world=None, infra_id=None):
        self.vehicle = vehicle
        self.carla_world = carla_world if carla_world is not None \
            else self.vehicle.get_world()
        self._map = self.carla_world.get_map()   # 获取地图
        self.id = infra_id if infra_id is not None else vehicle.id  # 设置ID

        # 从配置中读取参数
        self.activate = config_yaml['activate'] # 是否激活感知模块
        self.camera_visualize = config_yaml['camera']['visualize']  # 是否可视化相机
        self.camera_num = config_yaml['camera']['num'] # 相机数量
        self.lidar_visualize = config_yaml['lidar']['visualize']   # 是否可视化激光雷达
        self.global_position = config_yaml['global_position'] \
            if 'global_position' in config_yaml else None   # 全局位置

        self.cav_world = weakref.ref(cav_world)()  # 弱引用共享世界
        ml_manager = cav_world.ml_manager    # 获取机器学习管理器

        # 检查配置一致性
        if self.activate and data_dump:
            sys.exit("When you dump data, please deactivate the "
                     "detection function for precise label.")
        if self.activate and not ml_manager:
            sys.exit(
                'If you activate the perception module, '
                'then apply_ml must be set to true in'
                'the argument parser to load the detection DL model.')
        self.ml_manager = ml_manager  # 存储ML管理器

        # we only spawn the camera when perception module is activated or
        # camera visualization is needed
        # # 相机初始化(激活或需要可视化时)
        if self.activate or self.camera_visualize:
            self.rgb_camera = []
            mount_position = config_yaml['camera']['positions']
            assert len(mount_position) == self.camera_num, \
                "The camera number has to be the same as the length of the" \
                "relative positions list"  # "相机数量必须与位置列表长度一致"

            for i in range(self.camera_num):
                self.rgb_camera.append(
                    CameraSensor(
                        vehicle, self.carla_world, mount_position[i],
                        self.global_position))

        else:
            self.rgb_camera = None

        # we only spawn the LiDAR when perception module is activated or lidar
        # visualization is needed
        # 激光雷达初始化(激活或需要可视化时)
        if self.activate or self.lidar_visualize:
            self.lidar = LidarSensor(vehicle,
                                     self.carla_world,
                                     config_yaml['lidar'],
                                     self.global_position)
            self.o3d_vis = o3d_visualizer_init(self.id)   # 初始化Open3D可视化
        else:
            self.lidar = None
            self.o3d_vis = None

        # if data dump is true, semantic lidar is also spawned
        self.data_dump = data_dump
        if data_dump:
            self.semantic_lidar = SemanticLidarSensor(vehicle,
                                                      self.carla_world,
                                                      config_yaml['lidar'],
                                                      self.global_position)

        # count how many steps have been passed
        self.count = 0
        # ego position
        self.ego_pos = None

        # the dictionary contains all objects
        self.objects = {}
        # # 交通灯检测阈值
        self.traffic_thresh = config_yaml['traffic_light_thresh'] \
            if 'traffic_light_thresh' in config_yaml else 50

    def dist(self, a):
        """
        A fast method to retrieve the obstacle distance the ego
        vehicle from the server directly.
        计算自车与目标actor的距离
        Parameters
        ----------
        a : carla.actor
            The obstacle vehicle.

        Returns
        -------
        distance : float
            The distance between ego and the target actor.
        """
        return a.get_location().distance(self.ego_pos.location)

    def detect(self, ego_pos):
        """
        Detect surrounding objects. Currently only vehicle detection supported.
        检测周围物体(当前仅支持车辆检测)
        Parameters
        ----------
        ego_pos : carla.Transform
            Ego vehicle pose.

        Returns
        -------
        objects : list
            A list that contains all detected obstacle vehicles.

        """
        self.ego_pos = ego_pos

        objects = {'vehicles': [],
                   'traffic_lights': []}

        if not self.activate:
            objects = self.deactivate_mode(objects)  # 非激活模式检测

        else:
            objects = self.activate_mode(objects)   # 激活模式检测

        self.count += 1

        return objects

    def activate_mode(self, objects):
        """
        Use Yolov5 + Lidar fusion to detect objects.
        使用Yolov5+激光雷达融合检测物体
        Parameters
        ----------
        objects : dict
            The dictionary that contains all category of detected objects.
            The key is the object category name and value is its 3d coordinates
            and confidence.

        Returns
        -------
         objects: dict
            Updated object dictionary.
        """
        # # 获取相机和激光雷达数据
        rgb_images = []
        for rgb_camera in self.rgb_camera:
            while rgb_camera.image is None:
                continue
            rgb_images.append(
                cv2.cvtColor(
                    np.array(
                        rgb_camera.image),
                    cv2.COLOR_BGR2RGB))

        # yolo detection
        yolo_detection = self.ml_manager.object_detector(rgb_images)
        # rgb_images for drawing
        rgb_draw_images = []

        # 相机-激光雷达融合
        for (i, rgb_camera) in enumerate(self.rgb_camera):
            # lidar projection
            rgb_image, projected_lidar = st.project_lidar_to_camera(
                self.lidar.sensor,
                rgb_camera.sensor, self.lidar.data, np.array(
                    rgb_camera.image))
            rgb_draw_images.append(rgb_image)

            # camera lidar fusion
            objects = o3d_camera_lidar_fusion(
                objects,
                yolo_detection.xyxy[i],
                self.lidar.data,
                projected_lidar,
                self.lidar.sensor)

            # calculate the speed. current we retrieve from the server
            # directly.
            self.speed_retrieve(objects)

        # # 可视化处理
        if self.camera_visualize:
            for (i, rgb_image) in enumerate(rgb_draw_images):
                if i > self.camera_num - 1 or i > self.camera_visualize - 1:
                    break
                rgb_image = self.ml_manager.draw_2d_box(
                    yolo_detection, rgb_image, i)
                rgb_image = cv2.resize(rgb_image, (0, 0), fx=0.4, fy=0.4)
                cv2.imshow(
                    '%s-th camera of actor %d, perception activated' %
                    (str(i), self.id), rgb_image)
            cv2.waitKey(1)

        if self.lidar_visualize:
            while self.lidar.data is None:
                continue
            o3d_pointcloud_encode(self.lidar.data, self.lidar.o3d_pointcloud)
            o3d_visualizer_show(
                self.o3d_vis,
                self.count,
                self.lidar.o3d_pointcloud,
                objects)
        # add traffic light
        objects = self.retrieve_traffic_lights(objects)
        self.objects = objects

        return objects

    def deactivate_mode(self, objects):
        """
        Object detection using server information directly.
        直接从服务器获取物体信息
        Parameters
        ----------
        objects : dict
            The dictionary that contains all category of detected objects.
            The key is the object category name and value is its 3d coordinates
            and confidence.

        Returns
        -------
         objects: dict
            Updated object dictionary.
        """
        world = self.carla_world

        vehicle_list = world.get_actors().filter("*vehicle*")
        # todo: hard coded
        thresh = 50 if not self.data_dump else 120   # 检测范围阈值

        # # 过滤车辆
        vehicle_list = [v for v in vehicle_list if self.dist(v) < thresh and
                        v.id != self.id]

        # 数据转储时使用语义激光雷达进一步过滤
        if self.data_dump:
            vehicle_list = self.filter_vehicle_out_sensor(vehicle_list)

        # convert carla.Vehicle to opencda.ObstacleVehicle if lidar
        # visualization is required.
        if self.lidar:
            # 转换为ObstacleVehicle对象
            vehicle_list = [
                ObstacleVehicle(
                    None,
                    None,
                    v,
                    self.lidar.sensor,
                    self.cav_world.sumo2carla_ids) for v in vehicle_list]
        else:
            vehicle_list = [
                ObstacleVehicle(
                    None,
                    None,
                    v,
                    None,
                    self.cav_world.sumo2carla_ids) for v in vehicle_list]

        objects.update({'vehicles': vehicle_list})

        # 可视化处理
        if self.camera_visualize:
            # 在相机图像上绘制3D边界框
            while self.rgb_camera[0].image is None:
                continue

            names = ['front', 'right', 'left', 'back']

            for (i, rgb_camera) in enumerate(self.rgb_camera):
                if i > self.camera_num - 1 or i > self.camera_visualize - 1:
                    break
                # we only visualiz the frontal camera
                rgb_image = np.array(rgb_camera.image)
                # draw the ground truth bbx on the camera image
                rgb_image = self.visualize_3d_bbx_front_camera(objects,
                                                               rgb_image,
                                                               i)
                # resize to make it fittable to the screen
                rgb_image = cv2.resize(rgb_image, (0, 0), fx=0.4, fy=0.4)

                # show image using cv2
                cv2.imshow(
                    '%s camera of actor %d, perception deactivated' %
                    (names[i], self.id), rgb_image)
                cv2.waitKey(1)

        if self.lidar_visualize:
            # 可视化激光雷达点云
            while self.lidar.data is None:
                continue
            o3d_pointcloud_encode(self.lidar.data, self.lidar.o3d_pointcloud)
            # render the raw lidar
            o3d_visualizer_show(
                self.o3d_vis,
                self.count,
                self.lidar.o3d_pointcloud,
                objects)

        # add traffic light
        objects = self.retrieve_traffic_lights(objects)
        self.objects = objects

        return objects

    def filter_vehicle_out_sensor(self, vehicle_list):
        """
        By utilizing semantic lidar, we can retrieve the objects that
        are in the lidar detection range from the server.
        This function is important for collect training data for object
        detection as it can filter out the objects out of the senor range.
        使用语义激光雷达过滤传感器范围外的车辆
        Parameters
        ----------
        vehicle_list : list
            The list contains all vehicles information retrieves from the
            server.

        Returns
        -------
        new_vehicle_list : list
            The list that filters out the out of scope vehicles.

        """
        semantic_idx = self.semantic_lidar.obj_idx
        semantic_tag = self.semantic_lidar.obj_tag

        # label 10 is the vehicle  标签10是车辆
        vehicle_idx = semantic_idx[semantic_tag == 10]
        # each individual instance id
        vehicle_unique_id = list(np.unique(vehicle_idx))

        # 过滤不在传感器范围内的车辆
        new_vehicle_list = []
        for veh in vehicle_list:
            if veh.id in vehicle_unique_id:
                new_vehicle_list.append(veh)

        return new_vehicle_list

    def visualize_3d_bbx_front_camera(self, objects, rgb_image, camera_index):
        """
        Visualize the 3d bounding box on frontal camera image.

        Parameters
        ----------
        objects : dict
            The object dictionary.

        rgb_image : np.ndarray
            Received rgb image at current timestamp.

        camera_index : int
            Indicate the index of the current camera.

        """
        camera_transform = \
            self.rgb_camera[camera_index].sensor.get_transform()
        camera_location = \
            camera_transform.location
        camera_rotation = \
            camera_transform.rotation

        for v in objects['vehicles']:
            # we only draw the bounding box in the fov of camera
            _, angle = cal_distance_angle(
                v.get_location(), camera_location,
                camera_rotation.yaw)
            if angle < 60:
                bbx_camera = st.get_2d_bb(
                    v,
                    self.rgb_camera[camera_index].sensor,
                    camera_transform)
                cv2.rectangle(rgb_image,
                              (int(bbx_camera[0, 0]), int(bbx_camera[0, 1])),
                              (int(bbx_camera[1, 0]), int(bbx_camera[1, 1])),
                              (255, 0, 0), 2)

        return rgb_image

    def speed_retrieve(self, objects):
        """
        We don't implement any obstacle speed calculation algorithm.
        The speed will be retrieved from the server directly.

        Parameters
        ----------
        objects : dict
            The dictionary contains the objects.
        """
        if 'vehicles' not in objects:
            return

        world = self.carla_world
        vehicle_list = world.get_actors().filter("*vehicle*")
        vehicle_list = [v for v in vehicle_list if self.dist(v) < 50 and
                        v.id != self.id]

        # todo: consider the minimum distance to be safer in next version
        for v in vehicle_list:
            loc = v.get_location()
            for obstacle_vehicle in objects['vehicles']:
                obstacle_speed = get_speed(obstacle_vehicle)
                # if speed > 0, it represents that the vehicle
                # has been already matched.
                if obstacle_speed > 0:
                    continue
                obstacle_loc = obstacle_vehicle.get_location()
                if abs(loc.x - obstacle_loc.x) <= 3.0 and \
                        abs(loc.y - obstacle_loc.y) <= 3.0:
                    obstacle_vehicle.set_velocity(v.get_velocity())

                    # the case where the obstacle vehicle is controled by
                    # sumo
                    if self.cav_world.sumo2carla_ids:
                        sumo_speed = \
                            get_speed_sumo(self.cav_world.sumo2carla_ids,
                                           v.id)
                        if sumo_speed > 0:
                            # todo: consider the yaw angle in the future
                            speed_vector = carla.Vector3D(sumo_speed, 0, 0)
                            obstacle_vehicle.set_velocity(speed_vector)

                    obstacle_vehicle.set_carla_id(v.id)

    def retrieve_traffic_lights(self, objects):
        """
        Retrieve the traffic lights nearby from the server  directly.
        Next version may consider add traffic light detection module.
        从服务器获取附近交通灯信息
        Parameters
        ----------
        objects : dict
            The dictionary that contains all objects.

        Returns
        -------
        object : dict
            The updated dictionary.
        """
        world = self.carla_world
        tl_list = world.get_actors().filter('traffic.traffic_light*')

        vehicle_location = self.ego_pos.location
        vehicle_waypoint = self._map.get_waypoint(vehicle_location)

        # 获取激活的交通灯
        activate_tl, light_trigger_location = \
            self._get_active_light(tl_list, vehicle_location, vehicle_waypoint)

        objects.update({'traffic_lights': []})

        if activate_tl is not None:
            traffic_light = TrafficLight(activate_tl,
                                         light_trigger_location,
                                         activate_tl.get_state())
            objects['traffic_lights'].append(traffic_light)
        return objects

    def _get_active_light(self, tl_list, vehicle_location, vehicle_waypoint):
        for tl in tl_list:
            object_location = \
                TrafficLight.get_trafficlight_trigger_location(tl)
            object_waypoint = self._map.get_waypoint(object_location)

            if object_waypoint.road_id != vehicle_waypoint.road_id:
                continue

            ve_dir = vehicle_waypoint.transform.get_forward_vector()
            wp_dir = object_waypoint.transform.get_forward_vector()
            dot_ve_wp = ve_dir.x * wp_dir.x +\
                        ve_dir.y * wp_dir.y + \
                        ve_dir.z * wp_dir.z

            if dot_ve_wp < 0:
                continue
            while not object_waypoint.is_intersection:
                next_waypoint = object_waypoint.next(0.5)[0]
                if next_waypoint and not next_waypoint.is_intersection:
                    object_waypoint = next_waypoint
                else:
                    break

            return tl, object_waypoint.transform.location

        return None, None

    def destroy(self):
        """
        销毁所有传感器
        """
        if self.rgb_camera:
            for rgb_camera in self.rgb_camera:
                rgb_camera.sensor.destroy()

        if self.lidar:
            self.lidar.sensor.destroy()

        if self.camera_visualize:
            cv2.destroyAllWindows()

        if self.lidar_visualize:
            self.o3d_vis.destroy_window()

        if self.data_dump:
            self.semantic_lidar.sensor.destroy()
