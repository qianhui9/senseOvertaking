# -*- coding: utf-8 -*-
"""
Utilize scenario manager to manage CARLA simulation construction. This script
is used for carla simulation only, and if you want to manage the Co-simulation,
please use cosim_api.py.   # 这个脚本用于管理CARLA仿真构建，仅用于CARLA仿真，如需管理协同仿真请使用cosim_api.py
"""

import math
import random
import sys
import json
from random import shuffle
from omegaconf import OmegaConf
from omegaconf.listconfig import ListConfig

import carla
import numpy as np

from opencda.core.common.vehicle_manager import VehicleManager
from opencda.core.application.platooning.platooning_manager import \
    PlatooningManager
from opencda.core.common.rsu_manager import RSUManager
from opencda.core.common.cav_world import CavWorld
from opencda.scenario_testing.utils.customized_map_api import \
    load_customized_world, bcolors

# 定义车辆蓝图过滤函数，接收蓝图库和CARLA版本参数(默认0.9.11)
def car_blueprint_filter(blueprint_library, carla_version='0.9.11'):
    """
    Exclude the uncommon vehicles from the default CARLA blueprint library
    (i.e., isetta, carlacola, cybertruck, t2).  # 从默认CARLA蓝图库中排除不常见车辆

    Parameters
    ----------
    blueprint_library : carla.blueprint_library
        The blueprint library that contains all models.

    carla_version : str
        CARLA simulator version, currently support 0.9.11 and 0.9.12. We need
        this as since CARLA 0.9.12 the blueprint name has been changed a lot.

    Returns
    -------
    blueprints : list
        The list of suitable blueprints for vehicles.
    """

    if carla_version == '0.9.11':  # 如果是0.9.11版本，创建旧版车辆蓝图列表(包含奥迪A2等常见车型)
        print('old version')
        blueprints = [
            blueprint_library.find('vehicle.audi.a2'),
            blueprint_library.find('vehicle.audi.tt'),
            blueprint_library.find('vehicle.dodge_charger.police'),
            blueprint_library.find('vehicle.jeep.wrangler_rubicon'),
            blueprint_library.find('vehicle.chevrolet.impala'),
            blueprint_library.find('vehicle.mini.cooperst'),
            blueprint_library.find('vehicle.audi.etron'),
            blueprint_library.find('vehicle.mercedes-benz.coupe'),
            blueprint_library.find('vehicle.bmw.grandtourer'),
            blueprint_library.find('vehicle.toyota.prius'),
            blueprint_library.find('vehicle.citroen.c3'),
            blueprint_library.find('vehicle.mustang.mustang'),
            blueprint_library.find('vehicle.tesla.model3'),
            blueprint_library.find('vehicle.lincoln.mkz2017'),
            blueprint_library.find('vehicle.seat.leon'),
            blueprint_library.find('vehicle.nissan.patrol'),
            blueprint_library.find('vehicle.nissan.micra'),
        ]

    else:  # 其他版本(如0.9.12)创建新版车辆蓝图列表(车型名称有变化)
        blueprints = [
            blueprint_library.find('vehicle.audi.a2'),
            blueprint_library.find('vehicle.audi.tt'),
            blueprint_library.find('vehicle.dodge.charger_police'),
            blueprint_library.find('vehicle.dodge.charger_police_2020'),
            blueprint_library.find('vehicle.dodge.charger_2020'),
            blueprint_library.find('vehicle.jeep.wrangler_rubicon'),
            blueprint_library.find('vehicle.chevrolet.impala'),
            blueprint_library.find('vehicle.mini.cooper_s'),
            blueprint_library.find('vehicle.audi.etron'),
            blueprint_library.find('vehicle.mercedes.coupe'),
            blueprint_library.find('vehicle.mercedes.coupe_2020'),
            blueprint_library.find('vehicle.bmw.grandtourer'),
            blueprint_library.find('vehicle.toyota.prius'),
            blueprint_library.find('vehicle.citroen.c3'),
            blueprint_library.find('vehicle.ford.mustang'),
            blueprint_library.find('vehicle.tesla.model3'),
            blueprint_library.find('vehicle.lincoln.mkz_2017'),
            blueprint_library.find('vehicle.lincoln.mkz_2020'),
            blueprint_library.find('vehicle.seat.leon'),
            blueprint_library.find('vehicle.nissan.patrol'),
            blueprint_library.find('vehicle.nissan.micra'),
        ]

    return blueprints


# 定义多类别车辆蓝图过滤函数，接收类别标签、蓝图库和蓝图元数据
def multi_class_vehicle_blueprint_filter(label, blueprint_library, bp_meta):
    """
    Get a list of blueprints that have the class equals the specified label. #  获取指定类别的所有蓝图

    Parameters
    ----------
    label : str
        Specified blueprint.

    blueprint_library : carla.blueprint_library
        The blueprint library that contains all models.

    bp_meta : dict
        Dictionary of {blueprint name: blueprint class}.

    Returns
    -------
    blueprints : list
        List of blueprints that have the class equals the specified label.

    """
    # 用列表推导式找出所有类别匹配的蓝图
    blueprints = [
        blueprint_library.find(k)
        for k, v in bp_meta.items() if v["class"] == label
    ]
    return blueprints


# 定义场景管理器类，控制仿真构建、背景交通生成和CAV生成
class ScenarioManager:
    """
    The manager that controls simulation construction, backgound traffic
    generation and CAVs spawning.

    Parameters
    ----------
    scenario_params : dict
        The dictionary contains all simulation configurations.

    carla_version : str
        CARLA simulator version, it currently supports 0.9.11 and 0.9.12

    xodr_path : str
        The xodr file to the customized map, default: None.

    town : str
        Town name if not using customized map, eg. 'Town06'.

    apply_ml : bool
        Whether need to load dl/ml model(pytorch required) in this simulation.

    Attributes
    ----------
    client : carla.client
        The client that connects to carla server.

    world : carla.world
        Carla simulation server.

    origin_settings : dict
        The origin setting of the simulation server.

    cav_world : opencda object
        CAV World that contains the information of all CAVs.

    carla_map : carla.map
        Car;a HD Map.

    """

    # 初始化方法，接收场景参数、ML标志、CARLA版本等参数
    def __init__(self, scenario_params,
                 apply_ml,
                 carla_version,
                 xodr_path=None,
                 town=None,
                 cav_world=None):
        # 保存场景参数和CARLA版本
        self.scenario_params = scenario_params
        self.carla_version = carla_version

        simulation_config = scenario_params['world']

        # set random seed if stated,如果配置中有随机种子，设置随机种子
        if 'seed' in simulation_config:
            np.random.seed(simulation_config['seed'])
            random.seed(simulation_config['seed'])

        # 创建CARLA客户端连接，设置超时10秒
        self.client = \
            carla.Client('localhost', simulation_config['client_port'])
        self.client.set_timeout(10.0)

        # 如果有xodr路径加载自定义地图，否则加载指定城镇
        if xodr_path:
            self.world = load_customized_world(xodr_path, self.client)
        elif town:
            try:
                self.world = self.client.load_world(town)
            except RuntimeError:
                print(
                    f"{bcolors.FAIL} %s is not found in your CARLA repo! "
                    f"Please download all town maps to your CARLA "
                    f"repo!{bcolors.ENDC}" % town)
        else:
            self.world = self.client.get_world()

        if not self.world:
            sys.exit('World loading failed')

        self.origin_settings = self.world.get_settings()
        # 获取世界设置
        new_settings = self.world.get_settings()

        if simulation_config['sync_mode']:
            new_settings.synchronous_mode = True  #  启用同步模式
            #   设置固定时间步长
            new_settings.fixed_delta_seconds = \
                simulation_config['fixed_delta_seconds']
        else:
            sys.exit(
                'ERROR: Current version only supports sync simulation mode')

        self.world.apply_settings(new_settings)

        # set weather
        weather = self.set_weather(simulation_config['weather'])
        self.world.set_weather(weather)

        # Define probabilities for each type of blueprint
        self.use_multi_class_bp = scenario_params["blueprint"][
            'use_multi_class_bp'] if 'blueprint' in scenario_params else False
        if self.use_multi_class_bp:
            # bbx/blueprint meta
            with open(scenario_params['blueprint']['bp_meta_path']) as f:
                self.bp_meta = json.load(f)
            self.bp_class_sample_prob = scenario_params['blueprint'][
                'bp_class_sample_prob']

            # normalize probability
            self.bp_class_sample_prob = {
                k: v / sum(self.bp_class_sample_prob.values()) for k, v in
                self.bp_class_sample_prob.items()}

        # 保存CAV世界和CARLA地图
        self.cav_world = cav_world
        self.carla_map = self.world.get_map()
        self.apply_ml = apply_ml

    @staticmethod
    # 设置天气参数
    def set_weather(weather_settings):
        """
        Set CARLA weather params.

        Parameters
        ----------
        weather_settings : dict
            The dictionary that contains all parameters of weather.

        Returns
        -------
        The CARLA weather setting.
        """
        weather = carla.WeatherParameters(
            sun_altitude_angle=weather_settings['sun_altitude_angle'],
            cloudiness=weather_settings['cloudiness'],
            precipitation=weather_settings['precipitation'],
            precipitation_deposits=weather_settings['precipitation_deposits'],
            wind_intensity=weather_settings['wind_intensity'],
            fog_density=weather_settings['fog_density'],
            fog_distance=weather_settings['fog_distance'],
            fog_falloff=weather_settings['fog_falloff'],
            wetness=weather_settings['wetness']
        )
        return weather

    # 定义创建单个CAV车辆管理器的方法,application: 应用目的列表，如['single']或['platoon'];map_helper: 帮助在特定地图位置生成车辆的函数
    def create_vehicle_manager(self, application,
                               map_helper=None,
                               data_dump=True):
        """
        Create a list of single CAVs.

        Parameters
        ----------
        application : list
            The application purpose, a list, eg. ['single'], ['platoon'].

        map_helper : function
            A function to help spawn vehicle on a specific position in
            a specific map.

        data_dump : bool
            Whether to dump sensor data.

        Returns
        -------
        single_cav_list : list
            A list contains all single CAVs' vehicle manager.
        """
        print('Creating single CAVs.')
        # By default, we use lincoln as our cav model.
        default_model = 'vehicle.ford.mustang' \
            if self.carla_version == '0.9.11' else 'vehicle.ford.mustang'

        # 从蓝图库中获取默认车辆模型
        cav_vehicle_bp = \
            self.world.get_blueprint_library().find(default_model)
        # 初始化空列表存储车辆管理器
        single_cav_list = []

        # 遍历场景参数中的单个CAV配置列表
        for i, cav_config in enumerate(
                self.scenario_params['scenario']['single_cav_list']):
            # in case the cav wants to join a platoon later
            # it will be empty dictionary for single cav application
            # 创建车队基础配置;合并车辆基础配置、车队配置和当前CAV配置
            platoon_base = OmegaConf.create({'platoon': self.scenario_params.get('platoon_base',{})})
            cav_config = OmegaConf.merge(self.scenario_params['vehicle_base'],
                                         platoon_base,
                                         cav_config)
            # if the spawn position is a single scalar, we need to use map
            # helper to transfer to spawn transform
            if 'spawn_special' not in cav_config: # 如果没有特殊生成位置，直接从配置创建变换对象
                # 创建车辆生成位置和旋转的变换对象
                spawn_transform = carla.Transform(
                    carla.Location(
                        x=cav_config['spawn_position'][0],
                        y=cav_config['spawn_position'][1],
                        z=cav_config['spawn_position'][2]),
                    carla.Rotation(
                        pitch=cav_config['spawn_position'][5],
                        yaw=cav_config['spawn_position'][4],
                        roll=cav_config['spawn_position'][3]))
            else:  # 否则使用map_helper函数获取生成位置
                spawn_transform = map_helper(self.carla_version,
                                             *cav_config['spawn_special'])

            cav_vehicle_bp.set_attribute('color', '255, 0, 0')   # 设置车辆颜色为红色
            # 在世界中生成车辆
            vehicle = self.world.spawn_actor(cav_vehicle_bp, spawn_transform)

            # create vehicle manager for each cav
            vehicle_manager = VehicleManager(
                vehicle, cav_config, application,
                self.carla_map, self.cav_world,
                current_time=self.scenario_params['current_time'],
                data_dumping=data_dump)

            self.world.tick()

            #  设置V2X管理器无车队
            vehicle_manager.v2x_manager.set_platoon(None)

            # 设置目的地位置
            destination = carla.Location(x=cav_config['destination'][0],
                                         y=cav_config['destination'][1],
                                         z=cav_config['destination'][2])
            # 更新车辆信息
            vehicle_manager.update_info()

            # 设置干净的目的地路线 注意：测试自己环境时，下面的代码要注释掉，测试已有的代码时，下面的代码不可注释
            # vehicle_manager.set_destination(
            #     vehicle_manager.vehicle.get_location(),
            #     destination,
            #     clean=True)

            # 将车辆管理器添加到列表
            single_cav_list.append(vehicle_manager)

        # 返回车辆管理器列表
        return single_cav_list

    # 为ScenarioRunner创建的车辆创建管理器的方法,参数：ScenarioRunner创建的车辆对象
    def create_vehicle_manager_from_scenario_runner(self, vehicle):
        """
        Create a single CAV with a loaded ego vehicle from SR.
        Different from the create_vehicle_manager API creating Carla vehicle from scratch,
        SR creates on its own only supports 'single' vehicle.

        Parameters
        ----------
        vehicle:
            The Carla ego vehicle created by ScenarioRunner.

        Returns
        -------
        single_cav_list : list
            A list contains the singla CAV derived from the ego vehicle.
        """
        # 获取单个CAV参数
        single_cav_params = self.scenario_params['scenario']['single_cav_list']
        # 检查只支持一个ego车辆
        if len(single_cav_params) != 1:
            raise ValueError('Only support one ego vehicle for ScenarioRunner')

        cav_config = single_cav_params[0]
        # 创建并合并配置
        platoon_base = OmegaConf.create(
            {'platoon': self.scenario_params.get('platoon_base', {})})
        cav_config = OmegaConf.merge(self.scenario_params['vehicle_base'],
                                     platoon_base,
                                     cav_config)
        # 创建车辆管理器
        vehicle_manager = VehicleManager(
            vehicle, cav_config, ['single'], self.carla_map, self.cav_world)

        self.world.tick()  # 推进仿真

        vehicle_manager.v2x_manager.set_platoon(None)   #  设置无车队

        # 设置目的地和路线
        destination = carla.Location(x=cav_config['destination'][0],
                                     y=cav_config['destination'][1],
                                     z=cav_config['destination'][2])
        vehicle_manager.update_info()
        vehicle_manager.set_destination(
            vehicle_manager.vehicle.get_location(),
            destination,
            clean=True)

        # 返回包含单个车辆管理器的列表
        return [vehicle_manager]

    # 创建车队管理器的方法；参数：map_helper: 位置生成辅助函数；data_dump: 是否转储数据
    def create_platoon_manager(self, map_helper=None, data_dump=False):
        """
        Create a list of platoons.

        Parameters
        ----------
        map_helper : function
            A function to help spawn vehicle on a specific position in a
            specific map.

        data_dump : bool
            Whether to dump sensor data.

        Returns
        -------
        single_cav_list : list
            A list contains all single CAVs' vehicle manager.
        """
        print('Creating platoons/')
        # 初始化变量
        platoon_list = []
        self.cav_world = CavWorld(self.apply_ml)

        # we use lincoln as default choice since our UCLA mobility lab use the
        # same car
        # 设置默认车辆模型并获取蓝图
        default_model = 'vehicle.lincoln.mkz2017' \
            if self.carla_version == '0.9.11' else 'vehicle.lincoln.mkz_2017'

        cav_vehicle_bp = \
            self.world.get_blueprint_library().find(default_model)

        # create platoons 遍历每个车队配置
        for i, platoon in enumerate(
                self.scenario_params['scenario']['platoon_list']):
            # 合并配置
            platoon = OmegaConf.merge(self.scenario_params['platoon_base'],
                                      platoon)
            # 创建车队管理器实例
            platoon_manager = PlatooningManager(platoon, self.cav_world)
            for j, cav in enumerate(platoon['members']):
                platton_base = OmegaConf.create({'platoon': platoon})
                cav = OmegaConf.merge(self.scenario_params['vehicle_base'],
                                      platton_base,
                                      cav
                                      )
                if 'spawn_special' not in cav:
                    spawn_transform = carla.Transform(
                        carla.Location(
                            x=cav['spawn_position'][0],
                            y=cav['spawn_position'][1],
                            z=cav['spawn_position'][2]),
                        carla.Rotation(
                            pitch=cav['spawn_position'][5],
                            yaw=cav['spawn_position'][4],
                            roll=cav['spawn_position'][3]))
                else:
                    spawn_transform = map_helper(self.carla_version,
                                                 *cav['spawn_special'])

                cav_vehicle_bp.set_attribute('color', '0, 0, 255')
                vehicle = self.world.spawn_actor(cav_vehicle_bp,
                                                 spawn_transform)

                # create vehicle manager for each cav
                vehicle_manager = VehicleManager(
                    vehicle, cav, ['platooning'],
                    self.carla_map, self.cav_world,
                    current_time=self.scenario_params['current_time'],
                    data_dumping=data_dump)

                # add the vehicle manager to platoon 设置车队领头车和成员车
                if j == 0:
                    platoon_manager.set_lead(vehicle_manager)
                else:
                    platoon_manager.add_member(vehicle_manager, leader=False)

            self.world.tick()
            destination = carla.Location(x=platoon['destination'][0],
                                         y=platoon['destination'][1],
                                         z=platoon['destination'][2])

            # 设置目的地
            platoon_manager.set_destination(destination)
            # 更新车队成员顺序
            platoon_manager.update_member_order()
            # 添加到车队列表
            platoon_list.append(platoon_manager)

        return platoon_list   # 返回车队管理器列表

    # 创建RSU(路侧单元)管理器的方法
    def create_rsu_manager(self, data_dump):
        """
        Create a list of RSU.

        Parameters
        ----------
        data_dump : bool
            Whether to dump sensor data.

        Returns
        -------
        rsu_list : list
            A list contains all rsu managers..
        """
        print('Creating RSU.')
        rsu_list = []
        # 遍历RSU配置
        for i, rsu_config in enumerate(
                self.scenario_params['scenario']['rsu_list']):
            # 合并配置
            rsu_config = OmegaConf.merge(self.scenario_params['rsu_base'],
                                         rsu_config)
            # 创建RSU管理器并添加到列表
            rsu_manager = RSUManager(self.world, rsu_config,
                                     self.carla_map,
                                     self.cav_world,
                                     self.scenario_params['current_time'],
                                     data_dump)

            rsu_list.append(rsu_manager)

        return rsu_list   # 返回RSU管理器列表

    # 按列表生成交通车辆的方法，tm: CARLA交通管理器实例；traffic_config: 交通配置字典；bg_list: 背景车辆列表；
    def spawn_vehicles_by_list(self, tm, traffic_config, bg_list):
        """
        Spawn the traffic vehicles by the given list.

        Parameters
        ----------
        tm : carla.TrafficManager
            Traffic manager.

        traffic_config : dict
            Background traffic configuration.

        bg_list : list
            The list contains all background traffic.

        Returns
        -------
        bg_list : list
            Update traffic list.
        """

        # 获取蓝图库
        blueprint_library = self.world.get_blueprint_library()
        # 如果不使用多类别蓝图，使用基础过滤方法
        if not self.use_multi_class_bp:
            ego_vehicle_random_list = car_blueprint_filter(blueprint_library,
                                                           self.carla_version)
        else:  # 否则准备多类别蓝图的标签和概率列表
            label_list = list(self.bp_class_sample_prob.keys())
            prob = [self.bp_class_sample_prob[itm] for itm in label_list]

        # if not random select, we always choose lincoln.mkz with green color
        color = '0, 255, 0'  # 设置默认颜色为绿色
        default_model = 'vehicle.lincoln.mkz2017' \
            if self.carla_version == '0.9.11' else 'vehicle.lincoln.mkz_2017'
        # 获取默认车辆蓝图
        ego_vehicle_bp = blueprint_library.find(default_model)

        # 遍历车辆配置列表
        for i, vehicle_config in enumerate(traffic_config['vehicle_list']):
            # 创建车辆生成位置和旋转的变换对象
            spawn_transform = carla.Transform(
                carla.Location(
                    x=vehicle_config['spawn_position'][0],
                    y=vehicle_config['spawn_position'][1],
                    z=vehicle_config['spawn_position'][2]),
                carla.Rotation(
                    pitch=vehicle_config['spawn_position'][5],
                    yaw=vehicle_config['spawn_position'][4],
                    roll=vehicle_config['spawn_position'][3]))

            if not traffic_config['random']:   # 如果不随机选择，使用默认颜色
                ego_vehicle_bp.set_attribute('color', color)

            else:  # 否则随机选择车辆蓝图和颜色
                # sample a bp from various classes
                if self.use_multi_class_bp:
                    label = np.random.choice(label_list, p=prob)
                    # Given the label (class), find all associated blueprints in CARLA
                    ego_vehicle_random_list = multi_class_vehicle_blueprint_filter(
                        label, blueprint_library, self.bp_meta)
                ego_vehicle_bp = random.choice(ego_vehicle_random_list)

                if ego_vehicle_bp.has_attribute("color"):
                    color = random.choice(
                        ego_vehicle_bp.get_attribute(
                            'color').recommended_values)
                    ego_vehicle_bp.set_attribute('color', color)

            # 在世界中生成车辆
            # 优先用 try_spawn_actor，失败时它会返回 None 而不是抛异常
            vehicle = self.world.try_spawn_actor(ego_vehicle_bp, spawn_transform)
            if vehicle is None:
                # 跳过这个有碰撞风险的生成点
                print(f"[WARN] skip spawn at {spawn_transform.location} due to collision")
                continue
            # 设置车辆自动驾驶模式
            vehicle.set_autopilot(False)  # 禁用TM
            vehicle.set_target_velocity(carla.Vector3D(x=10.0, y=0.0, z=0.0))  # 固定速度前进

            if 'vehicle_speed_perc' in vehicle_config:  # 如果有速度百分比配置，设置车辆速度
                tm.vehicle_percentage_speed_difference(
                    vehicle, vehicle_config['vehicle_speed_perc'])
            # 设置车辆自动变道行为
            tm.auto_lane_change(vehicle, traffic_config['auto_lane_change'])

            # 将车辆添加到背景列表
            bg_list.append(vehicle)

        # 返回更新后的背景车辆列表
        return bg_list

    # 按范围生成交通车辆的方法
    def spawn_vehicle_by_range(self, tm, traffic_config, bg_list):
        """
        Spawn the traffic vehicles by the given range.

        Parameters
        ----------
        tm : carla.TrafficManager
            Traffic manager.

        traffic_config : dict
            Background traffic configuration.

        bg_list : list
            The list contains all background traffic.

        Returns
        -------
        bg_list : list
            Update traffic list.
        """
        blueprint_library = self.world.get_blueprint_library()
        if not self.use_multi_class_bp:
            ego_vehicle_random_list = car_blueprint_filter(blueprint_library,
                                                           self.carla_version)
        else:
            label_list = list(self.bp_class_sample_prob.keys())
            prob = [self.bp_class_sample_prob[itm] for itm in label_list]

        # if not random select, we always choose lincoln.mkz with green color
        color = '0, 255, 0'
        default_model = 'vehicle.lincoln.mkz2017' \
            if self.carla_version == '0.9.11' else 'vehicle.lincoln.mkz_2017'
        ego_vehicle_bp = blueprint_library.find(default_model)

        spawn_ranges = traffic_config['range']   # 获取生成范围配置
        # 初始化生成位置集合和计数器
        spawn_set = set()
        spawn_num = 0

        # 计算每个范围内的生成位置
        for spawn_range in spawn_ranges:
            spawn_num += spawn_range[6]
            x_min, x_max, y_min, y_max = \
                math.floor(spawn_range[0]), math.ceil(spawn_range[1]), \
                math.floor(spawn_range[2]), math.ceil(spawn_range[3])

            # 通过地图获取有效的路径点位置
            for x in range(x_min, x_max, int(spawn_range[4])):
                for y in range(y_min, y_max, int(spawn_range[5])):
                    location = carla.Location(x=x, y=y, z=0.3)
                    way_point = self.carla_map.get_waypoint(location).transform

                    spawn_set.add((way_point.location.x,
                                   way_point.location.y,
                                   way_point.location.z,
                                   way_point.rotation.roll,
                                   way_point.rotation.yaw,
                                   way_point.rotation.pitch))
        # 打乱生成位置列表
        count = 0
        spawn_list = list(spawn_set)
        shuffle(spawn_list)

        # 循环生成车辆直到达到数量或位置用完，获取下一个生成位置
        while count < spawn_num:
            if len(spawn_list) == 0:
                break

            coordinates = spawn_list[0]
            spawn_list.pop(0)

            spawn_transform = carla.Transform(carla.Location(x=coordinates[0],
                                                             y=coordinates[1],
                                                             z=coordinates[
                2] + 0.3),
                carla.Rotation(
                roll=coordinates[3],
                yaw=coordinates[4],
                pitch=coordinates[5]))
            # 处理车辆随机选择逻辑
            if not traffic_config['random']:
                ego_vehicle_bp.set_attribute('color', color)

            else:
                # sample a bp from various classes
                if self.use_multi_class_bp:
                    label = np.random.choice(label_list, p=prob)
                    # Given the label (class), find all associated blueprints in CARLA
                    ego_vehicle_random_list = multi_class_vehicle_blueprint_filter(
                        label, blueprint_library, self.bp_meta)
                ego_vehicle_bp = random.choice(ego_vehicle_random_list)
                if ego_vehicle_bp.has_attribute("color"):
                    color = random.choice(
                        ego_vehicle_bp.get_attribute(
                            'color').recommended_values)
                    ego_vehicle_bp.set_attribute('color', color)

            # 尝试生成车辆，失败则跳过
            vehicle = \
                self.world.try_spawn_actor(ego_vehicle_bp, spawn_transform)

            if not vehicle:
                continue

            # 设置车辆自动驾驶和行为参数
            vehicle.set_autopilot(True, 8000)
            tm.auto_lane_change(vehicle, traffic_config['auto_lane_change'])

            if 'ignore_lights_percentage' in traffic_config:
                tm.ignore_lights_percentage(vehicle,
                                            traffic_config[
                                                'ignore_lights_percentage'])

            # each vehicle have slight different speed
            tm.vehicle_percentage_speed_difference(
                vehicle,
                traffic_config['global_speed_perc'] + random.randint(-30, 30))

            # 添加车辆到列表并计数
            bg_list.append(vehicle)
            count += 1

        # 返回更新后的背景车辆列表
        return bg_list

    # 定义创建交通流的主方法
    def create_traffic_carla(self):
        """
        Create traffic flow.

        Returns
        -------
        tm : carla.traffic_manager
            Carla traffic manager.

        bg_list : list
            The list that contains all the background traffic vehicles.
        """
        print('Spawning CARLA traffic flow.')
        # 获取交通配置和交通管理器
        traffic_config = self.scenario_params['carla_traffic_manager']
        tm = self.client.get_trafficmanager()

        # 设置交通管理器全局参数
        tm.set_global_distance_to_leading_vehicle(
            traffic_config['global_distance'])
        tm.set_synchronous_mode(traffic_config['sync_mode'])
        tm.set_osm_mode(traffic_config['set_osm_mode'])
        tm.global_percentage_speed_difference(
            traffic_config['global_speed_perc'])

        # 根据配置类型选择生成方式
        bg_list = []
        if isinstance(traffic_config['vehicle_list'], list) or \
                isinstance(traffic_config['vehicle_list'], ListConfig):
            bg_list = self.spawn_vehicles_by_list(tm,
                                                  traffic_config,
                                                  bg_list)
        else:
            bg_list = self.spawn_vehicle_by_range(tm, traffic_config, bg_list)

        print('CARLA traffic flow generated.')
        return tm, bg_list

    # 推进仿真时间步的方法
    def tick(self):
        """
        Tick the server.
        """
        self.world.tick()

    # 销毁所有场景参与者的方法
    def destroyActors(self):
        """
        Destroy all actors in the world.
        """

        actor_list = self.world.get_actors()
        for actor in actor_list:
            actor.destroy()

    # 关闭仿真并恢复原始设置的方法
    def close(self):
        """
        Simulation close.
        """
        # restore to origin setting
        self.world.apply_settings(self.origin_settings)