# -*- coding: utf-8 -*-
"""
Scenario testing: merging vehicle joining a platoon in the
customized 2-lane freeway simplified map sorely with carla
"""

import os

import carla

import opencda.scenario_testing.utils.sim_api as sim_api
from opencda.core.common.cav_world import CavWorld
from opencda.scenario_testing.evaluations.evaluate_manager import \
    EvaluationManager
from opencda.scenario_testing.utils.yaml_utils import add_current_time, save_yaml


# 定义主函数，运行场景测试，接收选项参数和场景参数
def run_scenario(opt, scenario_params):
    try:
        # 为场景参数添加当前时间戳
        scenario_params = add_current_time(scenario_params)

        # create CAV world，参数决定是否应用机器学习
        cav_world = CavWorld(opt.apply_ml)

        # create scenario manager，传入场景参数、ML选项、版本、城镇名称和CAV世界
        scenario_manager = sim_api.ScenarioManager(scenario_params,
                                                   opt.apply_ml,
                                                   opt.version,
                                                   town='Town04',
                                                   cav_world=cav_world)

        if opt.record:  # 如果设置了记录选项，启动Carla记录器记录仿真过程
            scenario_manager.client. \
                start_recorder("single_town06_carla.log", True)

        # 创建单个CAV车辆管理器列表，设置应用类型为'single'并启用数据转储
        single_cav_list = \
            scenario_manager.create_vehicle_manager(application=['single'],
                                                    data_dump=True)
        # 创建RSU(路侧单元)管理器列表，并启用数据转储
        # rsu_list = \
        #     scenario_manager.create_rsu_manager(data_dump=True)

        # create background traffic in carla，返回交通管理器和背景车辆列表
        # 注释下面两行代码后就没有背景车辆生成了，否则会生成背景较多的车辆
        traffic_manager, bg_veh_list = \
            scenario_manager.create_traffic_carla()

        # create evaluation manager，传入CAV世界、脚本名称和当前时间
        # eval_manager = \
        #     EvaluationManager(scenario_manager.cav_world,
        #                       script_name='coop_town06',
        #                       current_time=scenario_params['current_time'])

        # 获取Carla世界的观察者(摄像机)
        spectator = scenario_manager.world.get_spectator()

        # save the data collection protocol to the folder
        current_path = os.path.dirname(os.path.realpath(__file__))   #  获取当前脚本的绝对路径
        # 构建保存数据协议的YAML文件路径
        save_yaml_name = os.path.join(current_path,
                                      '../../data_dumping',
                                      scenario_params['current_time'],
                                      'data_protocol.yaml')
        save_yaml(scenario_params, save_yaml_name)   #  将场景参数保存为YAML文件

        while True:
            scenario_manager.tick()   #   推进仿真时间步
            #  获取第一个CAV的变换信息(位置和旋转)
            transform = single_cav_list[0].vehicle.get_transform()
            # 设置观察者位置在车辆正上方70米处，俯视视角(pitch=-90)
            spectator.set_transform(carla.Transform(
                transform.location +
                carla.Location(
                    z=70),
                carla.Rotation(
                    pitch=-
                    90)))

            # 遍历所有CAV，更新信息，计算控制命令并应用到车辆
            for i, single_cav in enumerate(single_cav_list):
                single_cav.update_info()
                control = single_cav.run_step()
                single_cav.vehicle.apply_control(control)


            # 遍历所有RSU，更新信息并运行步骤
            # for rsu in rsu_list:
            #     rsu.update_info()
            #     rsu.run_step()

    finally:
        # 执行场景评估
        # eval_manager.evaluate()

        # 如果启用了记录，停止记录器
        if opt.record:
            scenario_manager.client.stop_recorder()

        # 关闭场景管理器
        scenario_manager.close()

        for v in single_cav_list:
            v.destroy()
        # for r in rsu_list:
        #     r.destroy()
        for v in bg_veh_list:
            v.destroy()
