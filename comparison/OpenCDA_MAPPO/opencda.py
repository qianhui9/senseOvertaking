# -*- coding: utf-8 -*-
"""
Script to run different scenarios.
"""

import argparse

import importlib
import os
import sys

from opencda.scenario_testing.single_2lanefree_carla_MyTest import runMARL, arg_parse  # 训练模型
# from opencda.scenario_testing.inference import runMARL
from opencda.version import __version__

# CarlaUE4.exe -carla-rpc-port=2000

if __name__ == '__main__':
    try:
        runMARL()
    except KeyboardInterrupt:
        print(' - Exited by user.')
