# -*- coding: utf-8 -*-


import argparse
import math
import os
import sys
import time

import numpy as np
from tqdm import tqdm

import torch
import open3d as o3d
from torch.utils.data import DataLoader

import opencood.hypes_yaml.yaml_utils as yaml_utils
from AB3DMOT.AB3DMOT_libs.io import load_detection, get_saving_dir, get_frame_det, save_affinity, save_results
from AB3DMOT.AB3DMOT_libs.utils import initialize, get_subfolder_seq, Config
from AB3DMOT.Xinshuo_PyToolbox.xinshuo_io import mkdir_if_missing
from AB3DMOT.Xinshuo_PyToolbox.xinshuo_miscellaneous import print_log, get_timestring
from AB3DMOT.main import parse_args
from opencood.tools import train_utils, inference_utils
from opencood.data_utils.datasets import build_dataset
from opencood.utils import eval_utils
from opencood.visualization import vis_utils
import matplotlib.pyplot as plt


def test_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Continued training path')
    parser.add_argument('--fusion_method', required=True, type=str,
                        default='intermediate',
                        help='late, early or intermediate')
    parser.add_argument('--show_vis', action='store_true',
                        help='whether to show image visualization result')
    parser.add_argument('--show_sequence', action='store_true',
                        help='whether to show video visualization result.'
                             'it can note be set true with show_vis together ')
    parser.add_argument('--save_vis', action='store_true',
                        help='whether to save visualization result')
    parser.add_argument('--save_npy', action='store_true',
                        help='whether to save prediction and gt result'
                             'in npy_test file')
    parser.add_argument('--global_sort_detections', action='store_true',
                        help='whether to globally sort detections by confidence score.'
                             'If set to True, it is the mainstream AP computing method,'
                             'but would increase the tolerance for FP (False Positives).')
    opt = parser.parse_args()
    return opt

def convert_detections(pred_box_tensor, pred_score):
    detections = []
    for i in range(pred_box_tensor.shape[0]):
        box = pred_box_tensor[i]
        # 计算中心点坐标
        center = torch.mean(box[:4], dim=0)
        x, y, z = center.tolist()

        # 计算长宽高
        h = torch.max(box[:, 2]) - torch.min(box[:, 2])
        w = torch.max(box[:, 1]) - torch.min(box[:, 1])
        l = torch.max(box[:, 0]) - torch.min(box[:, 0])

        # 计算θ（最长边法）：自动驾驶中通常使用右手坐标系（X前向，Y左向，Z上向），θ为绕Z轴的偏航角
        bottom_points = box[:4, :2]
        edges = [
            (bottom_points[0], bottom_points[1]),
            (bottom_points[1], bottom_points[2]),
            (bottom_points[2], bottom_points[3]),
            (bottom_points[3], bottom_points[0])
        ]
        max_length_sq = -1
        max_edge = edges[0]
        for edge in edges:
            p1, p2 = edge
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            length_sq = dx ** 2 + dy ** 2
            if length_sq > max_length_sq:
                max_length_sq = length_sq
                max_edge = (p1, p2)
        dx = max_edge[1][0] - max_edge[0][0]
        dy = max_edge[1][1] - max_edge[0][1]
        theta = math.atan2(dy, dx)

        score = pred_score[i].item()

        detection = [h.item(), w.item(), l.item(), x, y, z, theta, score]
        detections.append(detection)

    detections = np.array(detections)
    return detections


def main_per_cat(cfg, cat, ID_start):
    # ========== 参数定义 ==========
    # 模型路径（修改为你的模型路径）
    model_dir = 'voxelnet_attentive_fusion/voxelnet_attentive_fusion_compression'
    # model_dir = "second_attentive_fusion"

    # 融合方法，可选 'late', 'early', 'intermediate'
    fusion_method = 'intermediate'

    show_sequence = False  # 是否以视频流形式显示结果（不能与show_vis同时为True）

    global_sort_detections = False  # 是否全局按置信度排序检测结果

    assert fusion_method in ['late', 'early', 'intermediate']


    # 从模型目录加载配置文件
    hypes_path = os.path.join(model_dir, 'config.yaml')
    hypes = yaml_utils.load_yaml(hypes_path, None)

    print('Dataset Building')
    opencood_dataset = build_dataset(hypes, visualize=True, train=False)
    print(f"{len(opencood_dataset)} samples found.")
    data_loader = DataLoader(opencood_dataset,
                             batch_size=1,
                             # num_workers=16,
                             num_workers = 0,
                             collate_fn=opencood_dataset.collate_batch_test,
                             shuffle=False,
                             pin_memory=False,
                             drop_last=False,
                             )

    print('Creating Model')
    model = train_utils.create_model(hypes)
    if torch.cuda.is_available():
        model.cuda()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('Loading Model from checkpoint')
    _, model = train_utils.load_saved_model(model_dir, model)
    model.eval()

    result_stat = {0.3: {'tp': [], 'fp': [], 'gt': 0, 'score': []},                
                   0.5: {'tp': [], 'fp': [], 'gt': 0, 'score': []},                
                   0.7: {'tp': [], 'fp': [], 'gt': 0, 'score': []}}

    if show_sequence:
        vis = o3d.visualization.Visualizer()
        vis.create_window()

        vis.get_render_option().background_color = [0.05, 0.05, 0.05]
        vis.get_render_option().point_size = 1.0
        vis.get_render_option().show_coordinate_frame = True

        vis_pcd = o3d.geometry.PointCloud()
        vis_aabbs_gt = []
        vis_aabbs_pred = []
        for _ in range(50):
            vis_aabbs_gt.append(o3d.geometry.LineSet())
            vis_aabbs_pred.append(o3d.geometry.LineSet())

    ## 检测 + 跟踪  ##
    tracker = initialize(cfg, cat, ID_start)
    for i, batch_data in tqdm(enumerate(data_loader)):
        with torch.no_grad():
            batch_data = train_utils.to_device(batch_data, device)

            if fusion_method == 'late':
                pred_box_tensor, pred_score, gt_box_tensor = \
                    inference_utils.inference_late_fusion(batch_data,
                                                          model,
                                                          opencood_dataset)
            elif fusion_method == 'early':
                pred_box_tensor, pred_score, gt_box_tensor = \
                    inference_utils.inference_early_fusion(batch_data,
                                                           model,
                                                           opencood_dataset)
            elif fusion_method == 'intermediate':
                pred_box_tensor, pred_score, gt_box_tensor = \
                    inference_utils.inference_intermediate_fusion(batch_data,
                                                                  model,
                                                                  opencood_dataset)
            else:
                raise NotImplementedError('Only early, late and intermediate'
                                          'fusion is supported.')

            # 转换检测结果
            detections = convert_detections(pred_box_tensor, pred_score)

            # 提取检测信息
            dets = detections[:, :7]
            info = detections[:, 7:]

            dets_all = {'dets': dets, 'info': info}

            # 跟踪
            # results结果为（N，9）,N为检测到的目标数，9分别表示 h,w,l,x,y,z,theta, ID, score
            results, affi = tracker.track(dets_all)
            print(results)


        eval_utils.caluclate_tp_fp(pred_box_tensor,
                                   pred_score,
                                   gt_box_tensor,
                                   result_stat,
                                   0.3)
        eval_utils.caluclate_tp_fp(pred_box_tensor,
                                   pred_score,
                                   gt_box_tensor,
                                   result_stat,
                                   0.5)
        eval_utils.caluclate_tp_fp(pred_box_tensor,
                                   pred_score,
                                   gt_box_tensor,
                                   result_stat,
                                   0.7)
        if show_sequence:
            pcd, pred_o3d_box, gt_o3d_box = \
                vis_utils.visualize_inference_sample_dataloader(
                    pred_box_tensor,
                    gt_box_tensor,
                    batch_data['ego']['origin_lidar'],
                    vis_pcd,
                    mode='constant'
                    )
            if i == 0:
                vis.add_geometry(pcd)
                vis_utils.linset_assign_list(vis,
                                             vis_aabbs_pred,
                                             pred_o3d_box,
                                             update_mode='add')

                vis_utils.linset_assign_list(vis,
                                             vis_aabbs_gt,
                                             gt_o3d_box,
                                             update_mode='add')

            vis_utils.linset_assign_list(vis,
                                         vis_aabbs_pred,
                                         pred_o3d_box)
            vis_utils.linset_assign_list(vis,
                                         vis_aabbs_gt,
                                         gt_o3d_box)
            vis.update_geometry(pcd)
            vis.poll_events()
            vis.update_renderer()
            time.sleep(0.001)

    eval_utils.eval_final_results(result_stat,
                              model_dir,
                              global_sort_detections)
    if show_sequence:
        vis.destroy_window()

    return ID_start



if __name__ == '__main__':
    args = parse_args()

    config_path = '../../AB3DMOT/configs/%s.yml' % args.dataset
    cfg, settings_show = Config(config_path)

    # overwrite split and detection method
    if args.split is not '': cfg.split = args.split
    if args.det_name is not '': cfg.det_name = args.det_name

    ID_start = 1

    cat = 'Car'
    ID_start = main_per_cat(cfg, cat, ID_start)

