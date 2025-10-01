# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com

from __future__ import print_function
import matplotlib;

from AB3DMOT.AB3DMOT_libs.io import load_detection, get_saving_dir, get_frame_det, save_affinity, save_results
from AB3DMOT.AB3DMOT_libs.utils import get_subfolder_seq, initialize, Config
from AB3DMOT.Xinshuo_PyToolbox.xinshuo_io import mkdir_if_missing
from AB3DMOT.Xinshuo_PyToolbox.xinshuo_miscellaneous import print_log, get_timestring
from AB3DMOT.scripts.post_processing.combine_trk_cat import combine_trk_cat

matplotlib.use('Agg')
import os, numpy as np, time, sys, argparse

def parse_args():
    parser = argparse.ArgumentParser(description='AB3DMOT')
    parser.add_argument('--dataset', type=str, default='KITTI', help='KITTI, nuScenes')
    parser.add_argument('--split', type=str, default='', help='train, val, test')
    parser.add_argument('--det_name', type=str, default='', help='pointrcnn')
    args = parser.parse_args()
    return args

# 对每个类别进行跟踪处理
def main_per_cat(cfg, cat, log, ID_start):

	# result_sha：唯一标识符，组合检测模型名、目标类别和数据集划分，用于区分不同实验;
	# det_root/trk_root：加载检测结果和跟踪结果的根目录，符合MOT任务中“检测-跟踪”分离的流程
	result_sha = '%s_%s_%s' % (cfg.det_name, cat, cfg.split)
	det_root = os.path.join('./data', cfg.dataset, 'detection', result_sha)
	subfolder, det_id2str, hw, seq_eval, data_root = get_subfolder_seq(cfg.dataset, cfg.split)
	trk_root = os.path.join(data_root, 'tracking')
	# save_dir：保存多假设（Multi-Hypothesis）跟踪结果的目录，支持多假设跟踪以处理不确定性
	save_dir = os.path.join(cfg.save_root, result_sha + '_H%d' % cfg.num_hypo); mkdir_if_missing(save_dir)

	# 为每个假设（如不同运动模型或匹配策略）创建独立目录，支持多路径跟踪假设，避免漏检或误检导致的ID切换
	eval_dir_dict = dict()
	for index in range(cfg.num_hypo):
		eval_dir_dict[index] = os.path.join(save_dir, 'data_%d' % index); mkdir_if_missing(eval_dir_dict[index]) 		

	# loop every sequence 遍历每个序列，加载检测结果，如果没有检测结果则跳过该序列
	seq_count = 0
	total_time, total_frames = 0.0, 0
	# 序列遍历与检测加载
	for seq_name in seq_eval:
		seq_file = os.path.join(det_root, seq_name+'.txt')
		# load_detection：加载检测结果文件（如.txt格式），格式通常为[frame, ID, x, y, w, h, score]，与MOT Challenge标准一致
		seq_dets, flag = load_detection(seq_file) 				# load detection
		# 若检测结果为空则跳过，体现“检测驱动跟踪”范式
		if not flag: continue									# no detection

		# create folders for saving  为每个序列创建保存结果的文件夹
		eval_file_dict, save_trk_dir, affinity_dir, affinity_vis = \
			get_saving_dir(eval_dir_dict, seq_name, save_dir, cfg.num_hypo)	

		# initialize tracker;ID_start：全局ID计数器，避免跨类别ID冲突，符合MOT评估要求
		tracker, frame_list = initialize(cfg, trk_root, save_dir, subfolder, seq_name, cat, ID_start, hw, log)

		# 遍历每个帧，进行跟踪处理，记录处理时间，保存亲和矩阵和跟踪结果
		min_frame, max_frame = int(frame_list[0]), int(frame_list[-1])
		for frame in range(min_frame, max_frame + 1):
			# add an additional frame here to deal with the case that the last frame, although no detection
			# but should output an N x 0 affinity for consistency
			
			# logging
			print_str = 'processing %s %s: %d/%d, %d/%d   \r' % (result_sha, seq_name, seq_count, \
				len(seq_eval), frame, max_frame)
			sys.stdout.write(print_str)
			sys.stdout.flush()

			# tracking by detection
			dets_frame = get_frame_det(seq_dets, frame)
			since = time.time()
			# 核心跟踪逻辑;可能包含:预测阶段：卡尔曼滤波预测轨迹状态;关联阶段：通过匈牙利算法匹配检测与轨迹;更新阶段：更新匹配成功的轨迹状态
			results, affi = tracker.track(dets_frame, frame, seq_name)		
			total_time += time.time() - since

			# saving affinity matrix, between the past frame and current frame
			# e.g., for 000006.npy, it means affinity between frame 5 and 6
			# note that the saved value in affinity can be different in reality because it is between the 
			# original detections and ego-motion compensated predicted tracklets, rather than between the 
			# actual two sets of output tracklets
			save_affi_file = os.path.join(affinity_dir, '%06d.npy' % frame)
			save_affi_vis  = os.path.join(affinity_vis, '%06d.txt' % frame)
			# affi：亲和矩阵（相似度矩阵），记录检测与轨迹的匹配得分，用于级联匹配或后处理分析
			if (affi is not None) and (affi.shape[0] + affi.shape[1] > 0): 
				# save affinity as long as there are tracklets in at least one frame
				np.save(save_affi_file, affi)

				# cannot save for visualization unless both two frames have tracklets
				if affi.shape[0] > 0 and affi.shape[1] > 0:
					save_affinity(affi, save_affi_vis)

			# saving trajectories, loop over each hypothesis
			for hypo in range(cfg.num_hypo):
				save_trk_file = os.path.join(save_trk_dir[hypo], '%06d.txt' % frame)
				save_trk_file = open(save_trk_file, 'w')
				for result_tmp in results[hypo]:				# N x 15
					save_results(result_tmp, save_trk_file, eval_file_dict[hypo], \
						det_id2str, frame, cfg.score_threshold)
				save_trk_file.close()

			total_frames += 1
		seq_count += 1

		for index in range(cfg.num_hypo): 
			eval_file_dict[index].close()
			ID_start = max(ID_start, tracker.ID_count[index])

	print_log('%s, %25s: %4.f seconds for %5d frames or %6.1f FPS, metric is %s = %.2f' % \
		(cfg.dataset, result_sha, total_time, total_frames, total_frames / total_time, \
		tracker.metric, tracker.thres), log=log)

	# 返回最大的 ID 号
	return ID_start

def main(args):

	# load config files
	config_path = './configs/%s.yml' % args.dataset
	cfg, settings_show = Config(config_path)

	# overwrite split and detection method
	if args.split is not '': cfg.split = args.split
	if args.det_name is not '': cfg.det_name = args.det_name

	# print configs
	time_str = get_timestring()
	log = os.path.join(cfg.save_root, 'log/log_%s_%s_%s.txt' % (time_str, cfg.dataset, cfg.split))
	mkdir_if_missing(log); log = open(log, 'w')
	for idx, data in enumerate(settings_show):
		print_log(data, log, display=False)

	# global ID counter used for all categories, not start from 1 for each category to prevent different 
	# categories of objects have the same ID. This allows visualization of all object categories together
	# without ID conflicting, Also use 1 (not 0) as start because MOT benchmark requires positive ID
	# 初始化全局 ID 计数器为 1
	ID_start = 1

	# 对每个类别调用 main_per_cat 函数进行跟踪处理
	for cat in cfg.cat_list:
		# 	ID_start 全局递增，确保不同类别的轨迹ID唯一
		ID_start = main_per_cat(cfg, cat, log, ID_start)

	# 调用 combine_trk_cat 函数合并不同类别的跟踪结果
	print_log('\ncombining results......', log=log)
	# 将不同类别的跟踪结果合并为统一文件，符合MOT Challenge提交格式
	combine_trk_cat(cfg.split, cfg.dataset, cfg.det_name, 'H%d' % cfg.num_hypo, cfg.num_hypo)
	print_log('\nDone!', log=log)
	log.close()

if __name__ == '__main__':

	args = parse_args()
	main(args)