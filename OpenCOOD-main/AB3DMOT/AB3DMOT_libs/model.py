import numpy as np, os, copy, math

from AB3DMOT.AB3DMOT_libs.box import Box3D
from AB3DMOT.AB3DMOT_libs.kalman_filter import KF
from AB3DMOT.AB3DMOT_libs.kitti_oxts import get_ego_traj, egomotion_compensation_ID
from AB3DMOT.AB3DMOT_libs.matching import data_association
from AB3DMOT.AB3DMOT_libs.vis import vis_obj
from AB3DMOT.Xinshuo_PyToolbox.xinshuo_io import mkdir_if_missing
from AB3DMOT.Xinshuo_PyToolbox.xinshuo_miscellaneous import print_log
from AB3DMOT.Xinshuo_PyToolbox.xinshuo_visualization import random_colors

np.set_printoptions(suppress=True, precision=3)

# A Baseline of 3D Multi-Object Tracking
class AB3DMOT(object):			  	
	def __init__(self, cfg, cat, ID_init=0):

		# vis and log purposes
		# self.img_dir = img_dir  # 图像存储路径（用于可视化）
		# self.vis_dir = vis_dir  # 可视化结果存储路径
		self.vis = cfg.vis   # 是否启用可视化
		# self.hw = hw    # 图像高宽（height, width）

		#  跟踪器状态管理
		self.trackers = []  # 存储当前所有轨迹（Kalman滤波器实例）
		self.frame_count = 0  # 帧计数器
		self.ID_count = [ID_init]   # 全局ID计数器（初始值为ID_init）
		self.id_now_output = []   # 当前帧输出的有效轨迹ID

		#  算法配置参数
		self.cat = cat  # 目标类别（如Car、Pedestrian）
		self.ego_com = cfg.ego_com 			# 是否启用自车运动补偿
		# self.calib = calib   # 标定参数（相机-激光雷达坐标系转换）
		# self.oxts = oxts     # 自车运动数据（IMU/GPS）
		self.affi_process = cfg.affi_pro	# 亲和矩阵后处理开关
		self.get_param(cfg, cat)  # 加载类别相关参数（匹配算法、度量等）
		# self.print_param()

		# # 调试参数
		# self.debug_id = 2
		self.debug_id = None    # 设置某个轨迹ID以打印调试信息

		self.prev_bbox3D = None

	# 参数配置
	def get_param(self, cfg, cat):
		# get parameters for each dataset

		if cfg.dataset == 'KITTI':
			if cfg.det_name == 'pvrcnn':				# tuned for PV-RCNN detections
				if cat == 'Car': 			algm, metric, thres, min_hits, max_age = 'hungar', 'giou_3d', -0.2, 3, 2
				elif cat == 'Pedestrian': 	algm, metric, thres, min_hits, max_age = 'greedy', 'giou_3d', -0.4, 1, 4 		
				elif cat == 'Cyclist': 		algm, metric, thres, min_hits, max_age = 'hungar', 'dist_3d', 2, 3, 4
				else: assert False, 'error'
			elif cfg.det_name == 'pointrcnn':			# tuned for PointRCNN detections
				if cat == 'Car': 			algm, metric, thres, min_hits, max_age = 'hungar', 'giou_3d', -0.2, 3, 2
				elif cat == 'Pedestrian': 	algm, metric, thres, min_hits, max_age = 'greedy', 'giou_3d', -0.4, 1, 4 		
				elif cat == 'Cyclist': 		algm, metric, thres, min_hits, max_age = 'hungar', 'dist_3d', 2, 3, 4
				else: assert False, 'error'
			elif cfg.det_name == 'deprecated':			
				if cat == 'Car': 			algm, metric, thres, min_hits, max_age = 'hungar', 'dist_3d', 6, 3, 2
				elif cat == 'Pedestrian': 	algm, metric, thres, min_hits, max_age = 'hungar', 'dist_3d', 1, 3, 2		
				elif cat == 'Cyclist': 		algm, metric, thres, min_hits, max_age = 'hungar', 'dist_3d', 6, 3, 2
				else: assert False, 'error'
			else: assert False, 'error'
		elif cfg.dataset == 'nuScenes':
			if cfg.det_name == 'centerpoint':		# tuned for CenterPoint detections
				if cat == 'Car': 			algm, metric, thres, min_hits, max_age = 'greedy', 'giou_3d', -0.4, 1, 2
				elif cat == 'Pedestrian': 	algm, metric, thres, min_hits, max_age = 'greedy', 'giou_3d', -0.5, 1, 2
				elif cat == 'Truck': 		algm, metric, thres, min_hits, max_age = 'greedy', 'giou_3d', -0.4, 1, 2
				elif cat == 'Trailer': 		algm, metric, thres, min_hits, max_age = 'greedy', 'giou_3d', -0.3, 3, 2
				elif cat == 'Bus': 			algm, metric, thres, min_hits, max_age = 'greedy', 'giou_3d', -0.4, 1, 2
				elif cat == 'Motorcycle':	algm, metric, thres, min_hits, max_age = 'greedy', 'giou_3d', -0.7, 3, 2
				elif cat == 'Bicycle': 		algm, metric, thres, min_hits, max_age = 'greedy', 'dist_3d',    6, 3, 2
				else: assert False, 'error'
			elif cfg.det_name == 'megvii':			# tuned for Megvii detections
				if cat == 'Car': 			algm, metric, thres, min_hits, max_age = 'greedy', 'giou_3d', -0.5, 1, 2
				elif cat == 'Pedestrian': 	algm, metric, thres, min_hits, max_age = 'greedy', 'dist_3d',    2, 1, 2
				elif cat == 'Truck': 		algm, metric, thres, min_hits, max_age = 'greedy', 'giou_3d', -0.2, 1, 2
				elif cat == 'Trailer': 		algm, metric, thres, min_hits, max_age = 'greedy', 'giou_3d', -0.2, 3, 2
				elif cat == 'Bus': 			algm, metric, thres, min_hits, max_age = 'greedy', 'giou_3d', -0.2, 1, 2
				elif cat == 'Motorcycle':	algm, metric, thres, min_hits, max_age = 'greedy', 'giou_3d', -0.8, 3, 2
				elif cat == 'Bicycle': 		algm, metric, thres, min_hits, max_age = 'greedy', 'giou_3d', -0.6, 3, 2
				else: assert False, 'error'
			elif cfg.det_name == 'deprecated':		
				if cat == 'Car': 			metric, thres, min_hits, max_age = 'dist', 10, 3, 2
				elif cat == 'Pedestrian': 	metric, thres, min_hits, max_age = 'dist',  6, 3, 2	
				elif cat == 'Bicycle': 		metric, thres, min_hits, max_age = 'dist',  6, 3, 2
				elif cat == 'Motorcycle':	metric, thres, min_hits, max_age = 'dist', 10, 3, 2
				elif cat == 'Bus': 			metric, thres, min_hits, max_age = 'dist', 10, 3, 2
				elif cat == 'Trailer': 		metric, thres, min_hits, max_age = 'dist', 10, 3, 2
				elif cat == 'Truck': 		metric, thres, min_hits, max_age = 'dist', 10, 3, 2
				else: assert False, 'error'
			else: assert False, 'error'
		else: assert False, 'no such dataset'

		# add negative due to it is the cost
		if metric in ['dist_3d', 'dist_2d', 'm_dis']: thres *= -1
		# 统一设置参数,分别为：匹配算法（'hungar'为匈牙利算法或'greedy'为贪婪算法）；相似度度量（'giou_3d', 'dist_3d'等）；
		# 关联阈值（负值表示距离成本）；最大丢失帧数（超过则删除轨迹）；最小命中次数（达到后才输出轨迹）
		self.algm, self.metric, self.thres, self.max_age, self.min_hits = \
			algm, metric, thres, max_age, min_hits

		# define max/min values for the output affinity matrix
		if self.metric in ['dist_3d', 'dist_2d', 'm_dis']: self.max_sim, self.min_sim = 0.0, -100.
		elif self.metric in ['iou_2d', 'iou_3d']:   	   self.max_sim, self.min_sim = 1.0, 0.0
		elif self.metric in ['giou_2d', 'giou_3d']: 	   self.max_sim, self.min_sim = 1.0, -1.0

	def print_param(self):
		print_log('\n\n***************** Parameters for %s *********************' % self.cat, log=self.log, display=False)
		print_log('matching algorithm is %s' % self.algm, log=self.log, display=False)
		print_log('distance metric is %s' % self.metric, log=self.log, display=False)
		print_log('distance threshold is %f' % self.thres, log=self.log, display=False)
		print_log('min hits is %f' % self.min_hits, log=self.log, display=False)
		print_log('max age is %f' % self.max_age, log=self.log, display=False)
		print_log('ego motion compensation is %d' % self.ego_com, log=self.log, display=False)

	# 检测结果预处理：将输入的检测数组（格式如 [h,w,l,x,y,z,theta]）转换为 Box3D 对象，便于后续计算几何属性（如体积、交并比）
	def process_dets(self, dets):
		# convert each detection into the class Box3D 
		# inputs: 
		# 	dets - a numpy array of detections in the format [[h,w,l,x,y,z,theta],...]

		dets_new = []
		for det in dets:
			det_tmp = Box3D.array2bbox_raw(det)  # 将数组转换为Box3D对象
			dets_new.append(det_tmp)

		return dets_new

	def within_range(self, theta):
		# make sure the orientation is within a proper range

		if theta >= np.pi: theta -= np.pi * 2    # make the theta still in the range
		if theta < -np.pi: theta += np.pi * 2

		return theta

	# 方向角修正
	def orientation_correction(self, theta_pre, theta_obs):
		# update orientation in propagated tracks and detected boxes so that they are within 90 degree
		# 确保角度在 [-π, π] 范围内
		# make the theta still in the range
		theta_pre = self.within_range(theta_pre)
		theta_obs = self.within_range(theta_obs)

		# 如果预测与观测的角度差超过90度，调整预测角度使其对齐
		if abs(theta_obs - theta_pre) > np.pi / 2.0 and abs(theta_obs - theta_pre) < np.pi * 3 / 2.0:     
			theta_pre += np.pi       
			theta_pre = self.within_range(theta_pre)

		# 处理超过270度的情况；目标方向角（如车辆朝向）在 3D MOT 中至关重要，但直接使用原始角度可能导致匹配错误（如180度翻转）；
		# 通过强制预测与观测角度差为锐角，确保卡尔曼滤波更新时的方向一致性
		# now the angle is acute: < 90 or > 270, convert the case of > 270 to < 90
		if abs(theta_obs - theta_pre) >= np.pi * 3 / 2.0:
			if theta_obs > 0: theta_pre += np.pi * 2
			else: theta_pre -= np.pi * 2

		return theta_pre, theta_obs

	# 自车运动补偿；根据自车的IMU/GPS数据，将上一帧的轨迹位置转换到当前帧坐标系，消除自车运动带来的坐标系偏移；确保轨迹预测和检测匹配在统一坐标系下进行
	def ego_motion_compensation(self, frame, trks):
		# inverse ego motion compensation, move trks from the last frame of coordinate to the current frame for matching

		assert len(self.trackers) == len(trks), 'error'
		# 调用KITTI工具函数获取自车运动参数
		ego_xyz_imu, ego_rot_imu, left, right = get_ego_traj(self.oxts, frame, 1, 1, only_fut=True, inverse=True)
		# 对每个轨迹进行补偿
		for index in range(len(self.trackers)):
			trk_tmp = trks[index]
			xyz = np.array([trk_tmp.x, trk_tmp.y, trk_tmp.z]).reshape((1, -1))
			compensated = egomotion_compensation_ID(xyz, self.calib, ego_rot_imu, ego_xyz_imu, left, right)
			trk_tmp.x, trk_tmp.y, trk_tmp.z = compensated[0]

			# update compensated state in the Kalman filter
			try:
				# 更新卡尔曼滤波器状态
				self.trackers[index].kf.x[:3] = copy.copy(compensated).reshape((-1))
			except:
				self.trackers[index].kf.x[:3] = copy.copy(compensated).reshape((-1, 1))

		return trks

	def visualization(self, img, dets, trks, calib, hw, save_path, height_threshold=0):
		# visualize to verify if the ego motion compensation is done correctly
		# ideally, the ego-motion compensated tracks should overlap closely with detections
		import cv2 
		from PIL import Image

		dets, trks = copy.copy(dets), copy.copy(trks)
		img = np.array(Image.open(img))
		max_color = 20
		colors = random_colors(max_color)       # Generate random colors

		# 绘制检测框（黄色）
		for det_tmp in dets: 
			img = vis_obj(det_tmp, img, calib, hw, (255, 255, 0))				# yellow for detection
		
		# visualize color-specific tracks
		count = 0
		ID_list = [tmp.id for tmp in self.trackers]
		# 绘制轨迹框（随机颜色）
		for trk_tmp in trks: 
			ID_tmp = ID_list[count]
			color_float = colors[int(ID_tmp) % max_color]
			color_int = tuple([int(tmp * 255) for tmp in color_float])
			str_vis = '%d, %f' % (ID_tmp, trk_tmp.o)
			img = vis_obj(trk_tmp, img, calib, hw, color_int, str_vis)		# blue for tracklets
			count += 1
		
		img = Image.fromarray(img)
		img = img.resize((hw['image'][1], hw['image'][0]))
		img.save(save_path)

	def prediction(self):
		# get predicted locations from existing tracks

		trks = []
		for t in range(len(self.trackers)):
			
			# propagate locations
			kf_tmp = self.trackers[t]
			if kf_tmp.id == self.debug_id:
				print('\n before prediction')
				print(kf_tmp.kf.x.reshape((-1)))
				print('\n current velocity')
				print(kf_tmp.get_velocity())
			kf_tmp.kf.predict()  # 卡尔曼滤波预测步骤；根据运动模型（如匀速模型）预测目标在下一帧的状态
			if kf_tmp.id == self.debug_id:
				print('After prediction')
				print(kf_tmp.kf.x.reshape((-1)))
			kf_tmp.kf.x[3] = self.within_range(kf_tmp.kf.x[3])   # 修正角度

			# update statistics
			kf_tmp.time_since_update += 1 		
			trk_tmp = kf_tmp.kf.x.reshape((-1))[:7]
			trks.append(Box3D.array2bbox(trk_tmp))

		return trks

	def update(self, matched, unmatched_trks, dets, info):
		# update matched trackers with assigned detections
		
		dets = copy.copy(dets)
		for t, trk in enumerate(self.trackers):
			if t not in unmatched_trks:
				d = matched[np.where(matched[:, 1] == t)[0], 0]     # a list of index
				assert len(d) == 1, 'error'

				# update statistics
				trk.time_since_update = 0		# reset because just updated
				trk.hits += 1

				# update orientation in propagated tracks and detected boxes so that they are within 90 degree
				bbox3d = Box3D.bbox2array(dets[d[0]])
				trk.kf.x[3], bbox3d[3] = self.orientation_correction(trk.kf.x[3], bbox3d[3])

				if trk.id == self.debug_id:
					print('After ego-compoensation')
					print(trk.kf.x.reshape((-1)))
					print('matched measurement')
					print(bbox3d.reshape((-1)))

				# kalman filter update with observation
				trk.kf.update(bbox3d)   # 卡尔曼滤波更新步骤；利用匹配的检测结果修正预测状态，调整速度、位置等估计

				if trk.id == self.debug_id:
					print('after matching')
					print(trk.kf.x.reshape((-1)))
					print('\n current velocity')
					print(trk.get_velocity())

				trk.kf.x[3] = self.within_range(trk.kf.x[3])
				trk.info = info[d, :][0]

	# 为未匹配的检测初始化新跟踪器，生成唯一ID
	def birth(self, dets, info, unmatched_dets):
		# 为未匹配的检测创建并初始化新跟踪器
		new_id_list = list()  # new ID generated for unmatched detections
		for i in unmatched_dets:  # a scalar of index
			trk = KF(Box3D.bbox2array(dets[i]), info[i, :], self.ID_count[0])
			self.trackers.append(trk)
			new_id_list.append(trk.id)
			# print('track ID %s has been initialized due to new detection' % trk.id)

			self.ID_count[0] += 1

		return new_id_list  # 返回新生成的ID列表

	# 过滤并输出有效跟踪结果，移除失效跟踪器
	def output(self,dt=0.1):
		num_trks = len(self.trackers)
		results = []
		for trk in reversed(self.trackers):  # 逆序遍历跟踪器
			# change format from [x,y,z,theta,l,w,h] to [h,w,l,x,y,z,theta]
			# 转换跟踪器状态为原始检测格式
			d = Box3D.array2bbox(trk.kf.x[:7].reshape((7, )))     # bbox location self
			d = Box3D.bbox2array_raw(d)

			if self.prev_bbox3D is not None:
				prev_array = self.prev_bbox3D[0]  # 取出 numpy array
				matched = prev_array[prev_array[:, 7] == trk.id, :7]
				if matched.shape[0] > 0:
					prev_bbox = matched[0]
					dx = (d[0] - prev_bbox[0]) / dt
					dy = (d[1] - prev_bbox[1]) / dt
					dz = (d[2] - prev_bbox[2]) / dt
				else:
					dx, dy, dz = 1, 1, 0  # 没匹配到该 ID，给默认速度
			else:
				dx, dy, dz = 1, 1, 0

			velocity = np.array([[dx], [dy], [dz]]).reshape(-1)


			#  检查跟踪器是否有效（未过期且满足命中次数）
			# d 包含前7个参数（3D框参数）。[trk.id] 是跟踪ID。trk.info 包含检测时的附加信息（如score、alpha、2D框等）
			if ((trk.time_since_update < self.max_age) and (trk.hits >= self.min_hits or self.frame_count <= self.min_hits)):
				results.append(np.concatenate((d, [trk.id], velocity, trk.info)).reshape(1, -1))
			num_trks -= 1

			# 移除过期跟踪器
			# deadth, remove dead tracklet
			if (trk.time_since_update >= self.max_age): 
				self.trackers.pop(num_trks)

		return results

	# 调整亲和矩阵，使其表示活跃跟踪器间的关联
	def process_affi(self, affi, matched, unmatched_dets, new_id_list):
		###### determine the ID for each past track
		trk_id = self.id_past 			# 过去帧的跟踪器ID

		###### determine the ID for each current detection
		det_id = [-1 for _ in range(affi.shape[0])]		# 初始化当前检测的ID列表

		# 分配ID：匹配的检测继承跟踪器ID，未匹配的分配新ID
		for match_tmp in matched:		
			det_id[match_tmp[0]] = trk_id[match_tmp[1]]

		# assign the new birth ID to each unmatched detection
		count = 0
		assert len(unmatched_dets) == len(new_id_list), 'error'
		for unmatch_tmp in unmatched_dets:
			det_id[unmatch_tmp] = new_id_list[count] 	# new_id_list is in the same order as unmatched_dets
			count += 1
		assert not (-1 in det_id), 'error, still have invalid ID in the detection list'

		############################ update the affinity matrix based on the ID matching
		
		# transpose so that now row is past trks, col is current dets	
		affi = affi.transpose() 	  # 转置矩阵，行对应过去跟踪器，列对应当前检测

		###### compute the permutation for rows (past tracklets), possible to delete but not add new rows
		# 行置换：按过去活跃跟踪器ID的顺序调整行
		permute_row = list()
		for output_id_tmp in self.id_past_output:
			index = trk_id.index(output_id_tmp)
			permute_row.append(index)
		affi = affi[permute_row, :]	
		assert affi.shape[0] == len(self.id_past_output), 'error'

		# 列置换：按当前活跃跟踪器ID调整列，扩展矩阵处理新增跟踪器
		max_index = affi.shape[1]
		permute_col = list()
		to_fill_col, to_fill_id = list(), list() 		# append new columns at the end, also remember the ID for the added ones
		for output_id_tmp in self.id_now_output:
			try:
				index = det_id.index(output_id_tmp)  # 查找当前ID对应的列
			except:		# some output ID does not exist in the detections but rather predicted by KF # 若不存在（预测新增），扩展矩阵
				index = max_index
				max_index += 1
				to_fill_col.append(index); to_fill_id.append(output_id_tmp)
			permute_col.append(index)

		# 扩展矩阵并填充新增列的相似度（默认最小值）
		append = np.zeros((affi.shape[0], max_index - affi.shape[1]))
		append.fill(self.min_sim)
		affi = np.concatenate([affi, append], axis=1)

		# find out the correct permutation for the newly added columns of ID
		for count in range(len(to_fill_col)):
			fill_col = to_fill_col[count]
			fill_id = to_fill_id[count]
			row_index = self.id_past_output.index(fill_id)

			# construct one hot vector because it is proapgated from previous tracks, so 100% matching
			affi[row_index, fill_col] = self.max_sim		
		affi = affi[:, permute_col]    # 按列置换调整矩阵

		return affi

	# 整合整个跟踪流程，处理单帧数据
	def track(self, dets_all):
		"""
		Params:
		  	dets_all: dict
				dets - a numpy array of detections in the format [[h,w,l,x,y,z,theta],...]
				info: a array of other info for each det
			frame:    str, frame number, used to query ego pose
		Requires: this method must be called once for each frame even with empty detections.
		Returns the a similar array, where the last column is the object ID.

		NOTE: The number of objects returned may differ from the number of detections provided.
		"""
		dets, info = dets_all['dets'], dets_all['info']         # dets: N x 7, float numpy array
		# if self.debug_id: print('\nframe is %s' % frame)

		self.frame_count += 1

		# recall the last frames of outputs for computing ID correspondences during affinity processing
		# 记录上一帧的输出ID
		self.id_past_output = copy.copy(self.id_now_output)
		self.id_past = [trk.id for trk in self.trackers]

		# process detection format  # 预处理检测框格式
		dets = self.process_dets(dets)

		# tracks propagation based on velocity  # 预测跟踪器下一帧状态
		trks = self.prediction()

		# matching
		trk_innovation_matrix = None
		if self.metric == 'm_dis':
			trk_innovation_matrix = [trk.compute_innovation_matrix() for trk in self.trackers]
		# 数据关联：匹配检测与跟踪器
		matched, unmatched_dets, unmatched_trks, cost, affi = \
			data_association(dets, trks, self.metric, self.thres, self.algm, trk_innovation_matrix)

		# update trks with matched detection measurement
		self.update(matched, unmatched_trks, dets, info)   # 更新匹配的跟踪器

		# create and initialise new trackers for unmatched detections
		new_id_list = self.birth(dets, info, unmatched_dets)    # 为未匹配检测生成新跟踪器

		# output existing valid tracks
		results = self.output()   # 输出有效跟踪结果
		if len(results) > 0: results = [np.concatenate(results)]		# h,w,l,x,y,z,theta, ID, other info, confidence
		else:            	 results = [np.empty((0, 15))]
		self.id_now_output = results[0][:, 7].tolist()					# only the active tracks that are outputed

		# post-processing affinity to convert to the affinity between resulting tracklets   # 后处理亲和矩阵
		if self.affi_process:
			affi = self.process_affi(affi, matched, unmatched_dets, new_id_list)

		self.prev_bbox3D = results

		return results, affi
