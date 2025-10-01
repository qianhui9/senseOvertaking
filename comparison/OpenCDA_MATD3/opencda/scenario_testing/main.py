import carla
import numpy as np
import pandas as pd
import torch
import argparse
import os

import TD3
import opencda.scenario_testing.util


# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
from MARL.my_EnvCluster import LaneChangePredict
from opencda.scenario_testing import util
from opencda.scenario_testing.single_2lanefree_carla_MyTest import mainEnv, readSenseResult, physicsCorrection


def get_action(action1, action2, action3):
	# action1 单做离散动作来处理
	if action1 >= -1 and action1 < -1 / 3:
		actionP1 = -1
	elif action1 >= -1 / 3 and action1 < 1 / 3:
		actionP1 = 0
	else:
		actionP1 = 1

	# action2 单做连续动作来处理
	actionP2 = action2

	actionP3 = action3

	return actionP1, actionP2, actionP3



if __name__ == "__main__":
	
	parser = argparse.ArgumentParser()
	parser.add_argument("--policy", default="TD3")                  # Policy name (TD3, DDPG or OurDDPG)
	parser.add_argument("--episodes", default=2000)                 # Number of epsiodes
	parser.add_argument("--env", default="HalfCheetah-v2")          # OpenAI gym environment name
	parser.add_argument("--seed", default=0, type=int)              # Sets Gym, PyTorch and Numpy seeds
	parser.add_argument("--start_timesteps", default=25e3, type=int)# Time steps initial random policy is used
	parser.add_argument("--eval_freq", default=5e3, type=int)       # How often (time steps) we evaluate
	parser.add_argument("--max_timesteps", default=70, type=int)   # Max time steps to run environment
	parser.add_argument("--expl_noise", default=0.1, type=float)    # Std of Gaussian exploration noise
	parser.add_argument("--batch_size", default=32, type=int)      # Batch size for both actor and critic
	parser.add_argument("--discount", default=0.99, type=float)     # Discount factor
	parser.add_argument("--tau", default=0.005, type=float)         # Target network update rate 0.005
	parser.add_argument("--policy_noise", default=0.2)              # Noise added to target policy during critic update
	parser.add_argument("--noise_clip", default=0.5)                # Range to clip target policy noise
	parser.add_argument("--policy_freq", default=2, type=int)       # Frequency of delayed policy updates
	parser.add_argument("--save_model", action="store_true")        # Save model and optimizer parameters
	parser.add_argument("--load_model", default="")                 # Model load file name, "" doesn't load, "default" uses file_name
	args = parser.parse_args()

	file_name = f"{args.policy}_{args.env}_{args.seed}"
	print("---------------------------------------")
	print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
	print("---------------------------------------")

	if not os.path.exists("result"):
		os.makedirs("result")

	if args.save_model and not os.path.exists("./models"):
		os.makedirs("./models")

	N_Vehicle = 3

	env = LaneChangePredict()

	single_cav_list, scenario_manager, spectator, bg_veh_list = mainEnv()
	
	state_dim = env.n_features
	action_dim = 3
	max_action = float(np.max(env.actions[:,1]))  # 取动作空间的动作的最大值

	max_actionThreshold = 1

	kwargs = {
		"state_dim": state_dim,
		"action_dim": action_dim,
		"max_action": max_action,
		"discount": args.discount,
		"tau": args.tau,
	}

	# Initialize policy
	if args.policy == "TD3":
		# Target policy smoothing is scaled wrt the action scale
		kwargs["policy_noise"] = args.policy_noise * max_actionThreshold
		kwargs["noise_clip"] = args.noise_clip * max_actionThreshold
		kwargs["policy_freq"] = args.policy_freq
		policy = TD3.TD3(**kwargs)
	# elif args.policy == "OurDDPG":
	# 	policy = OurDDPG.DDPG(**kwargs)
	# elif args.policy == "DDPG":
	# 	policy = DDPG.DDPG(**kwargs)
	
	# Evaluate untrained policy
	# evaluations = [eval_policy(policy, args.env, args.seed)]

	done = False
	episode_reward = 0
	episode_timesteps = 0
	episode_num = 0

	replay_buffer = util.ReplayBuffer(state_dim, action_dim)


	for i_eps in range(args.episodes):

		readFlag = 0

		episode_reward = 0

		single_cav_list, spectator, bg_veh_list, cav_ids = env.reset(single_cav_list, scenario_manager, bg_veh_list)

		scenario_manager.tick()

		# 来确定目标车辆的ID
		deterTarIDVeh = 0

		action_all = np.zeros([N_Vehicle, env.n_actions], dtype=np.float32)  # [离散, 连续1, 连续2]
		state_all = np.zeros((N_Vehicle, env.n_features), dtype=np.float32)
		reward_all = np.zeros((N_Vehicle, 1), dtype=np.float32)
		next_state_all = np.zeros((N_Vehicle, env.n_features), dtype=np.float32)



		if single_cav_list[0].vehicle.is_alive:
			transform = single_cav_list[0].vehicle.get_transform()
			spectator.set_transform(carla.Transform(
				transform.location + carla.Location(z=70),
				carla.Rotation(pitch=-90)))
		else:
			print("[Warning] Vehicle actor is destroyed. Skipping this frame.")
			continue

		scores = []

		current_episode_data = {i: {'speedControl': [], 'ttc': []} for i in range(N_Vehicle)}

		for i_step in range(int(args.max_timesteps)):

			score = 0
			readFlag += 1

			s_all = []
			a_all = []

			action_all_idx = 0

			for i in range(N_Vehicle):
				single_cav_list[i].update_info()
				single_cav_list[i].savedData_dumper()

				# 物理修正
				physicsCorrection(i, single_cav_list, scenario_manager)

				# 获取环境信息
				information = readSenseResult(cav_ids[i], readFlag)
				if deterTarIDVeh == 0:
					env.deterTarIDVehID(information)
					deterTarIDVeh += 1

				state = np.array(env._findstate(i, information), dtype=np.float32)

				s_all.append(state)

				action1 = (
						policy.select_action(np.array(state))
						+ np.random.normal(0, max_actionThreshold * args.expl_noise, size=action_dim)
				).clip(-max_actionThreshold, max_actionThreshold)

				action2 = (
						policy.select_action(np.array(state))
						+ np.random.normal(0, max_actionThreshold * args.expl_noise, size=action_dim)
				).clip(-max_actionThreshold, max_actionThreshold)

				action3 = (
						policy.select_action(np.array(state))
						+ np.random.normal(0, max_actionThreshold * args.expl_noise, size=action_dim)
				).clip(-max_actionThreshold, max_actionThreshold)

				# 拿到离散和连续动作
				action, action1, action2 = get_action(action1[1], action2[1], action3[1])

				action_all[action_all_idx, 0] = action
				action_all[action_all_idx, 1] = action1
				action_all[action_all_idx, 2] = action2

				state_all[action_all_idx, :] = state

				action_all_idx += 1

			if single_cav_list[0].vehicle.is_alive:
				transform = single_cav_list[0].vehicle.get_transform()
				spectator.set_transform(carla.Transform(
					transform.location + carla.Location(z=70),
					carla.Rotation(pitch=-90)))
			else:
				print("[Warning] Vehicle actor is destroyed. Skipping this frame.")
				continue

			# s_all.append(state)
			all_next_state = np.array([])
			all_reward = np.array([])
			all_terminal = np.array([])

			scenario_manager.tick()

			for i in range(N_Vehicle):
				# 物理修正
				physicsCorrection(i, single_cav_list, scenario_manager)

				information = readSenseResult(cav_ids[i], readFlag)

				action1 = action_all[i, 0]
				action2 = action_all[i, 1]
				action3 = action_all[i, 2]
				env._findCurrentState(i, information)
				next_state, reward, terminal = env.step(single_cav_list[i], action1, action2, action3, i, readFlag + 1)

				ttc = env.getFinaTCC()
				speed = env.getFinaSpeed()
				current_episode_data[i]['speedControl'].append(speed)
				current_episode_data[i]['ttc'].append(ttc)

				all_next_state = np.append(all_next_state, next_state)
				all_reward = np.append(all_reward, reward)
				all_terminal = np.append(all_terminal, terminal)

				reward_all[i, :] = reward

				next_state_all[i, :] = next_state

			sumReward = sum(all_reward)

			# if episode_timesteps == args.max_timesteps - 1:
			# 	env.writer(0)

			terminal = int(any(element == 1 for element in all_terminal))
			if terminal:
				break
			done_bool = float(terminal)

			# Store data in replay
			for i in range(N_Vehicle):
				state = state_all[i]
				action = action_all[i]
				next_state = next_state_all[i]
				reward = reward_all[i]
				replay_buffer.add(state, action, next_state, reward, done_bool)


			# Train agent after collecting sufficient data
			# if t >= args.start_timesteps:

		# 最后，将最大平均回报轮次的数据写入文件
		for i in range(N_Vehicle):
			df1 = pd.DataFrame({'speedControl': current_episode_data[i]['speedControl']})
			df2 = pd.DataFrame({'ttc': current_episode_data[i]['ttc']})
			df1.to_csv(
				f'E:/postgraduate/V2X/senseOvertaking/comparison/OpenCDA_MATD3/opencda/scenario_testing/result/normal/agent{i + 1}_speedControl.csv',
				index=False,
				header=False)
			df2.to_csv(
				f'E:/postgraduate/V2X/senseOvertaking/comparison/OpenCDA_MATD3/opencda/scenario_testing/result/normal/agent{i + 1}_ttc.csv',
				index=False,
				header=False)



		policy.train(replay_buffer, args.batch_size)


		# scores.append(episode_reward / i_step)
		print(i_eps, "average_episode_reward：", sumReward / len(all_reward))
		df = pd.DataFrame({'Reward': sumReward / len(all_reward)}, index=[0])
		# 将数据保存为CSV文件，并去除标题行
		df.to_csv('result/td3/reward.csv', index=False, mode='a', header=False)

