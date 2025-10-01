from __future__ import division
import numpy as np
from numpy import random as nr
import time
import random
import math
import heapq
import scipy

np.set_printoptions(suppress=True)
np.random.seed(10)

class UAV:

    def __init__(self, id, start_position, start_direction, velocity, link):
        self.id = id
        self.position = start_position
        self.direction = start_direction
        self.velocity = velocity
        self.link = link
        self.neighbors = []
        self.destinations = []


class Environment:
    def __init__(self):
        self.T = 12
        self.n_uav = 4
        self.n_bs = 5
        self.n_user = 20
        self.X = 500
        self.height = 200
        self.c1 = 12.81
        self.c2 = 0.11395
        self.beta_LoS = 1.44544
        self.beta_NLoS = 199.526
        self.fc = 2  * pow(10, 9)  # 2GHz
        self.c = 3 * pow(10, 8)  # 光速
        self.bandwidth = 20
        self.channel = 20  # 信道个数
        self.max_power_UAV = 2  # 最大传输功率2W
        self.max_power_MBS = 4
        self.max_power_user = 0.6
        self.B = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        #self.power_bs = 2 / self.B  # 基站功率
        self.power_bs = np.zeros(self.n_bs)
        self.power_user = 0.2
        self.noise_level = -60  # 噪声等级-60dbW σ^2
        self.noise = pow(10, self.noise_level / 10)
        self.SIC = pow(10, 6)  # 自干扰取消60dB
        self.exponent = -2

        self.user_pos12 = nr.randint(-self.X + 100, self.X - 100, size=(self.n_user, 2))
        self.user_pos3 = np.ones((self.n_user, 1)) * 0
        self.user_pos = np.row_stack(([100, 15, 0],[30, 30, 0],[200, 200, 0],[300, 290, 0],[120, 150, 0],
                                      [50, -90, 0],[60, -170, 0],[240, -320, 0],[350, -20, 0],[110, -110, 0],
                                      [-45, -60, 0],[-20, -20, 0],[-150, -200, 0],[-180, -10, 0],[-200, -300, 0],
                                      [-5, 200, 0],[-190, 230, 0],[-310, 50, 0],[-270, 160, 0],[-100, 15, 0]))
        #np.column_stack((self.user_pos12, self.user_pos3))

        self.UAV_pos12 = nr.randint(-self.X, self.X, size=(self.n_bs - 1, 2))
        self.UAV_pos3 = np.ones((self.n_bs - 1, 1)) * self.height
        self.UAV_pos = np.column_stack((self.UAV_pos12, self.UAV_pos3))
        #self.bs_pos = np.row_stack(([0, 0, 0], self.UAV_pos))
        #self.bs_pos = np.row_stack(([0, 0, 0], [100, 100, 200],[100, -100, 200],[-100, -100, 200],[-100, 100, 200]))
        self.bs_pos = np.row_stack(([0, 0, 0], [100, 100, 200], [100, -100, 200], [-100, -100, 200], [-100, 100, 200]))

        self.dist_ub = np.zeros((self.n_user, self.n_bs))
        self.dist_uu = np.zeros((self.n_user, self.n_user))
        self.dist_bb = np.zeros((self.n_bs, self.n_bs))

        self.gain_ub = np.zeros((self.n_user, self.n_bs))
        self.gain_uu = np.zeros((self.n_user, self.n_user))
        self.gain_bb = np.zeros((self.n_bs, self.n_bs))

        self.interf_ub = np.zeros((self.n_bs, self.n_user))
        self.interf_bb = np.zeros((self.n_bs, self.n_bs))
        self.interf_ub_sum = np.zeros((self.n_bs, 1))
        self.interf_bb_sum = np.zeros((self.n_bs, 1))
        self.interf_b_sum = np.zeros((self.n_bs, 1))

        self.interf_bu = np.zeros((self.n_user, self.n_bs))
        self.interf_uu = np.zeros((self.n_user, self.n_user))
        self.interf_bu_sum = np.zeros((self.n_user, 1))
        self.interf_uu_sum = np.zeros((self.n_user, 1))
        self.interf_u_sum = np.zeros((self.n_user, 1))

        self.uav_backhaul_rate = np.zeros(self.n_bs - 1)

        self.rate_up = np.zeros(self.n_user)
        self.rate_down = np.zeros(self.n_user)
        self.rate = np.zeros(self.n_user)

        self.time = 1
        self.datasize = 300

        self.up_data = self.datasize * np.ones(self.n_user)
        self.down_data = self.datasize * np.ones(self.n_user)

        self.up_active_links = np.ones((self.n_user), dtype='bool')
        self.down_active_links = np.ones((self.n_user), dtype='bool')

    def judge_position(self, action, i):
        judge = 1
        b_x = self.bs_pos[i+1][0]
        b_y = self.bs_pos[i+1][1]
        b_x += action[0] * math.sin(math.radians(action[1]) * 90)
        b_y += action[0] * math.cos(math.radians(action[1]) * 90)
        if b_x < -400 or b_x > 400 or b_y < -400 or b_y > 400:
            judge = 0
        return judge

    def update_position(self, actions_all):
        for i in range(self.n_uav):
            self.bs_pos[i+1][0] += actions_all[i][0] * math.sin(math.radians(actions_all[i][1])*90)
            self.bs_pos[i+1][1] += actions_all[i][0] * math.cos(math.radians(actions_all[i][1])*90)

    def calc_reward(self, actions_all, actions_user_temp):

        up = actions_user_temp[0:20]
        down = actions_user_temp[20:40]

        for i in range(self.n_bs):
            if i == 0:
                self.power_bs[i] = self.max_power_MBS
            else:
                self.power_bs[i] = actions_all[i-1][2]

        self.interf_ub = np.zeros((self.n_bs, self.n_user))
        self.interf_bb = np.zeros((self.n_bs, self.n_bs))
        self.interf_ub_sum = np.zeros(self.n_bs)
        self.interf_bb_sum = np.zeros(self.n_bs)
        self.interf_b_sum = np.zeros(self.n_bs)

        self.interf_bu = np.zeros((self.n_user, self.n_bs))
        self.interf_uu = np.zeros((self.n_user, self.n_user))
        self.interf_bu_sum = np.zeros(self.n_user)
        self.interf_uu_sum = np.zeros(self.n_user)
        self.interf_u_sum = np.zeros(self.n_user)

        #  calc_distance
        for i in range(self.n_user):
            for j in range(self.n_bs):
                self.dist_ub[i, j] = np.linalg.norm(self.user_pos[i] - self.bs_pos[j])

        for i in range(self.n_user):
            for j in range(self.n_user):
                self.dist_uu[i, j] = np.linalg.norm(self.user_pos[i] - self.user_pos[j])

        for i in range(self.n_bs):
            for j in range(self.n_bs):
                self.dist_bb[i, j] = np.linalg.norm(self.bs_pos[i] - self.bs_pos[j])
        #print(self.bs_pos)
        #print(self.dist_bb)
        #  calc_gain
        for i in range(self.n_user):
            for j in range(self.n_bs):
                if j == 0:
                    theta = (180 / math.pi) * math.asin(0 / self.dist_ub[i, j])
                    P_LoS = 1 / (self.c1 * math.exp(-self.c2 * (theta - self.c1)))
                    alpha = pow(4 * math.pi * self.fc / self.c * self.dist_ub[i, j], self.exponent)
                    PL_LoS = alpha * self.beta_LoS
                    PL_NLoS = alpha * self.beta_NLoS
                    self.gain_ub[i, j] = P_LoS * PL_LoS + (1 - P_LoS) * PL_NLoS
                else:
                    theta = (180 / math.pi) * abs(math.asin(self.height / self.dist_ub[i, j]))
                    P_LoS = 1 / (self.c1 * math.exp(-self.c2 * (theta - self.c1)))
                    alpha = pow(4 * math.pi * self.fc / self.c * self.dist_ub[i, j], self.exponent)
                    PL_LoS = alpha * self.beta_LoS
                    PL_NLoS = alpha * self.beta_NLoS
                    self.gain_ub[i, j] = P_LoS * PL_LoS + (1 - P_LoS) * PL_NLoS

        for i in range(self.n_user):
            for j in range(self.n_user):
                theta = (180 / math.pi) * math.asin(0)
                P_LoS = 1 / (self.c1 * math.exp(-self.c2 * (theta - self.c1)))
                alpha = pow(4 * math.pi * self.fc / self.c * self.dist_uu[i, j], self.exponent)
                PL_LoS = alpha * self.beta_LoS
                PL_NLoS = alpha * self.beta_NLoS
                self.gain_uu[i, j] = P_LoS * PL_LoS + (1 - P_LoS) * PL_NLoS

        for i in range(self.n_bs):
            for j in range(self.n_bs):
                theta = (180 / math.pi) * math.asin(0)
                P_LoS = 1 / (self.c1 * math.exp(-self.c2 * (theta - self.c1)))
                alpha = pow(4 * math.pi * self.fc / self.c * self.dist_bb[i, j], self.exponent)
                PL_LoS = alpha * self.beta_LoS
                PL_NLoS = alpha * self.beta_NLoS
                self.gain_bb[i, j] = P_LoS * PL_LoS + (1 - P_LoS) * PL_NLoS

        # print('gain_bb',self.gain_bb)
        # print('gain_ub', self.gain_ub)
        # print('gain_uu', self.gain_uu)

        for i in range(self.n_bs - 1):
            self.uav_backhaul_rate[i] = 150 * math.log2(
                    1 + abs(self.max_power_MBS * self.gain_bb[0, i + 1] / self.noise))

        for k in range(self.n_bs):  # 上行时其他下行用户对基站的干扰
            for i in range(self.n_user):
                self.interf_ub[k, i] = self.power_user * self.gain_ub[i, k]
                self.interf_ub_sum[k] = self.interf_ub_sum[k] + self.interf_ub[k, i]
            for i in range(self.n_bs):
                if i != k:
                    self.interf_bb[k, i] = self.power_bs[i] * self.gain_bb[k, i]
                    self.interf_bb_sum[k] = self.interf_bb_sum[k] + self.interf_bb[k, i]
        for k in range(self.n_bs):  # 上行时基站受到的总干扰
            self.interf_b_sum[k] = self.interf_ub_sum[k] + self.interf_bb_sum[k] + (
                        self.power_bs[k] / self.SIC) + self.noise

        #  calc_interference  Downlink
        for k in range(self.n_user):  # 下行时其他上行用户对用户的干扰
            for i in range(self.n_user):
                if k != i:
                    self.interf_uu[k, i] = self.power_user * self.gain_uu[k, i]
                    self.interf_uu_sum[k] = self.interf_uu_sum[k] + self.interf_uu[k, i]

            for i in range(self.n_bs):
                self.interf_bu[k, i] = self.power_bs[i] * self.gain_ub[k, i]
                self.interf_bu_sum[k] = self.interf_bu_sum[k] + self.interf_bu[k, i]

        for k in range(self.n_user):  # 下行时用户受到的总干扰
            self.interf_u_sum[k] = self.interf_bu_sum[k] + self.interf_uu_sum[k] + (self.power_user / self.SIC) + self.noise

        for k in range(self.n_user):
            if np.isnan(up[k]) or np.isnan(down[k]):
                print("!!!")
                up[k] = int(1)
                down[k] = int(1)
            self.rate_up[k] = math.log2(
                1 + abs(self.power_user * self.gain_ub[k, int(up[k])] / self.interf_b_sum[int(up[k])]))
            self.rate_down[k] = math.log2(
                1 + abs(self.power_bs[int(down[k])] * self.gain_ub[k, int(down[k])] / self.interf_u_sum[k]))
            self.rate[k] = self.rate_up[k] + self.rate_down[k]

        for k in range(self.n_user):
            self.up_data[k] -= self.rate_up[k] * self.time * self.bandwidth
            self.down_data[k] -= self.rate_down[k] * self.time * self.bandwidth

        self.up_data[self.up_data < 0] = 0
        self.down_data[self.down_data < 0] = 0

        self.up_active_links[np.multiply(self.up_active_links, self.up_data <= 0)] = 0  #  transmission finished, turned to "inactive"
        self.down_active_links[np.multiply(self.down_active_links, self.down_data <= 0)] = 0

        punish1 = 0
        punish2 = 0
        punish3 = 0
        for i in range(self.n_bs - 1):
            uav_backhaul_rate_judge = 0
            for k in range(self.n_user):
                if up[k] == i + 1:
                    uav_backhaul_rate_judge += self.rate_up[k] * self.bandwidth
                if down[k] == i + 1:
                    uav_backhaul_rate_judge += self.rate_up[k] * self.bandwidth
            if uav_backhaul_rate_judge > self.uav_backhaul_rate[i]:
                #print(uav_backhaul_rate_judge, self.uav_backhaul_rate[i])
                punish1 = 50

        for i in range(self.n_uav):
            if self.bs_pos[i+1][0] < -400 or self.bs_pos[i+1][0] > 400 or self.bs_pos[i+1][1] < -400 or self.bs_pos[i+1][1] > 400:
                #print(self.bs_pos[i+1])
                punish2 = 500

        for i in range(self.n_uav):
            if self.power_bs[i+1] < self.max_power_UAV:
                for k in range(self.n_user):
                    if self.power_user < self.max_power_user:  # 此段代码由于self.power_user   [k+1]  前面定义是一个数，而不是列表，所以报错
                        punish3 = 50

        reward_up = self.rate_up.copy() * 10

        reward_down = self.rate_down.copy()

        return np.sum(reward_up) + np.sum(reward_down)

    def action_base(self):
        #  calc_distance
        for i in range(self.n_user):
            for j in range(self.n_bs):
                self.dist_ub[i, j] = np.linalg.norm(self.user_pos[i] - self.bs_pos[j])

        for i in range(self.n_user):
            for j in range(self.n_user):
                self.dist_uu[i, j] = np.linalg.norm(self.user_pos[i] - self.user_pos[j])

        for i in range(self.n_bs):
            for j in range(self.n_bs):
                self.dist_bb[i, j] = np.linalg.norm(self.bs_pos[i] - self.bs_pos[j])
        # print(self.bs_pos)
        # print(self.dist_bb)
        #  calc_gain
        for i in range(self.n_user):
            for j in range(self.n_bs):
                if j == 0:
                    theta = (180 / math.pi) * math.asin(0 / self.dist_ub[i, j])
                    P_LoS = 1 / (self.c1 * math.exp(-self.c2 * (theta - self.c1)))
                    alpha = pow(4 * math.pi * self.fc / self.c * self.dist_ub[i, j], self.exponent)
                    PL_LoS = alpha * self.beta_LoS
                    PL_NLoS = alpha * self.beta_NLoS
                    self.gain_ub[i, j] = P_LoS * PL_LoS + (1 - P_LoS) * PL_NLoS
                else:
                    theta = (180 / math.pi) * abs(math.asin(self.height / self.dist_ub[i, j]))
                    P_LoS = 1 / (self.c1 * math.exp(-self.c2 * (theta - self.c1)))
                    alpha = pow(4 * math.pi * self.fc / self.c * self.dist_ub[i, j], self.exponent)
                    PL_LoS = alpha * self.beta_LoS
                    PL_NLoS = alpha * self.beta_NLoS
                    self.gain_ub[i, j] = P_LoS * PL_LoS + (1 - P_LoS) * PL_NLoS
        action = np.zeros(self.n_user*2)
        for i in range(self.n_user):
            a = self.gain_ub[i].copy()[1:5]
            b = np.where(a==np.max(a))
            c = b[0][0] + 1
            if c == 1:
                if random.random() > 0.05:
                    action[i] = random.uniform(-0.59, -0.21)
                else:
                    action[i] = random.uniform(-1.0, 1.0)
            elif c == 2:
                if random.random() > 0.05:
                    action[i] = random.uniform(-0.19, 0.19)
                else:
                    action[i] = random.uniform(-1.0, 1.0)
            elif c == 3:
                if random.random() > 0.05:
                    action[i] = random.uniform(0.21, 0.59)
                else:
                    action[i] = random.uniform(-1.0, 1.0)
            elif c == 4:
                if random.random() > 0.05:
                    action[i] = random.uniform(0.61, 0.99)
                else:
                    action[i] = random.uniform(-1.0, 1.0)

        for j in range(self.n_bs):
            if j > 0:
                a = self.gain_ub[:,j].copy()
                b = heapq.nlargest(4, range(len(a)), a.take)
                c = j
                for i in range(len(b)):
                    if c == 1:
                        if random.random() > 0.05:
                            action[b[i]+20] = random.uniform(-0.59, -0.21)
                        else:
                            action[b[i]+20] = random.uniform(-1.0, 1.0)
                    elif c == 2:
                        if random.random() > 0.05:
                            action[b[i]+20] = random.uniform(-0.19, 0.19)
                        else:
                            action[b[i]+20] = random.uniform(-1.0, 1.0)
                    elif c == 3:
                        if random.random() > 0.05:
                            action[b[i]+20] = random.uniform(0.21, 0.59)
                        else:
                            action[b[i]+20] = random.uniform(-1.0, 1.0)
                    elif c == 4:
                        if random.random() > 0.05:
                            action[b[i]+20] = random.uniform(0.61, 0.99)
                        else:
                            action[b[i]+20] = random.uniform(-1.0, 1.0)
        for i in range(self.n_user * 2):
            if action[i] == 0:
                action[i] = random.uniform(-1.0, 1.0)

        return action



    def render(self, i_step):

        import matplotlib.pyplot as plt
        plt.scatter(self.bs_pos[1,0], self.bs_pos[1,1], c='#00CED1', alpha=0.4, label='UAV1')
        plt.scatter(self.bs_pos[2,0], self.bs_pos[2,1], c='#800080', alpha=0.4, label='UAV2')
        plt.scatter(self.bs_pos[3,0], self.bs_pos[3,1], c='#008B8B', alpha=0.4, label='UAV3')
        plt.scatter(self.bs_pos[4,0], self.bs_pos[4,1], c='#2E8B57', alpha=0.4, label='UAV4')

        plt.scatter(self.user_pos[:,0], self.user_pos[:,1], c='#DC143C', alpha=0.4, label='USER')
        plt.title("UAV"+str(i_step))
        plt.xlabel("X", fontsize=10)
        plt.ylabel("Y", fontsize=10)
        plt.xlim(-500, 500)
        plt.ylim(-500, 500)
        plt.tick_params(axis='both', labelsize=9)
        plt.show()

    def judge_episode_end(self, state):
        judge = 1
        for i in range(40):
            if state[i] != 0:
                judge = 0
        return judge

    def observe_uav(self, i):
        bs_pos1 = self.bs_pos.copy()
        user_pos1 = self.user_pos.copy()
        bs_pos1 = (bs_pos1 + 500) / 1000
        user_pos1 = (user_pos1 + 500) / 1000
        return np.concatenate((np.reshape(bs_pos1[1:5][:,0:2], -1), np.reshape(user_pos1[:,0:2], -1)))

    def cal_position_record(self, x, y):
        xx = int((x + 400)/10)
        yy = int((y + 400)/10)
        if xx >= 80:
            xx = 79
        if yy >= 80:
            yy = 79
        return xx, yy

# for i in range(10):
#     np.random.seed(10)
#     env = Environment()
#     i = 0
#     while i < 10:
#         actions_all = [[0,30,2],[20,60,2],[0,90,2],[0,270,2]]
#         env.update_position(actions_all)
#         #print(env.gain_ub[0])
#         re = env.calc_reward(actions_all)
#         #print(re)
#         i += 1
#     jj = env.observe_uav(1)
#     print(jj)




