import numpy as np
import torch

from MARL.utils import log_density
from utils import *
from hparams import HyperParams as hp


# 计算广义优势估计（Generalized Advantage Estimation, GAE）和回报值（returns）
# mask：表示是否是终止状态；svalues：值函数的估计值
def get_gae(rewards, masks, values):
    rewards = torch.Tensor(rewards)
    masks = torch.Tensor(masks)
    returns = torch.zeros_like(rewards) # 回报值
    advants = torch.zeros_like(rewards) # 优势估计

    running_returns = 0
    previous_value = 0
    running_advants = 0

    # 从最后一个时间步 t 反向遍历 rewards 和 masks
    for t in reversed(range(0, len(rewards))):
        # 表示累积的回报值
        running_returns = rewards[t] + hp.gamma * running_returns * masks[t]
        # 表示时序差分误差
        running_tderror = rewards[t] + hp.gamma * previous_value * masks[t] - \
                    values.data[t]
        # 表示累积的优势估计
        running_advants = running_tderror + hp.gamma * hp.lamda * \
                          running_advants * masks[t]

        # 更新 returns 和 previous_value
        returns[t] = running_returns
        previous_value = values.data[t]
        advants[t] = running_advants

    # 将 advants 标准化
    advants = (advants - advants.mean()) / advants.std()
    return returns, advants

# 计算 surrogate 损失和比率
def surrogate_loss(actor, advants, states, old_policy, actions, index):
    # 通过 actor 网络计算给定状态 states 的均值 mu、标准差 std 和对数标准差 logstd
    mu, std, logstd = actor(torch.Tensor(states))
    # 调用 log_density 函数计算新策略的对数概率
    new_policy = log_density(actions, mu, std, logstd)
    # 从 old_policy 中提取对应索引 index 处的对数概率
    old_policy = old_policy[index]

    # 计算比率 ratio，用于重要性采样
    ratio = torch.exp(new_policy - old_policy)
    # 计算 surrogate 损失，即 ratio 与 advants 的乘积
    surrogate = ratio * advants
    return surrogate, ratio

# 用于训练 Actor-Critic 模型
def train_model(actor, critic, memory, actor_optim, critic_optim):
    # 将 memory 转换为 NumPy 数组，并从中提取状态、动作、奖励、终止标志等信息
    # memory = np.array(memory)
    # states = np.vstack(memory[:, 0])
    # actions = list(memory[:, 1])
    # rewards = list(memory[:, 2])
    # masks = list(memory[:, 3])
    states = [item[0] for item in memory]
    actions = [item[1] for item in memory]
    rewards = [item[2] for item in memory]
    masks = [item[3] for item in memory]

    # 使用 critic 网络计算状态对应的值函数估计
    values = critic(torch.Tensor(states))

    # ----------------------------
    # step 1: get returns and GAEs and log probability of old policy
    # 调用 get_gae 函数计算回报值 returns 和广义优势估计 advants
    returns, advants = get_gae(rewards, masks, values)
    # 通过 actor 网络计算新策略的均值 mu、标准差 std 和对数标准差 logstd
    mu, std, logstd = actor(torch.Tensor(states))
    # 计算旧策略的对数概率 old_policy
    old_policy = log_density(torch.Tensor(actions), mu, std, logstd)

    # 使用均方误差损失函数计算值函数的损失
    criterion = torch.nn.MSELoss()
    n = len(states)
    arr = np.arange(n)

    # ----------------------------
    # step 2: get value loss and actor loss and update actor & critic
    # 在一个小批次上迭代进行更新（此处共进行 10 次迭代）
    for epoch in range(10):
        # 对数据进行随机洗牌
        np.random.shuffle(arr)

        for i in range(n // hp.batch_size):  # 将数据分成小批次，每批大小为 hp.batch_size
            batch_index = arr[hp.batch_size * i: hp.batch_size * (i + 1)]
            batch_index = torch.LongTensor(batch_index)
            inputs = torch.Tensor(states)[batch_index]
            returns_samples = returns.unsqueeze(1)[batch_index]
            advants_samples = advants.unsqueeze(1)[batch_index]
            actions_samples = torch.Tensor(actions)[batch_index]

            # 计算 surrogate_loss，获得 surrogate 损失和比率
            loss, ratio = surrogate_loss(actor, advants_samples, inputs,
                                         old_policy.detach(), actions_samples,
                                         batch_index)

            values = critic(inputs)
            # 采用随机梯度下降法更新值函数（critic）网络参数
            critic_loss = criterion(values, returns_samples)
            critic_optim.zero_grad()
            critic_loss.backward()
            critic_optim.step()
            # print(type(hp.clip_param))
            # 使用 clipped_ratio 对比率进行截断
            clipped_ratio = torch.clamp(ratio,
                                        1.0 - hp.clip_param,
                                        1.0 + hp.clip_param)
            #print(clipped_ratio)
            clipped_loss = clipped_ratio * advants_samples

            # 使用随机梯度下降法更新 actor 网络参数
            actor_loss = -torch.min(loss, clipped_loss).mean()

            actor_optim.zero_grad()
            actor_loss.backward()
            actor_optim.step()







