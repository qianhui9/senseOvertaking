import torch
import math
import numpy as np

# 定义了一系列的函数来计算离散动作和概率分布的相关操作

# 根据给定的连续动作，计算对应的离散动作
# 将每个连续动作值映射到对应的离散动作值
# 该函数将连续动作映射到40个离散动作
def discrete_action_user_coupled(action):
    action1 = np.zeros(40)
    for i in range(20):
        action1[i] = action_value_user(action[i])
        action1[i+20] = action_value_user(action[i])
    return action1

# 类似于上面的函数，将连续动作映射到40个离散动作
def discrete_action_user(action):
    action1 = np.zeros(40)
    for i in range(40):
        action1[i] = action_value_user(action[i])
    return action1

# 根据给定的连续动作值，计算对应的离散动作值
# 该函数定义了不同连续动作值范围对应的离散动作值
def action_value_user(a):
    if a < 1 and a >= 0.6:
        return 4
    elif a < 0.6 and a >= 0.2:
        return 3
    elif a < 0.2 and a >= -0.2:
        return 2
    elif a < -0.2 and a >= -0.6:
        return 1
    else:
        return 0

# 将连续动作分解为速度、方向和功率三个离散动作
# 分别调用对应的函数来进行映射
def discrete_action(action):
    # print(action)
    action1 = np.zeros(3)
    action1[0] = action_value_direction(action[0])
    action1[1] = action_value_speed(action[1])
    action1[2] = action_value_speed(action[2])
    return action1

# 速度动作
def action_value_speed(a):
    v_n = (np.tanh(a) + 1) * 10
    return v_n

# 转向动作
def action_value_direction(a):
    d = np.tanh(a)
    if d <= -0.5:
        d = -1
    elif d < 0.5 and d > 0.5:
        d = 0
    else:
        d = 1

    return d


# 根据给定的连续动作值，计算对应的功率离散动作值
# 分段函数，根据不同连续动作值范围映射到不同离散动作值
def action_value_power(a):
    if a >= -1.0 and a <= 1.0:
        return a + 1
    elif a < -1.0:
        return 0
    elif a > 1.0:
        return 2.0

# 从给定的均值和标准差中生成一个动作样本
# 采用正态分布生成随机样本，并将样本转换为NumPy数组返回
def get_action(mu, std):
    # 从一个正态分布中生成一个动作样本
    action = torch.normal(mu, std)
    # 将上面生成的张量转换为 NumPy 数组。 .data 属性用于获取张量的数据内容
    action = action.data.numpy()
    return action

# 计算给定样本在给定均值、标准差和对数标准差下的对数概率密度
# 使用正态分布的概率密度函数计算对数概率密度
# 返回的是对数概率密度的和，keepdim=True保持维度不变
def log_density(x, mu, std, logstd):
    var = std.pow(2)
    log_density = -(x - mu).pow(2) / (2 * var) \
                  - 0.5 * math.log(2 * math.pi) - logstd
    return log_density.sum(1, keepdim=True)

# 将给定列表中的梯度张量展平为一个一维向量
# 对每个梯度张量调用view函数，然后将它们拼接起来
# 返回的是展平后的一维向量
def flat_grad(grads):
    grad_flatten = []
    for grad in grads:
        grad_flatten.append(grad.view(-1))
    grad_flatten = torch.cat(grad_flatten)
    return grad_flatten

# 将给定列表中的Hessian矩阵展平为一个一维向量
# 对每个Hessian矩阵调用view函数，然后将它们拼接起来
# 返回的是展平后的一维向量
def flat_hessian(hessians):
    hessians_flatten = []
    for hessian in hessians:
        hessians_flatten.append(hessian.contiguous().view(-1))
    hessians_flatten = torch.cat(hessians_flatten).data
    return hessians_flatten

# 将给定模型的参数展平为一个一维向量
# 对每个参数调用view函数，然后将它们拼接起来
# 返回的是展平后的一维向量
def flat_params(model):
    params = []
    for param in model.parameters():
        params.append(param.data.view(-1))
    params_flatten = torch.cat(params)
    return params_flatten

# 更新给定模型的参数为新的参数
# 遍历模型的每个参数，将新参数复制到对应的参数中
# 将新的参数更新到给定的模型中。它遍历模型的每个参数，从 new_params 中提取对应长度的参数数据，并将其复制到模型参数中，以实现模型参数的更新。
def update_model(model, new_params):
    index = 0
    for params in model.parameters():
        params_length = len(params.view(-1))
        new_param = new_params[index: index + params_length]
        new_param = new_param.view(params.size())
        params.data.copy_(new_param)
        index += params_length

# 计算了两个概率分布（新策略和旧策略）之间的 KL 散度（Kullback-Leibler divergence）。KL 散度衡量了两个概率分布之间的差异
# 首先计算新旧策略的均值（mu）、标准差（std）和对数标准差（logstd）。然后，计算了 KL 散度的各个部分，包括对数标准差的差异、方差项和均值项。
# 最后，将计算得到的 KL 散度进行求和，以得到一个汇总的度量
def kl_divergence(new_actor, old_actor, states):
    mu, std, logstd = new_actor(torch.Tensor(states))
    mu_old, std_old, logstd_old = old_actor(torch.Tensor(states))
    mu_old = mu_old.detach()
    std_old = std_old.detach()
    logstd_old = logstd_old.detach()

    # kl divergence between old policy and new policy : D( pi_old || pi_new )
    # pi_old -> mu0, logstd0, std0 / pi_new -> mu, logstd, std
    # be careful of calculating KL-divergence. It is not symmetric metric
    kl = logstd_old - logstd + (std_old.pow(2) + (mu_old - mu).pow(2)) / \
         (2.0 * std.pow(2)) - 0.5
    return kl.sum(1, keepdim=True)


