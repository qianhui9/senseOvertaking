import torch
import torch.nn as nn
import torch.nn.functional as F
from hparams import HyperParams as hp


class Actor(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(num_inputs, hp.hidden)
        self.fc2 = nn.Linear(hp.hidden, hp.hidden)
        self.fc3 = nn.Linear(hp.hidden, num_outputs)
        # 对第三个全连接层的权重进行初始化，将其乘以0.1
        self.fc3.weight.data.mul_(0.1)
        # 对第三个全连接层的偏置进行初始化，将其乘以0.0
        self.fc3.bias.data.mul_(0.0)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # 将上一层的输出 x 经过第三个全连接层，得到均值 mu，表示策略网络输出的动作均值
        mu = self.fc3(x)
        # 创建一个与 mu 维度相同的张量，用值 -1 填充，表示对数标准差的初始值
        logstd = torch.zeros_like(mu) - 1
        # 计算标准差 std，通过对数标准差的指数化得到
        std = torch.exp(logstd)
        # 返回均值 mu、标准差 std 和对数标准差 logstd，这些值用于构建正态分布动作概率
        return mu, std, logstd


class Critic(nn.Module):
    def __init__(self, num_inputs):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(num_inputs, hp.hidden)
        self.fc2 = nn.Linear(hp.hidden, hp.hidden)
        self.fc3 = nn.Linear(hp.hidden, 1)
        self.fc3.weight.data.mul_(0.1)
        self.fc3.bias.data.mul_(0.0)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        v = self.fc3(x)
        # 返回 Critic 网络的值函数输出
        return v