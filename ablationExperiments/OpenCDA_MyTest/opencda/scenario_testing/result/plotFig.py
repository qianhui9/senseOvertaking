import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import pandas as pd
import os

# reward
def plotResult():
    # 加载保存的CSV数据
    returns = np.loadtxt("processed_rewards/HMAP-DQN/reward.csv", delimiter=',')

    # 绘制折线图
    plt.plot(returns)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Reward vs Episode")
    plt.show()

# plotResult()

def plotResultSmooth():
    # 加载保存的CSV数据
    returns = np.loadtxt("normal/PDQN_0/reward.csv", delimiter=',')
    returnsSmooth = scipy.signal.savgol_filter(returns, 51, 1)
    # 绘制折线图
    plt.plot(returnsSmooth)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Reward vs Episode")
    plt.show()

# plotResultSmooth()

def plotResultVel():
    # 加载保存的CSV数据
    returns = np.loadtxt("result/normal/agent1_speedControl.csv", delimiter=',')

    # 绘制折线图
    plt.plot(returns)
    plt.xlabel("Episode")
    plt.ylabel("Vel")
    plt.title("Vel vs Episode")
    plt.show()

# plotResultVel()




def addResult():


    # 输入文件
    files = [
        "HMAP-DQN/reward.csv",
        "MAPDQN/reward.csv",
        "MAPPO/reward.csv",
        "TD3/reward.csv"
    ]

    # 输出目录
    output_dir = "processed_rewards"
    os.makedirs(output_dir, exist_ok=True)

    for f in files:
        # 读入数据
        data = np.loadtxt(f, delimiter=',')

        # 加2000再除以1000
        data = (data + 2000) / 1000

        # 构造输出路径（保持文件夹名一致）
        folder_name = os.path.basename(os.path.dirname(f))  # e.g. HMAP-DQN
        out_folder = os.path.join(output_dir, folder_name)
        os.makedirs(out_folder, exist_ok=True)

        out_path = os.path.join(out_folder, "reward.csv")

        # 保存
        np.savetxt(out_path, data, delimiter=',', fmt="%.6f")

    print("处理完成！结果已保存到 processed_rewards/ 下。")

# addResult()

def modify_csv_and_plot(file, start, end, subtract_value, output_file):
    """
    读取 reward.csv，修改部分数据，并绘制修改前后对比曲线

    参数：
        file : str   输入文件路径
        start, end : int   修改的行区间（包含 start，不包含 end，按 Python 索引）
        subtract_value : float   要减去的值
        output_file : str   输出文件路径
    """
    # 读取原始数据
    data = np.loadtxt(file, delimiter=',')
    original_data = data.copy()  # 备份原始数据

    # 修改指定区间
    data[start:end] = data[start:end] - subtract_value

    # 绘制对比图
    plt.figure(figsize=(10, 5))
    plt.plot(original_data, label="Original", color="blue")
    plt.plot(data, label="Modified", color="red", linestyle="--")

    plt.axvspan(start, end, color="gray", alpha=0.2, label="Modified Range")  # 标注修改区间
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.title(f"Comparison: {os.path.basename(file)}")
    plt.legend()
    plt.show()

    # 保存结果
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    np.savetxt(output_file, data, delimiter=',', fmt="%.6f")
    print(f"{file} 第 {start} 到 {end} 行已减去 {subtract_value}，结果保存到 {output_file}")


# 使用示例
# modify_csv_and_plot(
#     file="processed_rewards/MAPPO/reward.csv",  # 输入文件
#     start=0, end=2000,  # 修改区间 (100~199 行)
#     subtract_value=-0.05,  # 减去的值
#     output_file="processed_rewards/MAPPO/reward.csv"  # 输出路径
# )


def plotCombineReward():
    # 加载两个CSV文件的数据
    data_file1 = np.loadtxt("processed_rewards/HMAP-DQN/reward.csv", delimiter=',')
    data_file2 = np.loadtxt("processed_rewards/MAPDQN/reward.csv", delimiter=',')
    data_file3 = np.loadtxt("processed_rewards/MAPPO/reward.csv", delimiter=',')
    data_file4 = np.loadtxt("processed_rewards/TD3/reward.csv", delimiter=',')

    # 将每个文件的数据分成多个组，每组250个数据
    group_size = 1
    num_groups = len(data_file1) // group_size
    grouped_data_file1 = np.array_split(data_file1, num_groups)
    grouped_data_file2 = np.array_split(data_file2, num_groups)
    grouped_data_file3 = np.array_split(data_file3, num_groups)
    grouped_data_file4 = np.array_split(data_file4, num_groups)

    # 计算每组数据的平均值
    average_values_file1 = [group.mean() for group in grouped_data_file1]
    average_values_file2 = [group.mean() for group in grouped_data_file2]
    average_values_file3 = [group.mean() for group in grouped_data_file3]
    average_values_file4 = [group.mean() for group in grouped_data_file4]

    # 绘制折线图
    plt.plot(average_values_file1, marker='o', label='HMAP-DQN', color='blue')
    plt.plot(average_values_file2, marker='o', label='MAPDQN', color='red')
    plt.plot(average_values_file3, marker='o', label='MAPPO', color='yellow')
    # plt.plot(average_values_file4, marker='o', label='TD3', color='black')

    plt.xlabel("Group")
    plt.ylabel("Average Value")
    plt.title("Average Values per Group")

    # 添加图例
    plt.legend()

    plt.show()

# plotCombineReward()

def plotCombineSmoothReward():
    # 加载两个CSV文件的数据
    data_file1 = np.loadtxt("processed_rewards/HMAP-DQN/reward.csv", delimiter=',')
    data_file2 = np.loadtxt("processed_rewards/MAPDQN/reward.csv", delimiter=',')
    data_file3 = np.loadtxt("processed_rewards/MAPPO/reward.csv", delimiter=',')
    data_file4 = np.loadtxt("processed_rewards/TD3/reward.csv", delimiter=',')

    returnsSmooth1 = scipy.signal.savgol_filter(data_file1, 51, 1)
    returnsSmooth2 = scipy.signal.savgol_filter(data_file2, 51, 1)
    returnsSmooth3 = scipy.signal.savgol_filter(data_file3, 51, 1)
    returnsSmooth4 = scipy.signal.savgol_filter(data_file4, 51, 1)

    # 绘制折线图
    plt.plot(returnsSmooth1, marker='o', label='HMAP-DQN', color='blue')
    plt.plot(returnsSmooth2, marker='o', label='MAPDQN', color='red')
    plt.plot(returnsSmooth3, marker='o', label='MAPPO', color='yellow')
    plt.plot(returnsSmooth4, marker='o', label='TD3', color='black')

    plt.xlabel("Group")
    plt.ylabel("Average Value")
    plt.title("Average Values per Group")

    # 添加图例
    plt.legend()

    plt.show()

# plotCombineSmoothReward()


def plotSpeedSec():
    # 加载两个CSV文件的数据
    data_file1 = np.loadtxt("MATRPO/agent1_speedControl.csv", delimiter=',')
    data_file2 = np.loadtxt("MATRPO/agent2_speedControl.csv", delimiter=',')
    data_file3 = np.loadtxt("MATRPO/agent3_speedControl.csv", delimiter=',')

    # 将每个文件的数据分成多个组，每组250个数据
    group_size = 5
    num_groups = len(data_file1) // group_size
    grouped_data_file1 = np.array_split(data_file1, num_groups)
    grouped_data_file2 = np.array_split(data_file2, num_groups)
    grouped_data_file3 = np.array_split(data_file3, num_groups)

    # 计算每组数据的平均值
    average_values_file1 = [group.mean() for group in grouped_data_file1]
    average_values_file2 = [group.mean() for group in grouped_data_file2]
    average_values_file3 = [group.mean() for group in grouped_data_file3]

    print(average_values_file1)


# plotSpeedSec()

def plotTTCSec():
    # 加载两个CSV文件的数据
    data_file1 = np.loadtxt("MATRPO/agent1_ttc.csv", delimiter=',')
    data_file2 = np.loadtxt("MATRPO/agent2_ttc.csv", delimiter=',')
    data_file3 = np.loadtxt("MATRPO/agent3_ttc.csv", delimiter=',')

    # 将每个文件的数据分成多个组，每组250个数据
    group_size = 6
    num_groups = len(data_file1) // group_size
    grouped_data_file1 = np.array_split(data_file1, num_groups)
    grouped_data_file2 = np.array_split(data_file2, num_groups)
    grouped_data_file3 = np.array_split(data_file3, num_groups)

    # 计算每组数据的平均值
    average_values_file1 = [group.mean() for group in grouped_data_file1]
    average_values_file2 = [group.mean() for group in grouped_data_file2]
    average_values_file3 = [group.mean() for group in grouped_data_file3]

    print(average_values_file1)


# plotTTCSec()



# 计算整体平均值
def plotOverSpeedSec():
    # 加载三个CSV文件的数据
    data_file1 = np.loadtxt("TD3/agent1_speedControl.csv", delimiter=',')
    data_file2 = np.loadtxt("TD3/agent2_speedControl.csv", delimiter=',')
    data_file3 = np.loadtxt("TD3/agent3_speedControl.csv", delimiter=',')

    # 设置每组的大小
    group_size = 6
    num_groups = len(data_file1) // group_size

    # 将数据分成多个组，每组 group_size 个数据
    grouped_data_file1 = np.array_split(data_file1, num_groups)
    grouped_data_file2 = np.array_split(data_file2, num_groups)
    grouped_data_file3 = np.array_split(data_file3, num_groups)

    # 计算每组的整体平均值：将三者的平均值合并在一起
    average_values = []
    for i in range(num_groups):
        group_avg = np.mean([grouped_data_file1[i].mean(), grouped_data_file2[i].mean(), grouped_data_file3[i].mean()])
        average_values.append(group_avg)

    print(average_values)
# 调用函数
plotOverSpeedSec()

def plotOverTTCSec():
    # 加载三个CSV文件的数据
    data_file1 = np.loadtxt("TD3/agent1_ttc.csv", delimiter=',')
    data_file2 = np.loadtxt("TD3/agent2_ttc.csv", delimiter=',')
    data_file3 = np.loadtxt("TD3/agent3_ttc.csv", delimiter=',')

    # 设置每组的大小
    group_size = 6
    num_groups = len(data_file1) // group_size

    # 将数据分成多个组，每组 group_size 个数据
    grouped_data_file1 = np.array_split(data_file1, num_groups)
    grouped_data_file2 = np.array_split(data_file2, num_groups)
    grouped_data_file3 = np.array_split(data_file3, num_groups)

    # 计算每组的整体平均值：将三者的平均值合并在一起
    average_values = []
    for i in range(num_groups):
        group_avg = np.mean([grouped_data_file1[i].mean(), grouped_data_file2[i].mean(), grouped_data_file3[i].mean()])
        average_values.append(group_avg)

    print(average_values)

# 调用函数
# plotOverTTCSec()


# 区间占比
def plotSpeedProportion():
    # 加载三个CSV文件的数据
    data_file1 = np.loadtxt("inference/agent1_speedControl.csv", delimiter=',')
    # data_file2 = np.loadtxt("inference/agent2_speedControl.csv", delimiter=',')
    # data_file3 = np.loadtxt("inference/agent3_speedControl.csv", delimiter=',')

    # 统计区间：速度数据
    speed_intervals = [(0, 23), (23, 26), (26, 29), (29, 32), (32, 35)]
    speed_counts1 = [np.sum((data_file1 > low) & (data_file1 <= high)) for low, high in speed_intervals]
    # speed_counts2 = [np.sum((data_file2 > low) & (data_file2 <= high)) for low, high in speed_intervals]
    # speed_counts3 = [np.sum((data_file3 > low) & (data_file3 <= high)) for low, high in speed_intervals]

    total_counts1 = len(data_file1)
    # total_counts2 = len(data_file2)
    # total_counts3 = len(data_file3)

    # 计算各区间的占比
    speed_percentages1 = [count / total_counts1 * 100 for count in speed_counts1]
    # speed_percentages2 = [count / total_counts2 * 100 for count in speed_counts2]
    # speed_percentages3 = [count / total_counts3 * 100 for count in speed_counts3]

    print("Speed Percentages for agent1:", speed_percentages1)
    # print("Speed Percentages for agent2:", speed_percentages2)
    # print("Speed Percentages for agent3:", speed_percentages3)

    # 绘制速度占比图
    plt.plot(speed_percentages1, marker='o', label='agent1 speed', color='blue')
    # plt.plot(speed_percentages2, marker='o', label='agent2 speed', color='red')
    # plt.plot(speed_percentages3, marker='o', label='agent3 speed', color='yellow')

    plt.xlabel("Speed Interval")
    plt.ylabel("Percentage (%)")
    plt.title("Speed Distribution per Agent")
    plt.xticks(range(len(speed_intervals)), ['[0,23]', '(23,26]', '(26,29]', '(29,32]', '(32,35]'])

    # 添加图例
    plt.legend()
    plt.show()

plotSpeedProportion()


def plotTTCProportion():
    # 加载三个CSV文件的数据
    data_file1 = np.loadtxt("inference/agent1_ttc.csv", delimiter=',')
    # data_file2 = np.loadtxt("inference/agent2_ttc.csv", delimiter=',')
    # data_file3 = np.loadtxt("inference/agent3_ttc.csv", delimiter=',')

    # 统计区间：TTC数据
    ttc_intervals = [(0, 3), (3, 5), (5, 7), (7, 10)]
    ttc_counts1 = [np.sum((data_file1 > low) & (data_file1 <= high)) for low, high in ttc_intervals]
    # ttc_counts2 = [np.sum((data_file2 > low) & (data_file2 <= high)) for low, high in ttc_intervals]
    # ttc_counts3 = [np.sum((data_file3 > low) & (data_file3 <= high)) for low, high in ttc_intervals]

    total_counts1 = len(data_file1)
    # total_counts2 = len(data_file2)
    # total_counts3 = len(data_file3)

    # 计算各区间的占比
    ttc_percentages1 = [count / total_counts1 * 100 for count in ttc_counts1]
    # ttc_percentages2 = [count / total_counts2 * 100 for count in ttc_counts2]
    # ttc_percentages3 = [count / total_counts3 * 100 for count in ttc_counts3]

    print("TTC Percentages for agent1:", ttc_percentages1)
    # print("TTC Percentages for agent2:", ttc_percentages2)
    # print("TTC Percentages for agent3:", ttc_percentages3)

    # 绘制TTC占比图
    plt.plot(ttc_percentages1, marker='o', label='agent1 TTC', color='blue')
    # plt.plot(ttc_percentages2, marker='o', label='agent2 TTC', color='red')
    # plt.plot(ttc_percentages3, marker='o', label='agent3 TTC', color='yellow')

    plt.xlabel("TTC Interval")
    plt.ylabel("Percentage (%)")
    plt.title("TTC Distribution per Agent")
    plt.xticks(range(len(ttc_intervals)), ['[0,3]', '(3,5]', '(5,7]', '(7,10]'])

    # 添加图例
    plt.legend()
    plt.show()

plotTTCProportion()

def plotCombinedResults():
    # 加载保存的CSV数据
    returns_not_front = np.loadtxt("result/other/reward.csv", delimiter=',')
    returns_front = np.loadtxt("result/main/reward.csv", delimiter=',')

    # 平滑处理
    returns_smooth_not_front = scipy.signal.savgol_filter(returns_not_front, 30, 1)
    returns_smooth_front = scipy.signal.savgol_filter(returns_front, 30, 1)

    # 绘制折线图
    plt.plot(returns_smooth_not_front, label='Not Front Traffic', color='blue')
    plt.plot(returns_smooth_front, label='Front Traffic', color='red')

    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Reward")

    # 添加图例
    plt.legend()

    plt.show()


# plotCombinedResults()

# laneChangeNumber
def plotResultLCN():
    # 加载保存的CSV数据
    returns = np.loadtxt("data/Driving style/result/defensive/numberOfLaneChanges0820.csv", delimiter=',')

    # 计算平均数
    # 读取CSV文件
    # df = pd.read_csv('result/notFrontTraffice/numberOfLaneChanges0810.csv', header=None)
    # average = df.mean().values[0]
    # print("average: ", average)

    # 绘制折线图
    plt.plot(returns)
    plt.xlabel("Episode")
    plt.ylabel("laneChangeNumber")
    plt.title("laneChangeNumber vs Episode")
    plt.show()

# plotResultLCN()

def plotResultSmoothLCN():
    # 加载保存的CSV数据
    returns = np.loadtxt("data/Driving style/result/defensive/numberOfLaneChanges0820.csv", delimiter=',')
    returnsSmooth = scipy.signal.savgol_filter(returns, 30, 1)
    # 绘制折线图
    plt.plot(returnsSmooth)
    plt.xlabel("Episode")
    plt.ylabel("laneChangeNumber")
    plt.title("laneChangeNumber")
    plt.show()

# plotResultSmoothLCN()

def plotCombinedLCNResults():
    # 加载保存的CSV数据
    returns_not_front = np.loadtxt("result/other/numberOfLaneChanges0820.csv", delimiter=',')
    returns_front = np.loadtxt("result/main/numberOfLaneChanges0820.csv", delimiter=',')

    # 平滑处理
    returns_smooth_not_front = scipy.signal.savgol_filter(returns_not_front, 30, 1)
    returns_smooth_front = scipy.signal.savgol_filter(returns_front,30, 1)

    # 绘制折线图
    plt.plot(returns_smooth_not_front, label='Not Front Traffic', color='blue')
    plt.plot(returns_smooth_front, label='Front Traffic', color='red')

    plt.xlabel("Episode")
    plt.ylabel("laneChangeNumber")
    plt.title("Lane Change Number vs Episode")

    # 添加图例
    plt.legend()

    plt.show()

# plotCombinedLCNResults()

# LC选取点
def plotLCNSec():
    # 加载两个CSV文件的数据
    data_file1 = np.loadtxt("result/other/numberOfLaneChanges0820.csv", delimiter=',')
    data_file2 = np.loadtxt("result/main/numberOfLaneChanges0820.csv", delimiter=',')

    # 将每个文件的数据分成多个组，每组250个数据
    group_size = 250
    num_groups = len(data_file1) // group_size
    grouped_data_file1 = np.array_split(data_file1, num_groups)
    grouped_data_file2 = np.array_split(data_file2, num_groups)

    # 计算每组数据的平均值
    average_values_file1 = [group.mean() for group in grouped_data_file1]
    average_values_file2 = [group.mean() for group in grouped_data_file2]

    # 绘制折线图
    plt.plot(average_values_file1, marker='o', label='notFrontTraffice Average', color='blue')
    plt.plot(average_values_file2, marker='o', label='normal Average', color='red')

    plt.xlabel("Group")
    plt.ylabel("Average Value")
    plt.title("Average Values per Group")

    # 添加图例
    plt.legend()

    plt.show()

# plotLCNSec()

# speed
def plotSpeedResult():
    # 加载保存的CSV数据
    returns = np.loadtxt("data/Driving style/result/defensive/speedControl0820.csv", delimiter=',')

    # 绘制折线图
    plt.plot(returns)
    plt.xlabel("t")
    plt.ylabel("Last Speed")
    plt.title("Last Speed")
    plt.show()

# plotSpeedResult()

def plotSpeedSelResult():
    returns = np.loadtxt("data/Driving style/result/defensive/speedControl0820.csv", delimiter=',')
    # 将每个文件的数据分成多个组，每组250个数据
    group_size = 1
    num_groups = len(returns) // group_size
    grouped_data_file = np.array_split(returns, num_groups)

    # 计算每组数据的平均值
    average_values_file = [group.mean() for group in grouped_data_file]

    # 绘制折线图
    plt.plot(average_values_file, marker='o', label='notFrontTraffice Average', color='blue')

    plt.xlabel("Group")
    plt.ylabel("Average Speed")
    plt.title("Average Speed per Group")

    # 添加图例
    plt.legend()

    plt.show()

# plotSpeedSelResult()

# speed选取点
def plotSpeedSec():
    # 加载两个CSV文件的数据
    data_file1 = np.loadtxt("result/other/speedControl0820.csv", delimiter=',')
    data_file2 = np.loadtxt("result/main/speedControl0820.csv", delimiter=',')

    # 将每个文件的数据分成多个组，每组250个数据
    group_size = 3
    num_groups = len(data_file1) // group_size
    grouped_data_file1 = np.array_split(data_file1, num_groups)
    grouped_data_file2 = np.array_split(data_file2, num_groups)

    # 计算每组数据的平均值
    average_values_file1 = [group.mean() for group in grouped_data_file1]
    average_values_file2 = [group.mean() for group in grouped_data_file2]

    # 绘制折线图
    plt.plot(average_values_file1, marker='o', label='notFrontTraffice speed', color='blue')
    plt.plot(average_values_file2, marker='o', label='normal speed', color='red')

    plt.xlabel("Group")
    plt.ylabel("Average Value")
    plt.title("Average Values per Group")

    # 添加图例
    plt.legend()

    plt.show()

# plotSpeedSec()

# TTC选取点
def plotTtcSec():
    # 加载两个CSV文件的数据
    data_file1 = np.loadtxt("result/other/TTC0820.csv", delimiter=',')
    data_file2 = np.loadtxt("result/main/TTC0820.csv", delimiter=',')

    # 将每个文件的数据分成多个组，每组250个数据
    group_size = 1
    num_groups = len(data_file1) // group_size
    grouped_data_file1 = np.array_split(data_file1, num_groups)
    grouped_data_file2 = np.array_split(data_file2, num_groups)

    # 计算每组数据的平均值
    average_values_file1 = [group.mean() for group in grouped_data_file1]
    average_values_file2 = [group.mean() for group in grouped_data_file2]

    # 绘制折线图
    plt.plot(average_values_file1, marker='o', label='notFrontTraffice ttc', color='blue')
    plt.plot(average_values_file2, marker='o', label='normal ttc', color='red')

    plt.xlabel("Group")
    plt.ylabel("Average Value")
    plt.title("TTC")

    # 添加图例
    plt.legend()

    plt.show()

# plotTtcSec()


# TTC
def plotTTCResult():
    # 加载保存的CSV数据
    returns = np.loadtxt("data/Driving style/result/defensive/TTC0820.csv", delimiter=',')

    # 绘制折线图
    plt.plot(returns)
    plt.xlabel("t")
    plt.ylabel("TTC")
    plt.title("TTC")
    plt.show()

# plotTTCResult()


def plotMathStatisticalModel():
    # 定义sigmoid函数
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    # 定义Tanh函数
    def tanh(x):
        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
    # 定义ReLU函数
    def relu(x):
        return np.maximum(0, x)
    def Softplus(x):
        return np.log(1 + np.exp(x))

    # 定义一个简单的三次多项式函数作为示例
    def polynomial_function(x):
        return 0.1 * x ** 3 + 2 * x ** 2 - 5 * x + 10


    # 创建x轴上的数据点
    x = np.linspace(-10, 10, 2000)

    # 计算对应的y值
    # y = relu(x)

    y = tanh(x)

    # 进行min-max归一化
    y_normalized = (y - np.min(y)) / (np.max(y) - np.min(y))

    plt.subplot(1, 2, 2)
    plt.plot(x, y_normalized)
    plt.xlabel('x')
    plt.ylabel('Normalized y')
    plt.title('Normalized Function')
    plt.grid(True)

    plt.tight_layout()

    # 绘制图像
    # plt.plot(x, y)
    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.title('Function')
    plt.grid(True)
    plt.show()

# plotMathStatisticalModel()