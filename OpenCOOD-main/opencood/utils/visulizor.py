import numpy as np
import matplotlib.pyplot as plt


# 定义绘制矩形框的函数
def draw_box_plt(boxes_dec, ax, color=None, linewidth_scale=1.0):
    """
    在给定的matplotlib坐标系中绘制3D边界框的2D投影（俯视图）
    :param boxes_dec: (N,5)或(N,7)矩阵，包含边界框参数（中心点坐标，尺寸，旋转角等）
    :param ax: matplotlib的坐标轴对象
    :param color: 框线颜色
    :param linewidth_scale: 线宽缩放因子
    :return: 添加了框线的坐标轴对象
    """
    # 检查输入是否为空
    if not len(boxes_dec)>0:
        return ax

    # 转换输入为numpy数组格式
    boxes_np= boxes_dec
    if not isinstance(boxes_np, np.ndarray):
        boxes_np = boxes_np.cpu().detach().numpy()   # 如果输入是PyTorch张量则转换

    # 处理不同维度的输入（适配不同数据格式）
    if boxes_np.shape[-1]>5:
        boxes_np = boxes_np[:, [0, 1, 3, 4, 6]]  # 选择需要的特征维度（x,y,dx,dy,yaw）
    # 解包边界框参数
    x = boxes_np[:, 0]  # 中心点x坐标
    y = boxes_np[:, 1]  # 中心点y坐标
    dx = boxes_np[:, 2]   # x轴方向长度
    dy = boxes_np[:, 3]  # y轴方向长度

    # 计算未旋转时的四个角点坐标
    x1 = x - dx / 2  # 左侧x坐标
    y1 = y - dy / 2   # 下侧y坐标
    x2 = x + dx / 2  # 右侧x坐标
    y2 = y + dy / 2   # 上侧y坐标
    theta = boxes_np[:, 4:5]  # 旋转角度（yaw）
    # bl, fl, fr, br
    # 构建初始角点矩阵（未旋转状态）
    corners = np.array([[x1, y1],[x1,y2], [x2,y2], [x2, y1]]).transpose(2, 0, 1)
    # 应用旋转矩阵进行坐标变换（绕z轴旋转）
    new_x = (corners[:, :, 0] - x[:, None]) * np.cos(theta) + (corners[:, :, 1]
              - y[:, None]) * (-np.sin(theta)) + x[:, None]
    new_y = (corners[:, :, 0] - x[:, None]) * np.sin(theta) + (corners[:, :, 1]
              - y[:, None]) * (np.cos(theta)) + y[:, None]
    # 重新组合旋转后的角点坐标
    corners = np.stack([new_x, new_y], axis=2)

    # 绘制每个边界框
    for corner in corners:
        # 绘制四周边框（闭合多边形）   # 连接顺序：0->1->2->3->0
        ax.plot(corner[[0,1,2,3,0], 0], corner[[0,1,2,3,0], 1], color=color, linewidth=0.5*linewidth_scale)
        # 强调绘制前边线（索引2到3的边）
        ax.plot(corner[[2, 3], 0], corner[[2, 3], 1], color=color, linewidth=2*linewidth_scale)
    return ax

# 定义绘制点云和边界框对比图的函数
def draw_points_pred_gt_boxes_plt_2d(pc_range, points=None, boxes_pred=None, boxes_gt=None):
    # 创建图形窗口和坐标轴
    ax = plt.figure(figsize=(14, 4)).add_subplot(1, 1, 1)
    ax.set_aspect('equal', 'box')  # 设置等比例坐标轴
    # 设置坐标范围（来自点云范围参数）
    ax.set(xlim=(pc_range[0], pc_range[3]),
           ylim=(pc_range[1], pc_range[4]))

    # 绘制点云数据（如果存在）
    if points is not None:
        ax.plot(points[:, 0], points[:, 1], 'y.', markersize=0.3)  # 黄色小点

    # 绘制真实框（绿色）
    if (boxes_gt is not None) and len(boxes_gt)>0:
        ax = draw_box_plt(boxes_gt, ax, color='green')

    # 绘制预测框（红色）
    if (boxes_pred is not None) and len(boxes_pred)>0:
        ax = draw_box_plt(boxes_pred, ax, color='red')

    # 添加坐标轴标签
    plt.xlabel('x')
    plt.ylabel('y')

    # 显示并关闭图形（防止内存泄漏）
    plt.show()
    plt.close()

# 定义通用绘图函数（可在现有坐标系上叠加内容）
def draw_points_boxes_plt_2d(ax, pc_range, points=None, boxes=None, color=None):
    # 绘制点云（可指定颜色）
    if points is not None:
        ax.plot(points[:, 0], points[:, 1], '.', markersize=0.3, color=color)
    # 绘制边界框（可指定颜色）
    if (boxes is not None) and len(boxes)>0:
        ax = draw_box_plt(boxes, ax, color=color)

    return ax