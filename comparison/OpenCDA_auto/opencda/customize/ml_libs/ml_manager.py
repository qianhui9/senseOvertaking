# -*- coding: utf-8 -*-

"""
Since multiple CAV normally use the same ML/DL model,
here we have this class to enable different CAVs share the same model to
 avoid duplicate memory consumption.
 这个类的目的是让多个CAV(联网自动驾驶车辆)共享同一个ML/DL模型，避免内存重复消耗
"""

import cv2
import torch
import numpy as np

# 用于集中管理所有要初始化的机器学习模型；类属性object_detector：从PyTorch加载的YOLOv5检测器
class MLManager(object):
    """
    A class that should contain all the ML models you want to initialize.

    Attributes
    -object_detector : torch_detector
        The YoloV5 detector load from pytorch.

    """
    # 初始化方法：从torch hub加载预训练的YOLOv5中等规模('yolov5m')模型
    def __init__(self):

        self.object_detector = torch.hub.load('ultralytics/yolov5', 'yolov5m')

    # 基于YOLO检测结果绘制2D边界框；result: YOLOv5的检测结果对象，rgb_image: 相机拍摄的RGB图像(numpy数组)，index: 指示索引
    def draw_2d_box(self, result, rgb_image, index):
        """
        Draw 2d bounding box based on the yolo detection.

        Args:
            -result (yolo.Result):Detection result from yolo 5.
            -rgb_image (np.ndarray): Camera rgb image.
            -index(int): Indicate the index.

        Returns:
            -rgb_image (np.ndarray): camera image with bbx drawn.
        """
        # torch.Tensor
        # 从结果中获取边界框坐标(x1,y1,x2,y2格式)
        bounding_box = result.xyxy[index]
        # 如果数据在GPU上，先转移到CPU再转为numpy数组
        if bounding_box.is_cuda:
            bounding_box = bounding_box.cpu().detach().numpy()
        else: # 否则直接转为numpy数组
            bounding_box = bounding_box.detach().numpy()

        # 遍历每个检测到的物体
        for i in range(bounding_box.shape[0]):
            detection = bounding_box[i]

            # the label has 80 classes, which is the same as coco dataset 获取类别标签和标签名称
            label = int(detection[5])
            label_name = result.names[label]

            # 如果是车辆类别(通过is_vehicle_cococlass判断)，统一显示为'vehicle'
            if is_vehicle_cococlass(label):
                label_name = 'vehicle'

            # 获取边界框坐标并转换为整数
            x1, y1, x2, y2 = int(
                detection[0]), int(
                detection[1]), int(
                detection[2]), int(
                detection[3])
            # 在图像上绘制绿色(0,255,0)矩形框，线宽为2
            cv2.rectangle(rgb_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # draw text on it；在边界框上方(x1,y1-10)位置添加标签文本；使用HERSHEY_SIMPLEX字体，大小0.9，颜色(36,255,12)，线宽1
            cv2.putText(rgb_image, label_name, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 1)

        # 返回绘制了边界框和标签的图像
        return rgb_image

# 根据COCO数据集判断标签是否属于车辆类别；参数：YOLO检测预测的标签(int)；返回：布尔值，表示是否属于车辆类别
def is_vehicle_cococlass(label):
    """
    Check whether the label belongs to the vehicle class according
    to coco dataset.
    Args:
        -label(int): yolo detection prediction.
    Returns:
        -is_vehicle: bool
            whether this label belongs to the vehicle class
    """
    # 定义车辆类别数组(COCO数据集中1:自行车,2:汽车,3:摩托车,5:公交车,7:卡车)
    vehicle_class_array = np.array([1, 2, 3, 5, 7], dtype=np.int)
    # 检查输入的label减去车辆类别数组后是否有0(即label是否在车辆类别数组中)；如果有则返回True，否则返回False
    return True if 0 in (label - vehicle_class_array) else False
