"""
本断裂线连接算法使用PyCharm开发平台，基于Python编程语言完成算法开发，具体过程涉及到Numpy、Matplotlib等第三方库。
计算机配置为Intel(R) Core(TM) i5-9300H CPU, Intel(R) UDH Graphics 630 GPU, 16.0GB RAM。运行操作系统为Windows 11。
版本：Python 3.10；numpy 1.24.2；matplotlib 0.1.9
230908完成实验的原理为：
a.将原始数据利用Numpy分割成两个数组，其中前两列为首端点数组head_lines和第三列和第四列为尾端点tail_lines，使用两层for循环，计算每一个首端点到所有尾端点的距离和角度；
b.但计算距离和角度时，还需要考虑不应该连接同一条直线上的两个端点，故设置布尔值数组to_delete，若计算完成，将值设置为False，则以后都不再计算该端点；
c.设置距离阈值distance_threshold和角度阈值angle_threshold，若计算的两个点满足距离和角度阈值，则视为应当连接点，置入连接点数组connected_lines_start和connected_lines_end；
d.利用Image库，读取tif影像作为底图并将其转换成Numpy数组，使用Matplotlib创建子图，分别显示tif影像，原始的断裂线(红色)，计算后得到的连接线(蓝色)。
"""
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# 读取.tif文件
img = Image.open('D:/python_work/研究生入学作业20210716/isprs58.tif')
# 读取断裂线txt文件，只提取前四列数据
broken_lines = np.loadtxt("D:/python_work/研究生入学作业20210716/ISPRS58_LSD.txt", usecols=(0, 1, 2, 3))
# 将图像转换为NumPy数组
img_array = np.array(img)
# 获取原始断裂线的端点数组和原始断裂线尾点数组
head_lines = broken_lines[:, :2]  # 首端点
third_column = broken_lines[:, 2]  # 第三列
fourth_column = broken_lines[:, 3]  # 第四列
tail_lines = np.column_stack((third_column, fourth_column))  # 尾端点

# 创建一个子图
fig, ax = plt.subplots()
# 显示提供的.tif影像
ax.imshow(img_array, cmap='gray')

# 计算图像的坐标范围
img_height, img_width = img_array.shape
ax.set_xlim(0, img_width)
ax.set_ylim(img_height, 0)  # 注意这里的坐标是从上到下递减的

# 连接线段的阈值
distance_threshold = 4  # 距离阈值
angle_threshold = 110  # 角度阈值，以度为单位

# 存储连接线段的起始点和结束点
connected_lines_start = []
connected_lines_end = []

# 绘制原始断裂线
for line in broken_lines:
    x1, y1, x2, y2 = line
    ax.plot([x1, x2], [y1, y2], color='red', linewidth=1)

# 创建四个布尔数组来标记不要再重复计算的点
to_delete1 = np.zeros(len(broken_lines), dtype=bool)
to_delete2 = np.zeros(len(broken_lines), dtype=bool)
to_delete3 = np.zeros(len(broken_lines), dtype=bool)
to_delete4 = np.zeros(len(broken_lines), dtype=bool)

# 遍历每个端点
for i in range(len(broken_lines)):
    x1, y1 = head_lines[i]
    x2, y2 = tail_lines[i]

    for j in range(len(broken_lines)):
        if i != j:
            x3, y3 = head_lines[j]
            x4, y4 = tail_lines[j]
            # 计算端点之间的距离
            distance1 = np.sqrt((x1 - x3) ** 2 + (y1 - y3) ** 2)
            distance2 = np.sqrt((x1 - x4) ** 2 + (y1 - y4) ** 2)
            distance3 = np.sqrt((x2 - x3) ** 2 + (y2 - y3) ** 2)
            distance4 = np.sqrt((x2 - x4) ** 2 + (y2 - y4) ** 2)

            if to_delete1[j] and to_delete2[j]:
                continue  # 如果该行已标记为要删除，则跳过

            if to_delete3[j] and to_delete4[j]:
                continue

            # 检查是否距离在阈值范围内
            if distance1 < distance_threshold and distance1 < distance2:
                # 计算两条线段的角度
                angle = np.degrees(np.arccos(
                    ((x2 - x1) * (x4 - x3) + (y2 - y1) * (y4 - y3)) /
                    (np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) * np.sqrt((x4 - x3) ** 2 + (y4 - y3) ** 2))
                ))
                # 检查角度是否在阈值范围内
                if angle < angle_threshold:
                    connected_lines_start.append((x1, y1))
                    connected_lines_end.append((x3, y3))
                    to_delete1[j] = True  # 标记该行要删除

            if distance2 < distance_threshold and distance2 < distance1:
                # 计算两条线段的角度
                angle = np.degrees(np.arccos(
                    ((x2 - x1) * (x4 - x3) + (y2 - y1) * (y4 - y3)) /
                    (np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) * np.sqrt((x4 - x3) ** 2 + (y4 - y3) ** 2))
                ))
                # 检查角度是否在阈值范围内
                if angle < angle_threshold:
                    connected_lines_start.append((x1, y1))
                    connected_lines_end.append((x4, y4))
                    to_delete2[j] = True  # 标记该行要删除

            if distance3 < distance_threshold and distance3 < distance4:
                # 计算两条线段的角度
                angle = np.degrees(np.arccos(
                    ((x2 - x1) * (x4 - x3) + (y2 - y1) * (y4 - y3)) /
                    (np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) * np.sqrt((x4 - x3) ** 2 + (y4 - y3) ** 2))
                ))
                # 检查角度是否在阈值范围内
                if angle < angle_threshold:
                    connected_lines_start.append((x2, y2))
                    connected_lines_end.append((x3, y3))
                    to_delete3[j] = True  # 标记该行要删除

            if distance4 < distance_threshold and distance4 < distance3:
                # 计算两条线段的角度
                angle = np.degrees(np.arccos(
                    ((x2 - x1) * (x4 - x3) + (y2 - y1) * (y4 - y3)) /
                    (np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) * np.sqrt((x4 - x3) ** 2 + (y4 - y3) ** 2))
                ))
                # 检查角度是否在阈值范围内
                if angle < angle_threshold:
                    connected_lines_start.append((x2, y2))
                    connected_lines_end.append((x4, y4))
                    to_delete4[j] = True  # 标记该行要删除

# # 删除已连接的行
# head_lines = head_lines[~(to_delete1 | to_delete2)]
# tail_lines = tail_lines[~(to_delete3 | to_delete4)]

# 绘制连接线段
for i in range(len(connected_lines_start)):
    x3, y3 = connected_lines_start[i]
    x4, y4 = connected_lines_end[i]
    ax.plot([x3, x4], [y3, y4], color='blue', linewidth=1)

# 显示图像
plt.show()