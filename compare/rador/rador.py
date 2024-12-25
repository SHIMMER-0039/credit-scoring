import numpy as np
import matplotlib.pyplot as plt
from math import pi

# 定义指标和模型
metrics = ['AUC', 'Type I Accuracy', 'Type II Accuracy', 'Total Accuracy', 'G-Mean', 'F-Measure']
models = ['Proposed', 'LDA', 'Log_R', 'SVM', 'BP', 'KNN', 'CT']

# 用随机数据填充（请用真实数据替换）
data = {
    'Proposed': [0.7, 0.8, 0.75, 0.85, 0.65, 0.7],
    'LDA': [0.6, 0.7, 0.6, 0.65, 0.6, 0.55],
    'Log_R': [0.58, 0.6, 0.7, 0.6, 0.55, 0.6],
    'SVM': [0.62, 0.68, 0.75, 0.7, 0.58, 0.65],
    'BP': [0.55, 0.6, 0.55, 0.6, 0.52, 0.5],
    'KNN': [0.5, 0.55, 0.52, 0.55, 0.5, 0.45],
    'CT': [0.65, 0.7, 0.68, 0.7, 0.6, 0.65]
}

# 定义变量数量（即指标数量）
num_metrics = len(metrics)

# 创建图像和极坐标轴
fig = plt.figure(figsize=(8, 8))
ax = plt.subplot(111, polar=True)

# 计算每个轴的角度
angles = [n / float(num_metrics) * 2 * pi for n in range(num_metrics)]
angles += angles[:1]  # 闭合图形

# 为每个模型绘制边线，并将线条变细
markers = ['o', 'D', 's', '^', 'v', 'P', 'h']  # 为不同模型选择不同标记
colors = ['red', 'purple', 'brown', 'green', 'gold', 'cyan', 'blue']  # 不同的颜色方案

for i, model in enumerate(models):
    values = data[model]
    values += values[:1]  # 重复第一个值以闭合雷达图
    ax.plot(angles, values, label=model, linewidth=1, linestyle='-', marker=markers[i], markersize=8, color=colors[i])  # 线条变细，linewidth 设置为 1

# 添加每个指标的标签
plt.xticks(angles[:-1], metrics, fontsize=12)

# 设置y标签和范围，确保外圈达到1.0
ax.set_rlabel_position(30)
plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ["0.2", "0.4", "0.6", "0.8", "1.0"], color="grey", size=8)  # 添加 1.0 标签
plt.ylim(0, 1)  # 确保 y 轴的范围覆盖到 1.0

# 自定义网格样式，使其更接近方形
ax.spines['polar'].set_visible(False)  # 隐藏外边框
ax.grid(color='grey', linestyle='-', linewidth=1)  # 设置直线网格

# 手动设置角度以增强锐利的多边形效果
ax.set_theta_offset(pi / 2)  # 旋转网格使其更整齐
ax.set_theta_direction(-1)   # 逆时针方向

# 将圆形网格替换为线性分布
ax.xaxis.set_tick_params(size=10, width=2)  # 使轴线更粗，更接近多边形的棱角
ax.yaxis.set_tick_params(size=10, width=2)

# 图例放置在图形外部
plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1), fontsize=10)

# 显示图表
plt.show()
