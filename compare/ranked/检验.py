import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# 数据
data = {
    'LDA': [15, 16, 16.75, 10.125],
    'LR': [16.5, 17.25, 15.5, 15.125],
    'DT': [15, 15.5, 14, 17],
    'KNN': [16, 17.5, 15.75, 12.375],
    'Adaboost': [14.75, 14.75, 14, 9.25],
    'RF': [10.125, 11.25, 10.5, 10.25],
    'GBDT': [5.625, 10, 8, 6],
    'XGboost': [9.75, 8, 5.5, 3.5],
    'Lightgbm': [13.75, 7.25, 4.25, 3.625],
    'AAESS': [6.875, 2.125, 1.875, 4],
    'HO-DT': [11, 10.75, 9.75, 12.375],
    'HO-LDA': [8.75, 14.25, 12, 13.75],
    'HO-LR': [12, 12.25, 12, 13.25],
    'HO-RF': [9.75, 8.75, 11.5, 15.25],
    'HO-XGB': [4.75, 3.75, 8.5, 12.25],
    'HO-LGB': [3, 7, 4.5, 7],
    'AAESS-LR': [4.5, 1.5, 10, 4.625],
    'HE-tree': [6.5, 5.75, 9.5, 12.25],
    'AAESS-TREE': [6.375, 6.375, 5.375, 8]
}

# 转换数据为DataFrame
df = pd.DataFrame(data)

# 计算每个算法的平均排名
average_ranks = df.mean(axis=0)

# 根据平均排名从低到高排序
sorted_ranks = average_ranks.sort_values()

# 定义不同显著性水平下的CD值
y_cd_005 = 10.22  # 例子中的 alpha = 0.05 的 CD 值
y_cd_01 = 9.11    # 例子中的 alpha = 0.1 的 CD 值
y_cd_001 = 12.78  # 例子中的 alpha = 0.01 的 CD 值

# 创建排序后的图表
fig, ax = plt.subplots(figsize=(20, 10))

# 更改颜色以使图表更加美观
colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(sorted_ranks)))

# 绘制排序后的平均排名柱状图，并为图例标注柱子信息
bars = ax.bar(sorted_ranks.index, sorted_ranks.values, color=colors, edgecolor='black', label='Rank bar')

# 为每个柱子顶端添加数值标签
for i, v in enumerate(sorted_ranks.values):
    ax.text(i, v + 0.2, f'{v:.2f}', color='black', ha='center', fontsize=12)

# 添加不同样式的临界差（CD）线
ax.hlines(y=y_cd_005, xmin=-0.5, xmax=len(sorted_ranks) - 0.5, color='red', linestyle='-', linewidth=2, label='α=0.05 ')
ax.hlines(y=y_cd_01, xmin=-0.5, xmax=len(sorted_ranks) - 0.5, color='blue', linestyle='-.', linewidth=2, label='α=0.1 ')
ax.hlines(y=y_cd_001, xmin=-0.5, xmax=len(sorted_ranks) - 0.5, color='purple', linestyle=':', linewidth=2, label='α=0.01 ')

# 在左侧添加临界差数值标注
ax.text(-0.5, y_cd_005, f'CD={y_cd_005:.2f}', va='bottom', ha='left', fontsize=12, color='red')
ax.text(-0.5, y_cd_01, f'CD={y_cd_01:.2f}', va='bottom', ha='left', fontsize=12, color='blue')
ax.text(-0.5, y_cd_001, f'CD={y_cd_001:.2f}', va='bottom', ha='left', fontsize=12, color='green')

# 调整y轴限制以适应CD线
ax.set_ylim(0, max(sorted_ranks) + 2)

# 添加轴标签和标题
ax.set_ylabel('Average Rank', fontsize=16)
ax.set_xlabel('Algorithms', fontsize=16)
ax.set_xticks(range(len(sorted_ranks)))
ax.set_xticklabels(sorted_ranks.index, rotation=45, fontsize=16)

# 将图例移动到左上角
ax.legend(loc='upper left', fontsize=14)

# 显示图表
plt.tight_layout()
plt.show()
