import os

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# 定义模型和数据 (Shandong 数据集)
data = {
    'DT': [0.95775, 0.91364, 0.03603, 0.57845, 0.00428, 0.57678],
    'LDA': [0.95175, 0.88010, 0.07697, 0.49907, 0.00003, 0.72285],
    'LR': [0.95925, 0.90037, 0.07198, 0.55701, 0.00054, 0.60300],
    'RF': [0.95675, 0.93630, 0.03454, 0.63584, 0.00107, 0.63296],
    'XGB': [0.96600, 0.94761, 0.02866, 0.65865, 0.00509, 0.43820],
    'LGB': [0.96550, 0.94786, 0.02844, 0.66789, 0.00375, 0.04644],
    'AAESS-LR': [0.96625, 0.94848, 0.02796, 0.67780, 0.00267, 0.46816],
    'extre-tree': [0.96475, 0.93756, 0.03019, 0.65693, 0.00241, 0.49438],
    'AAESS-TREE': [0.96275, 0.93196, 0.03141, 0.64775, 0.00289, 0.47048],
    'AAESS': [0.96650, 0.94484, 0.02788, 0.67619, 0.00294, 0.46816]
}

# 定义指标
metrics = ['Accuracy', 'AUC', 'BS', 'HM', 'e-I', 'e-II']

# 创建 DataFrame，将模型和指标结合起来
df = pd.DataFrame(data, index=metrics)

# 创建热图
plt.figure(figsize=(10, 8))
sns.heatmap(df.T, annot=True, fmt='.5f', cmap='coolwarm', linewidths=.5, annot_kws={"size": 10})

# 设置标题
plt.title('Model Performance Across Metrics')

# 检查目录是否存在，如果不存在则创建
save_path = r'D:\study\second\picture\compare\new'
if not os.path.exists(save_path):
    os.makedirs(save_path)

# 保存图片到指定路径
plt.savefig(os.path.join(save_path, 'shandong.png'), dpi=300, bbox_inches='tight')

# 显示图表
plt.show()
