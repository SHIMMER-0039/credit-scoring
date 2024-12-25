import os

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# 定义模型和数据 (Give 数据集)
data = {
    'DT': [0.93520, 0.86713, 0.04760, 0.26451, 0.00656, 0.83333],
    'LDA': [0.93906, 0.85372, 0.04880, 0.28257, 0.00784, 0.81707],
    'LR': [0.93866, 0.85597, 0.04916, 0.22689, 0.00506, 0.86280],
    'RF': [0.93753, 0.84515, 0.04927, 0.19105, 0.00624, 0.83841],
    'XGB': [0.93780, 0.86487, 0.04999, 0.30216, 0.01077, 0.79471],
    'LGB': [0.93873, 0.87347, 0.04884, 0.33646, 0.01198, 0.76321],
    'AAESS-LR': [0.94020, 0.87370, 0.04712, 0.28752, 0.00670, 0.81605],
    'extre-tree': [0.93940, 0.87014, 0.04748, 0.26157, 0.00613, 0.83638],
    'AAESS-TREE': [0.93980, 0.86690, 0.04765, 0.30162, 0.00813, 0.80182],
    'AAESS': [0.94060, 0.87341, 0.04705, 0.28890, 0.00627, 0.81605]
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
plt.savefig(os.path.join(save_path, 'give.png'), dpi=300, bbox_inches='tight')

# 显示图表
plt.show()

