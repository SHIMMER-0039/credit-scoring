import os

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# 定义模型和数据 (以Fannie数据为例)
data = {
    'DT': [0.92405, 0.79602, 0.06348, 0.25645, 0.00114, 0.98260],
    'LDA': [0.92424, 0.81332, 0.06243, 0.28137, 0.00023, 0.99077],
    'LR': [0.92359, 0.80162, 0.06384, 0.25953, 0.00191, 0.97905],
    'RF': [0.92318, 0.81727, 0.06291, 0.29096, 0.00066, 0.996],
    'XGB': [0.92392, 0.83276, 0.06094, 0.3240, 0.00332, 0.94797],
    'LGB': [0.92454, 0.83317, 0.0609, 0.32235, 0.00399, 0.94141],
    'AAESS-LR': [0.92518, 0.83118, 0.06070, 0.10029, 0.02960, 0.9453],
    'extre-tree': [0.92470, 0.81164, 0.06224, 0.03804, 0.00058, 0.98046],
    'AAESS-TREE': [0.92502, 0.83116, 0.06101, 0.09183, 0.00273, 0.95028],
    'AAESS': [0.92435, 0.83223, 0.06107, 0.09226, 0.00267, 0.95008]
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
plt.savefig(os.path.join(save_path, 'fannie.png'), dpi=300, bbox_inches='tight')

# 显示图表
plt.show()

