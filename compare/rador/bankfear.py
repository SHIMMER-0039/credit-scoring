import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import os

# 定义模型和数据
data = {
    'DT': [0.92413, 0.79116, 0.06402, 0.24228, 0.00108, 0.98189],
    'LDA': [0.92376, 0.81136, 0.06292, 0.27818, 0.00003, 0.99964],
    'LR': [0.92378, 0.80864, 0.06350, 0.27337, 0.000029, 0.99929],
    'RF': [0.83921, 0.88550, 0.11154, 0.49704, 0.00630, 0.65892],
    'XGB': [0.92202, 0.94740, 0.06303, 0.71342, 0.02565, 0.24632],
    'LGB': [0.92208, 0.94700, 0.06260, 0.71411, 0.02580, 0.24560],
    'AAESS-LR': [0.91679, 0.94051, 0.06463, 0.81120, 0.03252, 0.24623],
    'extre-tree': [0.87705, 0.90717, 0.09300, 0.67187, 0.01531, 0.46919],
    'AAESS-TREE': [0.91702, 0.94186, 0.06646, 0.81036, 0.03033, 0.25233],
    'AAESS': [0.92319, 0.94725, 0.06198, 0.82617, 0.02905, 0.23039]
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
plt.savefig(os.path.join(save_path, 'bankfear.png'), dpi=300, bbox_inches='tight')

# 显示图表
plt.show()
