import numpy as np
import pandas as pd

# 数据集：每个方法在4个指标上的排名，分别对应 acc, auc, bs, hm
data = {
    'lda': [15, 16, 16.75, 10.125],
    'lr': [16.5, 17.25, 15.5, 15.125],
    'dt': [15, 15.5, 14, 17],
    'knn': [16, 17.5, 15.75, 12.375],
    'adaboost': [14.75, 14.75, 14, 9.25],
    'rf': [10.125, 11.25, 10.5, 10.25],
    'gbdt': [5.625, 10, 8, 6],
    'xgboost': [9.75, 8, 5.5, 3.5],
    'lightgbm': [13.75, 7.25, 4.25, 3.625],
    'aaess': [6.875, 2.125, 1.875, 4],
    'ho-dt': [11, 10.75, 9.75, 12.375],
    'ho-lda': [8.75, 14.25, 12, 13.75],
    'ho-lr': [12, 12.25, 12, 13.25],
    'ho-rf': [9.75, 8.75, 11.5, 15.25],
    'ho-xgb': [4.75, 3.75, 8.5, 12.25],
    'ho-lgb': [3, 7, 4.5, 7],
    'aaess-lr': [4.5, 1.5, 10, 4.625],
    'he-tree': [6.5, 5.75, 9.5, 12.25],
    'aaess-tree': [6.375, 6.375, 5.375, 8]
}

# 将数据转换为 DataFrame
df = pd.DataFrame(data)

# 为方便起见，我们给每个指标命名
df.columns = [
    'lda', 'lr', 'dt', 'knn', 'adaboost', 'rf', 'gbdt', 'xgboost', 'lightgbm',
    'aaess', 'ho-dt', 'ho-lda', 'ho-lr', 'ho-rf', 'ho-xgb', 'ho-lgb',
    'aaess-lr', 'he-tree', 'aaess-tree'
]

# 我们将指标命名为 acc, auc, bs, hm，分别代表四个指标
df.index = ['acc', 'auc', 'bs', 'hm']

# 方法数量 (19个方法)
Nc = df.shape[1]

# 指标数量 (4个指标)
D = df.shape[0]

# 初始化一个列表存储每个指标的Friedman检验统计量
friedman_statistics_per_indicator = []

# 对于每个指标（每一行），计算Friedman统计量
for indicator in df.index:
    # 获取该指标下所有方法的排名
    indicator_data = df.loc[indicator].values

    # 对排名求和
    rank_sum = np.sum(indicator_data)

    # 计算排名平方和
    rank_sum_square = np.sum(indicator_data ** 2)

    # 使用Friedman公式计算该指标的卡方统计量
    term1 = (12 * 4) / (19 * (19 + 1))  # 常数部分
    term2 = rank_sum_square
    term3 = (19 * (19 + 1) ** 2) / 4  # 公式中的固定偏移量

    friedman_statistic = term1 * (term2 - term3)

    # 将该指标的Friedman统计量存储
    friedman_statistics_per_indicator.append((indicator, friedman_statistic))

# 输出每个指标的Friedman检验卡方统计量
for indicator, stat in friedman_statistics_per_indicator:
    print(f"Friedman test statistic for indicator {indicator}: {stat:.4f}")
