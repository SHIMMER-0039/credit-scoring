import numpy as np
import scipy.stats as stats
import scikit_posthocs as sp
from math import sqrt

# 数据准备
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

# 将数据转换为NumPy数组
data_values = np.array(list(data.values()))

# Friedman检验
friedman_stat, friedman_p = stats.friedmanchisquare(*data_values)

print(f'Friedman检验结果: 统计量={friedman_stat}, p值={friedman_p}')

# 如果Friedman检验显著，可以进行Nemenyi检验
if friedman_p < 0.05:
    nemenyi_result = sp.posthoc_nemenyi_friedman(data_values.T)
    print("Nemenyi检验结果:")
    print(nemenyi_result)

    # 计算不同显著性水平下的CD值 (Critical Difference)
    k = len(data)  # 算法的数量
    n = len(data_values[0])  # 测试的数量

    # 不同alpha下的q_alpha值
    q_alpha_values = {
        0.1: 2.291,
        0.05: 2.569,
        0.01: 3.213
    }

    for alpha, q_alpha in q_alpha_values.items():
        cd_value = q_alpha * sqrt(k * (k + 1) / (6.0 * n))
        print(f"CD值 (alpha={alpha}): {cd_value}")
