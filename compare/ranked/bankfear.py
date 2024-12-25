import pandas as pd

# 手动构造数据，或者你可以从CSV文件中读取
data = {
    'acc': [0.7662, 0.77614, 0.82429, 0.77723, 0.82995, 0.8544, 0.89118, 0.92023, 0.92117, 0.92319,
            0.92413, 0.92376, 0.92378, 0.83921, 0.92202, 0.92208, 0.91679, 0.87705, 0.92319],
    'auc': [0.62103, 0.76468, 0.77807, 0.63363, 0.80837, 0.90382, 0.92588, 0.94571, 0.94668, 0.94725,
            0.79116, 0.81136, 0.80864, 0.8855, 0.94715, 0.947, 0.94051, 0.90717, 0.94186],
    'bs': [0.17618, 0.15263, 0.13358, 0.1817, 0.23908, 0.10199, 0.08242, 0.06415, 0.06326, 0.06198,
           0.06402, 0.62922, 0.635, 0.11154, 0.6303, 0.0626, 0.66463, 0.093, 0.06646],
    'hm': [0.34744, 0.36993, 0.46552, 0.38933, 0.4659, 0.57401, 0.72044, 0.81586, 0.81825, 0.82617,
           0.24428, 0.27818, 0.27337, 0.49704, 0.71342, 0.71411, 0.8112, 0.67187, 0.81036],
}

# 算法名称
algorithms = ['lda', 'lr', 'dt', 'knn', 'adaboost', 'rf', 'gbdt', 'xgboost', 'lightgbm', 'aaess',
              'ho-dt', 'ho-lda', 'ho-lr', 'ho-rf', 'ho-xgb', 'ho-lgb', 'aaess-lr', 'he-tree', 'aaess-tree']

# 构造DataFrame
df = pd.DataFrame(data, index=algorithms)

# 对每个指标分别排名
df['rank_acc'] = df['acc'].rank(ascending=False, method='average')
df['rank_auc'] = df['auc'].rank(ascending=False, method='average')
df['rank_bs'] = df['bs'].rank(ascending=True, method='average')  # bs值越小排名越高
df['rank_hm'] = df['hm'].rank(ascending=False, method='average')

# 计算总排名
df['total_rank'] = (df['rank_acc'] + df['rank_auc'] + df['rank_bs'] + df['rank_hm']) / 4

# 输出结果
print(df[['total_rank']])

# 如果需要将结果保存到新的CSV文件
# df.to_csv('ranked_results.csv', index=True)
