import pandas as pd

# 构造数据
data = {
    'acc': [0.93786, 0.93786, 0.89966, 0.9352, 0.9388, 0.93993, 0.9392, 0.94013, 0.9402, 0.9406,
            0.9352, 0.93906, 0.93833, 0.93753, 0.9378, 0.93873, 0.9402, 0.9394, 0.9398],
    'auc': [0.85651, 0.84485, 0.61834, 0.73722, 0.86802, 0.8724, 0.87339, 0.87412, 0.87651, 0.87341,
            0.86713, 0.85372, 0.85597, 0.84515, 0.84487, 0.87337, 0.8737, 0.87014, 0.8669],
    'bs': [0.05325, 0.05023, 0.10005, 0.04754, 0.05583, 0.23372, 0.04758, 0.04731, 0.04725, 0.04705,
           0.0476, 0.488, 0.04916, 0.04927, 0.04999, 0.04884, 0.04712, 0.4748, 0.04765],
    'hm': [0.37449, 0.24959, 0.27189, 0.29257, 0.32599, 0.28321, 0.30909, 0.31554, 0.30085, 0.2889,
           0.26451, 0.28257, 0.22689, 0.19105, 0.30216, 0.33646, 0.28752, 0.26157, 0.30162],
}

# 构造DataFrame
df = pd.DataFrame(data)

# 对每个指标分别排名
df['rank_acc'] = df['acc'].rank(ascending=False, method='average')
df['rank_auc'] = df['auc'].rank(ascending=False, method='average')
df['rank_bs'] = df['bs'].rank(ascending=True, method='average')  # bs值越小排名越高
df['rank_hm'] = df['hm'].rank(ascending=False, method='average')

# 计算总排名
df['total_rank'] = (df['rank_acc'] + df['rank_auc'] + df['rank_bs'] + df['rank_hm']) / 4

# 计算每个算法的平均排名
df['mean_rank'] = df[['rank_acc', 'rank_auc', 'rank_bs', 'rank_hm']].mean(axis=1)

# 只输出排名数据，不包含算法名称
rankings = df[['total_rank']].values

print(rankings)

# 如果需要将结果保存到新的CSV文件
# pd.DataFrame(rankings, columns=['rank_acc', 'rank_auc', 'rank_bs', 'rank_hm', 'mean_rank', 'total_rank']).to_csv('ranked_results.csv', index=False)
