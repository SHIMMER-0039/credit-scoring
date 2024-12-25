import pandas as pd

# 构造数据
data = {
    'acc': [0.933, 0.93325, 0.93675, 0.928, 0.9515, 0.9625, 0.9555, 0.96375, 0.9402, 0.9665,
            0.95775, 0.95175, 0.95925, 0.95675, 0.966, 0.9655, 0.96625, 0.96475, 0.96275],
    'auc': [0.8283, 0.61332, 0.7592, 0.84829, 0.89905, 0.93493, 0.91692, 0.93326, 0.93863, 0.94484,
            0.91364, 0.8801, 0.90037, 0.9363, 0.94671, 0.94786, 0.94848, 0.93756, 0.93196],
    'bs': [0.05526, 0.06408, 0.06325, 0.5027, 0.2317, 0.04252, 0.03269, 0.03215, 0.0295, 0.02788,
           0.03603, 0.07697, 0.07198, 0.03454, 0.02866, 0.2844, 0.02796, 0.03019, 0.03141],
    'hm': [0.36364, 0.34251, 0.40474, 0.38721, 0.47849, 0.35258, 0.62829, 0.61691, 0.65882, 0.67619,
           0.57845, 0.49907, 0.55701, 0.63584, 0.65865, 0.66789, 0.6778, 0.65693, 0.67619],
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
rankings = df[[ 'total_rank']].values

print(rankings)

# 如果需要将结果保存到新的CSV文件
# pd.DataFrame(rankings, columns=['rank_acc', 'rank_auc', 'rank_bs', 'rank_hm', 'mean_rank', 'total_rank']).to_csv('ranked_results.csv', index=False)
