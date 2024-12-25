import pandas as pd

# 构造数据
data = {
    'acc': [0.92364, 0.92381, 0.92399, 0.91595, 0.92286, 0.92435, 0.92462, 0.92229, 0.92185, 0.92435,
            0.92405, 0.92424, 0.92359, 0.92318, 0.92392, 0.92454, 0.92518, 0.9247, 0.92502],
    'auc': [0.79425, 0.78749, 0.78932, 0.67061, 0.81346, 0.81699, 0.83194, 0.84309, 0.82642, 0.83223,
            0.79602, 0.81332, 0.80162, 0.81727, 0.83276, 0.83317, 0.83118, 0.81164, 0.83116],
    'bs': [0.06413, 0.06455, 0.06407, 0.07466, 0.23494, 0.0621, 0.06101, 0.06132, 0.06308, 0.06107,
           0.06348, 0.06243, 0.06384, 0.06291, 0.06094, 0.0609, 0.0607, 0.06224, 0.06101],
    'hm': [0.04665, 0.01539, 0.0175, 0.13537, 0.08715, 0.03522, 0.10482, 0.07894, 0.02103, 0.09226,
           0.25645, 0.28137, 0.26953, 0.29096, 0.324, 0.32235, 0.10029, 0.3804, 0.09183],
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

# 输出只有排名的数据，不包括算法名称
rankings = df[ 'total_rank'].values

print(rankings)

# 如果需要将结果保存到新的CSV文件
# pd.DataFrame(rankings, columns=['rank_acc', 'rank_auc', 'rank_bs', 'rank_hm', 'mean_rank', 'total_rank']).to_csv('ranked_results.csv', index=False)
