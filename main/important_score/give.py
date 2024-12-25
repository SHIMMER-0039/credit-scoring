import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import BayesianRidge
import xgboost as xgb
import lightgbm as lgb
from main.feature_selection import FeatureEvaluator, is_pareto_efficient, evaluate_model
from main.adaptive_bayesian_stacking import AdaptiveBayesianStacking

# 设置路径和加载数据


root_path = 'D:/study/Credit(1)/Credit/'
params_path = r'D:\study\Credit(1)\Credit\params/'
dataset_path = r'D:\study\credit_scoring_datasets/'
shuffle_path = r'D:\study\Credit(1)\Credit\shuffle_index/'
save_path = r'D:\study\second\outcome/'
os.makedirs(save_path, exist_ok=True)

data = pd.read_csv(r'D:\study\credit_scroing_datasets\give_me_some_credit_cleaned.csv')
features = data.drop('SeriousDlqin2yrs', axis=1).replace([-np.inf, np.inf, np.nan], 0)
labels = data['SeriousDlqin2yrs']

# 分割数据集 (保持不变)
train_size = int(features.shape[0] * 0.8)
valid_size = int(features.shape[0] * 0.1)
test_size = valid_size  # 假设测试集大小与验证集相同

with open(shuffle_path + 'give/shuffle_index.pickle', 'rb') as f:
    shuffle_index = pickle.load(f)
train_index = shuffle_index[:train_size]
valid_index = shuffle_index[train_size:(train_size + valid_size)]
test_index = shuffle_index[(train_size + valid_size):(train_size + valid_size + test_size)]

train_x, train_y = features.iloc[train_index, :], labels.iloc[train_index]
valid_x, valid_y = features.iloc[valid_index, :], labels.iloc[valid_index]
test_x, test_y = features.iloc[test_index, :], labels.iloc[test_index]

# 将训练集和验证集合并用于交叉验证
full_train_x = pd.concat([train_x, valid_x], axis=0)
full_train_y = pd.concat([train_y, valid_y], axis=0)


# 特征选择
feature_methods = ['ClassifierFE','CorrelationFE']
selected_features = set()

evaluator = FeatureEvaluator(method=feature_methods[0])
evaluator.fit(train_x.values, train_y)
importance_scores_1 = evaluator.scores_
threshold_1 = 0.05 * np.max(importance_scores_1)
features_1 = set(np.where(importance_scores_1 > threshold_1)[0])
selected_features = features_1

for method in feature_methods[0:]:
    evaluator = FeatureEvaluator(method=method)
    evaluator.fit(train_x.values[:, list(selected_features)], train_y)
    importance_scores = evaluator.scores_
    threshold = 0.05 * np.max(importance_scores)
    additional_features = set(np.where(importance_scores > threshold)[0])

    candidate_features = selected_features.union(additional_features)

    filtered_train_x = train_x.values[:, list(candidate_features)]
    filtered_valid_x = valid_x.values[:, list(candidate_features)]
    original_scores = evaluate_model(train_x.values[:, list(selected_features)], train_y,
                                     valid_x.values[:, list(selected_features)], valid_y)
    new_scores = evaluate_model(filtered_train_x, train_y, filtered_valid_x, valid_y)

    pareto_efficient = is_pareto_efficient(np.array([original_scores, new_scores]))
    if pareto_efficient[1]:
        selected_features = candidate_features

final_selected_features = list(selected_features)

# 使用最终选择的特征训练和测试模型
filtered_train_x = train_x.values[:, final_selected_features]
filtered_valid_x = valid_x.values[:, final_selected_features]
filtered_test_x = test_x.values[:, final_selected_features]

# 替换特征名称为x1, x2, x3等
feature_names = [f'x{i+1}' for i in range(len(final_selected_features))]

# 定义参数网格
param_grid = {
    'n_estimators': [600],
    'max_depth': [3],
    'learning_rate': [0.05],
}

all_results = []
best_results = None

# 迭代参数网格
for n_estimators in param_grid['n_estimators']:
    for max_depth in param_grid['max_depth']:
        for learning_rate in param_grid['learning_rate']:
            base_models = [
                RandomForestClassifier(n_estimators=n_estimators, min_samples_split=5, max_features='sqrt', n_jobs=-1,
                                       random_state=42),
                xgb.XGBClassifier(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate,
                                  n_jobs=-1, use_label_encoder=False, eval_metric='logloss', random_state=42,
                                  tree_method='hist', device='cuda'),
                lgb.LGBMClassifier(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate,
                                   n_jobs=-1, random_state=42, device='gpu'),
            ]

            # 在计算特征重要性之前，先进行模型拟合
            for model in base_models:
                model.fit(filtered_train_x, train_y)

            stacking_model = AdaptiveBayesianStacking(base_models=base_models, weight_model=BayesianRidge(), n_folds=5)
            stacking_model.fit(filtered_train_x, train_y)

            # 计算特征重要性
            feature_importance_dict = {name: 0 for name in feature_names}

            for model in base_models:
                if isinstance(model, RandomForestClassifier):
                    importance = model.feature_importances_
                elif isinstance(model, xgb.XGBClassifier):
                    if hasattr(model, "get_booster"):
                        booster = model.get_booster()
                        importance = booster.get_score(importance_type='gain')
                        importance = np.array([importance.get(f'f{i}', 0) for i in range(len(feature_names))])
                    else:
                        importance = np.zeros(len(feature_names))
                elif isinstance(model, lgb.LGBMClassifier):
                    if model.booster_ is not None:  # 检查是否训练成功
                        importance = model.booster_.feature_importance(importance_type='gain')  # 使用正确的方式获取 LGBMClassifier 的特征重要性
                    else:
                        importance = np.zeros(len(feature_names))  # 若模型未成功训练，返回零向量
                else:
                    continue
                normalized_importance = importance / np.sum(importance) if np.sum(importance) > 0 else importance
                for i, feature in enumerate(feature_names):
                    feature_importance_dict[feature] += normalized_importance[i]

            # 排序并获取前10个特征
            # 取Top10重要特征
            sorted_importance = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)[:10]
            labels, importance = zip(*sorted_importance)
            total_importance = sum(importance)
            importance_percentages = [(imp / total_importance) * 100 for imp in importance]

            # 可视化前10个特征的重要性
            plt.figure(figsize=(8, 6))  # 调小图表大小
            colors = sns.color_palette('pastel', len(importance_percentages))


            plt.barh(labels, importance_percentages, color=colors)
            plt.xlabel('Feature Importance Percentage', fontsize=14)
            plt.ylabel('Features', fontsize=14)
            plt.gca().invert_yaxis()

            # 使用tight_layout自动调整布局减少空白
            plt.tight_layout()

            # 保存图片
            plt.savefig(r'D:\study\Credit(1)\Credit\outcome\give\top10_feature_importance.png',
                        bbox_inches='tight', dpi=300)
            plt.show()

