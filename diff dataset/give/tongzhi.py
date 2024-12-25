import os
import time
import pickle
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import BayesianRidge, LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss, precision_score, recall_score, f1_score, brier_score_loss, average_precision_score, roc_curve
from scipy.stats import ks_2samp
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from main.feature_selection import FeatureEvaluator, is_pareto_efficient, evaluate_model
from main.adaptive_bayesian_stacking import AdaptiveBayesianStacking

# 记录开始时间
start_time = time.time()

# Setup directories and load data
root_path = 'D:/study/Credit(1)/Credit/'
params_path = r'D:\study\Credit(1)\Credit\params/'
dataset_path = r'D:\study\credit_scoring_datasets/'
shuffle_path = r'D:\study\Credit(1)\Credit\shuffle_index/'
save_path = r'D:\study\second\outcome/'
os.makedirs(save_path, exist_ok=True)

data = pd.read_csv(r'D:\study\credit_scroing_datasets\bankfear.csv', low_memory=True)
features = data.drop(['loan_status', 'member_id'], axis=1).replace([-np.inf, np.inf, -np.nan, np.nan], 0)
labels = data['loan_status']

train_size = int(features.shape[0] * 0.8)
valid_size = int(features.shape[0] * 0.1)
test_size = valid_size

with open(shuffle_path + 'bankfear/shuffle_index.pickle', 'rb') as f:
    shuffle_index = pickle.load(f)

train_index = shuffle_index[:train_size]
valid_index = shuffle_index[train_size:(train_size + valid_size)]
test_index = shuffle_index[(train_size + valid_size):(train_size + valid_size + test_size)]
remaining_index = shuffle_index[(valid_size + test_size):]

train_x, train_y = features.iloc[train_index, :], labels.iloc[train_index]
valid_x, valid_y = features.iloc[valid_index], labels.iloc[valid_index]
test_x, test_y = features.iloc[test_index], labels.iloc[test_index]
remaining_x, remaining_y = features.iloc[remaining_index], labels.iloc[remaining_index]

full_train_x = pd.concat([train_x, valid_x], axis=0)
full_train_y = pd.concat([train_y, valid_y], axis=0)

# Feature selection methods
feature_methods = ['ClassifierFE', 'CorrelationFE']
selected_features = set()  # Initialize selected feature set

# Initial feature selection
evaluator = FeatureEvaluator(method=feature_methods[0])
evaluator.fit(train_x.values, train_y)
importance_scores_1 = evaluator.scores_
threshold_1 = 0.05 * np.max(importance_scores_1)
features_1 = set(np.where(importance_scores_1 > threshold_1)[0])
selected_features = features_1

# Iterative selection using other methods
for method in feature_methods[0:]:
    # Apply new method on initially selected feature set
    evaluator = FeatureEvaluator(method=method)
    evaluator.fit(train_x.values[:, list(selected_features)], train_y)
    importance_scores = evaluator.scores_
    threshold = 0.05 * np.max(importance_scores)
    additional_features = set(np.where(importance_scores > threshold)[0])

    # Add new important features to the original feature set
    candidate_features = selected_features.union(additional_features)

    # Apply all selected features to model training and validation
    filtered_train_x = train_x.values[:, list(candidate_features)]
    filtered_valid_x = valid_x.values[:, list(candidate_features)]
    original_scores = evaluate_model(train_x.values[:, list(selected_features)], train_y, valid_x.values[:, list(selected_features)], valid_y)
    new_scores = evaluate_model(filtered_train_x, train_y, filtered_valid_x, valid_y)

    print(f"Method: {method}")
    print("Original scores:", original_scores)
    print("New scores:", new_scores)

    pareto_efficient = is_pareto_efficient(np.array([original_scores, new_scores]))
    if pareto_efficient[1]:  # If the new feature set performs better
        print("New feature set is Pareto efficient and better than the original.")
        selected_features = candidate_features  # Update feature set

# Select final feature set
final_selected_features = list(selected_features)
print(f"Final selected features (indices): {final_selected_features}")

# Train and test model using final selected features
filtered_train_x = train_x.values[:, final_selected_features]
filtered_valid_x = valid_x.values[:, final_selected_features]
filtered_test_x = test_x.values[:, final_selected_features]
def get_base_models(method):
    model_mapping = {
        'DT': [DecisionTreeClassifier(random_state=42) for _ in range(3)],
        'LDA': [LinearDiscriminantAnalysis() for _ in range(3)],
        'LR': [LogisticRegression(n_jobs=-1, random_state=42) for _ in range(3)],
        'RF': [RandomForestClassifier(n_jobs=-1, random_state=42) for _ in range(3)],
        'XGB': [XGBClassifier(n_jobs=-1, random_state=42, use_label_encoder=False, eval_metric='logloss') for _ in range(3)],
        'LGB': [LGBMClassifier(n_jobs=-1, random_state=42) for _ in range(3)]
    }
    return model_mapping.get(method.upper(), None)
# 定义基模型，这里使用多个随机森林模型，全部使用默认参数
# base_models = [
#     ('rf1', XGBClassifier(        n_jobs=-1, use_label_encoder=False, eval_metric='logloss', random_state=42,
#                                   tree_method='hist', device='cuda')),  # 使用默认参数的 LR 模型
#     ('rf2', XGBClassifier(        n_jobs=-1, use_label_encoder=False, eval_metric='logloss', random_state=42,
#                                   tree_method='hist', device='cuda')),  # 使用默认参数的 LR 模型
#     ('rf3', XGBClassifier(        n_jobs=-1, use_label_encoder=False, eval_metric='logloss', random_state=42,
#                                   tree_method='hist', device='cuda'))  # 使用默认参数的 LR 模型
# ]
# base_models = [
#     ('lgb1',LGBMClassifier( n_jobs=-1, random_state=42, device='gpu')),
#     ('lgb2',LGBMClassifier( n_jobs=-1, random_state=42, device='gpu')),
#     ('lgb3',LGBMClassifier(n_jobs=-1, random_state=42, device='gpu')),
# ]
base_models = [
    ('lgb1',LogisticRegression( n_jobs=-1, random_state=42, device='gpu')),
    ('lgb2',LogisticRegression( n_jobs=-1, random_state=42, device='gpu')),
    ('lgb3',LogisticRegression(n_jobs=-1, random_state=42, device='gpu')),
]

# 定义堆叠模型，使用 BayesianRidge 作为最终模型
stacking_model = StackingClassifier(
    estimators=base_models,
    final_estimator=LogisticRegression(),  # 可以改成其他模型，如 LogisticRegression()
    cv=5,  # 使用 5 折交叉验证
    n_jobs=-1  # 并行化
)

# 训练堆叠模型
stacking_model.fit(filtered_train_x, train_y)

# 进行预测
y_pred = stacking_model.predict(filtered_test_x)
y_pred_proba = stacking_model.predict_proba(filtered_test_x)[:, 1]

# 确保预测概率在 [0, 1] 范围内
y_pred_proba_clipped = np.clip(y_pred_proba, 0, 1)

# 计算性能指标
accuracy = accuracy_score(test_y, y_pred)
roc_auc = roc_auc_score(test_y, y_pred_proba_clipped)
logloss = log_loss(test_y, y_pred_proba_clipped)
ks = ks_2samp(y_pred_proba_clipped[test_y == 1], y_pred_proba_clipped[test_y != 1]).statistic
precision = precision_score(test_y, y_pred)
recall = recall_score(test_y, y_pred)
f1 = f1_score(test_y, y_pred)
brier_score = brier_score_loss(test_y, y_pred_proba_clipped)
average_precision = average_precision_score(test_y, y_pred_proba_clipped)
fprs, tprs, thresholds = roc_curve(test_y, y_pred_proba_clipped)
true_positive_rate = tprs
true_negative_rate = 1 - fprs
gmean = np.sqrt(true_positive_rate * true_negative_rate)

def h_mean(precision, recall):
    if precision + recall == 0:
        return 0
    return 2 * (precision * recall) / (precision + recall)

hm = h_mean(precision, recall)

def type1error(y_proba, y_true, threshold=0.5):
    y_pred = (y_proba >= threshold).astype(int)
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    return fp / (y_true == 0).sum()

def type2error(y_proba, y_true, threshold=0.5):
    y_pred = (y_proba >= threshold).astype(int)
    fn = ((y_pred == 0) & (y_true == 1)).sum()
    return fn / (y_true == 1).sum()

type1_error = type1error(y_pred_proba_clipped, test_y)
type2_error = type2error(y_pred_proba_clipped, test_y)

average_score = (accuracy + roc_auc + precision + recall) / 4

# 保存指标到结果字典中
results = {
    'params': stacking_model.get_params(),
    'auc': roc_auc,
    'acc': accuracy,
    'prec': precision,
    'rec': recall,
    'f1': f1,
    'bs': brier_score,
    'fprs': fprs,
    'tprs': tprs,
    'e1': type1_error,
    'e2': type2_error,
    'prec_rec': hm,
    'ap': average_precision,
    'tpr': true_positive_rate,
    'tnr': true_negative_rate,
    'gmean': gmean,
    'test_accs': accuracy,
    'average_score': average_score
}

all_results = [results]

for key, value in results.items():
    print(f"{key}: {value}")

# 找出最佳结果
best_result = max(all_results, key=lambda x: x['average_score'])
print("最佳结果：", best_result)

results_dict = {'results': all_results, 'best_result': best_result}

# 保存最佳结果
dataset = 'give'
method = 'tongzhirf'
file_path = os.path.join(save_path, f'{dataset}\\{method}_res.pickle')
os.makedirs(os.path.dirname(file_path), exist_ok=True)

with open(file_path, 'wb') as f:
    pickle.dump(results_dict, f)

print(f'This is ACC: {best_result["acc"]}')
print(f'This is AUC: {best_result["auc"]}')
print(f'The results of {method} on {dataset} have been calculated and saved.\n\n')

# 记录结束时间
end_time = time.time()

# 计算并打印总运行时间
total_time = end_time - start_time
print(f"Total running time: {total_time:.2f} seconds")