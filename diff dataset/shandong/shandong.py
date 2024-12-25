from sklearn.metrics import (accuracy_score, roc_auc_score, log_loss, precision_score, recall_score,
                             f1_score, brier_score_loss, average_precision_score, roc_curve, precision_recall_curve)

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import (accuracy_score, roc_auc_score, log_loss, precision_score, recall_score,
                             f1_score, brier_score_loss, average_precision_score, roc_curve, precision_recall_curve)
from scipy.stats import ks_2samp
import xgboost as xgb
import lightgbm as lgb
from main.feature_selection import FeatureEvaluator, is_pareto_efficient, evaluate_model
from main.adaptive_bayesian_stacking import AdaptiveBayesianStacking


def train_and_evaluate_model(model, train_x, train_y, test_x, test_y, save__path, method, dataset):
    model.fit(train_x, train_y)
    proba = model.predict_proba(test_x)[:, 1]
    pred = model.predict(test_x)

    # 确保预测概率在 [0, 1] 范围内
    proba = np.clip(proba, 0, 1)

    # 计算各项评估指标
    auc = roc_auc_score(test_y, proba)
    acc = accuracy_score(test_y, pred)
    prec = precision_score(test_y, pred)
    rec = recall_score(test_y, pred)
    f1 = f1_score(test_y, pred)
    bs = brier_score_loss(test_y, proba)
    ks = ks_2samp(proba[test_y == 1], proba[test_y != 1]).statistic
    fprs, tprs, _ = roc_curve(test_y, proba)
    prec_rec = precision_recall_curve(test_y, proba)
    ap = average_precision_score(test_y, proba)
    tpr = np.mean(tprs)
    tnr = 1 - np.mean(fprs)
    gmean = np.sqrt(tpr * tnr)

    # 自定义错误类型计算
    def type1error(y_proba, y_true, threshold=0.5):
        y_pred = (y_proba >= threshold).astype(int)
        fp = ((y_pred == 1) & (y_true == 0)).sum()
        return fp / (y_true == 0).sum()

    def type2error(y_proba, y_true, threshold=0.5):
        y_pred = (y_proba >= threshold).astype(int)
        fn = ((y_pred == 0) & (y_true == 1)).sum()
        return fn / (y_true == 1).sum()

    e1 = type1error(proba, test_y)
    e2 = type2error(proba, test_y)

    # 将结果保存到字典中
    results = {
        'auc': auc, 'acc': acc, 'prec': prec, 'rec': rec, 'f1': f1,
        'bs': bs, 'ks': ks, 'fprs': fprs, 'tprs': tprs,
        'prec_rec': prec_rec, 'ap': ap,
        'tpr': tpr, 'tnr': tnr, 'gmean': gmean, 'e1': e1, 'e2': e2
    }

    # 保存模型性能到文件
    file_path = os.path.join(save_path, f'{dataset}_{method}_res.pickle')
    with open(file_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"Results saved to {file_path}")

    return results


# 设置目录和加载数据
root_path = 'D:/study/Credit(1)/Credit/'
params_path = r'D:\study\Credit(1)\Credit\params/'
dataset_path = r'D:\study\credit_scoring_datasets/'
shuffle_path = r'D:\study\Credit(1)\Credit\shuffle_index/'
save_path = r'D:\study\second\outcome/'
os.makedirs(save_path, exist_ok=True)

data = pd.read_csv(r'D:\study\credit_scroing_datasets\shandong.csv', low_memory=True)
features = data.drop('label', axis=1).replace([-np.inf, np.inf], 0).fillna(0)
labels = data['label']

# 分割数据集
train_size = int(features.shape[0] * 0.8)
valid_size = int(features.shape[0] * 0.1)
test_size = valid_size  # 假设测试集大小与验证集相同

# 加载shuffle索引
with open(shuffle_path + 'shandong/shuffle_index.pickle', 'rb') as f:
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

# 特征选择方法
feature_methods = ['ClassifierFE', 'CorrelationFE']
selected_features = set()

# 初始特征选择
evaluator = FeatureEvaluator(method=feature_methods[0])
evaluator.fit(train_x.values, train_y)
importance_scores_1 = evaluator.scores_
threshold_1 = 0.05 * np.max(importance_scores_1)
features_1 = set(np.where(importance_scores_1 > threshold_1)[0])
selected_features = features_1

# 使用其他方法进行迭代选择
for method in feature_methods:
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

    print(f"Method: {method}")
    print("Original scores:", original_scores)
    print("New scores:", new_scores)

    pareto_efficient = is_pareto_efficient(np.array([original_scores, new_scores]))
    if pareto_efficient[1]:
        print("New feature set is Pareto efficient and better than the original.")
        selected_features = candidate_features

final_selected_features = list(selected_features)
print(f"Final selected features (indices): {final_selected_features}")

filtered_train_x = train_x.values[:, final_selected_features]
filtered_valid_x = valid_x.values[:, final_selected_features]
filtered_test_x = test_x.values[:, final_selected_features]

# 参数网格定义
param_grid = {
    'n_estimators': [400],
    'max_depth': [5],
    'learning_rate': [0.2],
}

all_results = []

# 参数网格迭代
for n_estimators in param_grid['n_estimators']:
    for max_depth in param_grid['max_depth']:
        for learning_rate in param_grid['learning_rate']:
            base_models = [
                RandomForestClassifier(
                    n_estimators=n_estimators,
                    min_samples_split=5,
                    max_features='sqrt',
                    n_jobs=-1,
                    random_state=42
                ),
                xgb.XGBClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    learning_rate=learning_rate,
                    n_jobs=-1,
                    use_label_encoder=False,
                    eval_metric='logloss',
                    random_state=42,
                    tree_method='hist',
                    gpu_id=0  # 指定使用 GPU
                ),
                lgb.LGBMClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    learning_rate=learning_rate,
                    n_jobs=-1,
                    random_state=42,
                    device_type='gpu',  # 使用 GPU
                    min_data_in_leaf=20,  # 避免分裂问题
                    min_split_gain=0.1  # 避免空分裂
                ),
            ]

            stacking_model = AdaptiveBayesianStacking(
                base_models=base_models,
                weight_model=BayesianRidge(),
                n_folds=5
            )
            stacking_model.fit(filtered_train_x, train_y)

            # 使用 train_and_evaluate_model 函数来评估模型
            method = 'AAESS'
            dataset = 'shandong'

            results = train_and_evaluate_model(
                model=stacking_model,
                train_x=filtered_train_x,
                train_y=train_y,
                test_x=filtered_test_x,
                test_y=test_y,
                save_path=save_path,
                method=method,
                dataset=dataset
            )

            all_results.append(results)

# 打印所有结果
print("所有结果：")
for result in all_results:
    print(result)

# 选择最佳结果
best_result = max(all_results, key=lambda x: x['acc'])  # 这里以准确率为准，可以根据需要更改
print("最佳结果：", best_result)

results_dict = {'results': all_results, 'best_result': best_result}

# 保存所有结果
file_path = os.path.join(save_path, f'{dataset}_{method}_all_res.pickle')
with open(file_path, 'wb') as f:
    pickle.dump(results_dict, f)

print(f'This is ACC: {best_result["acc"]}')
print(f'This is AUC: {best_result["auc"]}')
print(f'The results of {method} on {dataset} have been calculated and saved.\n\n')
