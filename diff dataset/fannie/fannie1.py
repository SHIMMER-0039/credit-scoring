import os
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import BayesianRidge
# from sklearn.linear import BayesianRidge
from sklearn.metrics import (accuracy_score, roc_auc_score, log_loss, precision_score, recall_score,
                             f1_score, brier_score_loss, average_precision_score, roc_curve)
from scipy.stats import ks_2samp
import xgboost as xgb
import lightgbm as lgb
from main.feature_selection import FeatureEvaluator, is_pareto_efficient, evaluate_model
from main.adaptive_bayesian_stacking import AdaptiveBayesianStacking

# 设置目录和加载数据
root_path = 'D:/study/Credit(1)/Credit/'
params_path = r'D:\study\Credit(1)\Credit\params/'
dataset_path = r'D:\study\credit_scoring_datasets/'
shuffle_path = r'D:\study\Credit(1)\Credit\shuffle_index/'
save_path = r'D:\study\second\outcome/'
os.makedirs(save_path, exist_ok=True)

data = pd.read_csv(r'D:\study\credit_scroing_datasets\FannieMae/2008q1.csv', low_memory=True)
features = data.drop(['DEFAULT', 'LOAN IDENTIFIER'], axis=1).replace([-np.inf, np.inf, np.nan], 0)
labels = data['DEFAULT']

train_size = int(features.shape[0] * 0.7)
valid_size = int(features.shape[0] * 0.15)
test_size = features.shape[0] - train_size - valid_size

with open(shuffle_path + 'fannie/shuffle_index.pickle', 'rb') as f:
    shuffle_index = pickle.load(f)
train_index = shuffle_index[:train_size]
valid_index = shuffle_index[train_size:(train_size + valid_size)]
test_index = shuffle_index[(train_size + valid_size):(train_size + valid_size + test_size)]

train_x, train_y = features.iloc[train_index, :], labels.iloc[train_index]
valid_x, valid_y = features.iloc[valid_index, :], labels.iloc[valid_index]
test_x, test_y = features.iloc[test_index, :], labels.iloc[test_index]

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
for method in feature_methods[0:]:
    evaluator = FeatureEvaluator(method=method)
    evaluator.fit(train_x.values[:, list(selected_features)], train_y)
    importance_scores = evaluator.scores_
    threshold = 0.05 * np.max(importance_scores)
    additional_features = set(np.where(importance_scores > threshold)[0])

    candidate_features = selected_features.union(additional_features)

    filtered_train_x = train_x.values[:, list(candidate_features)]
    filtered_valid_x = valid_x.values[:, list(candidate_features)]
    original_scores = evaluate_model(train_x.values[:, list(selected_features)], train_y, valid_x.values[:, list(selected_features)], valid_y)
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
    'n_estimators': [800],
    'max_depth': [3 ],
    'learning_rate': [0.1],
}

all_results = []
best_results = None

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
                    min_split_gain=0.1   # 避免空分裂
                ),
            ]

            stacking_model = AdaptiveBayesianStacking(
                base_models=base_models,
                weight_model=BayesianRidge(),
                n_folds=5
            )
            stacking_model.fit(filtered_train_x, train_y)

            # 提取模型权重
            weights = stacking_model.weight_model.coef_
            total_weight = np.sum(weights)
            proportions = weights / total_weight

            # 打印模型贡献比例
            for i, model in enumerate(base_models):
                print(f"Classifier {type(model).__name__} contributes {proportions[i]:.2%} to the final prediction.")

            # 预测和评估
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
            results = {
                'n_estimators': n_estimators,
                'max_depth': max_depth,
                'learning_rate': learning_rate,
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

            all_results.append(results)

            # 打印当前网格组合的结果
            print(f"Results for n_estimators={n_estimators}, max_depth={max_depth}, learning_rate={learning_rate}:")
            print(f"AUC: {roc_auc}, Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1: {f1}")
            print(f"KS Statistic: {ks}, Log Loss: {logloss}, Brier Score: {brier_score}")
            print(f"Average Precision: {average_precision}, G-Mean: {gmean}")
            print(f"Type 1 Error: {type1_error}, Type 2 Error: {type2_error}\n")

            if best_results is None or average_score > best_results['average_score']:
                best_results = results

        print("所有结果：")
        for result in all_results:
            print(result)

        best_result = max(all_results, key=lambda x: x['average_score'])
        print("最佳结果：", best_result)

        results_dict = {'results': all_results, 'best_result': best_result}

        dataset = 'fannie'
        method = 'AAESS'
        file_path = os.path.join(save_path, f'{dataset}\\{method}_res.pickle')
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, 'wb') as f:
            pickle.dump(results_dict, f)

        print(f'This is ACC: {best_result["acc"]}')
        print(f'This is AUC: {best_result["auc"]}')
        print(f'The results of {method} on {dataset} have been calculated and saved.\n\n')