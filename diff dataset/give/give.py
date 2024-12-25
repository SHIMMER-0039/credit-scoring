import os
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import BayesianRidge, HuberRegressor
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss, precision_score, recall_score, f1_score, brier_score_loss, average_precision_score, roc_curve
from scipy.stats import ks_2samp, zscore
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from main.feature_selection import FeatureEvaluator, is_pareto_efficient, evaluate_model
from main.adaptive_bayesian_stacking import AdaptiveBayesianStacking

# Setup directories and load data
root_path = 'D:/study/Credit(1)/Credit/'
params_path = r'D:\study\Credit(1)\Credit\params/'
dataset_path = r'D:\study\credit_scoring_datasets/'
shuffle_path = r'D:\study\Credit(1)\Credit\shuffle_index/'
save_path = r'D:\study\second\outcome/'
os.makedirs(save_path, exist_ok=True)

data = pd.read_csv(r'D:\study\credit_scroing_datasets\give_me_some_credit_cleaned.csv')
features = data.drop('SeriousDlqin2yrs', axis=1).replace([-np.inf, np.inf, np.nan], 0)
labels = data['SeriousDlqin2yrs']

# ------ 定义异常值检测方法 ------

# IQR方法
def iqr_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] < lower_bound) | (df[column] > upper_bound)].index

# Z-score方法
def zscore_outliers(df, threshold=3):
    z_scores = np.abs(zscore(df))
    return np.where(z_scores > threshold)[0]

# 孤立森林方法
def isolation_forest_outliers(X, contamination=0.05):
    iso_forest = IsolationForest(contamination=contamination, random_state=42)
    y_pred = iso_forest.fit_predict(X)
    return np.where(y_pred == -1)[0]

# LOF方法
def lof_outliers(X, n_neighbors=20, contamination=0.05):
    lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
    y_pred = lof.fit_predict(X)
    return np.where(y_pred == -1)[0]

# ------ 检测数据中的异常值 ------

# 使用IQR方法检测异常值
iqr_indices = set()
for col in features.columns:
    iqr_indices.update(iqr_outliers(features, col))

# 使用Z-score方法检测异常值
zscore_indices = set(zscore_outliers(features))

# 使用孤立森林方法检测异常值
isolation_indices = set(isolation_forest_outliers(features))

# 使用LOF方法检测异常值
lof_indices = set(lof_outliers(features))

# 将所有检测到的异常值索引合并
all_outliers = iqr_indices.union(zscore_indices).union(isolation_indices).union(lof_indices)

# 计算异常值比例
outlier_ratio = len(all_outliers) / features.shape[0]
print(f"Outlier ratio: {outlier_ratio:.2%}")

# ------ 根据异常值比例选择元分类器 ------
if outlier_ratio > 0.05:  # 假设5%为异常值比例的阈值
    print("Using Huber Regressor due to high outlier ratio.")
    weight_model = HuberRegressor()
else:
    print("Using Bayesian Ridge due to low outlier ratio.")
    weight_model = BayesianRidge()

# 分割数据集
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

# Feature selection methods
feature_methods = ['GainRFE']
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

# Define parameter grid
param_grid = {
    'n_estimators': [600],
    'max_depth': [3],
    'learning_rate': [0.05],
}

all_results = []
best_results = None

# Iterate over parameter grid
for n_estimators in param_grid['n_estimators']:
    for max_depth in param_grid['max_depth']:
        for learning_rate in param_grid['learning_rate']:
            base_models = [
                xgb.XGBClassifier(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate,
                                  n_jobs=-1, use_label_encoder=False, eval_metric='logloss', random_state=42,
                                  tree_method='hist', device='cuda'),
                lgb.LGBMClassifier(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate,
                                   n_jobs=-1, random_state=42, device='gpu'),
            ]

            stacking_model = AdaptiveBayesianStacking(base_models=base_models, weight_model=weight_model, n_folds=5)
            stacking_model.fit(filtered_train_x, train_y)

            # Extract model weights
            weights = stacking_model.weight_model.coef_
            total_weight = np.sum(weights)
            proportions = weights / total_weight

            # Print contribution of each classifier
            for i, model in enumerate(base_models):
                print(f"Classifier {type(model).__name__} contributes {proportions[i]:.2%} to the final prediction.")

            y_pred = stacking_model.predict(filtered_test_x)
            y_pred_proba = stacking_model.predict_proba(filtered_test_x)[:, 1]

            # Ensure predicted probabilities are in [0, 1] range
            y_pred_proba_clipped = np.clip(y_pred_proba, 0, 1)

            # Calculate performance metrics
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

            # Save metrics in the results dictionary
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
                'test_accs': accuracy,  # Assuming test_accs is same as accuracy
                'average_score': average_score
            }

            all_results.append(results)

            print(f"Results for n_estimators={n_estimators}, max_depth={max_depth}, learning_rate={learning_rate}:")
            for key, value in results.items():
                print(f"{key}: {value}")

            if best_results is None or average_score > best_results['average_score']:
                best_results = results

print("所有结果：")
for result in all_results:
    print(result)

best_result = max(all_results, key=lambda x: x['average_score'])
print("最佳结果：", best_result)

results_dict = {'results': all_results, 'best_result': best_result}

dataset = 'give'
method = 'AAESS'
file_path = os.path.join(save_path, f'{dataset}\\{method}_res.pickle')
os.makedirs(os.path.dirname(file_path), exist_ok=True)

with open(file_path, 'wb') as f:
    pickle.dump(results_dict, f)

print(f'This is ACC: {best_result["acc"]}')
print(f'This is AUC: {best_result["auc"]}')
print(f'The results of {method} on {dataset} have been calculated and saved.\n\n')