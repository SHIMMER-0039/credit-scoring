import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss, precision_score, recall_score, f1_score, \
    brier_score_loss, roc_curve, average_precision_score
from scipy.stats import ks_2samp
import os
import pickle
from xgboost import XGBClassifier

# 设置路径 (保持不变)
root_path = 'D:/study/Credit(1)/Credit/'
params_path = r'D:\study\Credit(1)\Credit\params/'
dataset_path = r'D:\study\credit_scoring_datasets/'
shuffle_path = r'D:\study\Credit(1)\Credit\shuffle_index/'
save_path = r'D:\study\second\outcome/'
os.makedirs(save_path, exist_ok=True)

# 加载新的数据集
data = pd.read_csv(r'D:\study\credit_scroing_datasets\FannieMae/2008q1.csv', low_memory=True)
features = data.drop(['DEFAULT', 'LOAN IDENTIFIER'], axis=1).replace([-np.inf, np.inf, np.nan], 0)
labels = data['DEFAULT']

# 分割数据集
train_size = int(features.shape[0] * 0.8)
valid_size = int(features.shape[0] * 0.1)
test_size = valid_size  # 假设测试集大小与验证集相同

# 加载shuffle索引
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

# 确保索引范围正确
print(f"Total data size: {features.shape[0]}")
print(f"Train indices: {train_index[:5]}...{train_index[-5:]}")
print(f"Valid indices: {valid_index[:5]}...{valid_index[-5:]}")
print(f"Test indices: {test_index[:5]}...{test_index[-5:]}")
print(f"Remaining indices: {remaining_index[:5]}...{remaining_index[-5:]}")

# 将训练集和验证集合并用于交叉验证
full_train_x = pd.concat([train_x, valid_x], axis=0)
full_train_y = pd.concat([train_y, valid_y], axis=0)

# 如果不需要任何预处理，则直接使用原始数据
full_train_x_transformed = full_train_x
test_x_transformed = test_x

# 使用默认参数训练模型
xgb_model = XGBClassifier()
xgb_model.fit(full_train_x_transformed, full_train_y)

default_params = xgb_model.get_params()
print("Default parameters of XGBClassifier:")
for param, value in default_params.items():
    print(f"{param}: {value}")
print(xgb_model.get_xgb_params())
# 预测和评估模型
preds_proba = xgb_model.predict_proba(test_x_transformed)[:, 1]
preds = xgb_model.predict(test_x_transformed)

# 计算评估指标
auc_score = roc_auc_score(test_y, preds_proba)
logloss = log_loss(test_y, preds_proba)
ks = ks_2samp(preds_proba[test_y == 1], preds_proba[test_y != 1]).statistic
accuracy = accuracy_score(test_y, preds)
precision = precision_score(test_y, preds)
recall = recall_score(test_y, preds)
f1 = f1_score(test_y, preds)
brier_score = brier_score_loss(test_y, preds_proba)
average_precision = average_precision_score(test_y, preds_proba)
fprs, tprs, thresholds = roc_curve(test_y, preds_proba)
true_positive_rate = tprs
true_negative_rate = 1 - fprs
gmean = np.sqrt(true_positive_rate * true_negative_rate)

# 计算 H-mean 和其他自定义指标
def h_mean(precision, recall):
    if precision + recall == 0:
        return 0
    return 2 * (precision * recall) / (precision + recall)

hm = h_mean(precision, recall)

# 计算 type1error 和 type2error
def type1error(y_proba, y_true, threshold=0.5):
    y_pred = (y_proba >= threshold).astype(int)
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    return fp / (y_true == 0).sum()

def type2error(y_proba, y_true, threshold=0.5):
    y_pred = (y_proba >= threshold).astype(int)
    fn = ((y_pred == 0) & (y_true == 1)).sum()
    return fn / (y_true == 1).sum()

type1_error = type1error(preds_proba, test_y)
type2_error = type2error(preds_proba, test_y)

# 计算 Acc AUC Prec Rec 的平均值
average_score = (accuracy + auc_score + precision + recall) / 4

# 将结果存入列表
results = [{
    'params': xgb_model.get_params(),
    'accuracy': accuracy,
    'auc_score': auc_score,
    'logloss': logloss,
    'ks_stat': ks,
    'precision': precision,
    'recall': recall,
    'f1': f1,
    'brier_score': brier_score,
    'average_precision': average_precision,
    'hm': hm,
    'gmean': gmean,
    'type1_error': type1_error,
    'type2_error': type2_error,
    'average_score': average_score
}]

# 输出结果
print("Results using default parameters:")
for result in results:
    print(result)

# 保存所有结果到字典并写入文件
results_dict = {'results': results}

dataset = 'bankfear'
method = 'XGBoost_Default'
file_path = os.path.join(save_path, f'{dataset}\\{method}_res.pickle')
os.makedirs(os.path.dirname(file_path), exist_ok=True)

with open(file_path, 'wb') as f:
    pickle.dump(results_dict, f)

print(f'This is ACC: {accuracy}')
print(f'This is AUC: {auc_score}')
print(f'The results of {method} on {dataset} have been calculated and saved.\n\n')
