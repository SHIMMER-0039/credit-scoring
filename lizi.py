import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss, precision_score, recall_score, f1_score, \
    brier_score_loss, roc_curve, average_precision_score
from scipy.stats import ks_2samp
import optuna
from optuna.samplers import TPESampler
import os
import pickle
from lightgbm import LGBMClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression

# 设置路径 (保持不变)
root_path = 'D:/study/Credit(1)/Credit/'
params_path = r'D:\study\Credit(1)\Credit\params/'
dataset_path = r'D:\study\credit_scoring_datasets/'
shuffle_path = r'D:\study\Credit(1)\Credit\shuffle_index/'
save_path = r'D:\study\second\outcome/'
os.makedirs(save_path, exist_ok=True)

data = pd.read_csv(r'D:\study\credit_scroing_datasets\bankfear.csv', low_memory=True)
features = data.drop(['loan_status', 'member_id'], axis=1).replace([-np.inf, np.inf, -np.nan, np.nan], 0)
labels = data['loan_status']

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

# 将训练集和验证集合并用于交叉验证
full_train_x = pd.concat([train_x, valid_x], axis=0)
full_train_y = pd.concat([train_y, valid_y], axis=0)


# 预处理类定义
class SimplifiedDataProcessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.robust_scaler = RobustScaler()
        self.selector = SelectFromModel(LogisticRegression(penalty="l1", solver="liblinear"))
        self.feature_mapping = {}

    def fit(self, X, y=None):
        self.scaler.fit(X)
        self.robust_scaler.fit(X)
        self.selector.fit(X, y)
        return self

    def transform(self, X):
        X_scaled = self.scaler.transform(X)
        X_robust = self.robust_scaler.transform(X)
        X_selected = self.selector.transform(X)

        self.record_feature_mapping(X, X_scaled, 'scaled')
        self.record_feature_mapping(X, X_robust, 'robust')
        self.record_feature_mapping(X, X_selected, 'selected')

        X_selected_expanded = self.expand_selected_features(X_selected, X.shape[1])

        X_transformed = np.hstack((X_scaled, X_robust, X_selected_expanded))
        X_transformed = pd.DataFrame(X_transformed, columns=[f'feature_{i}' for i in range(X_transformed.shape[1])])
        self.update_feature_mapping(X_transformed, X.columns)
        return X_transformed

    def record_feature_mapping(self, original_X, transformed_X, method):
        for i in range(transformed_X.shape[1]):
            original_feature = original_X.columns[i % original_X.shape[1]]
            self.feature_mapping[f'{method}_feature_{i}'] = original_feature

    def expand_selected_features(self, X_selected, n_features):
        expanded_features = np.zeros((X_selected.shape[0], n_features))
        selected_indices = np.where(self.selector.get_support())[0]
        for i, col_idx in enumerate(selected_indices):
            expanded_features[:, col_idx] = X_selected[:, i]
        return expanded_features

    def update_feature_mapping(self, X_transformed, original_columns):
        for i in range(X_transformed.shape[1]):
            original_feature = original_columns[i % len(original_columns)]
            self.feature_mapping[f'feature_{i}'] = original_feature

# 预处理数据
vp = SimplifiedDataProcessor()
vp.fit(full_train_x, full_train_y)
full_train_x_transformed = vp.transform(full_train_x)
test_x_transformed = vp.transform(test_x)

# 定义目标函数
def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 300, 900),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'max_depth': trial.suggest_int('max_depth', 2, 8),
        'reg_lambda': trial.suggest_int('reg_lambda', 3, 7),
        'subsample': trial.suggest_float('subsample', 0.5, 0.9),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.9),
        'objective': 'binary',
        'verbosity': 0,
        'device': 'gpu',
        'gpu_platform_id': 0,
        'gpu_device_id': 0
    }

    # 训练模型
    lgb_model = LGBMClassifier(**params)
    lgb_model.fit(full_train_x_transformed, full_train_y)

    # 预测和评估模型
    preds_proba = lgb_model.predict_proba(test_x_transformed)[:, 1]
    preds = lgb_model.predict(test_x_transformed)

    # 计算AUC和accuracy
    auc_score = roc_auc_score(test_y, preds_proba)
    # accuracy = accuracy_score(test_y, preds)

    # 返回AUC和accuracy的和
    return auc_score

# 使用Optuna进行超参数优化
study = optuna.create_study(direction='maximize', sampler=TPESampler())
study.optimize(objective, n_trials=50)

best_params = study.best_params
print("Best params: ", best_params)

# 使用最佳参数训练模型
best_lgb_model = LGBMClassifier(**best_params)
best_lgb_model.fit(full_train_x_transformed, full_train_y)

# 预测和评估模型
preds_proba = best_lgb_model.predict_proba(test_x_transformed)[:, 1]
preds = best_lgb_model.predict(test_x_transformed)

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

# 输出结果
print(f"Accuracy: {accuracy}, AUC: {auc_score}, Average Score: {average_score}")

# 保存所有结果到字典并写入文件
results_dict = {
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
}

dataset = 'bankfear'
method = 'Optuna'
file_path = os.path.join(save_path, f'{dataset}\\{method}_res.pickle')
os.makedirs(os.path.dirname(file_path), exist_ok=True)

with open(file_path, 'wb') as f:
    pickle.dump(results_dict, f)

print(f'This is ACC: {accuracy}')
print(f'This is AUC: {auc_score}')
print(f'The results of {method} on {dataset} have been calculated and saved.\n\n')
