import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss, precision_score, recall_score, f1_score, \
    brier_score_loss, roc_curve, average_precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
from sklearn.base import BaseEstimator, ClassifierMixin, clone
import os
import pickle


# 自定义 Stacking 类
class StackingClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, base_models, meta_model, n_folds=5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds

    def fit(self, X, y):
        self.base_models_ = [list() for _ in self.base_models]
        self.meta_model_ = clone(self.meta_model)
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)

        # 使用 K 折交叉验证训练基础模型
        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))
        for i, model in enumerate(self.base_models):
            for train_idx, holdout_idx in kfold.split(X, y):
                instance = clone(model)
                self.base_models_[i].append(instance)
                instance.fit(X[train_idx], y[train_idx])
                y_pred = instance.predict(X[holdout_idx])
                out_of_fold_predictions[holdout_idx, i] = y_pred

        # 使用基础模型的预测结果训练元学习器
        self.meta_model_.fit(out_of_fold_predictions, y)
        return self

    def predict(self, X):
        meta_features = np.column_stack([
            np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)
            for base_models in self.base_models_
        ])
        return self.meta_model_.predict(meta_features)

    def predict_proba(self, X):
        meta_features = np.column_stack([
            np.column_stack([model.predict_proba(X)[:, 1] for model in base_models]).mean(axis=1)
            for base_models in self.base_models_
        ])
        return self.meta_model_.predict_proba(meta_features)


# 设置路径
root_path = 'D:/study/Credit(1)/Credit/'
params_path = r'D:\study\Credit(1)\Credit\params/'
dataset_path = r'D:\study\credit_scoring_datasets/'
shuffle_path = r'D:\study\Credit(1)\Credit\shuffle_index/'
save_path = r'D:\study\second\outcome/'
os.makedirs(save_path, exist_ok=True)

# 加载新的数据集
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

# 确保索引范围正确
print(f"Total data size: {features.shape[0]}")
print(f"Train indices: {train_index[:5]}...{train_index[-5:]}")
print(f"Valid indices: {valid_index[:5]}...{valid_index[-5:]}")
print(f"Test indices: {test_index[:5]}...{test_index[-5:]}")
print(f"Remaining indices: {remaining_index[:5]}...{remaining_index[-5:]}")

# 将训练集和验证集合并用于交叉验证
full_train_x = pd.concat([train_x, valid_x], axis=0)
full_train_y = pd.concat([train_y, valid_y], axis=0)

# 定义基础模型和元学习器
base_models = [
    RandomForestClassifier(n_estimators=100, random_state=42),
    xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
    lgb.LGBMClassifier()
]
meta_model = LogisticRegression()

# 初始化并训练 Stacking 模型
stacking_model = StackingClassifier(base_models=base_models, meta_model=meta_model, n_folds=5)
stacking_model.fit(full_train_x.values, full_train_y.values)

# 预测并评估
y_pred = stacking_model.predict(test_x.values)
y_pred_proba = stacking_model.predict_proba(test_x.values)[:, 1]

accuracy = accuracy_score(test_y, y_pred)
roc_auc = roc_auc_score(test_y, y_pred_proba)
logloss = log_loss(test_y, y_pred_proba)
ks = ks_2samp(y_pred_proba[test_y == 1], y_pred_proba[test_y != 1]).statistic
precision = precision_score(test_y, y_pred)
recall = recall_score(test_y, y_pred)
f1 = f1_score(test_y, y_pred)
brier_score = brier_score_loss(test_y, y_pred_proba)
average_precision = average_precision_score(test_y, y_pred_proba)
fprs, tprs, thresholds = roc_curve(test_y, y_pred_proba)
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


type1_error = type1error(y_pred_proba, test_y)
type2_error = type2error(y_pred_proba, test_y)

# 计算 Acc AUC Prec Rec 的平均值
average_score = (accuracy + roc_auc + precision + recall) / 4

# 将结果存入列表
results = [{
    'params': stacking_model.get_params(),
    'accuracy': accuracy,
    'roc_auc': roc_auc,
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
print("Results using stacking model:")
for result in results:
    print(result)

# 保存所有结果到字典并写入文件
results_dict = {'results': results}

dataset = 'bankfear'
method = 'Stacking_XGB_LGB_RF'
file_path = os.path.join(save_path, f'{dataset}\\{method}_res.pickle')
os.makedirs(os.path.dirname(file_path), exist_ok=True)

with open(file_path, 'wb') as f:
    pickle.dump(results_dict, f)

print(f'This is ACC: {accuracy}')
print(f'This is AUC: {roc_auc}')
print(f'The results of {method} on {dataset} have been calculated and saved.\n\n')
