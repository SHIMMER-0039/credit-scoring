import os
import pickle
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss, precision_score, recall_score, f1_score, brier_score_loss, average_precision_score, roc_curve, mutual_info_score
from scipy.stats import ks_2samp
import xgboost as xgb
import lightgbm as lgb
from imblearn.over_sampling import SMOTE
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import mutual_info_classif
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors

# 离群检测函数
def detect_outliers_lof(X, n_neighbors=20):
    lof = LocalOutlierFactor(n_neighbors=n_neighbors)
    lof.fit(X)
    return -lof.negative_outlier_factor_

class FeatureEvaluator:
    def __init__(self, method='ClassifierFE'):
        self.method = method
        np.random.seed(42)  # Ensure reproducibility

    def fit(self, X, y):
        if self.method == 'ClassifierFE':
            self.scores_ = self._classifier_fe(X, y)
        elif self.method == 'CorrelationFE':
            self.scores_ = self._correlation_fe(X, y)
        elif self.method == 'InfoGainFE':
            self.scores_ = self._infogain_fe(X, y)
        elif self.method == 'ReliefFE':
            self.scores_ = self._relief_fe(X, y)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        return self

    def _classifier_fe(self, X, y):
        clf = DecisionTreeClassifier()
        clf.fit(X, y)
        return clf.feature_importances_

    def _correlation_fe(self, X, y):
        return np.array([pearsonr(X[:, i], y)[0] for i in range(X.shape[1])])

    def _infogain_fe(self, X, y):
        return mutual_info_classif(X, y)  # 并行化计算

    def _relief_fe(self, X, y):
        return self._relief(X, y)

    def _relief(self, X, y):
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        m = X.shape[0]
        scores = np.zeros(X.shape[1])
        for i in range(m):
            instance = X[i, :]
            same_class = X[y == y[i]]
            diff_class = X[y != y[i]]
            nearest_same = self._nearest(instance, same_class)
            nearest_diff = self._nearest(instance, diff_class)
            scores += np.abs(instance - nearest_diff) - np.abs(instance - nearest_same)
        return scores / m

    def _nearest(self, instance, data):
        if data.shape[0] == 0:
            return np.zeros(instance.shape)
        nbrs = NearestNeighbors(n_neighbors=1).fit(data)
        distances, indices = nbrs.kneighbors([instance])
        return data[indices[0][0]]

    def transform(self, X):
        return X[:, self.scores_.argsort()[::-1]]

class AdaptiveVotingStacking(BaseEstimator, ClassifierMixin):
    def __init__(self, base_models, weight_model=LinearRegression(), n_folds=5, feature_evaluator=None):
        self.base_models = base_models
        self.weight_model = weight_model
        self.n_folds = n_folds
        self.feature_evaluator = feature_evaluator

    def fit(self, X, y):
        if self.feature_evaluator:
            self.feature_evaluator.fit(X, y)
            X = self.feature_evaluator.transform(X)

        self.base_models_ = [list() for _ in self.base_models]
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)

        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))
        for i, model in enumerate(self.base_models):
            for train_idx, holdout_idx in kfold.split(X, y):
                instance = clone(model)
                self.base_models_[i].append(instance)
                instance.fit(X[train_idx], y[train_idx])
                y_pred = instance.predict_proba(X[holdout_idx])[:, 1]
                out_of_fold_predictions[holdout_idx, i] = y_pred

        lof_scores = detect_outliers_lof(out_of_fold_predictions)
        enhanced_features = pd.DataFrame({'LOF_score': lof_scores})

        scaler = StandardScaler()
        enhanced_features = pd.DataFrame(scaler.fit_transform(enhanced_features), columns=enhanced_features.columns)

        self.X_columns = np.hstack([['base_model_{}'.format(i) for i in range(out_of_fold_predictions.shape[1])], enhanced_features.columns])
        X_transformed = np.hstack([out_of_fold_predictions, enhanced_features.values])
        self.weight_model.fit(X_transformed, y)
        return self

    def attention_weights(self, X_transformed):
        weights = self.weight_model.coef_
        attention_scores = np.exp(weights) / np.sum(np.exp(weights))
        return attention_scores

    def predict(self, X):
        if self.feature_evaluator:
            X = self.feature_evaluator.transform(X)
        meta_features = np.column_stack([np.mean([model.predict_proba(X)[:, 1] for model in models], axis=0) for models in self.base_models_])
        lof_scores = detect_outliers_lof(meta_features)
        enhanced_features = pd.DataFrame({'LOF_score': lof_scores})

        scaler = StandardScaler()
        enhanced_features = pd.DataFrame(scaler.fit_transform(enhanced_features), columns=enhanced_features.columns)

        X_transformed = np.hstack([meta_features, enhanced_features.values])
        X_transformed = pd.DataFrame(X_transformed, columns=self.X_columns)
        attention_scores = self.attention_weights(X_transformed.values)
        weighted_predictions = np.sum(meta_features * attention_scores[:meta_features.shape[1]], axis=1)
        return (weighted_predictions > 0.5).astype(int)

    def predict_proba(self, X):
        if self.feature_evaluator:
            X = self.feature_evaluator.transform(X)
        meta_features = np.column_stack([np.mean([model.predict_proba(X)[:, 1] for model in models], axis=0) for models in self.base_models_])
        lof_scores = detect_outliers_lof(meta_features)
        enhanced_features = pd.DataFrame({'LOF_score': lof_scores})

        scaler = StandardScaler()
        enhanced_features = pd.DataFrame(scaler.fit_transform(enhanced_features), columns=enhanced_features.columns)

        X_transformed = np.hstack([meta_features, enhanced_features.values])
        X_transformed = pd.DataFrame(X_transformed, columns=self.X_columns)
        attention_scores = self.attention_weights(X_transformed.values)
        weighted_predictions = np.sum(meta_features * attention_scores[:meta_features.shape[1]], axis=1)
        weighted_predictions = np.clip(weighted_predictions, 0, 1)
        return np.vstack((1-weighted_predictions, weighted_predictions)).T


def is_pareto_efficient(costs):
    """
    Find the pareto-efficient points
    :param costs: An (n_points, n_costs) array
    :return: A boolean array of pareto efficient points
    """
    is_efficient = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        is_efficient[i] = np.all(np.any(costs[:i] > c, axis=1)) and np.all(np.any(costs[i+1:] > c, axis=1))
    return is_efficient

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

# 创建 SMOTE 实例
smote = SMOTE(random_state=42)

# 应用 SMOTE 进行过采样
train_x_resampled, train_y_resampled = smote.fit_resample(full_train_x, full_train_y)

# 定义参数组合
param_grid = {
    'n_estimators': [400, 800, 1000, 1500],
    'max_depth': [4, 6, 9],
    'learning_rate': [0.02, 0.05, 0.1, 0.2],
}

all_results = []
best_results = None
best_params = None

feature_methods = ['ClassifierFE', 'CorrelationFE', 'InfoGainFE']
# feature_methods = ['ClassifierFE']
feature_method_scores = []

for method in feature_methods:
    feature_evaluator = FeatureEvaluator(method=method)
    feature_evaluator.fit(train_x_resampled.values, train_y_resampled)
    transformed_train_x = feature_evaluator.transform(train_x_resampled.values)
    transformed_valid_x = feature_evaluator.transform(valid_x.values)

    # 使用一个简单模型评估特征选择的效果
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(transformed_train_x, train_y_resampled)
    valid_pred_proba = model.predict_proba(transformed_valid_x)[:, 1]
    valid_accuracy = accuracy_score(valid_y, model.predict(transformed_valid_x))
    valid_roc_auc = roc_auc_score(valid_y, valid_pred_proba)
    valid_precision = precision_score(valid_y, model.predict(transformed_valid_x))
    valid_recall = recall_score(valid_y, model.predict(transformed_valid_x))

    feature_method_scores.append([valid_accuracy, valid_roc_auc, valid_precision, valid_recall])

feature_method_scores = np.array(feature_method_scores)
pareto_efficient = is_pareto_efficient(-feature_method_scores)  # -scores because we want to maximize scores

best_method_idx = np.argmax(pareto_efficient)
best_method = feature_methods[best_method_idx]

print(f"Best feature selection method: {best_method}")

# 使用最佳特征选择方法
feature_evaluator = FeatureEvaluator(method=best_method)
feature_evaluator.fit(train_x_resampled.values, train_y_resampled)

for n_estimators in param_grid['n_estimators']:
    for max_depth in param_grid['max_depth']:
        for learning_rate in param_grid['learning_rate']:
            base_models = [
                RandomForestClassifier(n_estimators=n_estimators, min_samples_split=5, max_features='sqrt', n_jobs=-1, random_state=42),
                xgb.XGBClassifier(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, n_jobs=-1, use_label_encoder=False, eval_metric='logloss', random_state=42),
                lgb.LGBMClassifier(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, n_jobs=-1, random_state=42),
            ]

            stacking_model = AdaptiveVotingStacking(base_models=base_models, weight_model=LinearRegression(), n_folds=5, feature_evaluator=feature_evaluator)
            stacking_model.fit(train_x_resampled.values, train_y_resampled)

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

            type1_error = type1error(y_pred_proba, test_y)
            type2_error = type2error(y_pred_proba, test_y)

            average_score = (accuracy + roc_auc + precision + recall) / 4

            results = {
                'n_estimators': n_estimators,
                'max_depth': max_depth,
                'learning_rate': learning_rate,
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
            }

            all_results.append(results)

            print(f"Results for n_estimators={n_estimators}, max_depth={max_depth}, learning_rate={learning_rate}:")
            for key, value in results.items():
                print(f"{key}: {value}")

            if best_results is None or average_score > best_results['average_score']:
                best_results = results
                best_params = (n_estimators, max_depth, learning_rate)

print("所有结果：")
for result in all_results:
    print(result)

best_result = max(all_results, key=lambda x: x['average_score'])
print("最佳结果：", best_result)

results_dict = {'results': all_results, 'best_result': best_result}

save_path = r'D:\study\second\outcome/'  # 确保save_path正确
dataset = 'LC'
method = 'adaa'
file_path = os.path.join(save_path, f'{dataset}\\{method}_res.pickle')
os.makedirs(os.path.dirname(file_path), exist_ok=True)

with open(file_path, 'wb') as f:
    pickle.dump(results_dict, f)

print(f'This is ACC: {best_result["accuracy"]}')
print(f'This is AUC: {best_result["roc_auc"]}')
print(f'The results of {method} on {dataset} have been calculated and saved.\n\n')
