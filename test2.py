import os
import pickle
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, log_loss, precision_score, recall_score,
    f1_score, brier_score_loss, average_precision_score, roc_curve
)
from scipy.stats import ks_2samp
import xgboost as xgb
import lightgbm as lgb

class AdaptiveVotingStacking(BaseEstimator, ClassifierMixin):
    def __init__(self, base_models, weight_model=LinearRegression(), n_folds=5):
        self.base_models = base_models
        self.weight_model = weight_model
        self.n_folds = n_folds

    def fit(self, X, y):
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

        self.weight_model.fit(out_of_fold_predictions, y)
        return self

    def attention_weights(self, X):
        meta_features = np.column_stack([np.mean([model.predict_proba(X)[:, 1] for model in models], axis=0) for models in self.base_models_])
        weights = self.weight_model.coef_
        attention_scores = np.exp(weights) / np.sum(np.exp(weights))
        return attention_scores

    def predict(self, X):
        meta_features = np.column_stack([np.mean([model.predict_proba(X)[:, 1] for model in models], axis=0) for models in self.base_models_])
        attention_scores = self.attention_weights(X)
        weighted_predictions = np.sum(meta_features * attention_scores, axis=1)
        return (weighted_predictions > 0.5).astype(int)

    def predict_proba(self, X):
        meta_features = np.column_stack([np.mean([model.predict_proba(X)[:, 1] for model in models], axis=0) for models in self.base_models_])
        attention_scores = self.attention_weights(X)
        weighted_predictions = np.sum(meta_features * attention_scores, axis=1)
        weighted_predictions = np.clip(weighted_predictions, 0, 1)
        return np.vstack((1-weighted_predictions, weighted_predictions)).T

def is_pareto_efficient(costs):
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

param_grid = {
    # 'n_estimators': [400, 800, 1000, 1500],
    # 'max_depth': [4, 6, 9],
    # 'learning_rate': [0.02, 0.05, 0.1, 0.2],
    'n_estimators': [1000],
    'max_depth': [9],
    'learning_rate': [0.02],
}

all_results = []
best_results = None
best_params = None

for n_estimators in param_grid['n_estimators']:
    for max_depth in param_grid['max_depth']:
        for learning_rate in param_grid['learning_rate']:
            base_models = [
                RandomForestClassifier(n_estimators=n_estimators, min_samples_split=5, max_features='sqrt', n_jobs=-1, random_state=42),
                xgb.XGBClassifier(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, n_jobs=-1, use_label_encoder=False, eval_metric='logloss', random_state=42),
                lgb.LGBMClassifier(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, n_jobs=-1, random_state=42),
            ]

            stacking_model = AdaptiveVotingStacking(base_models=base_models, weight_model=LinearRegression(), n_folds=5)
            stacking_model.fit(full_train_x.values, full_train_y.values)

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

dataset = 'LC'
method = 'adaa'
file_path = os.path.join(save_path, f'{dataset}\\{method}_res.pickle')
os.makedirs(os.path.dirname(file_path), exist_ok=True)

with open(file_path, 'wb') as f:
    pickle.dump(results_dict, f)

print(f'This is ACC: {best_result["accuracy"]}')
print(f'This is AUC: {best_result["roc_auc"]}')
print(f'The results of {method} on {dataset} have been calculated and saved.\n\n')
