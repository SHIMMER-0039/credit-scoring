import os
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss, precision_score, recall_score, f1_score, \
    brier_score_loss, average_precision_score, roc_curve
from scipy.stats import ks_2samp
import xgboost as xgb
import lightgbm as lgb
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

# Load shuffle index
with open(shuffle_path + 'give/shuffle_index.pickle', 'rb') as f:
    shuffle_index = pickle.load(f)

# Define training proportions to test
train_proportions = [0.4, 0.5, 0.6, 0.7, 0.8]
results_list = []

# Define parameter grid for XGBoost
param_grid = {
    'n_estimators': 600,
    'max_depth': 3,
    'learning_rate': 0.05,
}

# Feature selection methods
feature_methods = ['ClassifierFE','CorrelationFE']

# Iterate over different training proportions
for train_proportion in train_proportions:
    print(f"Training with {train_proportion * 100}% of the data")

    # Split data based on the training proportion
    train_size = int(features.shape[0] * train_proportion)
    valid_size = int(features.shape[0] * 0.1)  # Keep validation size constant
    test_size = valid_size  # Assuming test size is the same as validation size

    train_index = shuffle_index[:train_size]
    valid_index = shuffle_index[train_size:(train_size + valid_size)]
    test_index = shuffle_index[(train_size + valid_size):(train_size + valid_size + test_size)]

    train_x, train_y = features.iloc[train_index, :], labels.iloc[train_index]
    valid_x, valid_y = features.iloc[valid_index, :], labels.iloc[valid_index]
    test_x, test_y = features.iloc[test_index, :], labels.iloc[test_index]

    # Combine train and valid for cross-validation
    full_train_x = pd.concat([train_x, valid_x], axis=0)
    full_train_y = pd.concat([train_y, valid_y], axis=0)

    # Initial feature selection
    selected_features = set()
    for method in feature_methods:
        evaluator = FeatureEvaluator(method=method)
        evaluator.fit(train_x.values, train_y)
        importance_scores = evaluator.scores_
        threshold = 0.05 * np.max(importance_scores)
        selected_features.update(np.where(importance_scores > threshold)[0])

    final_selected_features = list(selected_features)

    # Filter features
    filtered_train_x = train_x.values[:, final_selected_features]
    filtered_valid_x = valid_x.values[:, final_selected_features]
    filtered_test_x = test_x.values[:, final_selected_features]

    # Train the models using Adaptive Bayesian Stacking with homogeneous XGBoost models
    base_models = [
        # xgb.XGBClassifier(n_estimators=param_grid['n_estimators'], max_depth=param_grid['max_depth'],
        #                   learning_rate=param_grid['learning_rate'], n_jobs=-1, use_label_encoder=False,
        #                   eval_metric='logloss', random_state=42, tree_method='hist', device='cuda'),
        # lgb.LGBMClassifier(eval_metric='logloss', random_state=42, tree_method='hist', device='gpu'),
        # RandomForestClassifier(n_estimators=param_grid['n_estimators'], max_depth=param_grid['max_depth'], n_jobs=-1,
        #                        random_state=42)
        # xgb.XGBClassifier(n_estimators=param_grid['n_estimators'], max_depth=param_grid['max_depth'],
        #                   learning_rate=param_grid['learning_rate'], n_jobs=-1, use_label_encoder=False,
        #                   eval_metric='logloss', random_state=42, tree_method='hist', device='cuda'),
        lgb.LGBMClassifier(n_estimators=100, max_depth=param_grid['max_depth'],
                          learning_rate=param_grid['learning_rate'], n_jobs=-1, use_label_encoder=False,
                          eval_metric='logloss', random_state=42, tree_method='hist', device='gpu'),
        # RandomForestClassifier(n_estimators=param_grid['n_estimators'], max_depth=param_grid['max_depth'], n_jobs=-1,
        #                        random_state=42)

        # for _ in range(3)
    ]

    stacking_model = AdaptiveBayesianStacking(base_models=base_models, weight_model=BayesianRidge(), n_folds=5)
    stacking_model.fit(filtered_train_x, train_y)

    # Predict and calculate metrics
    y_pred = stacking_model.predict(filtered_test_x)
    y_pred_proba = stacking_model.predict_proba(filtered_test_x)[:, 1]
    y_pred_proba_clipped = np.clip(y_pred_proba, 0, 1)

    # Calculate performance metrics
    roc_auc = roc_auc_score(test_y, y_pred_proba_clipped)
    accuracy = accuracy_score(test_y, y_pred)
    precision = precision_score(test_y, y_pred)
    recall = recall_score(test_y, y_pred)
    f1 = f1_score(test_y, y_pred)
    brier_score = brier_score_loss(test_y, y_pred_proba_clipped)
    average_precision = average_precision_score(test_y, y_pred_proba_clipped)

    # Save results
    results = {
        'train_proportion': train_proportion,
        'auc': roc_auc,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'brier_score': brier_score,
        'average_precision': average_precision,
        'selected_features': final_selected_features
    }

    results_list.append(results)
    print(f"Results for training proportion {train_proportion}: AUC = {roc_auc:.4f}")

# Save results to file
results_file_path = os.path.join(save_path, 'give_lgb_train_proportion_auc_results_with_features.pickle')
with open(results_file_path, 'wb') as f:
    pickle.dump(results_list, f)

print("Results saved successfully.")
