import os
import pickle
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    RandomForestClassifier,
    GradientBoostingClassifier
)
from sklearn.metrics import (
    accuracy_score, roc_auc_score, log_loss,
    precision_score, recall_score, f1_score,
    brier_score_loss, average_precision_score, roc_curve
)
import xgboost as xgb
import lightgbm as lgb

# =========================================================
# 0. Paths Configuration
# =========================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

dataset_root = os.path.join(BASE_DIR, 'data-raw')
shuffle_root = os.path.join(BASE_DIR, 'data')
save_path = os.path.join(BASE_DIR, 'outcome', 'all_methods')

os.makedirs(save_path, exist_ok=True)

# =========================================================
# 1. Datasets Configuration
# =========================================================
datasets = [
    {
        'name': 'give',
        'file': os.path.join(dataset_root, 'give_me_some_credit_cleaned.csv'),
        'shuffle': os.path.join(shuffle_root, 'give_shuffle_index.pickle'),
        'label': 'SeriousDlqin2yrs',
        'drop': []
    },
    {
        'name': 'shandong',
        'file': os.path.join(dataset_root, 'shandong.csv'),
        'shuffle': os.path.join(shuffle_root, 'shandong_shuffle_index.pickle'),
        'label': 'label',
        'drop': []
    },
    {
        'name': 'fannie',
        'file': os.path.join(dataset_root, 'fannie.csv'),
        'shuffle': os.path.join(shuffle_root, 'fannie_shuffle_index.pickle'),
        'label': 'DEFAULT',
        'drop': ['LOAN IDENTIFIER']
    },
    {
        'name': 'bankfear',
        'file': os.path.join(dataset_root, 'bankfear.csv'),
        'shuffle': os.path.join(shuffle_root, 'bankfear_shuffle_index.pickle'),
        'label': 'loan_status',
        'drop': ['member_id']
    }
]


# =========================================================
# 2. Methods to evaluate
# =========================================================
def get_classifier(name):
    if name == 'LR':
        return LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
    if name == 'LDA':
        return LinearDiscriminantAnalysis()
    if name == 'DT':
        return DecisionTreeClassifier(random_state=42)
    if name == 'KNN':
        return KNeighborsClassifier(n_jobs=-1)
    if name == 'Adaboost':
        return AdaBoostClassifier(random_state=42)
    if name == 'RF':
        return RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    if name == 'GBDT':
        return GradientBoostingClassifier(random_state=42)
    if name == 'XGBOOST':
        return xgb.XGBClassifier(
            use_label_encoder=False,
            eval_metric='logloss',
            tree_method='hist',
            random_state=42
        )
    if name == 'LightGBM':
        return lgb.LGBMClassifier(random_state=42, n_jobs=-1, verbose=-1)  # 添加了 verbose=-1 防止刷屏
    raise ValueError(f'Unknown classifier: {name}')


methods = ['LR', 'LDA', 'DT', 'KNN', 'Adaboost', 'RF', 'GBDT', 'XGBOOST', 'LightGBM']


# =========================================================
# 3. Metric computation
# =========================================================
def compute_metrics(y_true, y_pred, y_proba):
    y_proba = np.clip(y_proba, 0, 1)
    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_proba)
    ll = log_loss(y_true, y_proba)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    bs = brier_score_loss(y_true, y_proba)
    ap = average_precision_score(y_true, y_proba)
    fprs, tprs, _ = roc_curve(y_true, y_proba)
    gmean = np.sqrt(tprs * (1 - fprs)).mean()
    hm = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

    pred = (y_proba >= 0.5).astype(int)
    denom1 = (y_true == 0).sum()
    denom2 = (y_true == 1).sum()
    e1 = ((pred == 1) & (y_true == 0)).sum() / denom1 if denom1 > 0 else 0.0
    e2 = ((pred == 0) & (y_true == 1)).sum() / denom2 if denom2 > 0 else 0.0

    return {
        'accuracy': acc, 'auc': auc, 'logloss': ll,
        'precision': prec, 'recall': rec, 'f1': f1,
        'brier': bs, 'avg_precision': ap,
        'gmean': gmean, 'h_measure': hm,
        'type1_error': e1, 'type2_error': e2
    }


# =========================================================
# 4. Main processing loop
# =========================================================
all_results = {}

for cfg in datasets:
    name = cfg['name']
    print(f"\n" + "=" * 40)
    print(f"--- Dataset: {name} ---")
    print("=" * 40)

    df = pd.read_csv(cfg['file'], low_memory=True)
    X = df.drop(cfg['drop'] + [cfg['label']], axis=1, errors='ignore') \
        .replace([-np.inf, np.inf, np.nan], 0).fillna(0)
    y = df[cfg['label']]

    with open(cfg['shuffle'], 'rb') as f:
        idx = pickle.load(f)

    n = len(idx)

    # train/valid/test split
    if name == 'fannie':
        train_end = int(n * 0.7)
        valid_end = train_end + int(n * 0.15)
    else:
        train_end = int(n * 0.8)
        valid_end = int(n * 0.9)

    train_idx = idx[:train_end]
    valid_idx = idx[train_end:valid_end]
    test_idx = idx[valid_end:]

    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_valid, y_valid = X.iloc[valid_idx], y.iloc[valid_idx]
    X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

    # combine train+valid for fitting
    X_fit = pd.concat([X_train, X_valid], axis=0)
    y_fit = pd.concat([y_train, y_valid], axis=0)

    # evaluate each method
    results = {}
    for m in methods:
        clf = get_classifier(m)

        X_fit_arr = X_fit.values
        y_fit_arr = y_fit.values
        X_test_arr = X_test.values

        clf.fit(X_fit_arr, y_fit_arr)
        y_pred = clf.predict(X_test_arr)
        y_proba = clf.predict_proba(X_test_arr)

        if y_proba.shape[1] == 2:
            y_proba = y_proba[:, 1]
        else:
            y_proba = y_proba[:, 0]

        metrics = compute_metrics(y_test.values, y_pred, y_proba)
        results[m] = metrics
        print(f"{m:10s} | Acc: {metrics['accuracy']:.4f}, AUC: {metrics['auc']:.4f}, Pre: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}")
    all_results[name] = results

# === Save results ===
save_file = os.path.join(save_path, 'all_methods_results.pkl')
with open(save_file, 'wb') as f:
    pickle.dump(all_results, f)
print(f"\n✅ All experiments completed. Results saved to {save_file}")
