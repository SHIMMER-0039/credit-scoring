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

# === Paths ===
root_path    = 'D:/study/Credit(1)/Credit/'
dataset_path = r'D:/study/credit_scoring_datasets/'
shuffle_path = r'D:/study/Credit(1)/Credit/shuffle_index/'
save_path    = r'D:/study/second/outcome/all_methods/'
os.makedirs(save_path, exist_ok=True)

# === Dataset configurations ===
datasets = [
    {
        'name': 'give',
        'file': 'give_me_some_credit_cleaned.csv',
        'label': 'SeriousDlqin2yrs',
        'drop': []
    },
    {
        'name': 'shandong',
        'file': 'shandong.csv',
        'label': 'label',
        'drop': []
    },
    {
        'name': 'fannie',
        'file': os.path.join('FannieMae','2008q1.csv'),
        'label': 'DEFAULT',
        'drop': ['LOAN IDENTIFIER']
    },
    {
        'name': 'bankfear',
        'file': 'bankfear.csv',
        'label': 'loan_status',
        'drop': ['member_id']
    }
]

# === Methods to evaluate ===
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
        return lgb.LGBMClassifier(random_state=42, n_jobs=-1)
    raise ValueError(f'Unknown classifier: {name}')

methods = ['LR','LDA','DT','KNN','Adaboost','RF','GBDT','XGBOOST','LightGBM']

# === Metric computation ===
def compute_metrics(y_true, y_pred, y_proba):
    y_proba = np.clip(y_proba, 0, 1)
    acc   = accuracy_score(y_true, y_pred)
    auc   = roc_auc_score(y_true, y_proba)
    ll    = log_loss(y_true, y_proba)
    prec  = precision_score(y_true, y_pred)
    rec   = recall_score(y_true, y_pred)
    f1    = f1_score(y_true, y_pred)
    bs    = brier_score_loss(y_true, y_proba)
    ap    = average_precision_score(y_true, y_proba)
    fprs, tprs, _ = roc_curve(y_true, y_proba)
    gmean = np.sqrt(tprs * (1 - fprs)).mean()
    hm    = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    # type I / II errors at threshold=0.5
    pred = (y_proba >= 0.5).astype(int)
    e1 = ((pred==1)&(y_true==0)).sum() / (y_true==0).sum()
    e2 = ((pred==0)&(y_true==1)).sum() / (y_true==1).sum()
    return {
        'accuracy': acc, 'auc': auc, 'logloss': ll,
        'precision': prec, 'recall': rec, 'f1': f1,
        'brier': bs, 'avg_precision': ap,
        'gmean': gmean, 'h_measure': hm,
        'type1_error': e1, 'type2_error': e2
    }

# === Main processing loop ===
all_results = {}

for cfg in datasets:
    name = cfg['name']
    print(f"\n--- Dataset: {name} ---")
    # load data
    df = pd.read_csv(os.path.join(dataset_path, cfg['file']), low_memory=True)
    X  = df.drop(cfg['drop'] + [cfg['label']], axis=1, errors='ignore') \
           .replace([-np.inf, np.inf, np.nan], 0).fillna(0)
    y  = df[cfg['label']]

    # load shuffled indices
    with open(os.path.join(shuffle_path, name, 'shuffle_index.pickle'), 'rb') as f:
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
    test_idx  = idx[valid_end:]

    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_valid, y_valid = X.iloc[valid_idx], y.iloc[valid_idx]
    X_test,  y_test  = X.iloc[test_idx],  y.iloc[test_idx]

    # combine train+valid for fitting
    X_fit = pd.concat([X_train, X_valid], axis=0)
    y_fit = pd.concat([y_train, y_valid], axis=0)

    # evaluate each method
    results = {}
    for m in methods:
        clf = get_classifier(m)
        clf.fit(X_fit, y_fit)
        y_pred  = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)[:, 1]
        metrics = compute_metrics(y_test, y_pred, y_proba)
        results[m] = metrics
        print(f"{m:10s} | Acc: {metrics['accuracy']:.4f}, AUC: {metrics['auc']:.4f}")
    all_results[name] = results

# === (Optional) Save results ===
with open(os.path.join(save_path, 'all_methods_results.pkl'), 'wb') as f:
    pickle.dump(all_results, f)
