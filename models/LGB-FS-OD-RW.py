from __future__ import annotations

import os
import json
import pickle
import random

import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import ks_2samp
from sklearn.metrics import (
    accuracy_score, roc_auc_score,
    precision_score, recall_score, f1_score,
    brier_score_loss, average_precision_score
)

from main.feature_selection import FeatureEvaluator, is_pareto_efficient, evaluate_model
from main.outlier_detection import OutlierDetector
from main.robust_weighting import RobustWeightingModule


# =========================================================
# 0. Config
# =========================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

DATASET_NAME = "give"
DATA_FILE = r'D:\study\credit_scoring_datasets\give_me_some_credit_cleaned.csv'
SHUFFLE_FILE = r'D:\study\Credit(1)\Credit\shuffle_index\give\shuffle_index.pickle'
SAVE_DIR = r'D:\study\second\outcome\give_grid_lgb_with_fs_od'

TARGET_COL = 'SeriousDlqin2yrs'
DROP_COLS = ['SeriousDlqin2yrs']

FEATURE_METHODS = [
    "ClassifierFE",
    "CorrelationFE",
    "GainRFE",
    "InfoGainFE",
    "ReliefFE",
]

REMOVE_OUTLIERS = True

ROBUST_MODE = "auto"

FAST_MODE = False
FAST_MAX_SAMPLES = 50000

os.makedirs(SAVE_DIR, exist_ok=True)


# =========================================================
# 1. Helpers
# =========================================================
def h_mean(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def type1error(y_proba, y_true, threshold=0.5):
    y_pred = (y_proba >= threshold).astype(int)
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    denom = (y_true == 0).sum()
    return fp / denom if denom > 0 else 0.0


def type2error(y_proba, y_true, threshold=0.5):
    y_pred = (y_proba >= threshold).astype(int)
    fn = ((y_pred == 0) & (y_true == 1)).sum()
    denom = (y_true == 1).sum()
    return fn / denom if denom > 0 else 0.0


def safe_precision(y_true, y_pred):
    return precision_score(y_true, y_pred, zero_division=0)


def safe_recall(y_true, y_pred):
    return recall_score(y_true, y_pred, zero_division=0)


def safe_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, zero_division=0)


def to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_serializable(v) for v in obj]
    return obj


# =========================================================
# 2. Feature Selection
# =========================================================
def run_feature_selection(train_x, train_y, valid_x, valid_y, methods):
    candidate_results = []
    for method in methods:
        evaluator = FeatureEvaluator(method=method, random_state=SEED)
        evaluator.fit(train_x.values, train_y)

        scores = np.asarray(evaluator.scores_, dtype=float)
        scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)

        threshold = 0.05 * np.max(scores)
        selected_indices = np.where(scores > threshold)[0].tolist()

        if len(selected_indices) == 0:
            selected_indices = np.argsort(scores)[::-1][:1].tolist()

        selected_indices = sorted(selected_indices)
        selected_names = train_x.columns[selected_indices].tolist()

        train_sub = train_x.iloc[:, selected_indices].values
        valid_sub = valid_x.iloc[:, selected_indices].values

        metric_values = evaluate_model(train_sub, train_y, valid_sub, valid_y)
        metrics = {
            "acc": float(metric_values[0]),
            "auc": float(metric_values[1]),
            "precision": float(metric_values[2]),
            "recall": float(metric_values[3]),
        }

        candidate_results.append({
            "method": method,
            "threshold": float(threshold),
            "selected_indices": selected_indices,
            "selected_names": selected_names,
            "n_features": len(selected_indices),
            "metrics": metrics,
        })
        print(f"[{method}] n_features={len(selected_indices)}, metrics={metrics}")

    score_matrix = np.array([
        [r["metrics"]["acc"], r["metrics"]["auc"], r["metrics"]["precision"], r["metrics"]["recall"]]
        for r in candidate_results
    ])

    pareto_mask = is_pareto_efficient(score_matrix)
    for i, flag in enumerate(pareto_mask):
        candidate_results[i]["pareto_efficient"] = bool(flag)

    pareto_results = [r for r in candidate_results if r["pareto_efficient"]]
    pareto_results = sorted(
        pareto_results,
        key=lambda r: (-r["metrics"]["auc"], -r["metrics"]["recall"], r["n_features"], -r["metrics"]["acc"])
    )

    best = pareto_results[0]

    print("\nSelected feature subset:")
    print(f"  method: {best['method']}")
    print(f"  n_features: {best['n_features']}")
    print(f"  features: {best['selected_names']}")
    print(f"  metrics: {best['metrics']}")

    return best, candidate_results


# =========================================================
# 3. Data loading / FS / Outlier Detection
# =========================================================
print(f"Loading data for {DATASET_NAME}...")
data = pd.read_csv(DATA_FILE, low_memory=True)
data = data.replace([-np.inf, np.inf, np.nan], 0)

if FAST_MODE and len(data) > FAST_MAX_SAMPLES:
    data = data.iloc[:FAST_MAX_SAMPLES].copy()

features = data.drop(DROP_COLS, axis=1)
labels = data[TARGET_COL]

train_size = int(features.shape[0] * 0.8)
valid_size = int(features.shape[0] * 0.1)
test_size = features.shape[0] - train_size - valid_size

with open(SHUFFLE_FILE, 'rb') as f:
    shuffle_index = pickle.load(f)

if FAST_MODE:
    shuffle_index = [i for i in shuffle_index if i < len(features)]

train_index = shuffle_index[:train_size]
valid_index = shuffle_index[train_size:(train_size + valid_size)]
test_index = shuffle_index[(train_size + valid_size):(train_size + valid_size + test_size)]

train_x, train_y = features.iloc[train_index, :], labels.iloc[train_index]
valid_x, valid_y = features.iloc[valid_index, :], labels.iloc[valid_index]
test_x, test_y = features.iloc[test_index, :], labels.iloc[test_index]

full_train_x = pd.concat([train_x, valid_x], axis=0)
full_train_y = pd.concat([train_y, valid_y], axis=0)

print("\nRunning Feature Selection...")
best_fs, fs_all_results = run_feature_selection(
    train_x=train_x, train_y=train_y, valid_x=valid_x, valid_y=valid_y, methods=FEATURE_METHODS
)

selected_indices = best_fs["selected_indices"]
selected_names = best_fs["selected_names"]

filtered_full_train_x = full_train_x.iloc[:, selected_indices].values
filtered_test_x = test_x.iloc[:, selected_indices].values

print("\nRunning Outlier Detection...")
outlier_detector = OutlierDetector(
    z_thresh=3.0,
    iqr_multiplier=1.5,
    contamination="auto",
    vote_threshold=3,
    random_state=SEED,
    lof_n_neighbors=20
)
outlier_detector.fit(filtered_full_train_x)
outlier_mask = outlier_detector.get_outlier_mask()

if REMOVE_OUTLIERS:
    inlier_mask = (outlier_mask == 0)
    filtered_full_train_x_clean = filtered_full_train_x[inlier_mask]
    full_train_y_clean = full_train_y.iloc[inlier_mask].reset_index(drop=True)
else:
    filtered_full_train_x_clean = filtered_full_train_x
    full_train_y_clean = full_train_y.reset_index(drop=True)


# =========================================================
# 4. Robust Weighting 
# =========================================================

robust_module = None
train_sample_weights = None
robust_report_file = None

outlier_ratio = float((outlier_mask == 1).sum()) / len(outlier_mask)

if ROBUST_MODE == "auto":
    if outlier_ratio > 0.05:
        selected_robust_mode = "Huber"
    else:
        selected_robust_mode = "BayesianRidge"


print(f"\nOutlier ratio: {outlier_ratio:.4%}")
print(f"Selected robust mode: {selected_robust_mode}")

if selected_robust_mode in {"Huber", "BayesianRidge"}:
    robust_report_file = os.path.join(
        SAVE_DIR, f"robust_weighting_{selected_robust_mode}_{DATASET_NAME}.json"
    )

    print(f"\nRunning Robust Weighting with {selected_robust_mode}...")
    robust_module = RobustWeightingModule(
        weighting_model=selected_robust_mode,
        huber_epsilon=5.0,
        normalize_weights=True,
        standardize_x=False,
        verbose=True,
    )
    robust_module.fit(filtered_full_train_x_clean, full_train_y_clean)
    train_sample_weights = robust_module.transform_sample_weights(filtered_full_train_x_clean)

    print("Robust sample weights summary:")
    print(f"  min : {train_sample_weights.min():.6f}")
    print(f"  max : {train_sample_weights.max():.6f}")
    print(f"  mean: {train_sample_weights.mean():.6f}")

    robust_module.save_report(robust_report_file)
    print(f"Robust weighting report saved to: {robust_report_file}")

elif selected_robust_mode == "None":
    print("\nRobust weighting disabled. Training without sample weights.")
    train_sample_weights = None

else:
    raise ValueError("ROBUST_MODE must be one of {'auto', 'Huber', 'BayesianRidge', 'None'}")



# =========================================================
# 5. Grid Search: LightGBM Hyperparameters
# =========================================================
PARAM_GRID = {
    "n_estimators": [100,200,300,400,500,600,700,800,900,1000,1100],
    "max_depth": [1,2,3,4,5,6,7,8,9],
    "learning_rate": [0.01,0.02,0.1,0.2]
}

all_results = []
best_result = None
best_score = -np.inf

print("\n" + "=" * 80)
print(f"{'Grid Search Starting':^80}")
print("=" * 80)

for n_estimators in PARAM_GRID["n_estimators"]:
    for max_depth in PARAM_GRID["max_depth"]:
        for learning_rate in PARAM_GRID["learning_rate"]:

            model_params = {
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "learning_rate": learning_rate,
                "n_jobs": -1,
                "random_state": SEED,
            }

            model = lgb.LGBMClassifier(**model_params)

            if train_sample_weights is not None:
                model.fit(
                    filtered_full_train_x_clean,
                    full_train_y_clean,
                    sample_weight=train_sample_weights
                )
            else:
                model.fit(filtered_full_train_x_clean, full_train_y_clean)

            y_proba = model.predict_proba(filtered_test_x)[:, 1]
            y_pred = model.predict(filtered_test_x)

            auc = roc_auc_score(test_y, y_proba)
            acc = accuracy_score(test_y, y_pred)
            prec = safe_precision(test_y, y_pred)
            rec = safe_recall(test_y, y_pred)
            f1 = safe_f1(test_y, y_pred)
            hm = h_mean(prec, rec)
            bs = brier_score_loss(test_y, y_proba)
            ap = average_precision_score(test_y, y_proba)
            ks = ks_2samp(y_proba[test_y == 1], y_proba[test_y != 1]).statistic
            e1 = type1error(y_proba, test_y, threshold=0.5)
            e2 = type2error(y_proba, test_y, threshold=0.5)

            res_entry = {
                "params": model_params,
                "outlier_detection": {
                    "enabled": bool(REMOVE_OUTLIERS),
                    "n_removed": int((outlier_mask == 1).sum()) if REMOVE_OUTLIERS else 0,
                    "n_remaining": int(len(full_train_y_clean)),
                },
                "robust_weighting": {
                    "enabled": ROBUST_MODE in {"Huber", "BayesianRidge"},
                    "mode": ROBUST_MODE,
                    "sample_weight_min": float(train_sample_weights.min()) if train_sample_weights is not None else None,
                    "sample_weight_max": float(train_sample_weights.max()) if train_sample_weights is not None else None,
                    "sample_weight_mean": float(train_sample_weights.mean()) if train_sample_weights is not None else None,
                },
                "metrics": {
                    "auc": float(auc),
                    "acc": float(acc),
                    "f1": float(f1),
                    "ks": float(ks),
                    "hm": float(hm),
                    "brier": float(bs),
                    "prec": float(prec),
                    "rec": float(rec),
                    "ap": float(ap),
                    "type1_err": float(e1),
                    "type2_err": float(e2)
                }
            }
            all_results.append(res_entry)

            print(
                f"Testing >> n_est: {n_estimators:4} | depth: {max_depth:1} | lr: {learning_rate:.2f} "
                f"==> AUC: {auc:.4f} | KS: {ks:.4f} | F1: {f1:.4f}"
            )

            if auc > best_score:
                best_score = auc
                best_result = res_entry


# =========================================================
# 6. Final Summary & Save
# =========================================================
print("\n" + "=" * 80)
print(f"{'BEST MODEL SUMMARY':^80}")
print("=" * 80)
bm = best_result["metrics"]
bp = best_result["params"]

print(f"Robust mode: {ROBUST_MODE}")
print(f"Optimal Params: n_est={bp['n_estimators']}, depth={bp['max_depth']}, lr={bp['learning_rate']}")
print("-" * 80)
print(f"AUC:      {bm['auc']:.6f} | Accuracy: {bm['acc']:.6f} | KS:   {bm['ks']:.6f}")
print(f"F1 Score: {bm['f1']:.6f} | H-Measure: {bm['hm']:.6f} | Brier: {bm['brier']:.6f}")
print(f"Type I:   {bm['type1_err']:.6f} | Type II: {bm['type2_err']:.6f}")
print("=" * 80)

output_final = {
    "dataset": DATASET_NAME,
    "best_overall": best_result,
    "all_combinations": all_results,
    "feature_selection_summary": best_fs,
    "selected_feature_names": selected_names,
    "remove_outliers": REMOVE_OUTLIERS,
    "robust_mode_input": ROBUST_MODE,
    "robust_mode_selected": selected_robust_mode,
    "outlier_ratio": outlier_ratio,
    "robust_report_file": robust_report_file,
}

save_file = os.path.join(SAVE_DIR, f"LGBM_Full_Grid_{ROBUST_MODE}_{DATASET_NAME}.json")
with open(save_file, 'w', encoding='utf-8') as f:
    json.dump(to_serializable(output_final), f, indent=4, ensure_ascii=False)

print(f"\nSaved result file: {save_file}")
