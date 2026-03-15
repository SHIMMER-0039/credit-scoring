from __future__ import annotations

import os
import json
import pickle
import random

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

import xgboost as xgb
import lightgbm as lgb

from main.feature_selection import FeatureEvaluator, is_pareto_efficient, evaluate_model
from main.aaess_attention_stacking import AAESSAttentionStacking


# =========================================================
# 0. Config
# =========================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)


DATA_FILE = r'D:\study\credit_scoring_datasets\FannieMae\2008q1.csv'
SHUFFLE_FILE = r'D:\study\Credit(1)\Credit\shuffle_index\fannie\shuffle_index.pickle'
SAVE_DIR = r'D:\study\second\outcome\fannie'
# =================================

TARGET_COL = 'DEFAULT'
DROP_COLS = ['DEFAULT', 'LOAN IDENTIFIER']

FEATURE_METHODS = [
    "ClassifierFE",
    "CorrelationFE",
    "GainRFE",
    "InfoGainFE",
    "ReliefFE",
]

PARAM_GRID = {
    "n_estimators": [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100],
    "max_depth": [1, 2, 3, 4, 5, 6, 7, 8, 9],
    "learning_rate": [0.01, 0.02, 0.1, 0.2]
}

FAST_MODE = False
FAST_MAX_SAMPLES = 50000

os.makedirs(SAVE_DIR, exist_ok=True)


# 1. Helpers

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


# 2. Feature selection

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
        [
            r["metrics"]["acc"],
            r["metrics"]["auc"],
            r["metrics"]["precision"],
            r["metrics"]["recall"],
        ]
        for r in candidate_results
    ])

    pareto_mask = is_pareto_efficient(score_matrix)
    for i, flag in enumerate(pareto_mask):
        candidate_results[i]["pareto_efficient"] = bool(flag)

    pareto_results = [r for r in candidate_results if r["pareto_efficient"]]
    if len(pareto_results) == 0:
        raise RuntimeError("No Pareto-efficient subset found.")

    pareto_results = sorted(
        pareto_results,
        key=lambda r: (
            -r["metrics"]["auc"],
            -r["metrics"]["recall"],
            r["n_features"],
            -r["metrics"]["acc"],
        )
    )

    best = pareto_results[0]

    print("\nSelected feature subset:")
    print(f"  method: {best['method']}")
    print(f"  n_features: {best['n_features']}")
    print(f"  features: {best['selected_names']}")
    print(f"  metrics: {best['metrics']}")

    return best, candidate_results


# 3. Load data

print("Loading data...")
data = pd.read_csv(DATA_FILE, low_memory=True)
data = data.replace([-np.inf, np.inf, np.nan], 0)

if FAST_MODE and len(data) > FAST_MAX_SAMPLES:
    data = data.iloc[:FAST_MAX_SAMPLES].copy()

features = data.drop(DROP_COLS, axis=1)
labels = data[TARGET_COL]

print(f"Data shape: {data.shape}")
print(f"Feature shape: {features.shape}")
print(f"Target distribution:\n{labels.value_counts(dropna=False)}")

train_size = int(features.shape[0] * 0.7)
valid_size = int(features.shape[0] * 0.15)
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

print(f"Train shape: {train_x.shape}")
print(f"Valid shape: {valid_x.shape}")
print(f"Test shape: {test_x.shape}")



# 4. Feature selection on train/valid only

print("\nRunning feature selection...")
best_fs, fs_all_results = run_feature_selection(
    train_x=train_x,
    train_y=train_y,
    valid_x=valid_x,
    valid_y=valid_y,
    methods=FEATURE_METHODS
)

selected_indices = best_fs["selected_indices"]
selected_names = best_fs["selected_names"]

filtered_train_x = train_x.iloc[:, selected_indices].values
filtered_valid_x = valid_x.iloc[:, selected_indices].values
filtered_test_x = test_x.iloc[:, selected_indices].values
filtered_full_train_x = full_train_x.iloc[:, selected_indices].values



# 5. Grid search AAESS

print("\nUsing parameter grid:")
print(PARAM_GRID)

pos_weight_train = int((train_y == 0).sum() / max(1, (train_y == 1).sum()))
pos_weight_full = int((full_train_y == 0).sum() / max(1, (full_train_y == 1).sum()))

all_results = []
best_result = None
best_score = -np.inf


def build_base_models(n_estimators, max_depth, learning_rate, pos_weight):
    return [
        xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_jobs=-1,
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=SEED,
            tree_method='hist',
            scale_pos_weight=pos_weight
        ),
        lgb.LGBMClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_jobs=-1,
            random_state=SEED,
            min_data_in_leaf=20,
            min_split_gain=0.1,
            class_weight='balanced'
        ),
        RandomForestClassifier(
            n_estimators=n_estimators,
            min_samples_split=5,
            max_features='sqrt',
            n_jobs=-1,
            random_state=SEED,
            class_weight='balanced'
        ),
        AdaBoostClassifier(
            n_estimators=min(n_estimators, 200),
            learning_rate=learning_rate,
            random_state=SEED
        )
    ]


for n_estimators in PARAM_GRID["n_estimators"]:
    for max_depth in PARAM_GRID["max_depth"]:
        for learning_rate in PARAM_GRID["learning_rate"]:

            print("\n" + "=" * 80)
            print(f"Testing params: n_estimators={n_estimators}, max_depth={max_depth}, learning_rate={learning_rate}")
            print("=" * 80)

            # -------------------------
            # Validation-stage fit
            # -------------------------
            valid_model = AAESSAttentionStacking(
                base_models=build_base_models(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    learning_rate=learning_rate,
                    pos_weight=pos_weight_train
                ),
                n_folds=5,
                random_state=SEED,
                use_original_features=False,
                verbose=True
            )
            valid_model.fit(filtered_train_x, train_y.values)

            valid_pred = valid_model.predict(filtered_valid_x)
            valid_proba = valid_model.predict_proba(filtered_valid_x)[:, 1]
            valid_proba = np.clip(valid_proba, 1e-7, 1 - 1e-7)

            valid_acc = accuracy_score(valid_y, valid_pred)
            valid_auc = roc_auc_score(valid_y, valid_proba)

            print("\nValidation results:")
            print(f"AUC: {valid_auc:.6f}")
            print(f"ACC: {valid_acc:.6f}")

            # -------------------------
            # Final fit on train+valid
            # -------------------------
            final_model = AAESSAttentionStacking(
                base_models=build_base_models(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    learning_rate=learning_rate,
                    pos_weight=pos_weight_full
                ),
                n_folds=5,
                random_state=SEED,
                use_original_features=False,
                verbose=True
            )
            final_model.fit(filtered_full_train_x, full_train_y.values)

            # -------------------------
            # Final test evaluation
            # -------------------------
            y_pred = final_model.predict(filtered_test_x)
            y_proba = final_model.predict_proba(filtered_test_x)[:, 1]
            y_proba = np.clip(y_proba, 1e-7, 1 - 1e-7)

            auc = roc_auc_score(test_y, y_proba)
            acc = accuracy_score(test_y, y_pred)

            print("\nFinal test results:")
            print(f"AUC: {auc:.6f}")
            print(f"ACC: {acc:.6f}")

            attention_summary = final_model.get_attention_summary()
            contributions = final_model.get_model_contributions()

            result = {
                "params": {
                    "n_estimators": n_estimators,
                    "max_depth": max_depth,
                    "learning_rate": learning_rate,
                },
                "validation_metrics": {
                    "auc": valid_auc,
                    "acc": valid_acc,
                },
                "test_metrics": {
                    "auc": auc,
                    "acc": acc,
                },
                "attention_summary": attention_summary,
                "model_contributions": contributions,
                "model_params": final_model.get_params(),
            }

            all_results.append(result)

         
            if valid_auc > best_score:
                best_score = valid_auc
                best_result = result


# 6. Save outputs

output = {
    "param_grid": PARAM_GRID,
    "feature_selection_summary": {
        "best": best_fs,
        "all_results": fs_all_results,
    },
    "selected_features": selected_indices,
    "selected_feature_names": selected_names,
    "best_result": best_result,
    "all_results": all_results,
}

pickle_path = os.path.join(SAVE_DIR, "AAESS_fannie_grid_results_auc_acc_only.pickle")
with open(pickle_path, "wb") as f:
    pickle.dump(output, f)

json_path = os.path.join(SAVE_DIR, "AAESS_fannie_grid_results_auc_acc_only.json")
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(to_serializable(output), f, indent=2, ensure_ascii=False)

print(f"\nSaved pickle: {pickle_path}")
print(f"Saved json:   {json_path}")

if best_result is not None:
    print("\nBest params found:")
    print(best_result["params"])
    print(f"Best validation AUC: {best_result['validation_metrics']['auc']:.6f}")
    print(f"Best validation ACC: {best_result['validation_metrics']['acc']:.6f}")
    print(f"Best test AUC: {best_result['test_metrics']['auc']:.6f}")
    print(f"Best test ACC: {best_result['test_metrics']['acc']:.6f}")
