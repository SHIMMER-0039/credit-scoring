from __future__ import annotations

import os
import json
import pickle
import random

import numpy as np
import pandas as pd

from scipy.stats import ks_2samp

from sklearn.base import clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    log_loss,
    precision_score,
    recall_score,
    f1_score,
    brier_score_loss,
    average_precision_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

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
SAVE_DIR = r'D:\study\second\outcome\table10'
os.makedirs(SAVE_DIR, exist_ok=True)

TARGET_COL = 'DEFAULT'
DROP_COLS = ['DEFAULT', 'LOAN IDENTIFIER']

FEATURE_METHODS = [
    "ClassifierFE",
    "CorrelationFE",
    "GainRFE",
    "InfoGainFE",
    "ReliefFE",
]

USE_FEATURE_SELECTION = True

# 先给一个固定参数，方便直接跑
N_ESTIMATORS = 300
MAX_DEPTH = 3
LEARNING_RATE = 0.1

# 调试开关
FAST_MODE = False
FAST_MAX_SAMPLES = 50000


# =========================================================
# 1. Helpers
# =========================================================
def safe_precision(y_true, y_pred):
    return precision_score(y_true, y_pred, zero_division=0)


def safe_recall(y_true, y_pred):
    return recall_score(y_true, y_pred, zero_division=0)


def safe_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, zero_division=0)


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


def compute_metrics(y_true, y_pred, y_proba):
    y_proba = np.clip(np.asarray(y_proba, dtype=float), 1e-7, 1 - 1e-7)

    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_proba)
    prec = safe_precision(y_true, y_pred)
    rec = safe_recall(y_true, y_pred)
    f1 = safe_f1(y_true, y_pred)
    ll = log_loss(y_true, y_proba)
    bs = brier_score_loss(y_true, y_proba)
    ap = average_precision_score(y_true, y_proba)
    ks = ks_2samp(y_proba[y_true == 1], y_proba[y_true != 1]).statistic
    hm = h_mean(prec, rec)
    e1 = type1error(y_proba, y_true)
    e2 = type2error(y_proba, y_true)

    return {
        "acc": acc,
        "auc": auc,
        "prec": prec,
        "rec": rec,
        "f1": f1,
        "logloss": ll,
        "bs": bs,
        "ap": ap,
        "ks": ks,
        "hm": hm,
        "e1": e1,
        "e2": e2,
    }


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
# 2. Data / Feature Selection
# =========================================================
def load_data():
    print("Loading data...")
    data = pd.read_csv(DATA_FILE, low_memory=True)
    data = data.replace([-np.inf, np.inf, np.nan], 0)

    if FAST_MODE and len(data) > FAST_MAX_SAMPLES:
        data = data.iloc[:FAST_MAX_SAMPLES].copy()

    features = data.drop(DROP_COLS, axis=1)
    labels = data[TARGET_COL]

    with open(SHUFFLE_FILE, "rb") as f:
        shuffle_index = pickle.load(f)

    if FAST_MODE:
        shuffle_index = [i for i in shuffle_index if i < len(features)]

    train_size = int(len(features) * 0.7)
    valid_size = int(len(features) * 0.15)
    test_size = len(features) - train_size - valid_size

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

    return train_x, train_y, valid_x, valid_y, test_x, test_y, full_train_x, full_train_y


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


# =========================================================
# 3. AAESS-LR (LR weighting)
# =========================================================
class LRWeightingModule:
    def __init__(self, standardize_x=True):
        self.standardize_x = standardize_x
        self.scaler_ = None
        self.model_ = None
        self.sample_weights_ = None
        self.sample_reliability_vector_ = None
        self.model_confidence_ = None

    def _prepare_x(self, X, fit=False):
        X = np.asarray(X, dtype=float)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        if not self.standardize_x:
            return X

        if fit:
            self.scaler_ = StandardScaler()
            return self.scaler_.fit_transform(X)

        return self.scaler_.transform(X)

    def fit(self, X, y):
        Xs = self._prepare_x(X, fit=True)
        y = np.asarray(y).ravel()

        model = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=SEED
        )
        model.fit(Xs, y)

        proba = model.predict_proba(Xs)[:, 1]

        sample_weights = 1.0 - np.abs(proba - y.astype(float))
        sample_weights = np.maximum(sample_weights, 1e-12)
        sample_weights = sample_weights / np.mean(sample_weights)

        reliability_vector = Xs * sample_weights[:, None]

        coef_abs = np.abs(model.coef_.ravel())
        coef_abs = np.maximum(coef_abs, 1e-12)

        self.model_ = model
        self.sample_weights_ = sample_weights
        self.sample_reliability_vector_ = reliability_vector
        self.model_confidence_ = coef_abs
        return self

    def transform_reliability_vector(self, X):
        Xs = self._prepare_x(X, fit=False)
        proba = self.model_.predict_proba(Xs)[:, 1]

        sample_weights = np.abs(proba - 0.5) * 2.0
        sample_weights = np.maximum(sample_weights, 1e-12)
        sample_weights = sample_weights / np.mean(sample_weights)

        return Xs * sample_weights[:, None]

    def get_model_confidence(self):
        return self.model_confidence_


class AAESSAttentionStackingLR:
    def __init__(
        self,
        base_models,
        n_folds=5,
        meta_model=None,
        use_original_features=False,
        random_state=SEED,
        verbose=False
    ):
        self.base_models = base_models
        self.n_folds = n_folds
        self.meta_model = meta_model if meta_model is not None else LogisticRegression(
            max_iter=1000,
            class_weight="balanced"
        )
        self.use_original_features = use_original_features
        self.random_state = random_state
        self.verbose = verbose

        self.weighting_module_ = None
        self.fitted_base_models_ = None
        self.meta_model_ = None
        self.model_confidences_ = None
        self.train_attention_weights_ = None

    @staticmethod
    def _softmax(x, axis=1):
        x = x - np.max(x, axis=axis, keepdims=True)
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

    def _generate_oof_predictions(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).ravel()

        n_samples = X.shape[0]
        n_models = len(self.base_models)
        oof_predictions = np.zeros((n_samples, n_models), dtype=float)

        skf = StratifiedKFold(
            n_splits=self.n_folds,
            shuffle=True,
            random_state=self.random_state
        )

        fitted_base_models = []

        for j, base_model in enumerate(self.base_models):
            for tr_idx, va_idx in skf.split(X, y):
                model = clone(base_model)
                model.fit(X[tr_idx], y[tr_idx])

                if hasattr(model, "predict_proba"):
                    pred = model.predict_proba(X[va_idx])[:, 1]
                else:
                    pred = model.predict(X[va_idx])

                oof_predictions[va_idx, j] = pred

            final_model = clone(base_model)
            final_model.fit(X, y)
            fitted_base_models.append(final_model)

        return oof_predictions, fitted_base_models

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        y = np.asarray(y).ravel()

        self.weighting_module_ = LRWeightingModule(standardize_x=True)
        self.weighting_module_.fit(X, y)

        reliability_vectors = self.weighting_module_.sample_reliability_vector_
        base_feature_confidence = self.weighting_module_.get_model_confidence()

        oof_predictions, fitted_base_models = self._generate_oof_predictions(X, y)

        model_confidences = []
        for j in range(oof_predictions.shape[1]):
            pred_j = np.clip(oof_predictions[:, j], 1e-12, 1 - 1e-12)
            bs = brier_score_loss(y, pred_j)
            multiplier = 1.0 / max(bs, 1e-12)
            gamma_j = base_feature_confidence * multiplier
            model_confidences.append(gamma_j)

        model_confidences = np.vstack(model_confidences)

        scores = reliability_vectors @ model_confidences.T
        attention_weights = self._softmax(scores, axis=1)
        weighted_oof_predictions = attention_weights * oof_predictions

        if self.use_original_features:
            meta_features = np.hstack([X, weighted_oof_predictions])
        else:
            meta_features = weighted_oof_predictions

        meta_model = clone(self.meta_model)
        meta_model.fit(meta_features, y)

        self.fitted_base_models_ = fitted_base_models
        self.meta_model_ = meta_model
        self.model_confidences_ = model_confidences
        self.train_attention_weights_ = attention_weights
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        n_samples = X.shape[0]
        n_models = len(self.fitted_base_models_)
        base_predictions = np.zeros((n_samples, n_models), dtype=float)

        for j, model in enumerate(self.fitted_base_models_):
            if hasattr(model, "predict_proba"):
                pred = model.predict_proba(X)[:, 1]
            else:
                pred = model.predict(X)
            base_predictions[:, j] = pred

        reliability_vectors = self.weighting_module_.transform_reliability_vector(X)
        scores = reliability_vectors @ self.model_confidences_.T
        attention_weights = self._softmax(scores, axis=1)
        weighted_predictions = attention_weights * base_predictions

        if self.use_original_features:
            meta_features = np.hstack([X, weighted_predictions])
        else:
            meta_features = weighted_predictions

        return self.meta_model_.predict_proba(meta_features)

    def predict(self, X, threshold=0.5):
        proba = self.predict_proba(X)[:, 1]
        return (proba >= threshold).astype(int)

    def get_model_contributions(self):
        mean_att = self.train_attention_weights_.mean(axis=0)
        total = np.sum(mean_att)
        if total <= 0:
            return mean_att
        return mean_att / total


# =========================================================
# 4. Model builders
# =========================================================
def make_dt():
    return DecisionTreeClassifier(max_depth=5, random_state=SEED)


def make_lda():
    return make_pipeline(StandardScaler(), LinearDiscriminantAnalysis())


def make_lr():
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED),
    )


def make_rf():
    return RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_features="sqrt",
        min_samples_split=5,
        n_jobs=-1,
        random_state=SEED,
        class_weight="balanced",
    )


def make_xgb(pos_weight):
    return xgb.XGBClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        learning_rate=LEARNING_RATE,
        n_jobs=-1,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=SEED,
        tree_method="hist",
        scale_pos_weight=pos_weight,
    )


def make_lgb():
    return lgb.LGBMClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        learning_rate=LEARNING_RATE,
        n_jobs=-1,
        random_state=SEED,
        min_data_in_leaf=20,
        min_split_gain=0.1,
        class_weight="balanced",
    )


def make_ada():
    return AdaBoostClassifier(
        n_estimators=min(N_ESTIMATORS, 200),
        learning_rate=LEARNING_RATE,
        random_state=SEED,
    )


def make_gbdt():
    return GradientBoostingClassifier(
        n_estimators=N_ESTIMATORS,
        learning_rate=LEARNING_RATE,
        max_depth=MAX_DEPTH,
        random_state=SEED,
    )


def build_homogeneous_stack(base_estimator, n_clones=3):
    base_models = [clone(base_estimator) for _ in range(n_clones)]
    return AAESSAttentionStacking(
        base_models=base_models,
        n_folds=5,
        random_state=SEED,
        use_original_features=False,
        meta_model=LogisticRegression(max_iter=1000, class_weight="balanced"),
        verbose=False,
    )


def build_table10_model(name, pos_weight):
    if name == "DT":
        return build_homogeneous_stack(make_dt())

    elif name == "LDA":
        return build_homogeneous_stack(make_lda())

    elif name == "LR":
        return build_homogeneous_stack(make_lr())

    elif name == "RF":
        return build_homogeneous_stack(make_rf())

    elif name == "XGB":
        return build_homogeneous_stack(make_xgb(pos_weight))

    elif name == "LGB":
        return build_homogeneous_stack(make_lgb())

    elif name == "AAESSLR":
        return AAESSAttentionStackingLR(
            base_models=[
                make_rf(),
                make_xgb(pos_weight),
                make_lgb(),
                make_ada(),
            ],
            n_folds=5,
            random_state=SEED,
            use_original_features=False,
            meta_model=LogisticRegression(max_iter=1000, class_weight="balanced"),
            verbose=False,
        )

    elif name == "extreetree":
        return AAESSAttentionStacking(
            base_models=[
                make_dt(),
                make_rf(),
                make_ada(),
            ],
            n_folds=5,
            random_state=SEED,
            use_original_features=False,
            meta_model=LogisticRegression(max_iter=1000, class_weight="balanced"),
            verbose=False,
        )

    elif name == "AAESSTREE":
        return AAESSAttentionStacking(
            base_models=[
                make_dt(),
                make_gbdt(),
                make_xgb(pos_weight),
            ],
            n_folds=5,
            random_state=SEED,
            use_original_features=False,
            meta_model=LogisticRegression(max_iter=1000, class_weight="balanced"),
            verbose=False,
        )

    elif name == "AAESS":
        return AAESSAttentionStacking(
            base_models=[
                make_rf(),
                make_xgb(pos_weight),
                make_lgb(),
                make_ada(),
            ],
            n_folds=5,
            random_state=SEED,
            use_original_features=False,
            meta_model=LogisticRegression(max_iter=1000, class_weight="balanced"),
            verbose=False,
        )

    else:
        raise ValueError(f"Unknown variant: {name}")


# =========================================================
# 5. Main
# =========================================================
print("Loading dataset...")
train_x, train_y, valid_x, valid_y, test_x, test_y, full_train_x, full_train_y = load_data()

if USE_FEATURE_SELECTION:
    print("\nRunning feature selection...")
    fs_best, fs_all = run_feature_selection(train_x, train_y, valid_x, valid_y, FEATURE_METHODS)
    selected_indices = fs_best["selected_indices"]

    train_x = train_x.iloc[:, selected_indices]
    valid_x = valid_x.iloc[:, selected_indices]
    test_x = test_x.iloc[:, selected_indices]
    full_train_x = full_train_x.iloc[:, selected_indices]
else:
    fs_best, fs_all = None, None

pos_weight = int((train_y == 0).sum() / max(1, (train_y == 1).sum()))

variant_names = [
    "DT", "LDA", "LR", "RF", "XGB", "LGB",
    "AAESSLR", "extreetree", "AAESSTREE", "AAESS"
]

results = []

for variant in variant_names:
    print("\n" + "=" * 80)
    print(f"Running variant: {variant}")

    valid_model = build_table10_model(variant, pos_weight)
    valid_model.fit(train_x.values, train_y.values)

    valid_pred = valid_model.predict(valid_x.values)
    valid_proba = valid_model.predict_proba(valid_x.values)[:, 1]
    valid_metrics = compute_metrics(valid_y.values, valid_pred, valid_proba)

    final_model = build_table10_model(variant, pos_weight)
    final_model.fit(full_train_x.values, full_train_y.values)

    test_pred = final_model.predict(test_x.values)
    test_proba = final_model.predict_proba(test_x.values)[:, 1]
    test_metrics = compute_metrics(test_y.values, test_pred, test_proba)

    row = {
        "variant": variant,
        "validation_auc": valid_metrics["auc"],
        "validation_acc": valid_metrics["acc"],
        "validation_prec": valid_metrics["prec"],
        "validation_rec": valid_metrics["rec"],
        **test_metrics,
    }

    if hasattr(final_model, "get_model_contributions"):
        try:
            row["model_contributions"] = final_model.get_model_contributions()
        except Exception:
            row["model_contributions"] = None

    results.append(row)

    print(
        f"Test -> AUC: {row['auc']:.6f}, "
        f"ACC: {row['acc']:.6f}, "
        f"PREC: {row['prec']:.6f}, "
        f"REC: {row['rec']:.6f}, "
        f"F1: {row['f1']:.6f}"
    )

results_df = pd.DataFrame(results)

csv_path = os.path.join(SAVE_DIR, "table10_results.csv")
results_df.to_csv(csv_path, index=False)

json_path = os.path.join(SAVE_DIR, "table10_results.json")
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(to_serializable(results), f, indent=2, ensure_ascii=False)

print("\nSaved files:")
print(csv_path)
print(json_path)
