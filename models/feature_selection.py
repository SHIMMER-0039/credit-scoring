from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score, 
)


class FeatureEvaluator:
    """
    Unified feature evaluator.

    Supported methods:
    - ClassifierFE   : RandomForest feature importance
    - CorrelationFE  : absolute Pearson correlation with target
    - GainRFE        : RandomForest importance as a gain-based proxy
    - InfoGainFE     : mutual information
    - ReliefFE       : simple class-separation score
    """

    def __init__(
        self,
        method: str = "ClassifierFE",
        random_state: int = 42,
        n_estimators: int = 200,
    ) -> None:
        self.method = method
        self.random_state = random_state
        self.n_estimators = n_estimators
        self.scores_: np.ndarray | None = None

    def fit(self, X, y) -> "FeatureEvaluator":
        X_np = self._to_numpy_x(X)
        y_np = self._to_numpy_y(y)

        if self.method == "ClassifierFE":
            self.scores_ = self._classifier_fe(X_np, y_np)

        elif self.method == "CorrelationFE":
            self.scores_ = self._correlation_fe(X_np, y_np)

        elif self.method == "GainRFE":
            self.scores_ = self._gain_rfe(X_np, y_np)

        elif self.method == "InfoGainFE":
            self.scores_ = self._info_gain_fe(X_np, y_np)

        elif self.method == "ReliefFE":
            self.scores_ = self._relief_fe(X_np, y_np)

        else:
            raise ValueError(f"Unsupported feature selection method: {self.method}")

        self.scores_ = np.asarray(self.scores_, dtype=float)
        self.scores_ = np.nan_to_num(self.scores_, nan=0.0, posinf=0.0, neginf=0.0)

        return self

    @staticmethod
    def _to_numpy_x(X) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            X = X.values
        X = np.asarray(X, dtype=float)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        return X

    @staticmethod
    def _to_numpy_y(y) -> np.ndarray:
        if isinstance(y, (pd.Series, pd.DataFrame)):
            y = np.asarray(y).ravel()
        else:
            y = np.asarray(y).ravel()
        return y

    def _classifier_fe(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=-1,
        )
        model.fit(X, y)
        return model.feature_importances_

    def _correlation_fe(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        scores = np.zeros(X.shape[1], dtype=float)

        for j in range(X.shape[1]):
            xj = X[:, j]

            if np.std(xj) == 0:
                scores[j] = 0.0
                continue

            corr = np.corrcoef(xj, y)[0, 1]
            if np.isnan(corr):
                corr = 0.0
            scores[j] = abs(corr)

        return scores

    def _gain_rfe(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Practical proxy implementation:
        use RandomForest feature importance as a gain-based ranking signal.
        """
        model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=-1,
        )
        model.fit(X, y)
        return model.feature_importances_

    def _info_gain_fe(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        scores = mutual_info_classif(
            X,
            y,
            random_state=self.random_state,
            discrete_features="auto",
        )
        return np.asarray(scores, dtype=float)

    def _relief_fe(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Lightweight Relief-style surrogate:
        class mean separation normalized by within-class variance.
        """
        pos_mask = (y == 1)
        neg_mask = (y == 0)

        if pos_mask.sum() == 0 or neg_mask.sum() == 0:
            return np.zeros(X.shape[1], dtype=float)

        X_pos = X[pos_mask]
        X_neg = X[neg_mask]

        pos_mean = np.mean(X_pos, axis=0)
        neg_mean = np.mean(X_neg, axis=0)

        pos_var = np.var(X_pos, axis=0)
        neg_var = np.var(X_neg, axis=0)

        scores = np.abs(pos_mean - neg_mean) / (pos_var + neg_var + 1e-12)
        return scores


def is_pareto_efficient(values: np.ndarray) -> np.ndarray:
    """
    Pareto efficiency under the convention: larger is better for every metric.

    Parameters
    ----------
    values : np.ndarray
        Shape (n_points, n_metrics)

    Returns
    -------
    np.ndarray
        Boolean mask of shape (n_points,), True if the point is Pareto efficient.
    """
    values = np.asarray(values, dtype=float)
    n_points = values.shape[0]
    efficient = np.ones(n_points, dtype=bool)

    for i in range(n_points):
        if not efficient[i]:
            continue

        dominates_i = np.all(values >= values[i], axis=1) & np.any(values > values[i], axis=1)
        dominates_i[i] = False

        if np.any(dominates_i):
            efficient[i] = False

    return efficient


def evaluate_model(train_x, train_y, valid_x, valid_y) -> np.ndarray:
    """
    Evaluate a validation backbone model for feature subset comparison.

    Returns metrics in this exact order:
    [Accuracy, AUC, Precision, Recall]
    """
    X_train = _to_numpy(train_x)
    y_train = _to_numpy_1d(train_y)
    X_valid = _to_numpy(valid_x)
    y_valid = _to_numpy_1d(valid_y)

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    pred = model.predict(X_valid)

    if hasattr(model, "predict_proba"):
        pred_proba = model.predict_proba(X_valid)[:, 1]
    else:
        pred_proba = pred.astype(float)

    pred_proba = np.clip(pred_proba, 1e-12, 1 - 1e-12)

    acc = accuracy_score(y_valid, pred)
    auc = roc_auc_score(y_valid, pred_proba)
    prec = precision_score(y_valid, pred, zero_division=0)
    rec = recall_score(y_valid, pred, zero_division=0)

    return np.array([acc, auc, prec, rec], dtype=float)


def _to_numpy(x) -> np.ndarray:
    if isinstance(x, pd.DataFrame):
        x = x.values
    x = np.asarray(x, dtype=float)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return x


def _to_numpy_1d(y) -> np.ndarray:
    if isinstance(y, (pd.Series, pd.DataFrame)):
        y = np.asarray(y).ravel()
    else:
        y = np.asarray(y).ravel()
    return y
