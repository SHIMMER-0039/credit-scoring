# main/aaess_attention_stacking.py

from __future__ import annotations

import numpy as np
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from main.outlier_detection import OutlierDetector
from main.robust_weighting import RobustWeightingModule


class AAESSAttentionStacking:
    """
    Adaptive Attention-Enhanced Sequential Stacking (AAESS)

    Workflow:
    1. Outlier detection on training data
    2. Select robust weighting model:
       - outlier ratio > 5%  -> Huber
       - otherwise           -> BayesianRidge
    3. Fit robust weighting module
    4. Generate OOF predictions from base learners
    5. Compute learner-specific confidence vectors
    6. Compute attention weights alpha_{k,i}
    7. Build attention-weighted meta features
    8. Train logistic regression meta-classifier
    """

    def __init__(
        self,
        base_models,
        n_folds: int = 5,
        meta_model=None,
        use_original_features: bool = True,
        outlier_detector=None,
        robust_weighting_module=None,
        random_state: int = 42,
        verbose: bool = True,
    ):
        self.base_models = base_models
        self.n_folds = n_folds
        self.meta_model = meta_model if meta_model is not None else LogisticRegression(max_iter=1000)
        self.use_original_features = use_original_features
        self.random_state = random_state
        self.verbose = verbose

        self.outlier_detector = outlier_detector
        self.robust_weighting_module = robust_weighting_module

        self.fitted_base_models_ = None
        self.meta_model_ = None

        self.oof_predictions_ = None
        self.model_confidences_ = None
        self.train_attention_weights_ = None
        self.test_attention_weights_ = None

        self.train_meta_features_ = None
        self.test_meta_features_ = None

        self.outlier_ratio_ = None
        self.selected_weighting_model_ = None

    def _log(self, msg: str):
        if self.verbose:
            print(msg)

    @staticmethod
    def _softmax(x: np.ndarray, axis: int = 1, temperature: float = 1.0) -> np.ndarray:
        x = x / temperature
        x = x - np.max(x, axis=axis, keepdims=True)
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

    def _ensure_numpy(self, x, y=None):
        x_np = x.values if hasattr(x, "values") else np.asarray(x)
        if y is None:
            return x_np.astype(float)
        y_np = y.values if hasattr(y, "values") else np.asarray(y)
        return x_np.astype(float), y_np.astype(float)

    def _build_default_outlier_detector(self):
        return OutlierDetector(
            z_thresh=3.0,
            iqr_multiplier=1.5,
            contamination="auto",
            vote_threshold=3,
            random_state=self.random_state,
            lof_n_neighbors=20,
            verbose=self.verbose,
        )

    def _build_default_weighting_module(self, weighting_model_name: str):
        return RobustWeightingModule(
            weighting_model=weighting_model_name,
            huber_epsilon=1.35,
            normalize_weights=True,
            standardize_x=True,
            verbose=self.verbose,
        )

    def _compute_attention_weights(
            self,
            reliability_vectors: np.ndarray,
            model_confidences: np.ndarray,
    ) -> np.ndarray:
        """
        reliability_vectors: (n_samples, n_features)
        model_confidences:   (n_models, n_features)
        returns:             (n_samples, n_models)
        """
        raw_scores = reliability_vectors @ model_confidences.T
        if hasattr(self, 'performance_biases_'):
            scores = raw_scores + self.performance_biases_
        else:
            scores = raw_scores
        return self._softmax(scores, axis=1)

    def _generate_oof_predictions(self, X: np.ndarray, y: np.ndarray):
        n_samples = X.shape[0]
        n_models = len(self.base_models)
        oof_predictions = np.zeros((n_samples, n_models), dtype=float)

        skf = StratifiedKFold(
            n_splits=self.n_folds,
            shuffle=True,
            random_state=self.random_state
        )

        fitted_base_models = []

        for model_idx, base_model in enumerate(self.base_models):
            self._log(f"Training base model {model_idx + 1}/{n_models}: {type(base_model).__name__}")

            for train_idx, valid_idx in skf.split(X, y):
                model = clone(base_model)
                model.fit(X[train_idx], y[train_idx])

                if hasattr(model, "predict_proba"):
                    pred = model.predict_proba(X[valid_idx])[:, 1]
                else:
                    pred = model.predict(X[valid_idx])

                oof_predictions[valid_idx, model_idx] = pred

            final_model = clone(base_model)
            final_model.fit(X, y)
            fitted_base_models.append(final_model)

        return oof_predictions, fitted_base_models

    def _build_model_confidences(
            self,
            y_true: np.ndarray,
            oof_predictions: np.ndarray,
            base_feature_confidence: np.ndarray,
    ) -> np.ndarray:
        """
        Build learner-specific confidence vectors based on ACC and AUC.
        """
        from sklearn.metrics import roc_auc_score, accuracy_score

        n_models = oof_predictions.shape[1]
        model_scores = []
        for j in range(n_models):
            pred_proba_j = np.clip(oof_predictions[:, j], 1e-12, 1 - 1e-12)
            pred_class_j = (pred_proba_j >= 0.5).astype(int)
            auc = roc_auc_score(y_true, pred_proba_j)
            acc = accuracy_score(y_true, pred_class_j)
            combined_score = 0.7 * auc + 0.3 * acc
            model_scores.append(combined_score)

        model_scores = np.array(model_scores)

        alpha = 20.0
        performance_biases = (model_scores - np.max(model_scores)) * alpha
        self.performance_biases_ = performance_biases
        model_confidences = np.tile(base_feature_confidence, (n_models, 1))

        return model_confidences

    def fit(self, X, y):
        X, y = self._ensure_numpy(X, y)

        self._log("Step 1: Outlier detection")
        if self.outlier_detector is None:
            self.outlier_detector = self._build_default_outlier_detector()

        self.outlier_detector.fit(X)
        self.outlier_ratio_ = self.outlier_detector.result_.outlier_ratio
        self.selected_weighting_model_ = self.outlier_detector.result_.selected_weighting_model

        self._log(f"Outlier ratio: {self.outlier_ratio_:.4%}")
        self._log(f"Selected weighting model: {self.selected_weighting_model_}")

        self._log("Step 2: Robust weighting")
        if self.robust_weighting_module is None:
            self.robust_weighting_module = self._build_default_weighting_module(
                self.selected_weighting_model_
            )

        self.robust_weighting_module.fit(X, y)
        reliability_vectors = self.robust_weighting_module.sample_reliability_vector_
        base_feature_confidence = self.robust_weighting_module.get_model_confidence()

        self._log("Step 3: OOF predictions")
        oof_predictions, fitted_base_models = self._generate_oof_predictions(X, y)

        self._log("Step 4: Learner-specific confidence vectors")
        model_confidences = self._build_model_confidences(
            y_true=y,
            oof_predictions=oof_predictions,
            base_feature_confidence=base_feature_confidence,
        )

        self._log("Step 5: Attention weights")
        attention_weights = self._compute_attention_weights(
            reliability_vectors=reliability_vectors,
            model_confidences=model_confidences,
        )

        weighted_oof_predictions = attention_weights * oof_predictions

        self._log("Step 6: Meta-features")
        if self.use_original_features:
            meta_features = np.hstack([X, weighted_oof_predictions])
        else:
            meta_features = weighted_oof_predictions

        self._log("Step 7: Train meta-classifier")
        meta_model = clone(self.meta_model)
        meta_model.fit(meta_features, y)

        self.fitted_base_models_ = fitted_base_models
        self.meta_model_ = meta_model
        self.oof_predictions_ = oof_predictions
        self.model_confidences_ = model_confidences
        self.train_attention_weights_ = attention_weights
        self.train_meta_features_ = meta_features

        return self

    def _predict_base_outputs(self, X: np.ndarray) -> np.ndarray:
        if self.fitted_base_models_ is None:
            raise RuntimeError("Model has not been fitted.")

        n_samples = X.shape[0]
        n_models = len(self.fitted_base_models_)
        base_predictions = np.zeros((n_samples, n_models), dtype=float)

        for j, model in enumerate(self.fitted_base_models_):
            if hasattr(model, "predict_proba"):
                pred = model.predict_proba(X)[:, 1]
            else:
                pred = model.predict(X)
            base_predictions[:, j] = pred

        return base_predictions

    def predict_proba(self, X):
        if self.meta_model_ is None:
            raise RuntimeError("Call fit() before predict_proba().")

        X = self._ensure_numpy(X)
        base_predictions = self._predict_base_outputs(X)

        reliability_vectors = self.robust_weighting_module.transform_reliability_vector(X)

        attention_weights = self._compute_attention_weights(
            reliability_vectors=reliability_vectors,
            model_confidences=self.model_confidences_,
        )

        weighted_predictions = attention_weights * base_predictions

        if self.use_original_features:
            meta_features = np.hstack([X, weighted_predictions])
        else:
            meta_features = weighted_predictions

        self.test_attention_weights_ = attention_weights
        self.test_meta_features_ = meta_features

        if hasattr(self.meta_model_, "predict_proba"):
            return self.meta_model_.predict_proba(meta_features)

        pred = self.meta_model_.predict(meta_features)
        return np.column_stack([1 - pred, pred])

    def predict(self, X, threshold: float = 0.5):
        proba = self.predict_proba(X)[:, 1]
        return (proba >= threshold).astype(int)

    def get_attention_summary(self):
        if self.train_attention_weights_ is None:
            raise RuntimeError("Call fit() first.")

        return {
            "mean_attention_train": self.train_attention_weights_.mean(axis=0),
            "std_attention_train": self.train_attention_weights_.std(axis=0),
            "selected_weighting_model": self.selected_weighting_model_,
            "outlier_ratio": self.outlier_ratio_,
        }

    def get_model_contributions(self):
        """
        Approximate learner contribution by average attention weight.
        """
        if self.train_attention_weights_ is None:
            raise RuntimeError("Call fit() first.")

        mean_att = self.train_attention_weights_.mean(axis=0)
        total = np.sum(mean_att)
        if total <= 0:
            return mean_att
        return mean_att / total

    def get_params(self, deep=True):
        return {
            "n_folds": self.n_folds,
            "use_original_features": self.use_original_features,
            "random_state": self.random_state,
            "base_models": [type(m).__name__ for m in self.base_models],
            "meta_model": type(self.meta_model).__name__,
            "selected_weighting_model": self.selected_weighting_model_,
            "outlier_ratio": self.outlier_ratio_,
        }
