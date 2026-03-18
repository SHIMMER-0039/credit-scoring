from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from typing import Dict, Optional
 
import numpy as np
import pandas as pd
from sklearn.linear_model import BayesianRidge, HuberRegressor
from sklearn.preprocessing import StandardScaler


@dataclass
class RobustWeightingResult:
    weighting_model: str
    n_samples: int
    n_features: int
    sample_weight_min: float
    sample_weight_max: float
    sample_weight_mean: float
    confidence_type: str


class RobustWeightingModule:
    """
    Robust weighting module aligned with the manuscript.

    Logic:
    - If outlier_ratio > 5%: use Huber regression during training
    - Else: use Bayesian Ridge Regression

    Outputs:
    - sample_weights_: shape (n_samples,)
    - sample_reliability_vector_: shape (n_samples, n_features)
    - model_confidence_: shape (n_features,) for BayesianRidge
                         shape (n_features,) broadcast scalar for Huber
    """

    def __init__(
        self,
        weighting_model: str,
        huber_epsilon: float = 1.35,
        normalize_weights: bool = True,
        standardize_x: bool = True,
        verbose: bool = True,
    ) -> None:
        if weighting_model not in {"Huber", "BayesianRidge"}:
            raise ValueError("weighting_model must be 'Huber' or 'BayesianRidge'")

        self.weighting_model = weighting_model
        self.huber_epsilon = huber_epsilon
        self.normalize_weights = normalize_weights
        self.standardize_x = standardize_x
        self.verbose = verbose

        self.scaler_: Optional[StandardScaler] = None
        self.model_: Optional[object] = None

        self.sample_weights_: Optional[np.ndarray] = None
        self.sample_reliability_vector_: Optional[np.ndarray] = None
        self.model_confidence_: Optional[np.ndarray] = None
        self.predictive_variance_: Optional[np.ndarray] = None
        self.result_: Optional[RobustWeightingResult] = None

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)

    @staticmethod
    def _to_numpy(x: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray):
        x_np = x.values if isinstance(x, pd.DataFrame) else np.asarray(x)
        y_np = y.values if isinstance(y, pd.Series) else np.asarray(y)
        return x_np.astype(float), y_np.astype(float)

    def _prepare_x(self, x: np.ndarray, fit: bool = False) -> np.ndarray:
        if not self.standardize_x:
            return x

        if fit:
            self.scaler_ = StandardScaler()
            return self.scaler_.fit_transform(x)

        if self.scaler_ is None:
            raise RuntimeError("Scaler not fitted.")
        return self.scaler_.transform(x)

    def _normalize_sample_weights(self, w: np.ndarray) -> np.ndarray:
        w = np.asarray(w, dtype=float)
        w = np.clip(w, a_min=0.05, a_max=None)

        if self.normalize_weights:
            w = w / np.mean(w)

        return w

    def fit(self, x: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray) -> "RobustWeightingModule":
        x_np, y_np = self._to_numpy(x, y)
        x_scaled = self._prepare_x(x_np, fit=True)

        if self.weighting_model == "Huber":
            self._fit_huber(x_scaled, y_np)
        else:
            self._fit_bayesian_ridge(x_scaled, y_np)

        self.result_ = RobustWeightingResult(
            weighting_model=self.weighting_model,
            n_samples=int(x_scaled.shape[0]),
            n_features=int(x_scaled.shape[1]),
            sample_weight_min=float(np.min(self.sample_weights_)),
            sample_weight_max=float(np.max(self.sample_weights_)),
            sample_weight_mean=float(np.mean(self.sample_weights_)),
            confidence_type="feature-wise" if self.weighting_model == "BayesianRidge" else "broadcast-scalar",
        )

        self._log(f"Robust weighting summary")
        self._log(f"  model: {self.result_.weighting_model}")
        self._log(f"  sample weight min: {self.result_.sample_weight_min:.6f}")
        self._log(f"  sample weight max: {self.result_.sample_weight_max:.6f}")
        self._log(f"  sample weight mean: {self.result_.sample_weight_mean:.6f}")
        self._log(f"  confidence type: {self.result_.confidence_type}")

        return self

    def _fit_huber(self, x_scaled: np.ndarray, y: np.ndarray) -> None:
        """
        Training-time robust weighting using Huber residuals.

        Paper-aligned intuition:
        residual small  -> weight 1
        residual large  -> weight down
        """
        model = HuberRegressor(epsilon=self.huber_epsilon)
        model.fit(x_scaled, y)
        y_hat = model.predict(x_scaled)

        residual = y - y_hat
        abs_res = np.abs(residual)

        # sklearn Huber uses sigma_ as robust scale estimate
        delta = float(model.scale_) if hasattr(model, "scale_") else 1.0
        delta = max(delta, 1e-12)

        sample_weights = np.where(abs_res <= delta, 1.0, delta / abs_res)
        sample_weights = self._normalize_sample_weights(sample_weights)

        # reliability-weighted feature vector w_i
        reliability_vector = x_scaled * sample_weights[:, None]

        # robust covariance approximation for coefficient uncertainty
        # simplified diagonal covariance proxy
        xtx = x_scaled.T @ x_scaled
        ridge = 1e-6 * np.eye(xtx.shape[0])
        cov_proxy = np.linalg.pinv(xtx + ridge)

        trace_cov = float(np.trace(cov_proxy))
        scalar_conf = 1.0 / max(trace_cov, 1e-12)
        model_confidence = np.full(x_scaled.shape[1], scalar_conf, dtype=float)

        self.model_ = model
        self.sample_weights_ = sample_weights
        self.sample_reliability_vector_ = reliability_vector
        self.model_confidence_ = model_confidence
        self.predictive_variance_ = None

    def _fit_bayesian_ridge(self, x_scaled: np.ndarray, y: np.ndarray) -> None:
        """
        Bayesian Ridge:
        sigma_i^2 = x_i^T Sigma x_i + alpha^{-1}
        reliability weight = 1 / sigma_i^2
        feature-wise confidence = diag(Sigma)^{-1}
        """
        model = BayesianRidge(compute_score=True)
        model.fit(x_scaled, y)

        sigma = model.sigma_  # posterior covariance of coefficients
        alpha_inv = 1.0 / max(model.alpha_, 1e-12)  # observation noise variance proxy

        # predictive variance for each sample
        quad = np.einsum("ij,jk,ik->i", x_scaled, sigma, x_scaled)
        predictive_variance = quad + alpha_inv
        predictive_variance = np.maximum(predictive_variance, 1e-12)

        sample_weights = 1.0 / predictive_variance
        sample_weights = self._normalize_sample_weights(sample_weights)

        reliability_vector = x_scaled * sample_weights[:, None]

        diag_sigma = np.diag(sigma)
        diag_sigma = np.maximum(diag_sigma, 1e-12)
        model_confidence = 1.0 / diag_sigma

        self.model_ = model
        self.sample_weights_ = sample_weights
        self.sample_reliability_vector_ = reliability_vector
        self.model_confidence_ = model_confidence
        self.predictive_variance_ = predictive_variance

    def transform_sample_weights(self, x: pd.DataFrame | np.ndarray) -> np.ndarray:
        """
        For inference-time weighting:
        - BayesianRidge: returns variance-based sample weights
        - Huber: returns all-ones weights, because true residuals are unavailable at inference
        """
        x_np = x.values if isinstance(x, pd.DataFrame) else np.asarray(x)
        x_scaled = self._prepare_x(x_np.astype(float), fit=False)

        if self.weighting_model == "Huber":
            sample_weights = np.ones(x_scaled.shape[0], dtype=float)
            return self._normalize_sample_weights(sample_weights)

        model = self.model_
        sigma = model.sigma_
        alpha_inv = 1.0 / max(model.alpha_, 1e-12)

        quad = np.einsum("ij,jk,ik->i", x_scaled, sigma, x_scaled)
        predictive_variance = quad + alpha_inv
        predictive_variance = np.maximum(predictive_variance, 1e-12)

        sample_weights = 1.0 / predictive_variance
        return self._normalize_sample_weights(sample_weights)

    def transform_reliability_vector(self, x: pd.DataFrame | np.ndarray) -> np.ndarray:
        x_np = x.values if isinstance(x, pd.DataFrame) else np.asarray(x)
        x_scaled = self._prepare_x(x_np.astype(float), fit=False)
        w = self.transform_sample_weights(x_np)
        return x_scaled * w[:, None]

    def get_model_confidence(self) -> np.ndarray:
        if self.model_confidence_ is None:
            raise RuntimeError("Call fit() first.")
        return self.model_confidence_

    def save_report(self, path: str) -> None:
        if self.result_ is None:
            raise RuntimeError("No result available. Call fit() first.")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self.result_), f, indent=2, ensure_ascii=False)
