from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor


@dataclass
class OutlierDetectionResult:
    n_samples: int
    n_features: int
    iqr_outliers: int
    zscore_outliers: int
    iforest_outliers: int
    lof_outliers: int
    majority_vote_outliers: int
    outlier_ratio: float
    selected_weighting_model: str
    vote_threshold: int = 3


class OutlierDetector:
    """
    Outlier detection module aligned with the manuscript.

    Methods:
    1. IQR
    2. Z-score
    3. Isolation Forest
    4. Local Outlier Factor (LOF)

    Aggregation:
    - Majority voting
    - A sample is labeled as an outlier if >= 3 of 4 methods flag it

    Switching rule:
    - outlier_ratio > 5%  -> Huber
    - otherwise           -> BayesianRidge
    """

    def __init__(
        self,
        z_thresh: float = 3.0,
        iqr_multiplier: float = 1.5,
        contamination: str | float = "auto",
        vote_threshold: int = 3,
        random_state: int = 42,
        lof_n_neighbors: int = 20,
        verbose: bool = True,
    ) -> None:
        self.z_thresh = z_thresh
        self.iqr_multiplier = iqr_multiplier
        self.contamination = contamination
        self.vote_threshold = vote_threshold
        self.random_state = random_state
        self.lof_n_neighbors = lof_n_neighbors
        self.verbose = verbose

        self.result_: Optional[OutlierDetectionResult] = None
        self.flags_: Optional[Dict[str, np.ndarray]] = None
        self.majority_vote_mask_: Optional[np.ndarray] = None
        self.selected_weighting_model_: Optional[str] = None

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)

    @staticmethod
    def _to_numpy(x: pd.DataFrame | np.ndarray) -> np.ndarray:
        if isinstance(x, pd.DataFrame):
            return x.values
        return np.asarray(x)

    def _iqr_flags(self, x: np.ndarray) -> np.ndarray:
        """
        Flag a sample as outlier if ANY feature falls outside:
        [Q1 - 1.5*IQR, Q3 + 1.5*IQR]
        """
        q1 = np.percentile(x, 25, axis=0)
        q3 = np.percentile(x, 75, axis=0)
        iqr = q3 - q1

        lower = q1 - self.iqr_multiplier * iqr
        upper = q3 + self.iqr_multiplier * iqr

        mask = ((x < lower) | (x > upper)).any(axis=1)
        return mask.astype(int)

    def _zscore_flags(self, x: np.ndarray) -> np.ndarray:
        """
        Flag a sample as outlier if ANY feature has |z| > z_thresh.
        """
        mean = np.mean(x, axis=0)
        std = np.std(x, axis=0)

        # avoid division by zero
        std = np.where(std == 0, 1e-12, std)

        z = np.abs((x - mean) / std)
        mask = (z > self.z_thresh).any(axis=1)
        return mask.astype(int)

    def _iforest_flags(self, x: np.ndarray) -> np.ndarray:
        """
        Isolation Forest:
        sklearn returns -1 for outliers, 1 for inliers
        """
        model = IsolationForest(
            contamination=self.contamination,
            random_state=self.random_state,
            n_estimators=200,
            n_jobs=-1,
        )
        pred = model.fit_predict(x)
        mask = (pred == -1)
        return mask.astype(int)

    def _lof_flags(self, x: np.ndarray) -> np.ndarray:
        """
        Local Outlier Factor:
        sklearn returns -1 for outliers, 1 for inliers
        """
        n_neighbors = min(self.lof_n_neighbors, max(2, x.shape[0] - 1))

        model = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            contamination=self.contamination,
            novelty=False,
            n_jobs=-1,
        )
        pred = model.fit_predict(x)
        mask = (pred == -1)
        return mask.astype(int)

    def fit(self, x: pd.DataFrame | np.ndarray) -> "OutlierDetector":
        x_np = self._to_numpy(x)

        iqr_flags = self._iqr_flags(x_np)
        zscore_flags = self._zscore_flags(x_np)
        iforest_flags = self._iforest_flags(x_np)
        lof_flags = self._lof_flags(x_np)

        vote_sum = iqr_flags + zscore_flags + iforest_flags + lof_flags
        majority_vote_mask = (vote_sum >= self.vote_threshold).astype(int)

        outlier_ratio = float(np.mean(majority_vote_mask))

        if outlier_ratio > 0.05:
            selected_weighting_model = "Huber"
        else:
            selected_weighting_model = "BayesianRidge"

        self.flags_ = {
            "iqr": iqr_flags,
            "zscore": zscore_flags,
            "iforest": iforest_flags,
            "lof": lof_flags,
            "majority_vote": majority_vote_mask,
            "vote_sum": vote_sum,
        }
        self.majority_vote_mask_ = majority_vote_mask
        self.selected_weighting_model_ = selected_weighting_model

        self.result_ = OutlierDetectionResult(
            n_samples=int(x_np.shape[0]),
            n_features=int(x_np.shape[1]),
            iqr_outliers=int(iqr_flags.sum()),
            zscore_outliers=int(zscore_flags.sum()),
            iforest_outliers=int(iforest_flags.sum()),
            lof_outliers=int(lof_flags.sum()),
            majority_vote_outliers=int(majority_vote_mask.sum()),
            outlier_ratio=outlier_ratio,
            selected_weighting_model=selected_weighting_model,
            vote_threshold=self.vote_threshold,
        )

        self._log("Outlier detection summary")
        self._log(f"  IQR outliers: {self.result_.iqr_outliers}")
        self._log(f"  Z-score outliers: {self.result_.zscore_outliers}")
        self._log(f"  IsolationForest outliers: {self.result_.iforest_outliers}")
        self._log(f"  LOF outliers: {self.result_.lof_outliers}")
        self._log(f"  Majority-vote outliers: {self.result_.majority_vote_outliers}")
        self._log(f"  Outlier ratio: {self.result_.outlier_ratio:.4%}")
        self._log(f"  Selected weighting model: {self.result_.selected_weighting_model}")

        return self
 
    def get_outlier_mask(self) -> np.ndarray:
        if self.majority_vote_mask_ is None:
            raise RuntimeError("Call fit() before get_outlier_mask().")
        return self.majority_vote_mask_

    def get_inlier_mask(self) -> np.ndarray:
        if self.majority_vote_mask_ is None:
            raise RuntimeError("Call fit() before get_inlier_mask().")
        return 1 - self.majority_vote_mask_

    def get_vote_sum(self) -> np.ndarray:
        if self.flags_ is None:
            raise RuntimeError("Call fit() before get_vote_sum().")
        return self.flags_["vote_sum"]

    def save_report(self, path: str) -> None:
        if self.result_ is None:
            raise RuntimeError("No result available. Call fit() first.")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self.result_), f, indent=2, ensure_ascii=False)
