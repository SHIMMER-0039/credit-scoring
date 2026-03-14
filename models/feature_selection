from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from main.feature_selection import FeatureEvaluator, is_pareto_efficient, evaluate_model


@dataclass
class FeatureSubsetResult:
    """Store one candidate subset and its evaluation results."""
    method: str
    selected_indices: List[int]
    selected_names: List[str]
    threshold: float
    n_features: int
    metrics: Dict[str, float]
    pareto_efficient: bool = False
    refinement_applied: bool = False


class PaperFeatureSelector:
    """
    Feature selection pipeline aligned with the manuscript description.

    Workflow:
    1. Run each feature selection method independently on the original feature space.
    2. Remove features whose importance score is below threshold.
    3. Evaluate each candidate subset on validation data using a common model.
    4. Apply Pareto efficiency test across all candidate subsets.
    5. Choose one subset from Pareto front using priority rules:
       - higher AUC
       - higher Recall
       - fewer features
    6. Optional final refinement using ReliefFE.

    Notes:
    - This implementation follows the paper description more closely than the
      earlier sequential-union implementation.
    - Threshold can be interpreted as absolute 0.05 or relative 5% of max score.
    """

    def __init__(
        self,
        methods: Optional[Sequence[str]] = None,
        threshold_mode: str = "relative",
        threshold_value: float = 0.05,
        final_refine_method: Optional[str] = "ReliefFE",
        min_features: int = 1,
        verbose: bool = True,
    ) -> None:
        """
        Args:
            methods: feature selection methods used to generate candidate subsets.
            threshold_mode: 'relative' or 'absolute'.
                - 'relative': threshold = threshold_value * max(score)
                - 'absolute': threshold = threshold_value
            threshold_value: threshold parameter.
            final_refine_method: final refinement method. Set to None to disable.
            min_features: minimum number of features to keep.
            verbose: print progress.
        """
        if methods is None:
            methods = [
                "ClassifierFE",
                "CorrelationFE",
                "GainRFE",
                "InfoGainFE",
                "ReliefFE",
            ]

        self.methods = list(methods)
        self.threshold_mode = threshold_mode
        self.threshold_value = threshold_value
        self.final_refine_method = final_refine_method
        self.min_features = min_features
        self.verbose = verbose

        self.results_: List[FeatureSubsetResult] = []
        self.best_result_: Optional[FeatureSubsetResult] = None
        self.selected_indices_: Optional[List[int]] = None
        self.selected_names_: Optional[List[str]] = None

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)

    def _compute_threshold(self, scores: np.ndarray) -> float:
        if self.threshold_mode == "relative":
            return float(self.threshold_value * np.max(scores))
        if self.threshold_mode == "absolute":
            return float(self.threshold_value)
        raise ValueError("threshold_mode must be either 'relative' or 'absolute'")

    def _safe_select_indices(self, scores: np.ndarray) -> List[int]:
        threshold = self._compute_threshold(scores)
        selected = np.where(scores > threshold)[0].tolist()

        # Ensure at least min_features are retained
        if len(selected) < self.min_features:
            ranked = np.argsort(scores)[::-1]
            selected = ranked[: self.min_features].tolist()

        return sorted(selected)

    def _evaluate_subset(
        self,
        method: str,
        feature_indices: List[int],
        feature_names: List[str],
        threshold: float,
        train_x: pd.DataFrame,
        train_y: pd.Series,
        valid_x: pd.DataFrame,
        valid_y: pd.Series,
    ) -> FeatureSubsetResult:
        train_sub = train_x.iloc[:, feature_indices].values
        valid_sub = valid_x.iloc[:, feature_indices].values

        metric_values = evaluate_model(train_sub, train_y, valid_sub, valid_y)

        # IMPORTANT:
        # We assume evaluate_model returns metrics in this order:
        # [Accuracy, AUC, Precision, Recall]
        # If your evaluate_model has another order, update the mapping below.
        metrics = {
            "acc": float(metric_values[0]),
            "auc": float(metric_values[1]),
            "precision": float(metric_values[2]),
            "recall": float(metric_values[3]),
        }

        return FeatureSubsetResult(
            method=method,
            selected_indices=feature_indices,
            selected_names=feature_names,
            threshold=float(threshold),
            n_features=len(feature_indices),
            metrics=metrics,
        )

    @staticmethod
    def _pareto_matrix(results: List[FeatureSubsetResult]) -> np.ndarray:
        """
        Build objective matrix for Pareto test.
        Larger is better for acc, auc, precision, recall.
        """
        return np.array(
            [
                [
                    r.metrics["acc"],
                    r.metrics["auc"],
                    r.metrics["precision"],
                    r.metrics["recall"],
                ]
                for r in results
            ]
        )

    @staticmethod
    def _choose_best_from_pareto(results: List[FeatureSubsetResult]) -> FeatureSubsetResult:
        """
        Choose final subset from Pareto front.
        Priority:
        1. higher AUC
        2. higher Recall
        3. fewer features
        4. higher Accuracy
        """
        pareto_results = [r for r in results if r.pareto_efficient]
        if not pareto_results:
            raise RuntimeError("No Pareto-efficient subset found.")

        pareto_results = sorted(
            pareto_results,
            key=lambda r: (
                -r.metrics["auc"],
                -r.metrics["recall"],
                r.n_features,
                -r.metrics["acc"],
            )
        )
        return pareto_results[0]

    def _run_one_method(
        self,
        method: str,
        train_x: pd.DataFrame,
        train_y: pd.Series,
        valid_x: pd.DataFrame,
        valid_y: pd.Series,
    ) -> FeatureSubsetResult:
        evaluator = FeatureEvaluator(method=method)
        evaluator.fit(train_x.values, train_y)

        scores = np.asarray(evaluator.scores_, dtype=float)
        threshold = self._compute_threshold(scores)
        selected_indices = self._safe_select_indices(scores)
        selected_names = train_x.columns[selected_indices].tolist()

        result = self._evaluate_subset(
            method=method,
            feature_indices=selected_indices,
            feature_names=selected_names,
            threshold=threshold,
            train_x=train_x,
            train_y=train_y,
            valid_x=valid_x,
            valid_y=valid_y,
        )

        self._log(
            f"[{method}] threshold={threshold:.6f}, "
            f"n_features={result.n_features}, metrics={result.metrics}"
        )
        return result

    def _refine_with_relief(
        self,
        base_result: FeatureSubsetResult,
        train_x: pd.DataFrame,
        train_y: pd.Series,
        valid_x: pd.DataFrame,
        valid_y: pd.Series,
    ) -> FeatureSubsetResult:
        """
        Final refinement using ReliefFE on the selected subset.
        Keeps only features re-confirmed by ReliefFE in the reduced space.
        """
        if self.final_refine_method is None:
            return base_result

        if self.final_refine_method != "ReliefFE":
            raise ValueError("Only ReliefFE refinement is currently supported.")

        if base_result.n_features <= self.min_features:
            return base_result

        reduced_train = train_x.iloc[:, base_result.selected_indices]

        evaluator = FeatureEvaluator(method="ReliefFE")
        evaluator.fit(reduced_train.values, train_y)

        scores = np.asarray(evaluator.scores_, dtype=float)
        threshold = self._compute_threshold(scores)
        local_indices = self._safe_select_indices(scores)

        refined_global_indices = [base_result.selected_indices[i] for i in local_indices]
        refined_global_names = train_x.columns[refined_global_indices].tolist()

        refined_result = self._evaluate_subset(
            method=f"{base_result.method}+ReliefFE",
            feature_indices=refined_global_indices,
            feature_names=refined_global_names,
            threshold=threshold,
            train_x=train_x,
            train_y=train_y,
            valid_x=valid_x,
            valid_y=valid_y,
        )
        refined_result.refinement_applied = True

        # Keep refined subset only if it is not worse in Pareto sense vs original best
        compare = np.array([
            [
                base_result.metrics["acc"],
                base_result.metrics["auc"],
                base_result.metrics["precision"],
                base_result.metrics["recall"],
            ],
            [
                refined_result.metrics["acc"],
                refined_result.metrics["auc"],
                refined_result.metrics["precision"],
                refined_result.metrics["recall"],
            ],
        ])
        mask = is_pareto_efficient(compare)

        self._log(
            f"[ReliefFE refinement] base={base_result.metrics}, refined={refined_result.metrics}, "
            f"accepted={bool(mask[1])}"
        )

        if bool(mask[1]):
            return refined_result
        return base_result

    def fit(
        self,
        train_x: pd.DataFrame,
        train_y: pd.Series,
        valid_x: pd.DataFrame,
        valid_y: pd.Series,
    ) -> "PaperFeatureSelector":
        """Run complete feature selection pipeline."""
        self.results_ = []

        # 1. Generate candidate subsets independently
        for method in self.methods:
            result = self._run_one_method(method, train_x, train_y, valid_x, valid_y)
            self.results_.append(result)

        # 2. Pareto test across all candidate subsets
        score_matrix = self._pareto_matrix(self.results_)
        pareto_mask = is_pareto_efficient(score_matrix)

        for i, flag in enumerate(pareto_mask):
            self.results_[i].pareto_efficient = bool(flag)

        self._log("\nPareto-efficient subsets:")
        for res in self.results_:
            if res.pareto_efficient:
                self._log(
                    f"  - {res.method}: auc={res.metrics['auc']:.6f}, "
                    f"recall={res.metrics['recall']:.6f}, n_features={res.n_features}"
                )

        # 3. Choose final subset from Pareto front
        best = self._choose_best_from_pareto(self.results_)

        # 4. Optional ReliefFE refinement
        best = self._refine_with_relief(best, train_x, train_y, valid_x, valid_y)

        self.best_result_ = best
        self.selected_indices_ = best.selected_indices
        self.selected_names_ = best.selected_names

        self._log(
            f"\nFinal selected subset: method={best.method}, "
            f"n_features={best.n_features}, features={best.selected_names}"
        )

        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        """Return data with selected features only."""
        if self.selected_indices_ is None:
            raise RuntimeError("fit must be called before transform.")
        return x.iloc[:, self.selected_indices_].copy()

    def fit_transform(
        self,
        train_x: pd.DataFrame,
        train_y: pd.Series,
        valid_x: pd.DataFrame,
        valid_y: pd.Series,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        self.fit(train_x, train_y, valid_x, valid_y)
        return self.transform(train_x), self.transform(valid_x)

    def save_report(self, path: str) -> None:
        """Save feature selection report for reproducibility."""
        if self.best_result_ is None:
            raise RuntimeError("No results available. Call fit first.")

        report = {
            "threshold_mode": self.threshold_mode,
            "threshold_value": self.threshold_value,
            "methods": self.methods,
            "final_refine_method": self.final_refine_method,
            "best_result": asdict(self.best_result_),
            "all_results": [asdict(r) for r in self.results_],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
