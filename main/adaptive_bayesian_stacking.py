# adaptive_bayesian_stacking.py

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.linear_model import BayesianRidge, HuberRegressor
from sklearn.model_selection import KFold

class AdaptiveBayesianStacking(BaseEstimator, ClassifierMixin):
    def __init__(self, base_models, weight_model=HuberRegressor(), n_folds=5, feature_evaluator=None):
        self.base_models = base_models
        self.weight_model = weight_model
        self.n_folds = n_folds
        self.feature_evaluator = feature_evaluator

    def fit(self, X, y):
        if self.feature_evaluator:
            self.feature_evaluator.fit(X, y)
            X = self.feature_evaluator.transform(X)[0]

        self.base_models_ = [list() for _ in self.base_models]
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)

        # Initialize for model output features
        intermediate_features = X

        # 确保 y 的索引是连续的
        y_values = y.values

        for i, model in enumerate(self.base_models):
            print(f"Training base model {i + 1}/{len(self.base_models)}")
            model_out_of_fold_predictions = np.zeros((X.shape[0],))

            for train_idx, holdout_idx in kfold.split(intermediate_features, y):
                instance = clone(model)
                self.base_models_[i].append(instance)
                instance.fit(intermediate_features[train_idx], y_values[train_idx])  # 使用 y 的值

                y_pred = instance.predict_proba(intermediate_features[holdout_idx])[:, 1]
                model_out_of_fold_predictions[holdout_idx] = y_pred

            # Add the current model's output as a new feature
            intermediate_features = np.column_stack((intermediate_features, model_out_of_fold_predictions))

        # Final input features for the meta-model
        self.final_features_ = intermediate_features

        # Train the HuberRegressor meta-model
        self.weight_model.fit(intermediate_features, y_values)  # 使用 y 的值
        return self

    def predict(self, X):
        if self.feature_evaluator:
            X = self.feature_evaluator.transform(X)[0]

        # Generate features through each model and concatenate
        meta_features = X
        for i, models in enumerate(self.base_models_):
            model_outputs = np.column_stack([model.predict_proba(meta_features)[:, 1] for model in models])
            meta_features = np.column_stack((meta_features, np.mean(model_outputs, axis=1)))

        weighted_predictions = self.weight_model.predict(meta_features)
        return (weighted_predictions > 0.5).astype(int)

    def predict_proba(self, X):
        if self.feature_evaluator:
            X = self.feature_evaluator.transform(X)[0]

        # Generate features through each model and concatenate
        meta_features = X
        for i, models in enumerate(self.base_models_):
            model_outputs = np.column_stack([model.predict_proba(meta_features)[:, 1] for model in models])
            meta_features = np.column_stack((meta_features, np.mean(model_outputs, axis=1)))

        weighted_predictions = self.weight_model.predict(meta_features)
        return np.vstack((1 - weighted_predictions, weighted_predictions)).T