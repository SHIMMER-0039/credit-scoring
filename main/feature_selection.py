import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from scipy.stats import pearsonr


# Feature Evaluator with different feature selection methods
class FeatureEvaluator:
    def __init__(self, method='ClassifierFE'):
        self.method = method
        np.random.seed(42)  # Ensure reproducibility

    def fit(self, X, y):
        print(f"Using feature selection method: {self.method}")
        if self.method == 'ClassifierFE':
            self.scores_ = self._classifier_fe(X, y)
        elif self.method == 'CorrelationFE':
            self.scores_ = self._correlation_fe(X, y)
        elif self.method == 'InfoGainFE':
            self.scores_ = self._infogain_fe(X, y)
        elif self.method == 'ReliefFE':
            self.scores_ = self._relief_fe(X, y)
        elif self.method == 'GainRFE':
            self.scores_ = self._gain_rfe(X, y)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        # 去除分数小于 0.005 的特征
        self._filter_features(X)
        return self

    def _classifier_fe(self, X, y):
        clf = DecisionTreeClassifier()
        clf.fit(X, y)
        print("Decision tree feature importances calculated.")
        return clf.feature_importances_

    def _correlation_fe(self, X, y):
        scores = np.array([pearsonr(X[:, i], y)[0] for i in range(X.shape[1])])
        print("Pearson correlation coefficients calculated.")
        return scores

    def _infogain_fe(self, X, y):
        scores = mutual_info_classif(X, y)
        print("Mutual information scores calculated.")
        return scores

    def _relief_fe(self, X, y):
        print("Calculating Relief feature scores...")
        return self._relief(X, y)

    def _relief(self, X, y):
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        m = X.shape[0]
        scores = np.zeros(X.shape[1])
        y = y.values  # 将 y 转换为 NumPy 数组
        for i in range(m):
            instance = X[i, :]
            same_class = X[y == y[i]]  # 这里 y[i] 现在是一个标量
            diff_class = X[y != y[i]]
            nearest_same = self._nearest(instance, same_class)
            nearest_diff = self._nearest(instance, diff_class)
            scores += np.abs(instance - nearest_diff) - np.abs(instance - nearest_same)
            if i % 100 == 0:  # 每100步输出进度
                print(f"Processed {i + 1}/{m} instances.")
        print("Relief scores calculated.")
        return scores / m

    def _nearest(self, instance, data):
        if data.shape[0] == 0:
            return np.zeros(instance.shape)
        nbrs = NearestNeighbors(n_neighbors=1).fit(data)
        distances, indices = nbrs.kneighbors([instance])
        return data[indices[0][0]]

    def _gain_rfe(self, X, y):
        print("Calculating GainRFE scores...")
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        rfe = RFE(estimator=clf, n_features_to_select=1, step=1)
        rfe.fit(X, y)
        scores = rfe.ranking_  # 获取特征排名（1表示最重要）
        max_score = np.max(scores)
        normalized_scores = (max_score - scores) / max_score  # 归一化到 [0, 1] 之间
        return normalized_scores

    def _filter_features(self, X):
        # 过滤掉分数小于 0.005 的特征
        mask = self.scores_ >= 0.005
        self.scores_ = self.scores_[mask]
        if isinstance(X, pd.DataFrame):
            feature_names = X.columns[mask]
            print(f"Features selected by {self.method}: {list(feature_names)}")
        else:
            selected_indices = np.arange(X.shape[1])[mask]
            print(f"Feature indices selected by {self.method}: {selected_indices}")

    def transform(self, X):
        ordered_features = self.scores_.argsort()[::-1]
        selected_features = ordered_features

        # 如果 X 是 DataFrame，则输出特征名称
        if isinstance(X, pd.DataFrame):
            feature_names = X.columns[selected_features]
            print(f"Final selected feature names in order: {list(feature_names)}")
            return X.iloc[:, selected_features], selected_features
        else:
            print(f"Final selected feature indices in order: {selected_features}")
            return X[:, selected_features], selected_features


# Define Pareto efficiency function
def is_pareto_efficient(costs):
    """
    Find the pareto-efficient points
    :param costs: An (n_points, n_costs) array
    :return: A boolean array of pareto efficient points
    """
    is_efficient = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        is_efficient[i] = np.all(np.any(costs[:i] > c, axis=1)) and np.all(np.any(costs[i + 1:] > c, axis=1))
    return is_efficient


# Train model and evaluate
def evaluate_model(train_x, train_y, valid_x, valid_y):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(train_x, train_y)
    valid_pred = model.predict(valid_x)
    valid_pred_proba = model.predict_proba(valid_x)[:, 1]
    accuracy = accuracy_score(valid_y, valid_pred)
    roc_auc = roc_auc_score(valid_y, valid_pred_proba)
    precision = precision_score(valid_y, valid_pred)
    recall = recall_score(valid_y, valid_pred)
    f1 = f1_score(valid_y, valid_pred)

    # Display each score
    print(f"Accuracy: {accuracy}")
    print(f"ROC AUC: {roc_auc}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")

    return np.array([accuracy, roc_auc, precision, recall, f1])
