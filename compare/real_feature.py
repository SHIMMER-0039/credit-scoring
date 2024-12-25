import os
import pickle
from sys import path
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import (roc_auc_score, accuracy_score, precision_score, recall_score,
                             f1_score, brier_score_loss, precision_recall_curve, average_precision_score,
                             roc_curve, log_loss)
from scipy.stats import ks_2samp
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from scipy.stats import pearsonr



# 定义特征选择类
class FeatureEvaluator:
    def __init__(self, method='ClassifierFE'):
        self.method = method
        np.random.seed(42)  # Ensure reproducibility

    def fit(self, X, y):
        if self.method == 'ClassifierFE':
            self.scores_ = self._classifier_fe(X, y)
        elif self.method == 'CorrelationFE':
            self.scores_ = self._correlation_fe(X, y)
        elif self.method == 'InfoGainFE':
            self.scores_ = self._infogain_fe(X, y)
        elif self.method == 'ReliefFE':
            self.scores_ = self._relief_fe(X, y)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        return self

    def _classifier_fe(self, X, y):
        clf = DecisionTreeClassifier()
        clf.fit(X, y)
        return clf.feature_importances_

    def _correlation_fe(self, X, y):
        return np.array([pearsonr(X[:, i], y)[0] for i in range(X.shape[1])])

    def _infogain_fe(self, X, y):
        return mutual_info_classif(X, y)

    def _relief_fe(self, X, y):
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
        return scores / m

    def _nearest(self, instance, data):
        if data.shape[0] == 0:
            return np.zeros(instance.shape)
        nbrs = NearestNeighbors(n_neighbors=1).fit(data)
        distances, indices = nbrs.kneighbors([instance])
        return data[indices[0][0]]

    def transform(self, X):
        return X[:, self.scores_.argsort()[::-1]], self.scores_.argsort()[::-1]

# 定义 Pareto 效率函数
def is_pareto_efficient(costs):
    is_efficient = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        is_efficient[i] = np.all(np.any(costs[:i] > c, axis=1)) and np.all(np.any(costs[i+1:] > c, axis=1))
    return is_efficient

# 评估模型
def evaluate_model(train_x, train_y, valid_x, valid_y):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(train_x, train_y)
    valid_pred = model.predict(valid_x)
    valid_pred_proba = model.predict_proba(valid_x)[:, 1]
    accuracy = accuracy_score(valid_y, valid_pred)
    roc_auc = roc_auc_score(valid_y, valid_pred_proba)
    precision = precision_score(valid_y, valid_pred)
    recall = recall_score(valid_y, valid_pred)
    f1 = f1_score(valid_y, valid_pred)
    return np.array([accuracy, roc_auc, precision, recall, f1])

# 将路径加入系统路径中
# path.append(r'D:\study\credit_scoring_papers&code\code')


# 定义数据加载函数
def load_data(dataset_name):
    dataset_path = '/home/server02/MLY/dataset/'
    shuffle_path = '/home/server02/MLY/shuffle_path/shuffle_index/'

    if dataset_name == 'give':
        data = pd.read_csv(os.path.join(dataset_path, 'give_me_some_credit_cleaned.csv'))
        features = data.drop('SeriousDlqin2yrs', axis=1).replace([-np.inf, np.inf], 0).replace([-np.nan, np.nan], 0)
        labels = data['SeriousDlqin2yrs']
    # 其他数据集加载逻辑...
    if dataset_name == 'bankfear':
        data = pd.read_csv(os.path.join(dataset_path, 'bankfear.csv'))
        features = data.drop(['loan_status', 'member_id'], axis=1).replace([-np.inf, np.inf, -np.nan, np.nan], 0)
        labels = data['loan_status']
    if dataset_name == 'shandong':
        data = pd.read_csv(os.path.join(dataset_path, 'shandong.csv'))
        features = data.drop('label', axis=1).replace([-np.inf, np.inf], 0).fillna(0)
        labels = data['label']
    if dataset_name == 'fannie':
        data = pd.read_csv(os.path.join(dataset_path, '/FannieMae/2008q1.csv', low_memory=True))
        features = data.drop(['DEFAULT', 'LOAN IDENTIFIER'], axis=1).replace([-np.inf, np.inf, np.nan], 0)
        labels = data['DEFAULT']
    if dataset_name == 'lc':
        data = pd.read_csv(r'D:\study\credit_scroing_datasets\lending club/2007-2015.csv', low_memory=True)
        features = data.drop('loan_status', axis=1)
        labels = data['loan_status']

    with open(os.path.join(shuffle_path, f'{dataset_name}/shuffle_index.pickle'), 'rb') as f:
        shuffle_index = pickle.load(f)

    train_size = int(features.shape[0] * 0.8)
    valid_size = int(features.shape[0] * 0.1)

    train_index = shuffle_index[:train_size]
    valid_index = shuffle_index[train_size:train_size + valid_size]
    test_index = shuffle_index[train_size + valid_size:]

    train_x, train_y = features.iloc[train_index, :], labels.iloc[train_index]
    valid_x, valid_y = features.iloc[valid_index, :], labels.iloc[valid_index]
    test_x, test_y = features.iloc[test_index, :], labels.iloc[test_index]

    return train_x, train_y, valid_x, valid_y, test_x, test_y

# 定义模型训练与评估函数
def train_and_evaluate_model(model, train_x, train_y, test_x, test_y, save_path, method, dataset):
    # 标准化数据
    scaler = StandardScaler()
    train_x = scaler.fit_transform(train_x)
    test_x = scaler.transform(test_x)

    model.fit(train_x, train_y)
    proba = model.predict_proba(test_x)[:, 1]
    pred = model.predict(test_x)

    # 计算各项评估指标
    auc = roc_auc_score(test_y, proba)
    acc = accuracy_score(test_y, pred)
    prec = precision_score(test_y, pred)
    rec = recall_score(test_y, pred)
    f1 = f1_score(test_y, pred)
    bs = brier_score_loss(test_y, proba)
    ks = ks_2samp(proba[test_y == 1], proba[test_y != 1]).statistic
    fprs, tprs, _ = roc_curve(test_y, proba)
    prec_rec = precision_recall_curve(test_y, proba)
    ap = average_precision_score(test_y, proba)

    def h_mean(precision, recall):
        if precision + recall == 0:
            return 0
        return 2 * (precision * recall) / (precision + recall)

    # 计算调和平均数
    hm = h_mean(prec, rec)

    def type1error(y_proba, y_true, threshold=0.5):
        y_pred = (y_proba >= threshold).astype(int)
        fp = ((y_pred == 1) & (y_true == 0)).sum()
        return fp / (y_true == 0).sum()

    def type2error(y_proba, y_true, threshold=0.5):
        y_pred = (y_proba >= threshold).astype(int)
        fn = ((y_pred == 0) & (y_true == 1)).sum()
        return fn / (y_true == 1).sum()

    # 计算第一类错误率和第二类错误率
    e1 = type1error(pred, test_y)
    e2 = type2error(pred, test_y)


    # 将结果保存到字典中
    results = {
        'auc': auc, 'acc': acc, 'prec': prec, 'rec': rec, 'f1': f1,
        'bs': bs, 'ks': ks, 'fprs': fprs, 'tprs': tprs,
        'prec_rec': prec_rec, 'ap': ap,'hm':hm,
         'e1': e1, 'e2': e2
    }

    # 保存模型性能到文件
    file_path = os.path.join(save_path, f'feature_{dataset}_{method}_res.pickle')
    with open(file_path, 'wb') as f:
        pickle.dump(results, f)
# 主程序
# 主程序
datasets = ['shandong']
methods = ['lr','lda','dt','knn','adaboost','rf','gbdt','xgboost','lightgbm']

base_path = r'D:\study\second\outcome'
params_path = '/home/server02/MLY/shuffle_path/shuffle_index/'
# 动态生成路径
for dataset in datasets:
    save_path = f"{base_path}\\{dataset}"
    print(save_path)

os.makedirs(save_path, exist_ok=True)

for dataset in datasets:
    train_x, train_y, valid_x, valid_y, test_x, test_y = load_data(dataset)

    for method in methods:
        params_loaded = False  # 标志，用于跟踪是否成功加载参数
        try:
            # 构建参数文件路径
            params_file = params_path + '{0}/{0}_{1}_params.pickle'.format(dataset, method)

            # 尝试加载参数
            with open(params_file, 'rb') as f:
                params = pickle.load(f)
                params_loaded = True

            # 根据不同方法创建模型并应用参数
            if method == 'lr':
                model = LogisticRegression(**params)
            elif method == 'rf':
                model = RandomForestClassifier(**params)
            elif method == 'lda':
                model = LinearDiscriminantAnalysis(**params)
            elif method == 'dt':
                model = DecisionTreeClassifier(**params)
            elif method == 'knn':
                model = KNeighborsClassifier(**params)
            elif method == 'adaboost':
                model = AdaBoostClassifier(**params)
            elif method == 'xgb':
                model = XGBClassifier(**params)
            elif method == 'lightgbm':
                model = LGBMClassifier(**params)
            elif method == 'nn':
                model = MLPClassifier(**params)
            elif method == 'gbdt':
                model = GradientBoostingClassifier(**params)

            if params_loaded:
                print(
                    f"Parameters for method {method} in dataset {dataset} were successfully loaded from {params_file}.")

        except FileNotFoundError:
            print(f"Params file for {method} in dataset {dataset} not found, using default parameters.")
            # 根据不同方法创建模型，使用默认参数
            if method == 'lr':
                model = LogisticRegression(random_state=42)
            elif method == 'rf':
                model = RandomForestClassifier(random_state=42)
            elif method == 'lda':
                model = LinearDiscriminantAnalysis()
            elif method == 'dt':
                model = DecisionTreeClassifier(random_state=42)
            elif method == 'knn':
                model = KNeighborsClassifier()
            elif method == 'adaboost':
                model = AdaBoostClassifier(random_state=42)
            elif method == 'xgb':
                model = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, use_label_encoder=False,
                                      eval_metric='logloss', random_state=42)
            elif method == 'lightgbm':
                model = LGBMClassifier(random_state=42)
            elif method == 'nn':
                model = MLPClassifier(random_state=42)
            elif method == 'gbdt':
                model = GradientBoostingClassifier(random_state=42)

        except Exception as e:
            print(f"An error occurred while initializing model {method} for dataset {dataset}: {e}")
            continue

        if not params_loaded:
            print(f"Default parameters used for method {method} in dataset {dataset}.")

        print(f"Training {method} model on {dataset} dataset...")
        train_and_evaluate_model(model, train_x, train_y, test_x, test_y, save_path, method, dataset)
        print(f"Finished training {method} model on {dataset} dataset.\n")
