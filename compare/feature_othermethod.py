import os
import pickle
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import (roc_auc_score, accuracy_score, precision_score, recall_score,
                             f1_score, brier_score_loss, precision_recall_curve, average_precision_score,
                             roc_curve, log_loss)
from scipy.stats import ks_2samp

# 定义评估指标函数
def TPR(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return tp / (tp + fn)

def TNR(y_true, y_pred):
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    return tn / (tn + fp)

def g_mean(y_true, y_pred):
    return np.sqrt(TPR(y_true, y_pred) * TNR(y_true, y_pred))

# 加载数据并划分训练、验证和测试集
def load_data(dataset_name):
    dataset_path = r'D:\study\credit_scroing_datasets'
    shuffle_path = r'D:\study\Credit(1)\Credit\shuffle_index'

    if dataset_name == 'give':
        data = pd.read_csv(os.path.join(dataset_path, 'give_me_some_credit_cleaned.csv'))
        features = data.drop('SeriousDlqin2yrs', axis=1).replace([-np.inf, np.inf], 0).replace([-np.nan, np.nan], 0)
        labels = data['SeriousDlqin2yrs']
    elif dataset_name == 'bankfear':
        data = pd.read_csv(r'D:\study\credit_scroing_datasets\bankfear.csv', low_memory=True)
        features = data.drop(['loan_status', 'member_id'], axis=1).replace([-np.inf, np.inf, -np.nan, np.nan], 0)
        labels = data['loan_status']
    elif dataset_name == 'shandong':
        data = pd.read_csv(r'D:\study\credit_scroing_datasets\shandong.csv', low_memory=True)
        features = data.drop('label', axis=1).replace([-np.inf, np.inf], 0).fillna(0)
        labels = data['label']
    elif dataset_name == 'fannie':
        data = pd.read_csv(r'D:\study\credit_scroing_datasets\FannieMae/2008q1.csv', low_memory=True)
        features = data.drop(['DEFAULT', 'LOAN IDENTIFIER'], axis=1).replace([-np.inf, np.inf, np.nan], 0)
        labels = data['DEFAULT']
    elif dataset_name == 'lc':
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
    tpr = TPR(test_y, pred)
    tnr = TNR(test_y, pred)
    gmean = g_mean(test_y, pred)

    def h_mean(precision, recall):
        if precision + recall == 0:
            return 0
        return 2 * (precision + recall) / (precision + recall)

    hm = h_mean(prec, rec)

    def type1error(y_proba, y_true, threshold=0.5):
        y_pred = (y_proba >= threshold).astype(int)
        fp = ((y_pred == 1) & (y_true == 0)).sum()
        return fp / (y_true == 0).sum()

    def type2error(y_proba, y_true, threshold=0.5):
        y_pred = (y_proba >= threshold).astype(int)
        fn = ((y_pred == 0) & (y_true == 1)).sum()
        return fn / (y_true == 1).sum()

    e1 = type1error(proba, test_y)
    e2 = type2error(proba, test_y)

    results = {
        'auc': auc, 'acc': acc, 'prec': prec, 'rec': rec, 'f1': f1,
        'bs': bs, 'ks': ks, 'fprs': fprs, 'tprs': tprs,
        'prec_rec': prec_rec, 'ap': ap, 'hm': hm,
        'tpr': tpr, 'tnr': tnr, 'gmean': gmean, 'e1': e1, 'e2': e2
    }

    file_path = os.path.join(save_path, f'{dataset}_{method}_res.pickle')
    with open(file_path, 'wb') as f:
        pickle.dump(results, f)

# 主程序
datasets = ['shandong']
methods = ['lr','lda','dt','knn','adaboost','rf','gbdt','xgboost','lightgbm']
base_path = r'D:\study\second\outcome'
params_path = r'D:\study\Credit(1)\Credit\params/'

for dataset in datasets:
    save_path = f"{base_path}\\{dataset}"
    os.makedirs(save_path, exist_ok=True)

    train_x, train_y, valid_x, valid_y, test_x, test_y = load_data(dataset)

    for method in methods:
        params_loaded = False
        try:
            params_file = params_path + '{0}/{0}_{1}_params.pickle'.format(dataset, method)
            with open(params_file, 'rb') as f:
                params = pickle.load(f)
                params_loaded = True

            if method == 'lr':
                params.pop('covariance_estimator', None)
            elif method == 'gbdt':
                params.pop('gamma', None)

            if method == 'lr':
                model = LogisticRegression()
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
                model = GradientBoostingClassifier()

            if params_loaded:
                print(f"Parameters for method {method} in dataset {dataset} were successfully loaded from {params_file}.")

        except FileNotFoundError:
            print(f"Params file for {method} in dataset {dataset} not found, using default parameters.")
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
