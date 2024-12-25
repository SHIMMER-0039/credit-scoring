import os
import pickle
# import imblearn
from sys import path


from lightgbm import LGBMClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier


path.append('D:\study\credit_scoring_papers&code\code')
# import matlab.engine
import pandas as pd
import numpy as np
import xgboost as xgb
from math import sqrt
from sklearn.impute import SimpleImputer
from scipy.stats import ks_2samp
#from keras.models import load_model
from sklearn.linear_model import LogisticRegression
from xgboost.sklearn import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from imblearn.under_sampling import (
    ClusterCentroids,
    NearMiss,
    RandomUnderSampler,
    EditedNearestNeighbours,
    AllKNN,
    TomekLinks,
    OneSidedSelection,
    CondensedNearestNeighbour,
    NeighbourhoodCleaningRule,
)
from imblearn.over_sampling import (
    RandomOverSampler, SMOTE, ADASYN,
    BorderlineSMOTE,
)
from imblearn.combine import (
    SMOTEENN, SMOTETomek,
)
# from imbalanced_ensemble.ensemble import SMOTEBoostClassifier
# from imbalanced_ensemble.ensemble import SMOTEBaggingClassifier
# from imbalanced_ensemble.ensemble import RUSBoostClassifier
# from imbalanced_ensemble.ensemble import UnderBaggingClassifier
# from imbalanced_ensemble.ensemble import BalanceCascadeClassifier
#
# #from imbalanced_ensemble.ensemble.reweighting.adacost import AdaCostClassifier
# from AWDF.AdaCost import AdaCostClassifier
# from imbalanced_ensemble.ensemble.reweighting.adauboost import AdaUBoostClassifier
# from imbalanced_ensemble.ensemble.reweighting.asymmetric_boost import AsymBoostClassifier
#
# from imblearn.ensemble import BalancedRandomForestClassifier
# from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
# from sklearn.linear_model import LogisticRegression
# from lightgbm.sklearn import LGBMClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score, roc_curve,\
     accuracy_score,precision_score,recall_score,\
     f1_score,brier_score_loss,precision_recall_curve,\
     average_precision_score,log_loss
# from AWDF.MetaCost import MetaCost
#from AdaCost import AdaCostClassifier
#from SMOTEBoost import SMOTEBoost
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier

'''
random over sampling: "ros"
random under sampling : rus 
smote: smote
rusboost: rusboost


cost-sesnitive learning methods for comparison:
    AdaCost 
    MetaCost
Cost sensitive algorithms that not implemented in this study:
    weighted
'''


'''
resampling-based methods:
    (1) Undersampling: 'RUS', 'CNN', 'ENN', 'NCR', 'Tomek', 'ALLKNN', 'OSS', 'NM', 'CC'  
        efficient: 'RUS','ENN','NCR'
    (2) Oversampling: 'SMOTE', 'ADASYN', 'BorderSMOTE'
    (3) Combined: 'SMOTEENN', 'SMOTETomek'
'''

undersamp = False
oversamp = False
hybrid = False
ensemble = False
cost_sensitive = True
reweighting = False
if undersamp is True:
#    imb_methods = ['RUS']    ### Efficient methods:  RUS, Tomek
    imb_methods = ['RUS', 'CNN', 'ENN', 'NCR', 'Tomek', 'ALLKNN', 'OSS', 'NM', 'CC']

if oversamp is True:
    imb_methods = ['ROS','SMOTE', 'ADASYN', 'BorderSMOTE']
if cost_sensitive is True:
    imb_methods = ['AdaUBoost','AsymBoost']
#    imb_methods = ['AdaCost','MetaCost','AdaUBoost','AsymBoost']
if hybrid is True:
    imb_methods = ['SMOTETomek']
#    imb_methods = ['SMOTEENN','SMOTETomek']
if ensemble is True:
    imb_methods = ['RUSBoost','UnderBagging','BalanceCascade','BCRF']   ### SMOTEBagging is memory inefficient.
#    imb_methods = ['SMOTEBoost','SMOTEBagging','RUSBoost','UnderBagging','BalanceCascade','BCRF']
root_path = 'D:/study/Credit(1)/Credit/'
params_path = r'D:\study\Credit(1)\Credit\params/'
dataset_path = r'D:\study\credit_scoring_datasets/'
shuffle_path = r'D:\study\Credit(1)\Credit\shuffle_index/'
save_path = os.path.join(root_path, 'performance/', 'imbalanced/','dataset')
os.makedirs(save_path, exist_ok=True)
random_state = 2021
#datasets = ['prosper','lendingclub','ppdai_2017']
datasets = ['shandong']
methods = ['xgb','lr','dt','rf','gbdt','lightgbm','lda']
n_jobs = -1

def sigmoid(x):
    return 1./(1. +  np.exp(-x))


def AP_eval(y_true, y_pred):
    p = 1 / (1 + np.exp(-y_pred))
    scores = average_precision_score(y_true, p)
    return 'AP', scores, False


def AUC(y_true, y_pred):
    scores = roc_auc_score(y_true, y_pred)
    return 'AUC', scores, False


def RECALL(y_true, y_pred):
    pred = [1 if y_pred[i] > 0.5 or y_pred[i] == 0.5 else 0 for i in range(y_pred.shape[0])]
    scores = recall_score(y_true, pred)
    return "RECALL", scores, False


def auroc(y_true, y_pred):
    return tf.py_func(roc_auc_score, (y_true, y_pred), tf.double)


def auch(preds, y):
    labels = y
    labels = list(labels)
    pred_matlab1 = list(preds)
    y_test_matlab1 = matlab.double(labels)
    pred_matlab1 = matlab.double(pred_matlab1)
    y_test_matlab = eng.transpose(y_test_matlab1)
    pred_matlab = eng.transpose(pred_matlab1)
    AUCH = eng.hmeasure(y_test_matlab, pred_matlab)
    AUCH = AUCH['H']
    return float(AUCH)


def type1error(preds, y):
    labels = y
    count_pos = len(labels[labels == 1])
    count_neg = len(labels[labels == 0])
    preds1 = np.copy(preds)
    preds1[preds1 >= 0.5] = 1
    preds1[preds1 < 0.5] = 0
    temp = labels - preds1
    r = len(temp[temp == -1]) / count_neg
    return float(r)


def type2error(preds, y):
    labels = y
    count_pos = len(labels[labels == 1])
    count_neg = len(labels[labels == 0])
    preds1 = np.copy(preds)
    preds1[preds1 >= 0.5] = 1
    preds1[preds1 < 0.5] = 0
    temp = labels - preds1
    return float((len(temp[temp == 1])) / count_pos)
def TPR(label,pred):
    return recall_score(label,pred)

def TNR(label,pred):
    tnr = sum([1 for i in range(pred.shape[0]) if (pred[i] == 0 and np.array(label)[i] == 0)]) /  len(np.where(label == 0)[0])
    return tnr

def g_mean(label,pred):
    # proba = sigmoid(proba)
    # pred = np.rint(proba)
    TPR = recall_score(np.array(label).astype(np.int16),pred)
    TNR = sum([1 for i in range(pred.shape[0]) if (pred[i] == 0 and np.array(label)[i] == 0)]) /  len(np.where(label == 0)[0])
    g_mean = sqrt(TPR * TNR)
    return g_mean

def auch(preds,y):
    labels = y
    labels=list(labels)
    pred_matlab1=list(preds)
    y_test_matlab1=matlab.double(labels)
    pred_matlab1=matlab.double(pred_matlab1)
    y_test_matlab=eng.transpose(y_test_matlab1)
    pred_matlab=eng.transpose(pred_matlab1)
    AUCH=eng.hmeasure(y_test_matlab,pred_matlab)
    AUCH=AUCH['H']
    return float(AUCH)

def type1error(preds, y):
    labels = y
    count_pos=len(labels[labels==1])
    count_neg=len(labels[labels==0])
    preds1 = np.copy(preds)
    preds1[preds1>=0.5]=1
    preds1[preds1<0.5]=0
    temp=labels-preds1
    r= len(temp[temp==-1])/count_neg
    return  float(r)

def type2error(preds, y):
    labels = y
    count_pos=len(labels[labels==1])
    count_neg=len(labels[labels==0])
    preds1 = np.copy(preds)
    preds1[preds1>=0.5]=1
    preds1[preds1<0.5]=0
    temp=labels-preds1
    return float((len(temp[temp==1]))/count_pos)

import numpy as np


def ghm_loss_xgb(preds, dtrain, bins=10, alpha=0.75, gamma=2.0):
    labels = dtrain.get_label()
    preds = 1.0 / (1.0 + np.exp(-preds))  # Sigmoid转换预测值为概率

    # 计算梯度和海森矩阵
    grad = preds - labels
    hess = preds * (1.0 - preds)

    # 计算梯度的绝对值
    g_abs = np.abs(grad)

    # 将梯度的绝对值分配到bins中
    edges = np.linspace(0, 1, bins + 1)
    g_bins = np.digitize(g_abs, edges) - 1
    g_bins = np.clip(g_bins, 0, bins - 1)

    # 计算每个bin的样本数
    bin_counts = np.bincount(g_bins, minlength=bins)

    # 计算每个样本的权重
    effective_num = 1.0 / np.clip(bin_counts, 1, np.max(bin_counts))
    weights = effective_num[g_bins]

    # 应用GHM权重调整
    grad *= weights
    hess *= weights

    return grad, hess


def ghm_loss_xgb1(preds, dtrain, bins=7, beta=0.65):
    """
    简化版的GHM损失函数用于XGBoost。

    参数:
    ------
    preds: numpy.ndarray
        模型预测的结果，一个包含预测值的数组。
    dtrain: xgboost.DMatrix
        XGBoost的DMatrix数据格式，包含真实标签。
    bins: int
        用于梯度密度估计的bins数量。
    beta: float
        用于计算样本权重的调和参数。
    """
    labels = dtrain.get_label()  # 从DMatrix获取真实标签
    p = 1 / (1 + np.exp(-preds))  # 使用sigmoid函数计算概率
    gradient = np.abs(p - labels)  # 计算梯度的近似大小

    # 初始化bins并统计每个bin中的样本数
    max_gradient = np.max(gradient)
    edges = np.linspace(0, max_gradient, bins + 1)
    bin_idx = np.digitize(gradient, edges) - 1
    bin_idx = np.clip(bin_idx, 0, bins - 1)  # 修正，确保bin_idx不超出范围
    bin_counts = np.bincount(bin_idx, minlength=bins)

    # 防止bin_counts中的任何值为0，以避免除以0的错误
    bin_counts[bin_counts == 0] = 1

    # 计算每个样本的权重
    effective_num = 1.0 / (np.power(bin_counts[bin_idx], beta))
    weights = (effective_num / np.sum(effective_num)) * len(labels)  # 修正了weights的计算

    # 计算加权的二分类交叉熵损失的梯度和Hessian
    grad = (p - labels) * weights
    hess = p * (1 - p) * weights

    return grad, hess


def custom_logloss(preds, dtrain):
    labels = dtrain.get_label()
    preds = 1.0 / (1.0 + np.exp(-preds))  # Sigmoid函数将预测值转换为概率
    grad = preds - labels  # 梯度
    hess = preds * (1.0 - preds)  # 海森矩阵
    return grad, hess

def custom_eval(preds, dtrain):
    labels = dtrain.get_label()
    preds = 1.0 / (1.0 + np.exp(-preds))
    return 'error', float(sum(labels != (preds > 0.5))) / len(labels)


datasets = ['shandong']
methods = ['xgb','lr','dt','rf','gbdt','lightgbm','lda']
data = pd.read_csv(r'D:\study\credit_scroing_datasets\shandong.csv', low_memory=True)  #### data path
features = data.drop('label', axis=1)
labels = data['label']
features = features.replace([-np.inf, np.inf], 0)
features = features.replace([-np.nan, np.nan], 0)
with open(shuffle_path + 'shandong/shuffle_index.pickle', 'rb') as f:
    shuffle_index = pickle.load(f)
train_size = int(features.shape[0] * 0.8)
valid_size = int(features.shape[0] * 0.1)
train_index = shuffle_index[:train_size]
valid_index = shuffle_index[train_size:(train_size + valid_size)]
test_index = shuffle_index[(train_size + valid_size):]
train_x, train_y = features.iloc[train_index, :], labels.iloc[train_index]
valid_x, valid_y = features.iloc[valid_index, :], labels.iloc[valid_index]
test_x, test_y = features.iloc[test_index, :], labels.iloc[test_index]
for dataset in datasets:
    for method in methods:
        if method == 'lr':
            clf = LogisticRegression()
        if method == 'lda':
            clf = LDA()
        if method == 'gbdt':
            with open(params_path + '{0}/{0}_{1}_params.pickle'.format(dataset, method), 'rb') as f:
                params = pickle.load(f)
            print(params)
            clf = GradientBoostingClassifier(**params)
        if method == 'rf':
            with open(params_path + '{0}/{0}_{1}_params.pickle'.format(dataset, method), 'rb') as f:
                params = pickle.load(f)
            clf = RandomForestClassifier(**params, n_jobs=n_jobs, random_state=random_state)
        if method == 'lightgbm':
            with open(params_path + '{0}/{1}_params.pickle'.format(dataset, method), 'rb') as f:
                params = pickle.load(f)
            clf = LGBMClassifier(**params,
                                 boosting_type='gbdt',
                                 objective='binary',
                                 n_jobs=n_jobs,
                                 random_state=random_state)
            eval_set = [(train_x, train_y), (valid_x, valid_y)]
        # if method == 'lgb':
        #     with open(params_path + '{0}/{1}_params.pickle'.format(dataset, method), 'rb') as f:
        #         params, _ = pickle.load(f)
        #     clf = LGBMClassifier(**params,
        #                          boosting_type='gbdt',
        #                          objective='binary',
        #                          n_jobs=n_jobs,
        #                          random_state=random_state)
        #     eval_set = [(train_x, train_y), (valid_x, valid_y)]
        if method == 'xgb':
            with open(params_path + '{0}/{1}_params.pickle'.format(dataset, method), 'rb') as f:
                params = pickle.load(f)
            clf = XGBClassifier(**params)
        if method == 'dt':
            with open(params_path + '{0}/{1}.pickle'.format(dataset, method), 'rb') as f:
                params = pickle.load(f)
            clf = DecisionTreeClassifier(**params)

        clf.fit(train_x, train_y)  # Fit the classifier
        #         # 初始化 XGBClassifier 实例
        #         # 请根据加载的 params 设置适当的参数
        #         xgb_model = XGBClassifier(**params)
        #
        #         # 初始化 Boruta 特征选择器，使用 XGBClassifier
        #
        #         # 使用过滤后的特征重新创建 DMatrix
        #         dtrain = xgb.DMatrix(train_x_filtered, label=train_y)
        #         dtest = xgb.DMatrix(test_x_filtered, label=test_y)
        #
        #         # 设置其他模型参数
        #         params = {
        #             # 根据需要设置其他参数
        #             'objective': 'binary:logistic',  # 使用自定义损失函数时，此设置将被覆盖
        #         }
        #         # 训练模型
        #         bst = xgb.train(params, dtrain, num_boost_round=100, obj=ghm_loss_xgb1, feval=custom_eval)
        #
        #         # 进行预测
        #         y_pred_prob = bst.predict(dtest)
        #         y_pred = np.round(y_pred_prob)
        #
        #     # 计算评估指标
        #     accuracy = accuracy_score(test_y, y_pred)
        #     logloss = log_loss(test_y, y_pred_prob)
        #     auc = roc_auc_score(test_y, y_pred_prob)
        #     precision = precision_score(test_y, y_pred)
        #     recall = recall_score(test_y, y_pred)
        #     f1 = f1_score(test_y, y_pred)
        #     brier = brier_score_loss(test_y, y_pred_prob)
        #     type1_err = type1error(y_pred, test_y)
        #     type2_err = type2error(y_pred, test_y)
        #     average_precision = average_precision_score(test_y, y_pred_prob)
        #     true_positive_rate = TPR(test_y, y_pred)
        #     true_negative_rate = TNR(test_y, y_pred)
        #     geometric_mean = g_mean(test_y, y_pred)
        #
        #     # 打印评估指标
        #     print(f"Accuracy: {accuracy}")
        #     print(f"Log Loss: {logloss}")
        # if method == 'dt':
        #     with open(params_path + '{0}/{0}_{1}_params.pickle'.format(dataset, method), 'rb') as f:
        #         params = pickle.load(f)
        #     clf = DecisionTreeClassifier(**params)
        # score = g_mean(valid_y, pred)
        # print('this is Gmean score:', score)
        # if best_score < score:
        #     best_score = score
        # eng = matlab.engine.start_matlab()#
        # eng.cd(r'/media/shom/老毛桃U盘/LINUX/writing/FDP/code/',nargout = 0)
        # eng.ls(nargout=0)
        dtest = xgb.DMatrix(test_x)

        # 使用训练好的模型进行预测
        # 注意：predict 方法返回的是预测为正类的概率
        proba = clf.predict_proba(test_x)[:,
                1]  # For binary classification, get probabilities for the positive class
        pred = (proba > 0.5).astype(int)  # Class predictions

        # Calculate metrics
        auc = roc_auc_score(test_y, proba)
        acc = accuracy_score(test_y, pred)
        logloss = log_loss(test_y, proba)
        prec = precision_score(test_y, pred)
        rec = recall_score(test_y, pred)
        f1 = f1_score(test_y, pred)
        bs = brier_score_loss(test_y, proba)
        e1 = type1error(proba, test_y)
        e2 = type2error(proba, test_y)
        prec_rec = precision_recall_curve(test_y, proba)
        ap = average_precision_score(test_y, proba)
        tpr = TPR(test_y, pred)  # 确保你已经定义了TPR函数
        tnr = TNR(test_y, pred)  # 确保你已经定义了TNR函数
        # feature_importance = clf.get_score(importance_type='weight')
        gmean = g_mean(test_y, pred)
        print(acc)

        file_path = os.path.join(save_path, '{0}\{1}_res11.pickle'.format(dataset, method))
        with open(file_path, 'wb') as f:
            pickle.dump({'auc': auc, 'acc': acc, 'prec': prec,
                         'rec': rec, 'f1': f1, 'bs': bs,
                          'e1': e1, 'e2': e2,
                         'prec_rec': prec_rec, 'ap': ap, 'tpr': tpr, 'tnr': tnr,
                         'gmean': gmean}, f)
            print('this is AUC:', auc)
            print(
                'The results of {0} on {1} is calculated ...\n\n'.format(method, dataset)
            )



