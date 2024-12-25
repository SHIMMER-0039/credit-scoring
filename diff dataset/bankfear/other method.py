import os
import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import (roc_auc_score, accuracy_score, precision_score, recall_score,
                             f1_score, brier_score_loss, precision_recall_curve, average_precision_score,
                             roc_curve, confusion_matrix)
from scipy.stats import ks_2samp
from sklearn.model_selection import train_test_split
import tensorflow as tf


# 定义 TPR, TNR, 和 g_mean 函数
def TPR(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    TP = cm[1, 1]
    FN = cm[1, 0]
    return TP / (TP + FN)

def TNR(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    TN = cm[0, 0]
    FP = cm[0, 1]
    return TN / (TN + FP)

def g_mean(y_true, y_pred):
    tpr = TPR(y_true, y_pred)
    tnr = TNR(y_true, y_pred)
    return np.sqrt(tpr * tnr)




# Paths
root_path = 'D:/study/Credit(1)/Credit/'
params_path = r'D:\study\Credit(1)\Credit\params/'
dataset_path = r'D:\study\credit_scoring_datasets/'
shuffle_path = '/home/server02/MLY/shuffle_path/shuffle_index/'
save_path = r'D:\study\second\outcome/'
os.makedirs(save_path, exist_ok=True)




data = pd.read_csv('/home/server02/MLY/dataset/shandong.csv', low_memory=True)

features = data.drop('label', axis=1).replace([-np.inf, np.inf], 0).fillna(0)
labels = data['label']

train_size = int(features.shape[0] * 0.8)
valid_size = int(features.shape[0] * 0.1)
test_size = valid_size

with open(shuffle_path + 'shandong/shuffle_index.pickle', 'rb') as f:
    shuffle_index = pickle.load(f)

train_index = shuffle_index[:train_size]
valid_index = shuffle_index[train_size:(train_size + valid_size)]
test_index = shuffle_index[(train_size + valid_size):(train_size + valid_size + test_size)]
remaining_index = shuffle_index[(valid_size + test_size):]

train_x, train_y = features.iloc[train_index, :], labels.iloc[train_index]
valid_x, valid_y = features.iloc[valid_index], labels.iloc[valid_index]
test_x, test_y = features.iloc[test_index], labels.iloc[test_index]
remaining_x, remaining_y = features.iloc[remaining_index], labels.iloc[remaining_index]

full_train_x = pd.concat([train_x, valid_x], axis=0)
full_train_y = pd.concat([train_y, valid_y], axis=0)

model = Sequential([
    Dense(128, input_dim=full_train_x.shape[1], activation='relu'),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer=Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(full_train_x, full_train_y, epochs=5, batch_size=32, validation_data=(valid_x, valid_y))

proba = model.predict(test_x).flatten()  # 确保输出为一维数组
pred = (proba > 0.5).astype(int)

# 计算指标
auc = roc_auc_score(test_y, proba)
acc = accuracy_score(test_y, pred)
prec = precision_score(test_y, pred, zero_division=1)  # 防止未定义精度
rec = recall_score(test_y, pred)
f1 = f1_score(test_y, pred)
bs = brier_score_loss(test_y, proba)
fprs, tprs, _ = roc_curve(test_y, proba)
ks = ks_2samp(proba[test_y == 1], proba[test_y != 1]).statistic
prec_rec = precision_recall_curve(test_y, proba)
ap = average_precision_score(test_y, proba)

tpr = TPR(test_y, pred)
tnr = TNR(test_y, pred)
gmean = g_mean(test_y, pred)

performance_file = save_path + 'model_run_res.pickle'
with open(performance_file, 'wb') as f:
    pickle.dump({'auc': auc, 'acc': acc, 'prec': prec,
                 'rec': rec, 'f1': f1, 'bs': bs, 'fprs': fprs,
                 'tprs': tprs, 'ks': ks,
                 'prec_rec': prec_rec, 'ap': ap, 'tpr': tpr, 'tnr': tnr,
                 'gmean': gmean}, f)
