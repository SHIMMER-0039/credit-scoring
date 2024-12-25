from sys import path

from sklearn.svm import SVC

path.append('D:/study/credit_scroing_datasets')
# import matlab.engine
import pickle
import numpy as np
import pandas as pd 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from xgboost.sklearn  import XGBClassifier 
from lightgbm.sklearn import LGBMClassifier 
from keras.models import Sequential 
from keras.layers import Dense 
from keras.models import load_model
# from keras.optimizers import Adam
from sklearn.metrics import roc_auc_score, roc_curve,\
     accuracy_score,precision_score,recall_score,\
     f1_score,brier_score_loss,precision_recall_curve,\
     average_precision_score
def auch(preds,y):
        labels = y
        labels=list(labels)
        pred_matlab1=list(preds)
        # y_test_matlab1=matlab.double(labels)
        # pred_matlab1=matlab.double(pred_matlab1)
        # y_test_matlab=eng.transpose(y_test_matlab1)
        pred_matlab=eng.transpose(pred_matlab1) 
        # AUCH=eng.hmeasure(y_test_matlab,pred_matlab)
        # AUCH=AUCH['H']
        # return float(AUCH)

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


datasets = ['full_t-1']
models = ['lr','lda','ada','dt','gbdt','rf','knn','xgb','lgb']
#models = ['nn']
for dataset in datasets:
    if dataset == 'fannie':
        data = pd.read_csv(r'D:\study\credit_scroing_datasets\FannieMae')
        features = data.drop('DEFAULT',axis = 1)
        features = features.replace([-np.inf,np.inf],0)
        features = features.replace([-np.nan,np.nan],0)
        labels=data['DEFAULT']
        print(features.shape)
        with open(r'D:\study\Credit(1)\Credit\shuffle_index\fannie\shuffle_index.pickle','rb') as f:
            shuffle_index = pickle.load(f)
        train_size = int(features.shape[0] * 0.7)
        valid_size = int(features.shape[0] * 0.15)
        train_index = shuffle_index[:train_size]
        valid_index = shuffle_index[train_size:(train_size + valid_size)]
        test_index = shuffle_index[(train_size + valid_size):]
        train_x,train_y = features.iloc[train_index,:],labels.iloc[train_index]
        valid_x,valid_y = features.iloc[valid_index,:],labels.iloc[valid_index]
        test_x,test_y = features.iloc[test_index,:],labels.iloc[test_index]
    
    
    elif dataset == 'lc':
        data = pd.read_csv(r'D:\study\credit_scroing_datasets\2007-2015.csv',low_memory = True)
        features = data.drop('loan_status',axis = 1)
        labels = data['loan_status']
        with open(r'D:\study\Credit(1)\Credit\shuffle_index\lc\shuffle_index.pickle','rb') as f:
            shuffle_index = pickle.load(f)
        train_size = int(features.shape[0] * 0.8)
        valid_size = int(features.shape[0] * 0.1)
        train_index = shuffle_index[:train_size]
        valid_index = shuffle_index[train_size:(train_size + valid_size)]
        test_index = shuffle_index[(train_size + valid_size):]
        train_x,train_y = features.iloc[train_index,:],labels.iloc[train_index]
        valid_x,valid_y = features.iloc[valid_index,:],labels.iloc[valid_index]
        test_x,test_y = features.iloc[test_index,:],labels.iloc[test_index]
        print(train_x.shape,train_y.shape)
    elif dataset == 'bankfear':
        data = pd.read_csv(r'D:\study\credit_scroing_dataset\sbankfear.csv',low_memory = True)
        features = data.drop(['loan_status','member_id'],axis = 1)
        features = features.replace([-np.inf,np.inf],0)
        features = features.replace([-np.nan,np.nan],0)
        labels = data['loan_status']
        with open(r'D:\study\Credit(1)\Credit\shuffle_index\bankfear\shuffle_index.pickle','rb') as f:
            shuffle_index = pickle.load(f)
        train_size = int(features.shape[0] * 0.8)
        valid_size = int(features.shape[0] * 0.1)
        train_index = shuffle_index[:train_size]
        valid_index = shuffle_index[train_size:(train_size + valid_size)]
        test_index = shuffle_index[(train_size + valid_size):]
        train_x,train_y = features.iloc[train_index,:],labels.iloc[train_index]
        valid_x,valid_y = features.iloc[valid_index,:],labels.iloc[valid_index]
        test_x,test_y = features.iloc[test_index,:],labels.iloc[test_index]
        print(train_x.shape,train_y.shape)
    elif dataset == 'give':
        data = pd.read_csv(r'D:\study\credit_scroing_datasets\give_me_some_credit_cleaned.csv')
        features= data.drop('SeriousDlqin2yrs',axis = 1)
        labels = data['SeriousDlqin2yrs']
        features = features.replace([-np.inf,np.inf],0)
        features = features.replace([-np.nan,np.nan],0)
        with open(r'D:\study\Credit(1)\Credit\shuffle_index\give\shuffle_index.pickle','rb') as f:
            shuffle_index = pickle.load(f)
        train_size = int(features.shape[0] * 0.8)
        valid_size = int(features.shape[0] * 0.1)
        train_index = shuffle_index[:train_size]
        valid_index = shuffle_index[train_size:(train_size + valid_size)]
        test_index = shuffle_index[(train_size + valid_size):]
        train_x = features.iloc[train_index]
        train_y = labels.iloc[train_index]
        valid_x = features.iloc[valid_index]
        valid_y = labels.iloc[valid_index]
        test_x = features.iloc[test_index]
        test_y = labels.iloc[test_index]

    elif 'full' in dataset:

        data = pd.read_csv('D:/study/credit_scroing_datasets/T时刻/{}.csv'.format(dataset[-4:]),encoding='ISO-8859-1',low_memory = True) #### data path
        features = data.iloc[:,:-1]
        labels = data.iloc[:,-1]
        features = features.replace([-np.inf,np.inf],0)
        features = features.replace([-np.nan,np.nan],0)
        with open('/home/shom/hardware/C/projects/YAO/shuffle_index1.pickle','rb') as f:
            shuffle_index = pickle.load(f)
        train_size = int(features.shape[0] * 0.8)
        valid_size = int(features.shape[0] * 0.1)
        train_index = shuffle_index[:train_size]
        valid_index = shuffle_index[train_size:(train_size + valid_size)]
        test_index = shuffle_index[(train_size + valid_size):]
        train_x,train_y = features.iloc[train_index,:],labels.iloc[train_index]
        valid_x,valid_y = features.iloc[valid_index,:],labels.iloc[valid_index]
        test_x,test_y = features.iloc[test_index,:],labels.iloc[test_index]
        print(train_x.shape,train_y.shape)
    elif dataset == 'shandong':
        data = pd.read_csv(r'D:\study\credit_scroing_datasets\shandong.csv',low_memory = True) #### data path
        features = data.drop('label',axis = 1)
        labels = data['label']
        features = features.replace([-np.inf,np.inf],0)
        features = features.replace([-np.nan,np.nan],0)
        with open(r'D:\study\Credit(1)\Credit\shuffle_index\shandong\shuffle_index.pickle','rb') as f:
            shuffle_index = pickle.load(f)
        train_size = int(features.shape[0] * 0.8)
        valid_size = int(features.shape[0] * 0.1)
        train_index = shuffle_index[:train_size]
        valid_index = shuffle_index[train_size:(train_size + valid_size)]
        test_index = shuffle_index[(train_size + valid_size):]
        train_x,train_y = features.iloc[train_index,:],labels.iloc[train_index]
        valid_x,valid_y = features.iloc[valid_index,:],labels.iloc[valid_index]
        test_x,test_y = features.iloc[test_index,:],labels.iloc[test_index]
        print(train_x.shape,train_y.shape)
    elif dataset == 'PPDai':
        with open(r'D:\study\credit_scroing_datasets\PPDai\dataset.pickle','rb') as f:
            data = pickle.load(f)
        with open(r'D:\study\Credit(1)\Credit\shuffle_index\PPDai\shuffle_index.pickle','rb') as f:
            shuffle_index = pickle.load(f)
        X = data['X']
        y = data['y']
        train_size = int(X.shape[0] * 0.8)
        valid_size = int(X.shape[0] * 0.1)
        train_index = shuffle_index[:train_size]
        valid_index = shuffle_index[train_size:(train_size + valid_size)]
        test_index = shuffle_index[(train_size + valid_size):]
        train_x,train_y = X.iloc[train_index,:],y.iloc[train_index]
        valid_x,valid_y = X.iloc[valid_index,:],y.iloc[valid_index]
        test_x,test_y = X.iloc[test_index,:],y.iloc[test_index]

    for model in models:
        if model != 'nn':
            with open('/home/dhu/NW/dataset/res/{0}/{1}_{2}_params.pickle'.format(dataset,dataset,model),'rb') as f:
                best_params = pickle.load(f)
        print('this is the best_params:',best_params)
        if model == 'lda':
            clf = LDA()
        if model == 'lr':
            clf = LogisticRegression(max_iter = 1000,solver = 'liblinear',penalty = 'l1')
        if model == 'dt':
            clf = DecisionTreeClassifier(**best_params)
        if model == 'rf':
            clf = RandomForestClassifier(**best_params,n_jobs = 16)
        if model == 'knn':
            clf = KNeighborsClassifier(**best_params,n_jobs = 16)
        if model == 'xgb':
            clf = XGBClassifier(**best_params,max_delta_step = 10,  #10步不降则停止
                                objective="binary:logistic",
                                n_jobs = 16)
        if model == 'lgb':
            clf = LGBMClassifier(**best_params,boosting_type = 'gbdt',
                                 objective = 'binary',n_jobs = 16)
#        if model == 'nn':
#            num_layers = best_params['num_layer']
#            hidden_node = best_params['hidden_node']
#            learning_rate = best_params['learning_rate']
#            opt = Adam(lr = learning_rate)
#            clf = Sequential()
#            clf.add(Dense(input_shape = (train_x.shape[1],),units = hidden_node,activation = 'relu'))
#            for i in range(num_layers - 1):
#                clf.add(Dense(units = hidden_node,activation = 'relu'))
#            clf.add(Dense(units = 1,activation = 'sigmoid'))
#            clf.compile(loss = 'binary_crossentropy',optimizer = opt)
        if model == 'ada':
            clf = AdaBoostClassifier(**best_params)
        if model == 'svm':
            clf = SVC(probability = True)
        if model == 'nn':
            scaler = StandardScaler()
            train_x = scaler.fit_transform(train_x)
            valid_x = scaler.transform(valid_x)
            test_x = scaler.transform(test_x)
            clf = load_model('/home/dhu/NW/dataset/res/{0}/{1}_{2}_params.h5'.format(dataset,dataset,model))
#            hist = clf.fit(train_x,train_y,validation_data = (valid_x,valid_y),epochs = 100,batch_size = 3000)
            proba = clf.predict(test_x)
            pred = [1 if proba[i] > 0.5 or proba[i] == 0.5 else 0 for i in range(proba.shape[0])]
            proba = np.concatenate([1-proba,proba],axis = 1)
        else:
            clf.fit(train_x,train_y)
            proba = clf.predict_proba(test_x)
            pred = clf.predict(test_x)
        # eng = matlab.engine.start_matlab()#启动matlab
        # eng.cd(r'/home/dhu/NW/dataset/',nargout = 0)
        # eng.ls(nargout=0)
        fpr,tpr,_ = roc_curve(test_y,proba[:,1])
        auc = roc_auc_score(test_y,proba[:,1])
        acc = accuracy_score(test_y,pred)
        prec = precision_score(test_y,pred)
        rec = recall_score(test_y,pred)
        f1 = f1_score(test_y,pred)
        bs = brier_score_loss(test_y,proba[:,1])
        e1 = type1error(proba[:,1],test_y)
        e2 = type2error(proba[:,1],test_y)
        hm = auch(proba[:,1],test_y)
        prec_rec = precision_recall_curve(test_y,proba[:,1])
        ap = average_precision_score(test_y,proba[:,1])
        with open('/home/dhu/NW/dataset/res/{1}_{2}_res.pickle'.format(dataset,dataset,model),'wb') as f:
            pickle.dump({'auc':auc,'acc':acc,'prec':prec,
                         'rec':rec,'f1':f1,'bs':bs,'fpr':fpr,
                         'tpr':tpr,'e1':e1,'e2':e2,'hm':hm,
                        'prec_rec':prec_rec,'ap':ap},f)
        print('this is AUC:',auc)
        print('The results of {0} on {1} is claculated ...\n\n'.format(model,dataset))