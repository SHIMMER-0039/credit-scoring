import numpy as np
import pandas as pd
import pickle 
import random
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
# from sklearn.cross_validation import cross_val_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score
from hyperopt import fmin, tpe, hp,space_eval,rand,Trials,partial,STATUS_OK
import matplotlib.pyplot as plt 
import matlab.engine 
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier 
from lightgbm import LGBMClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier 
from sklearn.neighbors import KNeighborsClassifier
# from sklearn.cross_validation import cross_val_predict as cvp
from sklearn.model_selection import cross_val_predict as cvp
from functools import reduce
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_auc_score, roc_curve,\
     accuracy_score,precision_score,recall_score,\
     f1_score,brier_score_loss,precision_recall_curve,\
     average_precision_score
     
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

class CascadeForest():
    def __init__(self, base_estimator, params_list, k_fold = 3, evaluate = lambda pre,y: roc_auc_score(y,pre)):
        if k_fold > 1: #use cv
            self.params_list = params_list
        else:#use oob
            self.params_list = [params.update({'oob_score':True}) or params for params in params_list]
        self.k_fold = k_fold
        self.evaluate = evaluate
        self.base_estimator = base_estimator
#         base_class = base_estimator.__class__
#         global prob_class
#         class prob_class(base_class): #to use cross_val_predict, estimator's predict method should be predict_prob
#             def predict(self, X):
#                 return base_class.predict_proba(self, X)
#         self.base_estimator = prob_class()

    def fit(self,X_train,y_train):
        self.n_classes = len(np.unique(y_train))
        self.estimators_levels = []
        klass = [self.base_estimator[i].__class__ for i in range(len(self.base_estimator))]
        predictions_levels = []
        self.classes = np.unique(y_train)

        #first level
        estimators = [klass[i](**self.params_list[i]) for i in range(len(self.params_list))]
        self.estimators_levels.append(estimators)
        predictions = []
        for estimator in estimators:
            estimator.fit(X_train, y_train)
            if self.k_fold > 1:# use cv
                predict_ = cvp(estimator, X_train, y_train, cv=self.k_fold, n_jobs = -1)
            else:#use oob
                predict_ = estimator.oob_decision_function_
                #fill default value if meet nan
                inds = np.where(np.isnan(predict_))
                predict_[inds] = 1./self.n_classes
            predictions.append(predict_)
        attr_to_next_level = np.hstack(predictions)
#         y_pre = self.classes.take(np.argmax(np.array(predictions).mean(axis=0),axis=1),axis=0)
        y_pre = np.array(predictions).mean(axis = 0)[:,1]
        self.max_accuracy = self.evaluate(y_pre,y_train)

        #cascade step
        while True:
            print('level {}, CV accuracy: {}'.format(len(self.estimators_levels),self.max_accuracy))
            estimators = [klass[i](**self.params_list[i]) for i in range(len(self.params_list))]
            self.estimators_levels.append(estimators)
            predictions = []
            X_train_step = np.hstack((attr_to_next_level,X_train))
            for estimator in estimators:
                estimator.fit(X_train_step, y_train)
                if self.k_fold > 1:# use cv
                    predict_ = cvp(estimator, X_train_step, y_train, cv=self.k_fold, n_jobs = -1)
                else:#use oob
                    predict_ = estimator.oob_decision_function_
                    #fill default value if meet nan
                    inds = np.where(np.isnan(predict_))
                    predict_[inds] = 1./self.n_classes
                predictions.append(predict_)
            attr_to_next_level = np.hstack(predictions)
            y_pre = self.classes.take(np.argmax(np.array(predictions).mean(axis=0),axis=1),axis=0)
            accuracy = self.evaluate(y_pre,y_train)
            if accuracy > self.max_accuracy:
                self.max_accuracy = accuracy
            else:
                self.estimators_levels.pop()
                break

    def predict_proba_staged(self,X):
        #init ouput, shape = nlevel * nsample * nclass
        self.proba_staged = np.zeros((len(self.estimators_levels),len(X),self.n_classes))

        #first level
        estimators = self.estimators_levels[0]
        predictions = []
        for estimator in estimators:
            predict_ = estimator.predict(X)
            predictions.append(predict_)
        attr_to_next_level = np.hstack(predictions)
        self.proba_staged[0] = np.array(predictions).mean(axis=0) #不同estimator求平均

        #cascade step
        for i in range(1,len(self.estimators_levels)):
            estimators = self.estimators_levels[i]
            predictions = []
            X_step = np.hstack((attr_to_next_level,X))
            for estimator in estimators:
                predict_ = estimator.predict(X_step)
                predictions.append(predict_)
            attr_to_next_level = np.hstack(predictions)
            self.proba_staged[i] = np.array(predictions).mean(axis=0)

        return self.proba_staged
    
    def predict_proba(self,X):
        return self.predict_proba_staged(X)[-1]
    
    def predict_staged(self,X):
        proba_staged = self.predict_proba_staged(X)
        predictions_staged = np.apply_along_axis(lambda proba: self.classes.take(np.argmax(proba),axis=0),
                                                 2, 
                                                 proba_staged)
        return predictions_staged

    def predict(self,X):
        proba = self.predict_proba(X)
        predictions = self.classes.take(np.argmax(proba,axis=1),axis=0) #平均值最大的index对应的class
        return predictions


dataset = 'fannie'
if dataset == 'fannie':
    data = pd.read_csv('/home/shom/hardware/C/projects/dataset/FannieMae/2008q1.csv')
    features = data.drop('DEFAULT',axis = 1)
    features.drop('LOAN IDENTIFIER',axis = 1,inplace = True)
    features = features.replace([-np.inf,np.inf],0)
    features = features.replace([-np.nan,np.nan],0)
    labels=data['DEFAULT']
    print(features.shape)
    with open('/home/shom/hardware/C/projects/shuffle_index/fannie/shuffle_index.pickle','rb') as f:
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
    data = pd.read_csv('/home/shom/hardware/C/projects/dataset/lending club/2007-2015.csv',low_memory = True)
    features = data.drop('loan_status',axis = 1)
    labels = data['loan_status']
    with open('/home/shom/hardware/C/projects/XGBFocal/shuffle_index/lc/shuffle_index.pickle','rb') as f:
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
    data = pd.read_csv('/home/shom/hardware/C/projects/dataset/bankfear.csv',low_memory = True)
    features = data.drop(['loan_status','member_id'],axis = 1)
    features = features.replace([-np.inf,np.inf],0)
    features = features.replace([-np.nan,np.nan],0)
    labels = data['loan_status']
    with open('/home/shom/hardware/C/projects/XGBFocal/shuffle_index/bankfear/shuffle_index.pickle','rb') as f:
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
    data = pd.read_csv('/home/shom/hardware/C/projects/dataset/give_me_some_credit_cleaned.csv')
    features= data.drop('SeriousDlqin2yrs',axis = 1)
    labels = data['SeriousDlqin2yrs']
    features = features.replace([-np.inf,np.inf],0)
    features = features.replace([-np.nan,np.nan],0)
    with open('/home/shom/hardware/C/projects/XGBFocal/shuffle_index/give/shuffle_index.pickle','rb') as f:
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



# cascade_forest_params1 = RandomForestClassifier(n_estimators=300,min_samples_split=11,max_features=1,n_jobs=-1).get_params()
# cascade_forest_params2 = RandomForestClassifier(n_estimators=300,min_samples_split=11,max_features='sqrt',n_jobs=-1).get_params()

# cascade_forest_params1 = XGBClassifier(n_estimators = 100,max_depth = 6,tree_method="hist",n_jobs=-1).get_params()
# cascade_forest_params2 = XGBClassifier(n_estimators = 100,max_depth = 7,tree_method="hist",n_jobs=-1).get_params()

#### 异质
num_estimators = [100,200,400,800,1000,1500]
for num_est in num_estimators:
    cascade_forest_params1 = RandomForestClassifier(n_estimators = num_est,min_samples_split = 11,max_features = 1,n_jobs = -1).get_params()
    cascade_forest_params2 = RandomForestClassifier(n_estimators = num_est,min_samples_split = 21,max_features = 'sqrt',n_jobs = -1).get_params()
    cascade_forest_params3 = XGBClassifier(n_estimators = num_est,num_leaves = 50,n_jobs = -1).get_params()
    cascade_forest_params4 = LGBMClassifier(n_estimators = num_est,num_leaves = 100,n_jobs = -1).get_params()
    
    cascade_params_list = [cascade_forest_params1,cascade_forest_params2,cascade_forest_params3,cascade_forest_params4]
    
    def calc_accuracy(pre,y):
        return float(sum(pre==y))/len(y)
    class ProbRandomForestClassifier(RandomForestClassifier):
        def predict(self, X):
            return RandomForestClassifier.predict_proba(self, X)
    class ProbXGBClassifier(XGBClassifier):
        def predict(self,X):
            return XGBClassifier.predict_proba(self,X)
    class ProbLGBMClassifier(LGBMClassifier):
        def predict(self,X):
            return LGBMClassifier.predict_proba(self,X)
    train_size = train_x.shape[0]
    # gcForest 
    
    
    # CascadeRF baseline
    # BaseCascadeRF = CascadeForest(ProbRandomForestClassifier(),cascade_params_list,k_fold=9)
    # BaseCascadeRF = CascadeForest(ProbXGBClassifier(),cascade_params_list,k_fold = 5)
    # BaseCascadeRF = CascadeForest(ProbLGBMClassifier(),cascade_params_list,k_fold = 5)
    BaseCascadeRF = CascadeForest([ProbRandomForestClassifier(),ProbRandomForestClassifier(),
                                   ProbXGBClassifier(),ProbLGBMClassifier()],cascade_params_list,
                                  k_fold = 5)
    BaseCascadeRF.fit(train_x[:train_size], train_y[:train_size])
    y_pre_staged = BaseCascadeRF.predict_staged(test_x)
    test_accuracy_staged = np.apply_along_axis(lambda y_pre: calc_accuracy(y_pre,test_y), 1, y_pre_staged)
    print('\n'.join('level {}, test accuracy: {}'.format(i+1,test_accuracy_staged[i]) for i in range(len(test_accuracy_staged))))
    staged_proba = BaseCascadeRF.predict_proba_staged(test_x)
    proba = staged_proba[-1]
    pred = np.argmax(proba,axis = 1)
    for i in range(len(staged_proba)):
        auc = roc_auc_score(test_y,staged_proba[i][:,1])
        print('auc:',auc)
    eng = matlab.engine.start_matlab()#启动matlab
    eng.cd(r'/home/shom/hardware/C/projects/TEM/',nargout = 0)
    eng.ls(nargout=0)
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
    with open('/home/shom/hardware/C/projects/HeterDF/res/{0}/result/{1}_{2}_res.pickle'.format(dataset,dataset,model),'wb') as f:
        pickle.dump({'auc':auc,'acc':acc,'prec':prec,
                     'rec':rec,'f1':f1,'bs':bs,'fpr':fpr,
                     'tpr':tpr,'e1':e1,'e2':e2,'hm':hm,
                    'prec_rec':prec_rec,'ap':ap},f)
    print('this is AUC:',auc)
    print('The results of {0} on {1} is claculated ...\n\n'.format(model,dataset))