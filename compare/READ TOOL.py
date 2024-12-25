import pickle

file_paths = [
    r'D:\study\second\outcome\give\lr_res.pickle',
    r'D:\study\second\outcome\give\lda_res.pickle',
    r'D:\study\second\outcome\give\dt_res.pickle',
    r'D:\study\second\outcome\give\knn_res.pickle',
    r'D:\study\second\outcome\give\adaboost_res.pickle',
    r'D:\study\second\outcome\give\rf_res.pickle',
    r'D:\study\second\outcome\give\gbdt_es.pickle',
    r'D:\study\second\outcome\give\lgb_res.pickle',
    r'D:\study\second\outcome\give\xgb_res.pickle',
    r'D:\study\second\outcome\give\AAESS_res.pickle',



]

# keys = ['auc', 'acc', 'prec', 'rec', 'f1', 'bs', 'fprs', 'tprs', 'e1', 'e2', 'prec_rec', 'ap', 'tpr', 'tnr', 'gmean']
keys = ['prec']
for file_path in file_paths:
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            print(f"\nLoaded data from {file_path}")
            # print("Available keys in the data:", list(data.keys()))

            # 读取每个键的具体值
            for key in keys:
                if key in data:
                    print(f"{key}: {data[key]}")
                else:
                    print(f"{key} not found in data")

    except Exception as e:
        print(f"Failed to load data from {file_path}: {e}")
