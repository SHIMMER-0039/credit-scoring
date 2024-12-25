import pickle

def load_pickle(file_path, label, keys):
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)

        # 结果字典
        result_data = {}

        # 如果label是'AAESS'，处理特定情况
        if label == 'AAESS':
            if 'results' in data and isinstance(data['results'], list) and len(data['results']) > 0:
                result = data['results'][0]  # 假设你想要使用第一个结果，可以根据需要调整
                for key in keys:
                    if key in result:
                        result_data[key] = result[key]
                    else:
                        print(f"Warning: Missing '{key}' in results for {label} in {file_path}")
            else:
                print(f"Warning: 'results' key is missing or empty for {label} in {file_path}")
        else:
            for key in keys:
                if key in data:
                    result_data[key] = data[key]
                else:
                    print(f"Warning: Missing '{key}' for {label} in {file_path}")

        return result_data

    except Exception as e:
        print(f"Failed to load data from {file_path} for {label}: {str(e)}")
        return None

# 示例使用：
file_path = r'D:\study\second\outcome\give\AAESS_res.pickle'
label = 'AAESS'
# keys = ['auc', 'acc', 'prec', 'rec', 'f1', 'bs', 'fprs', 'tprs', 'e1', 'e2', 'prec_rec', 'ap', 'tpr', 'tnr', 'gmean']
keys = ['auc', 'acc', 'rec', 'prec', 'e1', 'e2']
result_data = load_pickle(file_path, label, keys)

if result_data:
    for key, value in result_data.items():
        print(f"{key}: {value}")
