import pickle

# 加载pickle文件
file_path = r'D:\study\second\outcome\shandong\AAESS_res.pickle'

with open(file_path, 'rb') as f:
    data = pickle.load(f)

# 查看可用的键
print(f"Available keys in the data: {list(data.keys())}")

# 从 'results' 中提取指标
if 'best_results' in data:
    results = data['results']  # 'results' 可能是一个包含多个结果的列表

    for i, result in enumerate(results):
        print(f"Result {i+1}:")
        print(f"AUC: {result.get('auc')}")
        print(f"Accuracy: {result.get('acc')}")
        print(f"Precision: {result.get('prec')}")
        print(f"Recall: {result.get('rec')}")
        print(f"F1 Score: {result.get('f1')}")
        print(f"Brier Score: {result.get('bs')}")
        print(f"KS Statistic: {result.get('ks')}")
        # 继续打印其他你需要的指标
        print()

# 从 'best_result' 中提取指标
if 'best_result' in data:
    best_result = data['best_result']
    print("Best Result:")
    print(f"AUC: {best_result.get('auc')}")
    print(f"Accuracy: {best_result.get('acc')}")
    print(f"Precision: {best_result.get('prec')}")
    print(f"Recall: {best_result.get('rec')}")
    print(f"F1 Score: {best_result.get('f1')}")
    print(f"Brier Score: {best_result.get('bs')}")
    print(f"KS Statistic: {best_result.get('ks')}")
    # 继续打印其他你需要的指标
