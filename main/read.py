import os
import pickle

# 设置读取路径
save_path = r'D:\study\second\outcome/'
dataset = 'bankfear'
method = 'AdaptiveVotingStacking_Best'
file_path = os.path.join(save_path, f'{dataset}\\{method}_res.pickle')

# 检查文件是否存在
if not os.path.exists(file_path):
    raise FileNotFoundError(f"No results file found at {file_path}")

# 读取文件内容
with open(file_path, 'rb') as f:
    results_dict = pickle.load(f)

# 假设 results_dict 里包含所有结果，以字典形式存储
# 打印所有结果
print("All Results:")
all_results = results_dict.get('all_results', [])
if all_results:
    for idx, result in enumerate(all_results):
        print(f"\nResult {idx + 1}:")
        for key, value in result.items():
            print(f"{key}: {value}")
else:
    print("No results found.")
