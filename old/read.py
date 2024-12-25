import os
import pickle

folder_path = r'D:\study\second\outcome'
results = []

for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)

        # 获取各个指标的值，确保即使有缺失的数据也能正常处理
        keys_of_interest = ['auc', 'e1', 'e2', 'hm', 'bs', 'acc']  # 更新为你实际需要的键列表
        metrics = {key: data.get(key, None) for key in keys_of_interest}

        # 添加到结果列表
        results.append({'filename': filename, **metrics})

    except EOFError as e:
        print(f"EOFError - File {filename} is empty or corrupted: {e}")
    except (OSError, IOError) as e:
        print(f"Error opening file {filename}: {e}")
    except KeyError as e:
        print(f"Missing key in file {filename}: {e}")

# 打印结果，使用.get()避免KeyError
for result in results:
    print("File:", result['filename'])
    for key in keys_of_interest:
        print(f"{key.upper()}:", result.get(key, 'N/A'))  # 使用get方法安全访问字典键值
    print()
