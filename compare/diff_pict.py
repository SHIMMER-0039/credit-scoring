import pickle
import matplotlib.pyplot as plt

# 定义不同的结果文件路径
results_paths = {
    'xgb': 'D:/study/second/outcome/give_xgb_train_proportion_auc_results_with_features.pickle',
    'LightGBM': 'D:/study/second/outcome/give_lgb_train_proportion_auc_results_with_features.pickle',
    'RandomForest': 'D:/study/second/outcome/give_rf_train_proportion_auc_results_with_features.pickle',
    'AAESS': 'D:/study/second/outcome/give_AAESS_train_proportion_auc_results_with_features.pickle'
    # 'xgb': 'D:/study/second/outcome/fannie_xgb_train_proportion_auc_results_with_features.pickle',
    # 'LightGBM': 'D:/study/second/outcome/fannie_lgb_train_proportion_auc_results_with_features.pickle',
    # 'RandomForest': 'D:/study/second/outcome/fannie_rf_train_proportion_auc_results_with_features.pickle',
    # 'AAESS': 'D:/study/second/outcome/fannie_AAESS_train_proportion_auc_results_with_features.pickle'
    # 'xgb': 'D:/study/second/outcome/shandong_xgb_train_proportion_auc_results_with_features.pickle',
    # 'LightGBM': 'D:/study/second/outcome/shandong_lgb_train_proportion_auc_results_with_features.pickle',
    # 'RandomForest': 'D:/study/second/outcome/shandong_rf_train_proportion_auc_results_with_features.pickle',
    # 'AAESS': 'D:/study/second/outcome/shandong_AAESS_train_proportion_auc_results_with_features.pickle'
}

# 设置图形大小
plt.figure(figsize=(10, 7))

# 为每个结果文件绘制一条线
for model_name, file_path in results_paths.items():
    # 读取保存的 AUC 结果
    with open(file_path, 'rb') as f:
        auc_results = pickle.load(f)

    # 提取训练集比例和对应的 AUC 值
    train_proportions = [result['train_proportion'] for result in auc_results]
    auc_values = [result['auc'] for result in auc_results]

    # 绘制 AUC 随训练集比例变化的图
    plt.plot(train_proportions, auc_values, marker='o', linestyle='-', label=model_name)

# 添加标题和标签
plt.title('AUC vs. Training Set Proportion (All Models)')
plt.xlabel('Training Set Proportion')
plt.ylabel('AUC')
plt.grid(True)

# 反转X轴，从0.8到0.4
plt.gca().invert_xaxis()

# 添加图例
plt.legend()

# 显示图表
plt.show()
