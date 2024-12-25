import matplotlib.pyplot as plt
import numpy as np

# 模拟一些数据作为 accuracy、roc_auc、precision 和 recall
np.random.seed(42)
accuracy = np.random.uniform(0.7, 0.95, 10)
roc_auc = np.random.uniform(0.6, 0.9, 10)
precision = np.random.uniform(0.5, 0.85, 10)
recall = np.random.uniform(0.55, 0.9, 10)

# 对数据按照 accuracy 升序排序
sorted_indices = np.argsort(accuracy)
accuracy = accuracy[sorted_indices]
roc_auc = roc_auc[sorted_indices]
precision = precision[sorted_indices]
recall = recall[sorted_indices]

# 创建子图
fig, axs = plt.subplots(3, 1, figsize=(8, 12))

# 绘制 accuracy vs roc_auc 并连线
axs[0].scatter(accuracy, roc_auc, color='blue', label='ROC AUC')
axs[0].plot(accuracy, roc_auc, color='blue')  # 连接点
axs[0].set_xlabel('Accuracy')
axs[0].set_ylabel('ROC AUC')
axs[0].set_title('Accuracy vs ROC AUC')

# 绘制 accuracy vs precision 并连线
axs[1].scatter(accuracy, precision, color='green', label='Precision')
axs[1].plot(accuracy, precision, color='green')  # 连接点
axs[1].set_xlabel('Accuracy')
axs[1].set_ylabel('Precision')
axs[1].set_title('Accuracy vs Precision')

# 绘制 accuracy vs recall 并连线
axs[2].scatter(accuracy, recall, color='red', label='Recall')
axs[2].plot(accuracy, recall, color='red')  # 连接点
axs[2].set_xlabel('Accuracy')
axs[2].set_ylabel('Recall')
axs[2].set_title('Accuracy vs Recall')

# 调整布局
plt.tight_layout()

# 保存图像
# plt.savefig('/mnt/data/pareto_accuracy_vs_others_with_lines.png')

# 显示图像
plt.show()
