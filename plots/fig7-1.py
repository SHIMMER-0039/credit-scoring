import matplotlib.pyplot as plt
import numpy as np


train_prop = [0.4, 0.5, 0.6, 0.7, 0.8]
auc_values = [0.94335, 0.94422, 0.94458, 0.94483, 0.94502]

plt.figure(figsize=(8,5))

plt.plot(train_prop, auc_values, marker='o', linewidth=2, label='AAESS')

for i in range(len(train_prop)-1):
    plt.annotate(
        '',
        xy=(train_prop[i+1], auc_values[i+1]),
        xytext=(train_prop[i], auc_values[i]),
        arrowprops=dict(
            arrowstyle='->',
            color='#1f77b4',   
            lw=2
        )
    )

plt.title("AAESS: AUC vs. Training Set Proportion with Arrows")
plt.xlabel("Training Set Proportion")
plt.ylabel("AUC")

plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()
