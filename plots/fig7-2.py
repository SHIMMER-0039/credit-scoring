import matplotlib.pyplot as plt

# Training set proportions
train_prop = [0.4, 0.5, 0.6, 0.7, 0.8]

auc_values = [0.87722, 0.87834, 0.87848, 0.87962, 0.87978]

plt.figure(figsize=(8,5))

line, = plt.plot(
    train_prop,
    auc_values,
    marker='o',
    color='green',
    linewidth=2,
    markersize=7,
    label='AAESS'
)

for i in range(len(train_prop)-1):
    plt.annotate(
        '',
        xy=(train_prop[i+1], auc_values[i+1]),
        xytext=(train_prop[i], auc_values[i]),
        arrowprops=dict(
            arrowstyle='->',
            color=line.get_color(),
            lw=2
        )
    )

plt.title("AAESS: AUC vs. Training Set Proportion")
plt.xlabel("Training Set Proportion")
plt.ylabel("AUC")

plt.grid(True, linestyle='--', alpha=0.6)

plt.legend()

plt.tight_layout()

plt.savefig("AAESS_training_proportion_auc.pdf", dpi=600, bbox_inches='tight')

plt.show()
