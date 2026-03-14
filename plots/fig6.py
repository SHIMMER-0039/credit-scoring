import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# data
data = {
    'LDA': [15, 16, 16.75, 10.125],
    'LR': [16.5, 17.25, 15.5, 15.125],
    'DT': [15, 15.5, 14, 17],
    'KNN': [16, 17.5, 15.75, 12.375],
    'Adaboost': [14.75, 14.75, 14, 9.25],
    'RF': [10.125, 11.25, 10.5, 10.25],
    'GBDT': [5.625, 10, 8, 6],
    'XGboost': [9.75, 8, 5.5, 3.5],
    'Lightgbm': [13.75, 7.25, 4.25, 3.625],
    'AAESS': [1.875, 2.125, 1.875, 4],
    'HO-DT': [11, 10.75, 9.75, 12.375],
    'HO-LDA': [8.75, 14.25, 12, 13.75],
    'HO-LR': [12, 12.25, 12, 13.25],
    'HO-RF': [10.75, 8.75, 11.5, 15.25],
    'HO-XGB': [5.75, 3.75, 8.5, 12.25],
    'HO-LGB': [4, 7, 4.5, 7],
    'AAESS-LR': [5.5, 1.5, 10, 4.625],
    'HE-tree': [7.5, 5.75, 9.5, 12.25],
    'AAESS-TREE': [6.375, 6.375, 5.375, 8]
}

# Convert data to DataFrame
df = pd.DataFrame(data)

# Calculate the average rank of each algorithm
average_ranks = df.mean(axis=0)

# Sort by average ranking from lowest to highest
sorted_ranks = average_ranks.sort_values()

# Define CD values ​​at different significance levels
y_cd_005 = 10.22  # The CD value for alpha = 0.05 in the example.
y_cd_01 = 9.11    # The CD value for alpha = 0.1 in the example.
y_cd_001 = 12.78  # The CD value for alpha = 0.01 in the example.

# Create sorted charts
fig, ax = plt.subplots(figsize=(20, 10))

# Change the colors to make the chart more visually appealing
colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(sorted_ranks)))

# Draw a bar chart of the average ranking after sorting, and add a legend label to the bar information.
bars = ax.bar(sorted_ranks.index, sorted_ranks.values, color=colors, edgecolor='black', label='Rank bar')

# Add numerical labels to the top of each column.
for i, v in enumerate(sorted_ranks.values):
    ax.text(i, v + 0.2, f'{v:.2f}', color='black', ha='center', fontsize=12)

# Add different styles of critical difference (CD) lines
ax.hlines(y=y_cd_005, xmin=-0.5, xmax=len(sorted_ranks) - 0.5, color='red', linestyle='-', linewidth=2, label='α=0.05 ')
ax.hlines(y=y_cd_01, xmin=-0.5, xmax=len(sorted_ranks) - 0.5, color='blue', linestyle='-.', linewidth=2, label='α=0.1 ')
ax.hlines(y=y_cd_001, xmin=-0.5, xmax=len(sorted_ranks) - 0.5, color='purple', linestyle=':', linewidth=2, label='α=0.01 ')

# Add critical difference numerical labels on the left.
ax.text(-0.5, y_cd_005, f'CD={y_cd_005:.2f}', va='bottom', ha='left', fontsize=12, color='red')
ax.text(-0.5, y_cd_01, f'CD={y_cd_01:.2f}', va='bottom', ha='left', fontsize=12, color='blue')
ax.text(-0.5, y_cd_001, f'CD={y_cd_001:.2f}', va='bottom', ha='left', fontsize=12, color='green')

# Adjust the y-axis limit to fit the CD line.
ax.set_ylim(0, max(sorted_ranks) + 2)

# Add axis labels and titles
ax.set_ylabel('Average Rank', fontsize=16)
ax.set_xlabel('Algorithms', fontsize=16)
ax.set_xticks(range(len(sorted_ranks)))
ax.set_xticklabels(sorted_ranks.index, rotation=45, fontsize=16)

# Move the legend to the top left corner.
ax.legend(loc='upper left', fontsize=14)

# Show chart
plt.tight_layout()
plt.show()
