import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

# 设置保存路径
save_path = r'D:\study\Credit(1)\Credit\outcome\bankfear\\'
os.makedirs(save_path, exist_ok=True)

# 1. Feature importance scores extracted from the trained AAESS model
# Rankings are sorted by information gain in descending order
data = {
    'Feature_Code': ['x18', 'x19', 'x5', 'x6', 'x23', 'x8', 'x21', 'x15', 'x13', 'x20'],
    'Importance': [23.5, 19.5, 13.0, 11.0, 10.2, 6.2, 5.0, 4.5, 3.2, 3.1]
}

# 2. Map anonymized feature codes to descriptive variable names
# Mapping based on the original data dictionary and column indexing
label_mapping = {
    'x18': 'Total Current Balance',      # tot_cur_bal
    'x19': 'Total Rev. Hi Limit',        # total_rev_hi_lim
    'x5': 'Funded Amount (Inv)',         # funded_amnt_inv
    'x6': 'Sub Grade',                   # sub_grade
    'x23': 'Loan Term',                  # term (推断)
    'x8': 'Annual Income',               # annual_inc
    'x21': 'Recoveries',                 # recoveries
    'x15': 'Total Accounts',             # total_acc
    'x13': 'Revolving Balance',          # revol_bal
    'x20': 'Total Collection Amt'        # tot_coll_amt
}

# 3. Prepare plotting data
df = pd.DataFrame(data)
df['Feature_Label'] = df['Feature_Code'].map(label_mapping)

# 4. Drawing
plt.figure(figsize=(10, 6))
# Using the Pastel color palette
colors = sns.color_palette('pastel', len(df))

# Draw a horizontal bar chart
bars = plt.barh(df['Feature_Label'], df['Importance'], color=colors)

# Set labels and titles
plt.xlabel('Feature Importance Percentage', fontsize=14)
plt.ylabel('Features', fontsize=14)

# Reverse the Y-axis to display the most important information at the top.
plt.gca().invert_yaxis()

plt.tick_params(axis='y', labelsize=12)
plt.tick_params(axis='x', labelsize=12)

plt.tight_layout()

# 5. Save the image
save_file = os.path.join(save_path, 'bankfear_top10_feature_importance_descriptive_manual.png')
plt.savefig(save_file, bbox_inches='tight', dpi=300)
plt.show()
