import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

save_path = r'D:\study\Credit(1)\Credit\outcome\fannie\\'
os.makedirs(save_path, exist_ok=True)


data = {
    'Feature_Code': ['x9', 'x1', 'x10', 'x16', 'x4', 'x7', 'x2', 'x3', 'x6', 'x5'],
    'Importance': [16.5, 15.5, 15.0, 12.5, 8.2, 7.2, 7.2, 6.5, 5.8, 5.5]
}

label_mapping = {
    'x9': 'Number of Units',          # Number of property units
    'x1': 'Orig. Interest Rate',      # Original Interest Rate
    'x10': 'Min. Credit Score',       # Minimum Credit Score
    'x16': 'Occupancy Status',        # Occupancy Type
    'x4': 'Original LTV',             # Loan-to-Value (LTV)
    'x7': 'Debt-to-Income (DTI)',     # Debt-To-Income Ratio
    'x2': 'Original UPB',             # Unpaid Principal Balance
    'x3': 'Original Loan Term',       # Loan Term
    'x6': 'Num. of Borrowers',        # Number of Borrowers
    'x5': 'Original CLTV'             # Combined Loan-to-Value
}

df = pd.DataFrame(data)
df['Feature_Label'] = df['Feature_Code'].map(label_mapping)

plt.figure(figsize=(10, 6))

colors = sns.color_palette('pastel', len(df))


bars = plt.barh(df['Feature_Label'], df['Importance'], color=colors)

plt.xlabel('Feature Importance Percentage', fontsize=14)
plt.ylabel('Features', fontsize=14)


plt.gca().invert_yaxis()


plt.tick_params(axis='y', labelsize=12)
plt.tick_params(axis='x', labelsize=12)

plt.tight_layout()

save_file = os.path.join(save_path, 'fannie_top10_feature_importance_descriptive_manual.png')
plt.savefig(save_file, bbox_inches='tight', dpi=300)

plt.show()
