import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

save_path = r'D:\study\Credit(1)\Credit\outcome\shandong\\'
os.makedirs(save_path, exist_ok=True)


data = {
    'Feature_Code': ['x11', 'x8', 'x9', 'x10', 'x14', 'x2', 'x13', 'x12', 'x1', 'x5'],
    'Importance': [18.8, 17.8, 15.4, 14.3, 8.7, 8.3, 7.2, 6.5, 1.5, 0.5]
}


label_mapping = {
    'x11': 'Account Status',          # GRZHZT (Personal Account Status)
    'x8': 'Employer Eco. Type',       # DWJJLX (Employer Economic Type)
    'x9': 'Industry Sector',          # DWSSHY (Industry Sector)
    'x10': 'Contribution Base',       # GRJCJS (Contribution Base)
    'x14': 'Pers. Monthly Deposit',   # GRYJCE (Personal Monthly Deposit)
    'x2': 'Age',                      # CSNY (Birth Date -> Age)
    'x13': 'Cur. Year Payment',       # GRZHDNGJY (Current Year Payment)
    'x12': 'Account Balance',         # GRZHYE (Personal Account Balance)
    'x1': 'Gender',                   # XINGBIE (Gender)
    'x5': 'Prof. Title'               # ZHICHEN (Professional Title)
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

save_file = os.path.join(save_path, 'shandong_top10_feature_importance_descriptive_manual.png')
plt.savefig(save_file, bbox_inches='tight', dpi=300)
plt.show()
