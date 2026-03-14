import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

save_path = r'D:\study\Credit(1)\Credit\outcome\give\\'
os.makedirs(save_path, exist_ok=True)

data = {
    'Feature_Code': ['x1', 'x7', 'x3', 'x9', 'x2', 'x4', 'x5', 'x6', 'x8', 'x10'],
    'Importance': [27.0, 24.5, 12.5, 10.0, 6.2, 6.1, 5.8, 4.2, 2.3, 1.4]
}

label_mapping = {
    'x1': 'Revol. Util. of Unsecured Lines',    # RevolvingUtilizationOfUnsecuredLines
    'x7': '90+ Days Past Due',                  # NumberOfTimes90DaysLate
    'x3': '30-59 Days Past Due',                # NumberOfTime30-59DaysPastDueNotWorse
    'x9': '60-89 Days Past Due',                # NumberOfTime60-89DaysPastDueNotWorse
    'x2': 'Age',                                # age
    'x4': 'Debt Ratio',                         # DebtRatio
    'x5': 'Monthly Income',                     # MonthlyIncome
    'x6': 'Number of Open Credit Lines',        # NumberOfOpenCreditLinesAndLoans
    'x8': 'Real Estate Loans',                  # NumberRealEstateLoansOrLines
    'x10': 'Number of Dependents'               # NumberOfDependents
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

save_file = os.path.join(save_path, 'give_top10_feature_importance_descriptive_manual.png')
plt.savefig(save_file, bbox_inches='tight', dpi=300)
plt.show()
