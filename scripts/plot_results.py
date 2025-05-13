import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import f_oneway, ttest_ind
from itertools import combinations

csv_path = 'scripts/accuracies_linear.csv'  # Path to your CSV
selected_bands = [
    '100000',
    '010000',
    '001000',
    '000100',
    '000010',
    '000001'
] # Replace with your desired band IDs
# selected_bands = [
#     '110000',
#     '011000',
#     '001100',
#     '000110',
#     '000011'][::-1] # Replace with your desired band IDs

selected_bands = [
    '110000',
    '001100',
    '000011'][::-1]


df = pd.read_csv(csv_path, dtype={"band": str})
print(df.head())
df_filtered = df[df['band'].isin(selected_bands)].reset_index(drop=True)

if df_filtered.empty:
    raise ValueError(f"No matching rows found for bands: {selected_bands}")

data = [df_filtered.iloc[i, 1:].astype(float).values for i in range(len(df_filtered))]
labels = df_filtered['band'].tolist()

# anova
anova_stat, anova_p = f_oneway(*data)
print(f"\nANOVA: F={anova_stat:.4f}, p={anova_p:.4e}")

# pairwise t tests
print("\nPairwise t-tests with Bonferroni correction:")
n_tests = len(data) * (len(data) - 1) / 2
for (i, j) in combinations(range(len(data)), 2):
    stat, p = ttest_ind(data[i], data[j])
    corrected_p = p * n_tests
    print(f"{labels[i]} vs {labels[j]}: t={stat:.4f}, p={p:.4e}, corrected_p={corrected_p:.4e}")

# plottign
means = [np.mean(d) for d in data]
stds = [np.std(d) for d in data]

plt.figure(figsize=(10, 6))
plt.bar(labels, means, yerr=stds, capsize=5)
plt.xlabel('Band')
plt.ylabel('Accuracy')
plt.title('Mean ± Std Accuracy per Band')
plt.tight_layout()
plt.show()
