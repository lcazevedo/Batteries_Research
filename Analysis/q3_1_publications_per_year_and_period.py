import utils
import config
import re
import pandas as pd
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt

tqdm.pandas() 

df_data = utils.get_df_data()

#df_data = df_data[df_data[config.COLUMN_YEAR] <2005]

# Font sizes
fs1 = 30  # Title
fs2 = 20  # Axis labels and ticks
fs3 = 18  # Data labels

# Chart style
sns.set_theme(style="whitegrid")

# Data
years = df_data[config.COLUMN_YEAR].dropna()

# Boxplot statistics
q1 = years.quantile(0.25)
q2 = years.median()
q3 = years.quantile(0.75)
iqr = q3 - q1
lower_whisker = max(years.min(), q1 - 1.5 * iqr)
upper_whisker = min(years.max(), q3 + 1.5 * iqr)

year_counts = years.value_counts().sort_index().to_dict()


# Figure
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True, gridspec_kw={'height_ratios': [3, 1]})

# Histogram
sns.histplot(years, bins=range(int(years.min()), int(years.max()) + 1), ax=ax1, color="blue")
ax1.set_ylabel("Frequency", fontsize=fs2)
ax1.set_title("Distribution of Documents by Year", fontsize=fs1)
ax1.tick_params(axis='both', labelsize=fs2)  # X and Y axis tick size for histogram

# Boxplot
sns.boxplot(x=years, ax=ax2, color="lightgray")
ax2.set_xlabel("Year", fontsize=fs2)
ax2.set_ylabel("")
ax2.tick_params(axis='both', labelsize=fs2)  # X and Y axis tick size for boxplot

# Labels on boxplot (values only, rotated)
for value in [lower_whisker, q1, q2, q3, upper_whisker]:
    ax2.text(value, 0.02, f"{year_counts[int(value)]:.0f} ({value:.0f})", ha='left', va='bottom', fontsize=fs3, rotation=75)

# Layout adjustment
#plt.tight_layout()

plt.savefig(f"{config.RESULTS_FOLDER}/q3_1_distribution_all_period.pdf", bbox_inches='tight') #, dpi=300)


plt.show()
