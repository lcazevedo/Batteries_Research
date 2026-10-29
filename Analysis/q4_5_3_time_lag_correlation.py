import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import config
import utils

# 1. Loading (Maintaining standard setup)
df_data = utils.get_df_data()
df_phrases = utils.get_df_data_phrases()
df_merged = df_phrases.merge(df_data[[config.COLUMN_DOI, config.COLUMN_YEAR]], 
                             left_on='doi', right_on=config.COLUMN_DOI, how='inner')

# 2. CREATING FREQUENCY MATRICES
# Creates a table where columns are the 15 groups and rows are the years
challenges_pivot = df_merged.pivot_table(index=config.COLUMN_YEAR, 
                                        columns='challenges_group', 
                                        aggfunc='size', fill_value=0)

directions_pivot = df_merged.pivot_table(index=config.COLUMN_YEAR, 
                                        columns='directions_group', 
                                        aggfunc='size', fill_value=0)

# 3. CHOOSING WHAT TO COMPARE
# For the Lag, take the most frequent category of each to test correlation
# Or replace with a specific category name of interest
top_challenge_name = challenges_pivot.sum().idxmax()
top_direction_name = directions_pivot.sum().idxmax()

series_x = challenges_pivot[top_challenge_name]
series_y = directions_pivot[top_direction_name]

# 4. Lag Calculation (same logic as before)
lags = range(-5, 6)
correlations = [series_x.corr(series_y.shift(l)) for l in lags]
best_lag = lags[np.argmax(np.abs(correlations))]

# 5. Professional Visualization
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

# Top: Slice comparison (Example with Top Categories)
ax1.plot(series_x.index, series_x, label=f'Challenge: {top_challenge_name}', marker='o', color='teal')
ax1.plot(series_y.index, series_y, label=f'Direction: {top_direction_name}', marker='s', color='orange')
ax1.set_title(f'Temporal Evolution: {top_challenge_name} vs {top_direction_name}')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Bottom: Correlation
ax2.stem(lags, correlations)
ax2.axvline(best_lag, color='red', linestyle='--', label=f'Optimal Lag: {best_lag} years')
ax2.set_title(f'Cross-Correlation for Top Categories')
ax2.set_xlabel('Lag (Years)')
ax2.set_ylabel('Correlation')
ax2.legend()

plt.tight_layout()
plt.savefig(f'{config.RESULTS_FOLDER}/q4_5_3_time_lag_correlation.pdf', dpi=300)

print(f"Top Challenge: {top_challenge_name}")
print(f"Top Direction: {top_direction_name}")
print(f"The strongest link between these specific topics has a lag of {best_lag} years.")
