import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

# --- CZĘŚĆ OBLICZENIOWA ---
df = pd.read_parquet("raw_data.parquet")

# Kolumny numeryczne do eksploracji
cols = [
    'popularity', 'duration_ms', 'danceability', 'energy', 'loudness',
    'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo'
]
stats = df[cols].describe()
corr_matrix = df[cols].corr()


# --- CZĘŚĆ WIZUALNA STREAMLIT ---
st.title("🎵 Spotify Track Explorer")
st.markdown("Explore the distribution and relationships between audio features.")

# Heatmapa korelacji
st.subheader("🔗 Feature Correlation Matrix")
fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", ax=ax_corr)
st.pyplot(fig_corr)

# Wykresy rozkładu
st.subheader("📊 Feature Distributions")
num_features_to_plot = len(cols) # Should be 11 based on the 'cols' list

# Hardcode the subplot grid dimensions for 11 plots:
# 3 rows of 3 plots, and 1 row of 2 plots (total 4 rows in the grid)
subplot_grid_rows = 4
subplot_grid_cols = 3

fig, axs = plt.subplots(subplot_grid_rows, subplot_grid_cols, figsize=(18, subplot_grid_rows * 4))
axs = axs.flatten() # Flatten the array of axes for easy iteration

for i, col_name in enumerate(cols):
    sns.histplot(df[col_name], kde=True, ax=axs[i], color="skyblue")
    axs[i].set_title(col_name) # Plot title is at the top of each subplot

# Hide any unused subplots in the grid
# Total subplots in grid = subplot_grid_rows * subplot_grid_cols
# We have num_features_to_plot actual plots
for i in range(num_features_to_plot, subplot_grid_rows * subplot_grid_cols):
    fig.delaxes(axs[i])

plt.tight_layout()
st.pyplot(fig)


# Statystyki opisowe
st.subheader("📈 Descriptive Statistics")
st.dataframe(stats.T.round(2))
