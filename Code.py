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
num_cols = len(cols)
n_rows = (num_cols + 2) // 3  # Oblicz liczbę wierszy, aby zmieścić wszystkie wykresy (3 na wiersz)
fig, axs = plt.subplots(n_rows, 3, figsize=(18, n_rows * 4)) # Dostosuj figsize do liczby wierszy
axs = axs.flatten() # Spłaszcz tablicę osi dla łatwiejszej iteracji

for i, col_name in enumerate(cols):
    sns.histplot(df[col_name], kde=True, ax=axs[i], color="skyblue")
    axs[i].set_title(col_name)

# Ukryj nieużywane osie, jeśli liczba wykresów nie jest wielokrotnością 3
for i in range(num_cols, n_rows * 3):
    fig.delaxes(axs[i])

plt.tight_layout()
st.pyplot(fig)


# Statystyki opisowe
st.subheader("📈 Descriptive Statistics")
st.dataframe(stats.T.round(2))
