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
st.set_page_config(page_title="Spotify Track Explorer", layout="wide")
st.title("🎵 Spotify Track Explorer")
st.markdown("Explore the distribution and relationships between audio features.")

# Heatmapa korelacji
st.subheader("🔗 Feature Correlation Matrix")
fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", ax=ax_corr)
st.pyplot(fig_corr)

# Wykresy rozkładu (3 na jednym płótnie)
st.subheader("📊 Feature Distributions (with potential outliers)")
num_cols = len(cols)
rows = (num_cols + 2) // 3

for i in range(rows):
    fig, axs = plt.subplots(1, 3, figsize=(18, 4))
    for j in range(3):
        idx = i * 3 + j
        if idx < num_cols:
            sns.histplot(df[cols[idx]], kde=True, ax=axs[j], color="skyblue")
            axs[j].set_title(cols[idx])
    st.pyplot(fig)

# Statystyki opisowe
st.subheader("📈 Descriptive Statistics")
st.dataframe(stats.T.round(2))
