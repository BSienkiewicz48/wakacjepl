import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# --- CZĘŚĆ OBLICZENIOWA ---
df = pd.read_parquet("raw_data.parquet")
df['duration_s'] = df['duration_ms'] / 1000
df = df.drop(columns=['duration_ms'])

# Feature engineering
df['mood_score'] = df['valence'] * df['energy']
df['vocals_strength'] = 1 - df['instrumentalness'] - df['speechiness']
df['vocals_strength'] = df['vocals_strength'].clip(0, 1)  # ograniczenie do [0,1]

# Przycięcie instrumentalness do 95 percentyla
perc95_instr = df['instrumentalness'].quantile(0.95)
df['instrumentalness'] = np.where(
    df['instrumentalness'] > perc95_instr, 1, df['instrumentalness']
)

# Podobnie przycinamy speechiness
perc95_speech = df['speechiness'].quantile(0.95)
df['speechiness'] = np.where(
    df['speechiness'] > perc95_speech, perc95_speech, df['speechiness']
)

# Lista cech docelowych
features_for_similarity = [
    'danceability', 'energy', 'valence', 'loudness',
    'acousticness', 'tempo', 'mood_score', 'vocals_strength'
]

# Normalizacja
df_norm = df.copy()

# reszta – min-max
minmax_cols = features_for_similarity
scaler = MinMaxScaler()
df_norm[minmax_cols] = scaler.fit_transform(df[minmax_cols])

# --- Dla eksploracji ---
cols = [
    'popularity', 'duration_s', 'danceability', 'energy', 'loudness',
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
num_features_to_plot = len(cols)

fig, axs = plt.subplots(4, 3, figsize=(18, 16))
axs = axs.flatten()

for i, col_name in enumerate(cols):
    sns.histplot(df[col_name], ax=axs[i], color="skyblue")
    axs[i].set_title(col_name)
    axs[i].set_xlabel("") 

for i in range(num_features_to_plot, 12):
    fig.delaxes(axs[i])

plt.tight_layout()
st.pyplot(fig)

# Statystyki opisowe
st.subheader("📈 Descriptive Statistics")
st.dataframe(stats.T.round(2))

# Podsumowanie decyzji
st.markdown("### 💡 Feature Engineering & Normalization Decisions")
st.markdown("""
- **Odrzucono** `duration_ms` – nie wpływa znacząco na odbiór muzyki.
- **Dodano** cechy:
    - `mood_score = valence * energy`
    - `vocals_strength = 1 - instrumentalness - speechiness`
- **Zastosowano transformacje**:
    - `instrumentalness` i `speechiness`: przycięcie do 95 percentyla
    - `loudness`: standaryzacja (Z-score)
    - pozostałe cechy: min-max scaling
- **Wybrane cechy do podobieństw**:
    - `danceability`, `energy`, `valence`, `loudness`, `acousticness`, `tempo`, `mood_score`, `vocals_strength`
""")

# Wykresy rozkładu po normalizacji
st.subheader("📊 Normalized Feature Distributions (used for similarity search)")

num_features_to_plot_norm = len(features_for_similarity)
fig_norm, axs_norm = plt.subplots(3, 3, figsize=(18, 16))
axs_norm = axs_norm.flatten()

for i, col_name in enumerate(features_for_similarity):
    sns.histplot(df_norm[col_name], ax=axs_norm[i], color="lightgreen")
    axs_norm[i].set_title(col_name)
    axs_norm[i].set_xlabel("")  # Usunięcie podpisów pod osią x

for i in range(num_features_to_plot_norm, 9):
    fig_norm.delaxes(axs_norm[i])

plt.tight_layout()
st.pyplot(fig_norm)
