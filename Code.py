import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.neighbors import NearestNeighbors

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
# min-max
minmax_cols = features_for_similarity
scaler = MinMaxScaler()
df_norm[minmax_cols] = scaler.fit_transform(df[minmax_cols])

cols = [
    'popularity', 'duration_s', 'danceability', 'energy', 'loudness',
    'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo'
]
stats = df[cols].describe()
corr_matrix = df[cols].corr()

def find_similar_tracks(selected_title, df_raw, df_norm, features, k=5):
    # Wybieramy rząd odpowiadający wybranemu utworowi
    query_index = df_raw[df_raw['track_name'] == selected_title].index[0]
    query_vector = df_norm.loc[query_index, features].values.reshape(1, -1)
    
    model = NearestNeighbors(n_neighbors=k+1, metric='euclidean')  # +1, bo pierwszy wynik to on sam
    model.fit(df_norm[features])
    distances, indices = model.kneighbors(query_vector)

    # Pomijamy pierwszy wynik (ten sam utwór), zwracamy kolejne
    similar_indices = indices[0][1:]
    similar_distances = distances[0][1:]
    
    results = df_raw.loc[similar_indices, ['track_name', 'artists']].copy()
    results['distance'] = similar_distances
    return results


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

st.subheader("🎯 Find Similar Tracks")
selected_track = st.selectbox(
    "Choose a track:",
    options=df['track_name'].unique(),
    index=0,
    placeholder="Start typing..."
)

if st.button("🔍 Find Similar"):
    results_df = find_similar_tracks(selected_track, df, df_norm, features_for_similarity, k=5)
    st.write(f"Top 5 tracks similar to **{selected_track}**:")
    st.dataframe(results_df.reset_index(drop=True))