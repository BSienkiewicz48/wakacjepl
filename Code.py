import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.neighbors import NearestNeighbors

# --- CZĘŚĆ OBLICZENIOWA ---
@st.cache_data
def load_data():
    return pd.read_parquet("raw_data.parquet")

df = load_data()
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
    # Usuń utwory o popularności 0
    mask_popular = df_raw['popularity'] > 0
    df_raw = df_raw[mask_popular].reset_index(drop=True)
    df_norm = df_norm[mask_popular].reset_index(drop=True)

    # Jeśli jest kilka utworów o tym samym tytule, wybierz ten z największą popularnością
    candidates = df_raw[df_raw['track_name'] == selected_title]
    if len(candidates) > 1:
        query_index = candidates['popularity'].idxmax()
    else:
        query_index = candidates.index[0]

    query_vector = df_norm.loc[query_index, features].values.reshape(1, -1)

    model = NearestNeighbors(n_neighbors=len(df_norm), metric='euclidean')
    model.fit(df_norm[features])
    distances, indices = model.kneighbors(query_vector)

    # Pomijamy wyniki z distance == 0 (te same lub prawie identyczne utwory)
    mask = distances[0] > 0
    filtered_indices = indices[0][mask]
    filtered_distances = distances[0][mask]

    results = df_raw.loc[filtered_indices, ['track_name', 'artists', 'popularity']].copy()
    results['distance'] = filtered_distances

    # Usuwanie duplikatów: jeśli są te same tytuły i ten sam distance, zostaw z większą popularnością
    results = results.sort_values(['track_name', 'distance', 'popularity'], ascending=[True, True, False])
    results = results.drop_duplicates(subset=['track_name', 'distance'], keep='first')

    # Zwróć tylko k najlepszych
    results = results.head(k)
    return results[['track_name', 'artists', 'distance']]


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