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

# Dodaj do danych kolumnę łączoną: tytuł – artysta
df['title_artist'] = df['track_name'] + " – " + df['artists']

# Filtrowanie utworów o popularności > 0
df = df[df['popularity'] > 0].reset_index(drop=True)
df_norm = df_norm.loc[df.index].reset_index(drop=True)

# Funkcja wyszukiwania podobnych utworów
def find_similar_tracks(row_index, df_raw, df_norm, features, k=5):
    query_vector = df_norm.loc[row_index, features].values.reshape(1, -1)
    model = NearestNeighbors(n_neighbors=k+10, metric='euclidean')
    model.fit(df_norm[features])
    distances, indices = model.kneighbors(query_vector)

    # Pomijamy siebie + duplikaty tytułów
    mask = distances[0] > 0
    filtered_indices = indices[0][mask]
    filtered_distances = distances[0][mask]

    results = df_raw.loc[filtered_indices, ['track_name', 'artists', 'popularity']].copy()
    results['distance'] = filtered_distances

    results = results.sort_values(['track_name', 'distance', 'popularity'], ascending=[True, True, False])
    results = results.drop_duplicates(subset=['track_name', 'distance'], keep='first')
    return results.head(k)[['track_name', 'artists', 'distance']]


# --- CZĘŚĆ WIZUALNA STREAMLIT ---
st.title("🎵 Spotify Track Explorer")
st.markdown("Explore the distribution and relationships between audio features.")

# Heatmapa korelacji
st.subheader("🔗 Feature Correlation Matrix")
fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", ax=ax_corr)
st.pyplot(fig_corr)

# Wykresy rozkładu
@st.cache_data
def plot_feature_distributions(df, cols):
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
    return fig

st.subheader("📊 Feature Distributions")
fig = plot_feature_distributions(df, cols)
st.pyplot(fig)

# Statystyki opisowe
@st.cache_data
def get_stats(df, cols):
    return df[cols].describe()

st.subheader("📈 Descriptive Statistics")
stats = get_stats(df, cols)
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

# Wykresy rozkładu po normalizacji (z cache)
@st.cache_data
def plot_normalized_feature_distributions(df_norm, features):
    num_features_to_plot_norm = len(features)
    fig_norm, axs_norm = plt.subplots(3, 3, figsize=(18, 16))
    axs_norm = axs_norm.flatten()

    for i, col_name in enumerate(features):
        sns.histplot(df_norm[col_name], ax=axs_norm[i], color="lightgreen")
        axs_norm[i].set_title(col_name)
        axs_norm[i].set_xlabel("")

    for i in range(num_features_to_plot_norm, 9):
        fig_norm.delaxes(axs_norm[i])

    plt.tight_layout()
    return fig_norm

st.subheader("📊 Normalized Feature Distributions (used for similarity search)")
fig_norm = plot_normalized_feature_distributions(df_norm, features_for_similarity)
st.pyplot(fig_norm)

st.subheader("🎯 Find Similar Tracks")

selected_combo = st.selectbox("Choose a track:", df['title_artist'].unique())
selected_index = df[df['title_artist'] == selected_combo].index[0]

if st.button("🔍 Find Similar"):
    results_df = find_similar_tracks(selected_index, df, df_norm, features_for_similarity, k=5)
    st.write(f"Top 5 tracks similar to **{selected_combo}**:")
    st.dataframe(results_df[['track_name', 'artists', 'distance']].reset_index(drop=True), hide_index=True)


