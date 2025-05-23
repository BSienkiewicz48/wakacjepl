import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors

# --- ŁADOWANIE I PRZETWARZANIE DANYCH ---
@st.cache_data
def load_and_prepare_data():
    df = pd.read_parquet("raw_data.parquet")
    df = df[df['popularity'] > 0].copy()

    df['duration_s'] = df['duration_ms'] / 1000
    df = df.drop(columns=['duration_ms'])

    df['mood_score'] = df['valence'] * df['energy']
    df['vocals_strength'] = 1 - df['instrumentalness'] - df['speechiness']
    df['vocals_strength'] = df['vocals_strength'].clip(0, 1)

    df['instrumentalness'] = np.where(
        df['instrumentalness'] > df['instrumentalness'].quantile(0.95), 1, df['instrumentalness']
    )
    df['speechiness'] = np.where(
        df['speechiness'] > df['speechiness'].quantile(0.95),
        df['speechiness'].quantile(0.95),
        df['speechiness']
    )

    features = [
        'danceability', 'energy', 'valence', 'loudness',
        'acousticness', 'tempo', 'mood_score', 'vocals_strength'
    ]

    df_norm = df.copy()
    df_norm[features] = MinMaxScaler().fit_transform(df[features])

    df['title_artist'] = df['track_name'] + " – " + df['artists']

    cols = [
        'popularity', 'duration_s', 'danceability', 'energy', 'loudness',
        'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo'
    ]
    stats = df[cols].describe()
    corr = df[cols].corr()

    return df.reset_index(drop=True), df_norm.reset_index(drop=True), features, stats, corr, cols


# --- FUNKCJA REKOMENDACJI ---
def find_similar_tracks(row_index, df_raw, df_norm, features, k=5):
    query_vector = df_norm.loc[row_index, features].values.reshape(1, -1)
    model = NearestNeighbors(n_neighbors=k+10, metric='euclidean')
    model.fit(df_norm[features])
    distances, indices = model.kneighbors(query_vector)

    mask = distances[0] > 0
    filtered_indices = indices[0][mask]
    filtered_distances = distances[0][mask]

    results = df_raw.loc[filtered_indices, ['track_name', 'artists', 'popularity']].copy()
    results['distance'] = filtered_distances

    results = results.sort_values(['track_name', 'distance', 'popularity'], ascending=[True, True, False])
    results = results.drop_duplicates(subset=['track_name', 'distance'], keep='first')
    return results.head(k)[['track_name', 'artists', 'distance']]


# --- STRONA WIZUALNA - STREAMLIT ---
df, df_norm, features_for_similarity, stats, corr_matrix, cols = load_and_prepare_data()

st.title("🎵 Spotify Track Explorer")
st.markdown("Explore the distribution and relationships between audio features.")

# Heatmapa korelacji
st.subheader("🔗 Feature Correlation Matrix")
fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", ax=ax_corr)
st.pyplot(fig_corr)

# Wykresy rozkładu (oryginalne cechy)
st.subheader("📊 Feature Distributions")
fig, axs = plt.subplots(4, 3, figsize=(18, 16))
axs = axs.flatten()
for i, col_name in enumerate(cols):
    sns.histplot(df[col_name], ax=axs[i], color="skyblue")
    axs[i].set_title(col_name)
    axs[i].set_xlabel("")
for i in range(len(cols), 12):
    fig.delaxes(axs[i])
plt.tight_layout()
st.pyplot(fig)

# Statystyki opisowe
st.subheader("📈 Descriptive Statistics")
st.dataframe(stats.T.round(2))

# Wykresy po normalizacji
st.subheader("📊 Normalized Feature Distributions (used for similarity search)")
fig_norm, axs_norm = plt.subplots(3, 3, figsize=(18, 16))
axs_norm = axs_norm.flatten()
for i, col_name in enumerate(features_for_similarity):
    sns.histplot(df_norm[col_name], ax=axs_norm[i], color="lightgreen")
    axs_norm[i].set_title(col_name)
    axs_norm[i].set_xlabel("")
for i in range(len(features_for_similarity), 9):
    fig_norm.delaxes(axs_norm[i])
plt.tight_layout()
st.pyplot(fig_norm)

st.markdown("### 🧮 Notes on Normalized Features and Outlier Handling")
st.markdown("""
- All features used for similarity search are **scaled to the [0, 1] range** using **Min-Max normalization**.
- This ensures that each feature contributes equally to the distance calculation.
- **Outliers in `instrumentalness` and `speechiness`** are handled by:
  - Clipping values above the 95th percentile to a fixed cap.
  - This prevents rare extreme values from skewing similarity scores.
- The final feature set includes engineered metrics like:
  - `mood_score = valence * energy`
  - `vocals_strength = 1 - instrumentalness - speechiness`
- This preprocessing step improves the **accuracy and interpretability** of recommendations.
""")

# Interfejs wyszukiwania podobnych utworów
st.subheader("🎯 Find Similar Tracks")
selected_combo = st.selectbox("Choose a track:", df['title_artist'].unique())
selected_index = df[df['title_artist'] == selected_combo].index[0]

if st.button("🔍 Find Similar"):
    results_df = find_similar_tracks(selected_index, df, df_norm, features_for_similarity, k=5)
    st.write(f"Top 5 tracks similar to **{selected_combo}**:")
    st.dataframe(results_df.reset_index(drop=True), hide_index=True)

# Opis działania
st.markdown("### 🔎 How Similar Tracks Are Selected")
st.markdown("""
- We analyze **8 normalized audio features** that capture musical style and mood:
  - `danceability`, `energy`, `valence`, `loudness`, `acousticness`, `tempo`, `mood_score`, `vocals_strength`
- A selected track is represented as a **feature vector** in this multi-dimensional space.
- We use **K-Nearest Neighbors (KNN)** with **Euclidean distance** to find the closest tracks.
- For each comparison:
  - The **Euclidean distance** is calculated as:  
    $\\text{distance} = \\sqrt{(a_1 - b_1)^2 + (a_2 - b_2)^2 + \\dots + (a_8 - b_8)^2}$
  - **Lower distance = higher similarity**.
- Tracks with **distance = 0** (identical or nearly identical) are excluded.
- If multiple results share the same `track_name` and distance, the one with **higher popularity** is kept.
- Final result: **Top 5 most similar tracks**, sorted by distance (smallest first).
""")
