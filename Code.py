import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
import openai
import random

api_key = st.secrets["API_KEY"]

# --- USTAWIENIA METRYKI HYBRYDOWEJ ---
ALPHA = 0.60  # waga części cosine (moja rekomendacja dla tego zadania)
RANDOM_SEED = 42  # dla powtarzalności próbkowania w elbow

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

# --- POMOCNICZE: LICZENIE HYBRYDOWEJ PODOBIEŃSTWA ---
def _combined_similarity(query_vec: np.ndarray, matrix: np.ndarray, alpha: float) -> np.ndarray:
    """
    query_vec: (n_features,)
    matrix: (n_items, n_features)
    Zwraca wektor similarity w [0,1], im większa wartość tym większe podobieństwo.
    """
    # Cosine similarity
    q = query_vec.reshape(1, -1)
    q_norm = np.linalg.norm(q, axis=1, keepdims=True) + 1e-12
    m_norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-12
    cosine_sim = (matrix @ q.T).flatten() / (m_norms.flatten() * q_norm.flatten())

    # Normalized Euclidean -> similarity
    diff = matrix - q
    euclid = np.linalg.norm(diff, axis=1)  # [0, sqrt(n_features)]
    max_euclid = np.sqrt(matrix.shape[1])
    norm_euclid = euclid / (max_euclid + 1e-12)  # [0,1]
    euclid_sim = 1.0 - norm_euclid  # [0,1], większe = bliżej

    # Hybryda
    sim = alpha * cosine_sim + (1.0 - alpha) * euclid_sim
    # Clip na wszelki wypadek numeryczny
    sim = np.clip(sim, 0.0, 1.0)
    return sim

# --- FUNKCJA REKOMENDACJI ---
def find_similar_tracks(row_index, df_raw, df_norm, features, k=5, sort_by_popularity=True, alpha=ALPHA):
    # Macierz cech (już [0,1])
    feature_matrix = df_norm[features].values.astype(float)
    query_vector = feature_matrix[row_index]

    # Podobieństwo hybrydowe do wszystkich
    sims = _combined_similarity(query_vector, feature_matrix, alpha)

    # Wyklucz sam siebie
    sims[row_index] = -np.inf

    # Weź kandydatów (k + zapas)
    extra = 10
    top_idx = np.argpartition(-sims, kth=min(k+extra, len(sims)-1))[:k+extra]
    # Posortuj dokładnie po similarity malejąco
    top_idx = top_idx[np.argsort(-sims[top_idx])]

    results = df_raw.loc[top_idx, ['track_name', 'artists', 'popularity']].copy()
    results['similarity'] = sims[top_idx]

    # Sortowanie
    if sort_by_popularity:
        results = results.sort_values(['popularity'], ascending=[False])
    else:
        results = results.sort_values('similarity', ascending=False)

    # Usuń duplikaty po nazwie (zachowaj jeden o większej popularności bądź większym similarity w zależności od sortowania)
    key_order = ['track_name', 'similarity'] if not sort_by_popularity else ['track_name', 'popularity']
    results = results.drop_duplicates(subset=['track_name'], keep='first')

    return results.head(k)[['track_name', 'artists', 'similarity', 'popularity']]

def evaluate_similarity_with_ai(selected_track, recommended_df, alpha=ALPHA):
    # Formatowanie listy utworów
    recommended_list = "\n".join(
        f"{row['track_name']} – {row['artists']} (similarity: {row['similarity']:.2f})"
        for _, row in recommended_df.iterrows()
    )

    # Prompt z kontekstem działania algorytmu (zaktualizowany do metryki hybrydowej)
    prompt = f"""
You are a music expert. Analyze whether the following song recommendations are musically valid.

Songs are selected using a **hybrid similarity** computed on normalized audio features:
- cosine similarity (directional similarity in feature space)
- and normalized Euclidean similarity (1 - Euclidean/√n_features)

Final score: similarity = α * cosine + (1 - α) * (1 - normalized Euclidean), with α = {alpha:.2f}.

Keep in mind:
- This method does **not** account for genre or artist metadata — only measurable musical attributes.
- A high similarity doesn't guarantee that songs will feel similar to every listener.
- Your task is to evaluate **musical similarity**, not algorithm correctness.

Selected track:
{selected_track}

Recommended tracks:
{recommended_list}

For each recommendation evaluate:
- genre proximity
- mood / emotional tone
- instrumentation
- overall musical feel

For each recommendation:
- Provide a short, clear assessment (1–2 sentences).
- Conclude with **Relevant** or **Not relevant** in bold.
"""

    client = openai.OpenAI(api_key=api_key)

    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that evaluates music similarity recommendations."},
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content.strip()

# --- STRONA WIZUALNA - STREAMLIT ---
df, df_norm, features_for_similarity, stats, corr_matrix, cols = load_and_prepare_data()

st.title("🎵 Spotify Track Explorer")
st.markdown("Explore the distribution and relationships between audio features.")

st.markdown("### Assumptions and Analytical Goals")
st.markdown(f"""
This project assumes that:

- The goal is to recommend songs based on **audio similarity**.
- We focus on **intrinsic musical properties** rather than metadata like genre, album or release date.
- **Popularity is excluded from similarity scoring**, since it reflects external factors — but is still used to filter and prioritize results.
- **Track duration** is excluded as well, due to high variance and the presence of non-musical content (e.g. podcasts or live sets).
- Engineered features (`mood_score`, `vocals_strength`) are introduced to better reflect emotional and vocal characteristics of tracks.
- The similarity is computed using a **hybrid metric**:

  **similarity = α·cosine + (1−α)·(1−normalized Euclidean)** with **α = {ALPHA:.2f}**.

- The user will be able to explore the data, understand how similarity is defined, and receive transparent recommendations.
""")

# Heatmapa korelacji
st.subheader("Feature Correlation Matrix")
fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", ax=ax_corr)
st.pyplot(fig_corr)

st.markdown("### Insights from the Correlation Matrix")
st.markdown("""
The correlation matrix reveals several important relationships between audio features:

- **`energy` and `loudness`** are strongly positively correlated — which makes sense, as more energetic tracks tend to be louder.
- **`danceability`** shows a moderate positive correlation with both **`valence`** and **`loundness`**, indicating that upbeat, louder songs are often more danceable.
- **`acousticness`** is **negatively correlated** with most other features — especially `energy` and `loudness` — suggesting that acoustic tracks tend to be calmer and quieter.
- **`instrumentalness`** and **`speechiness`** are not strongly correlated with other features, making them useful for describing niche aspects of tracks (e.g. vocals vs. instrumental).

These patterns support the selection of features used in the similarity search — as they capture distinct and meaningful dimensions of a song's sound profile.
""")

# Wykresy rozkładu (oryginalne cechy)
st.markdown("### Feature Distributions")
st.markdown("""
Before building the recommendation logic, I explored the **raw distribution of available features** to understand their behavior and spot outliers, skews or irregularities.

These visualizations helped guide key preprocessing choices like **feature selection**, **normalization**, and **outlier handling**.
""")

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

st.markdown("### Feature Distributions: Key Observations")
st.markdown("""
- `duration_s` shows a sharp skew due to the presence of long-format content (e.g. podcasts).  
  This strong right skew indicates that while most tracks are relatively short, a small number of very long items (e.g. podcasts or live sets) greatly exceed the typical duration and could distort distance-based metrics.
- `instrumentalness` and `speechiness` contain extreme outliers — capped at the 95th percentile in preprocessing.  
  These values were clipped before normalization because their long tails would have distorted the min-max scaling process and reduced the discriminative power of these features in similarity comparisons.
- `loudness` has a narrow peak, suggesting a typical mastering level in most tracks.
- `energy`, `valence`, and `danceability` are well distributed and highly relevant for listener perception.
- Several features (e.g. `liveness`, `acousticness`) are heavily concentrated near 0, indicating the dominance of studio-produced tracks in the dataset.

""")

# Statystyki opisowe
st.subheader("Descriptive Statistics")
st.dataframe(stats.T.round(2))

st.markdown("### Feature Selection Strategy for Similarity Search")
st.markdown("""
- The dataset includes a wide range of audio and metadata features from Spotify — such as `popularity`, `duration`, `danceability`, `valence`, `loudness`, and more.
- For the purposes of recommending **similar-sounding tracks**, I focused on **intrinsic audio characteristics** — those that describe the sound and feel of the track, not its popularity or context.
- I excluded features like:
  - `popularity`: reflects user behavior, not audio similarity,
  - `duration_s`: varies heavily, with some tracks resembling **podcast-length content**, which could distort similarity scoring.
- Based on correlation analysis and musical intuition, I selected the following features:
  - `danceability`, `energy`, `valence`, `loudness`, `acousticness`, `tempo`, `mood_score` (*see below*), and `vocals_strength` (*see below*)
- These cover the essential dimensions of musical experience:
  - **rhythm** (`tempo`, `danceability`),
  - **intensity and production** (`energy`, `loudness`, `acousticness`),
  - **emotional tone** (`valence`, `mood_score`),
  - **vocal character** (`vocals_strength`)
- This combination of features forms a **compact yet expressive vector representation** of each track, suited for similarity search.
""")

# Wykresy po normalizacji
st.subheader("Normalized Feature Distributions (used for similarity search)")
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

st.markdown("### Notes on Normalized Features and Outlier Handling")
st.markdown(
    "- All features used for similarity search are **scaled to the [0, 1] range** using **Min-Max normalization**.\n"
    "- This ensures that each feature contributes equally to the distance calculation.\n"
    "- **Outliers in `instrumentalness` and `speechiness`** are handled by:\n"
    "  - Clipping values above the 95th percentile to a fixed cap.\n"
    "  - This prevents rare extreme values from skewing similarity scores.\n"
    "- The final feature set includes engineered metrics like:\n"
    "  - `mood_score = valence * energy`\n"
    "  - `vocals_strength = 1 - instrumentalness - speechiness`\n"
    "- This preprocessing step improves the **accuracy and interpretability** of recommendations."
)

# Opis działania (ZAKTUALIZOWANY DO METRYKI HYBRYDOWEJ)
st.markdown("### How Similar Tracks Are Selected")
st.markdown(f"""
- I analyze **8 normalized audio features** that capture musical style and mood:
  - `danceability`, `energy`, `valence`, `loudness`, `acousticness`, `tempo`, `mood_score`, `vocals_strength`
- A selected track is represented as a **feature vector** in this multi-dimensional space.
- For the selected track, I compute its similarity to **all other tracks** using a **hybrid metric**:

  \\[
  \\text{{similarity}} = \\alpha\\cdot\\text{{cosine}} + (1-\\alpha)\\cdot(1 - \\text{{normalized Euclidean}})
  \\]

  where:
  - \\(\\text{{cosine}}(a,b) = \\frac{{a\\cdot b}}{{\\lVert a\\rVert\\,\\lVert b\\rVert}}\\)
  - \\(\\text{{normalized Euclidean}}(a,b) = \\frac{{\\lVert a-b\\rVert}}{{\\sqrt{{n}}}}\\) (features are in \\([0,1]\\), so max distance is \\(\\sqrt{{n}}\\))
  - **α = {ALPHA:.2f}** (higher weight on cosine to emphasize directional similarity)

- Tracks are ranked by **similarity** (higher is better). Identical items are excluded.
- Optionally, within the candidate pool, results can be **re-sorted by popularity**.
- Final result: **Top 5 most similar tracks**.
""")

# --- ELBOW PLOT: dla metryki hybrydowej (na próbie) ---
st.subheader("Elbow Plot for Nearest Neighbors (Hybrid Dissimilarity)")

@st.cache_data
def compute_elbow_hybrid(df_norm, features, alpha=ALPHA, sample_size=1000, k_max=15, seed=RANDOM_SEED):
    rng = np.random.default_rng(seed)
    X = df_norm[features].values.astype(float)
    n = X.shape[0]
    m = min(sample_size, n)
    idxs = rng.choice(n, size=m, replace=False)

    k_range = list(range(1, k_max + 1))
    sums = np.zeros(len(k_range), dtype=float)

    for idx in idxs:
        sims = _combined_similarity(X[idx], X, alpha)
        # Dissimilarity = 1 - similarity; wyklucz sam siebie
        dis = 1.0 - sims
        dis[idx] = np.inf
        dis_sorted = np.sort(dis)  # rosnąco: najbliżsi pierwsi
        # Dodaj dystanse k-tego najbliższego sąsiada
        for j, k in enumerate(k_range):
            if k - 1 < len(dis_sorted):
                sums[j] += dis_sorted[k - 1]
            else:
                sums[j] += dis_sorted[-1]
    means = (sums / m).tolist()
    return k_range, means

k_values, avg_diss = compute_elbow_hybrid(df_norm, features_for_similarity, alpha=ALPHA, sample_size=800, k_max=15)
fig_elbow, ax_elbow = plt.subplots()
ax_elbow.plot(k_values, avg_diss, marker="o")
ax_elbow.set_xlabel("k")
ax_elbow.set_ylabel("Avg dissimilarity (1 - similarity) to k-th neighbor")
ax_elbow.set_title("Elbow Plot (Hybrid Metric)")
st.pyplot(fig_elbow)

st.markdown("""
### Elbow Plot Analysis for Optimal `k`

The plot above illustrates how the **average hybrid dissimilarity** (1 − similarity) to the *k*-th nearest neighbor changes with `k`.  
As with classic KNN elbow logic, the curve typically slows down around a small `k`. In this dataset, **k ≈ 5** remains a sensible default: it yields compact, relevant sets without drifting too far into less similar neighbors.
""")

# Interfejs wyszukiwania podobnych utworów
st.subheader("Find Similar Tracks")
selected_combo = st.selectbox("Choose a track:", df['title_artist'].unique())
selected_index = df[df['title_artist'] == selected_combo].index[0]

use_popularity = st.checkbox("Sort by popularity within results", value=True)

if st.button("🔍 Find Similar"):
    results_df = find_similar_tracks(
        selected_index, df, df_norm, features_for_similarity, k=5, sort_by_popularity=use_popularity, alpha=ALPHA
    )
    st.write(f"Top 5 tracks similar to **{selected_combo}** (hybrid metric, α={ALPHA:.2f}):")
    st.dataframe(results_df[['track_name', 'artists', 'similarity', 'popularity']].reset_index(drop=True), hide_index=True)

    # AI Evaluation
    ai_feedback = evaluate_similarity_with_ai(
        selected_combo, results_df[['track_name', 'artists', 'similarity']], alpha=ALPHA
    )
    st.markdown("### AI Evaluation of Recommendations")
    st.markdown("""
AI analyzes the musical similarity of the recommended tracks with each search. Here's the generated feedback:
""")
    st.write(ai_feedback)

# Podsumowanie i potencjalne ulepszenia
st.markdown("""
### Thought Process & Ideas Worth Exploring

- **Dynamic Feature Weighting**  
  It could be beneficial to emphasize features that are particularly distinctive for a given track.  
  For example, if a song has an exceptionally high `danceability` while other features are closer to the mean, that feature could be given more weight — ensuring that highly danceable songs return recommendations with similar characteristics.

- **Two-Stage Filtering Approach**  
  Use the hybrid similarity to select a candidate pool (e.g. top 20), then apply additional filtering: e.g. same artist, same album, or shared unusual feature values. This combines numerical proximity with semantic context.

- **Popularity-Aware Re-Ranking**  
  With higher `k` values, broader recommendation sets may include more marginal tracks. Lightly favoring more popular tracks within this pool may increase user satisfaction.  
  Currently, `popularity` is only used to exclude value 0 and for optional re-sorting.

- **Avoid Using Popularity as a Similarity Feature**  
  Adding `popularity` directly to the similarity vector might distort results. Two tracks with the same number of streams (e.g., disco and metal) are not necessarily musically alike. Popularity should be considered secondary, not part of the audio feature space.

- **Artist and Album Awareness**  
  In real-world use, users often explore more songs from the same artist or album. Enhancing the algorithm to optionally favor such continuity — for at least one suggestion — could feel more intuitive.

- **Segmenting Long-Form Content**  
  Tracks significantly longer than typical songs are often podcasts, interviews or DJ sets. These could be analyzed in a separate similarity space to avoid mismatches between spoken word and music.

- **Hybrid Similarity (Implemented)**  
  The app now uses a hybrid of **cosine** and **normalized Euclidean** similarities with **α = """ + f"{ALPHA:.2f}" + """** by default.  
  Depending on your dataset and goals, you can tune α to emphasize direction (cosine) vs. absolute proximity (Euclidean).
            
- **Clustering** 
  It is possible to cluster tracks based on their musical genre or sonic profile. Once such clusters are created, the system could ensure that at least one of the recommended tracks comes from the same cluster as the base track. This may help avoid numerically close but musically distant suggestions.

One of the biggest difficulties I encountered was evaluating the actual quality of recommendations. Judging whether one song is truly "similar" to another is highly subjective, especially across genres and moods.

To reduce bias, the app leverages an objective AI-based evaluation: GPT-4.1 analyzes the musical similarity of track recommendations with each search.
""")
