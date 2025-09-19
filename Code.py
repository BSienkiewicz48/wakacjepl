import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
import openai

api_key = st.secrets["API_KEY"]

# ===============================
# USTAWIENIA METRYKI I ADAPTACJI
# ===============================
ALPHA = 0.60  # waga części cosine (możesz zmienić)
EPS = 1e-12

# Adaptive Feature Weighting:
# - im bardziej "wyjątkowa" cecha utworu bazowego (odchylenie od średniej), tym większa waga
ADAPTIVE_BETA = 0.50   # maksymalny dodatni boost ~ +50% (przed renormalizacją do średniej = 1)
ADAPTIVE_ZCAP = 2.0    # z-score >= 2 traktujemy jako w pełni "wyjątkowy"

# ================
# DANE I PREP
# ================
@st.cache_data
def load_and_prepare_data():
    df = pd.read_parquet("raw_data.parquet")
    df = df[df['popularity'] > 0].copy()

    df['duration_s'] = df['duration_ms'] / 1000
    df = df.drop(columns=['duration_ms'])

    df['mood_score'] = df['valence'] * df['energy']
    df['vocals_strength'] = 1 - df['instrumentalness'] - df['speechiness']
    df['vocals_strength'] = df['vocals_strength'].clip(0, 1)

    # lekkie obcięcie ogonów
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

# ===========================================
# ADAPTIVE FEATURE WEIGHTING — FUNKCJE
# ===========================================
def compute_adaptive_weights(feature_matrix: np.ndarray,
                             query_vec: np.ndarray,
                             beta: float = ADAPTIVE_BETA,
                             z_cap: float = ADAPTIVE_ZCAP,
                             eps: float = EPS) -> np.ndarray:
    """
    Wyznacza wektory wag dla cech na podstawie "wyjątkowości" profilu utworu bazowego.
    - liczymy z-score względem średniej i odchylenia std w zbiorze,
    - przeskalowujemy do [0,1] poprzez podział przez z_cap,
    - wagi = 1 + beta * exceptionalness,
    - renormalizacja do średniej 1 (sum(w)=n_features), żeby zachować skalę metryk.
    """
    means = feature_matrix.mean(axis=0)
    stds = feature_matrix.std(axis=0) + eps
    z = np.abs(query_vec - means) / stds
    exceptional = np.clip(z / max(z_cap, eps), 0.0, 1.0)
    w_raw = 1.0 + beta * exceptional

    # renormalizacja do średniej 1
    w = w_raw * (feature_matrix.shape[1] / (np.sum(w_raw) + eps))
    return w

# ===========================================
# METRYKA HYBRYDOWA (COSINE + NORM EUCLIDEAN)
# z wagami adaptacyjnymi
# ===========================================
def _combined_similarity(query_vec: np.ndarray,
                         matrix: np.ndarray,
                         alpha: float,
                         weights: np.ndarray | None = None,
                         eps: float = EPS) -> np.ndarray:
    """
    Zwraca wektor similarity w [0,1], im większa wartość tym większe podobieństwo.
    Używa wag 'weights' (jeśli None -> wektor jedynek).
    """
    n_features = matrix.shape[1]
    if weights is None:
        weights = np.ones(n_features, dtype=float)

    # Weighted cosine
    # cos_w(a,b) = (sum_i w_i a_i b_i) / (sqrt(sum_i w_i a_i^2) * sqrt(sum_i w_i b_i^2))
    q = query_vec.reshape(1, -1)
    num = (matrix * weights) @ q.T  # (n_items, 1)
    denom_q = np.sqrt(np.sum(weights * (q ** 2), axis=1, keepdims=True)) + eps
    denom_m = np.sqrt(np.sum(weights * (matrix ** 2), axis=1, keepdims=True)) + eps
    cosine_sim = (num.flatten()) / (denom_q.flatten() * denom_m.flatten())

    # Weighted Euclidean (znormalizowany do [0,1])
    # euclid_w = sqrt( sum_i w_i (a_i - b_i)^2 )
    # normalized_euclid_w = euclid_w / sqrt(sum_i w_i)  (sum_i w_i = n_features, bo średnia wag = 1)
    diff = matrix - q
    euclid_w = np.sqrt(np.sum(weights * (diff ** 2), axis=1))
    norm_euclid_w = euclid_w / np.sqrt(np.sum(weights))

    euclid_sim = 1.0 - norm_euclid_w  # [0,1]

    sim = alpha * cosine_sim + (1.0 - alpha) * euclid_sim
    sim = np.clip(sim, 0.0, 1.0)
    return sim

# =====================
# REKOMENDACJE
# =====================
def find_similar_tracks(row_index, df_raw, df_norm, features, k=5, sort_by_popularity=True, alpha=ALPHA):
    # Macierz cech (już [0,1])
    X = df_norm[features].values.astype(float)
    q = X[row_index]

    # wagi adaptacyjne zależne od profilu utworu bazowego
    w = compute_adaptive_weights(X, q, beta=ADAPTIVE_BETA, z_cap=ADAPTIVE_ZCAP)

    # Podobieństwo hybrydowe do wszystkich
    sims = _combined_similarity(q, X, alpha, weights=w)

    # Wyklucz sam siebie
    sims[row_index] = -np.inf

    # Kandydaci (k + bufor)
    extra = 10
    k_eff = min(k + extra, len(sims) - 1)
    top_idx = np.argpartition(-sims, kth=k_eff)[:k_eff]
    top_idx = top_idx[np.argsort(-sims[top_idx])]

    results = df_raw.loc[top_idx, ['track_name', 'artists', 'popularity']].copy()
    results['similarity'] = sims[top_idx]

    # Sortowanie
    if sort_by_popularity:
        results = results.sort_values(['popularity'], ascending=[False])
    else:
        results = results.sort_values('similarity', ascending=False)

    # Duplikaty tytułów
    results = results.drop_duplicates(subset=['track_name'], keep='first')
    return results.head(k)[['track_name', 'artists', 'similarity', 'popularity']]

# =====================
# OCENA AI
# =====================
def evaluate_similarity_with_ai(selected_track, recommended_df, alpha=ALPHA):
    rec_list = "\n".join(
        f"{row['track_name']} – {row['artists']} (similarity: {row['similarity']:.2f})"
        for _, row in recommended_df.iterrows()
    )

    prompt = f"""
You are a music expert. Analyze whether the following song recommendations are musically valid.

Songs are selected using a **hybrid similarity** on normalized audio features with **Adaptive Feature Weighting**:
- Weighted cosine (directional similarity)
- and weighted normalized Euclidean similarity (1 - Euclidean/√(∑w_i))

Final score:
similarity = α * cosine_w + (1 - α) * (1 - normalized Euclidean_w), with α = {alpha:.2f}.

Weights emphasize features that are "exceptional" for the base track (larger deviation from dataset mean → larger weight; weights are renormalized to average 1).

Keep in mind:
- This method does **not** use genre/artist metadata — only measurable musical attributes.
- A high similarity doesn't guarantee that songs will feel similar to every listener.
- Your task is to evaluate **musical similarity**, not algorithm correctness.

Selected track:
{selected_track}

Recommended tracks:
{rec_list}

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

# =====================
# STRONA — STREAMLIT
# =====================
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
- Similarity is computed with a **hybrid metric** and **Adaptive Feature Weighting** (features that are exceptional for the base track get higher weight).
- Default parameters: **α = {ALPHA:.2f}**, adaptive boost β = {ADAPTIVE_BETA:.2f}.
""")

# --- Heatmap korelacji ---
st.subheader("Feature Correlation Matrix")
fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", ax=ax_corr)
st.pyplot(fig_corr)

st.markdown("### Insights from the Correlation Matrix")
st.markdown("""
The correlation matrix reveals several important relationships between audio features:

- **`energy` and `loudness`** are strongly positively correlated — which makes sense, as more energetic tracks tend to be louder.
- **`danceability`** shows a moderate positive correlation with both **`valence`** and **`loudness`**, indicating that upbeat, louder songs are often more danceable.
- **`acousticness`** is **negatively correlated** with most other features — especially `energy` and `loudness` — suggesting that acoustic tracks tend to be calmer and quieter.
- **`instrumentalness`** and **`speechiness`** are not strongly correlated with other features, making them useful for describing niche aspects of tracks (e.g. vocals vs. instrumental).

These patterns support the selection of features used in the similarity search — as they capture distinct and meaningful dimensions of a song's sound profile.
""")

# --- Rozkłady cech (raw) ---
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
- `instrumentalness` and `speechiness` contain extreme outliers — capped at the 95th percentile in preprocessing.
- `loudness` has a narrow peak, suggesting a typical mastering level in most tracks.
- `energy`, `valence`, and `danceability` are well distributed and highly relevant for listener perception.
- Several features (`liveness`, `acousticness`) are heavily concentrated near 0, indicating the dominance of studio-produced tracks in the dataset.
""")

# --- Statystyki opisowe ---
st.subheader("Descriptive Statistics")
st.dataframe(stats.T.round(2))

st.markdown("### Feature Selection Strategy for Similarity Search")
st.markdown("""
- We focus on **intrinsic audio characteristics** that describe the sound and feel of the track.
- Excluded: `popularity` (behavioral), `duration_s` (very high variance, long-form content).
- Selected features: `danceability`, `energy`, `valence`, `loudness`, `acousticness`, `tempo`, `mood_score`, `vocals_strength`.
- They cover **rhythm**, **intensity/production**, **emotional tone**, and **vocal character**.
""")

# --- Rozkłady po normalizacji ---
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
    "- All features are **scaled to [0, 1]** via **Min-Max**.\n"
    "- Outliers in `instrumentalness` and `speechiness` are clipped at 95th percentile.\n"
    "- Engineered metrics: `mood_score = valence * energy`, `vocals_strength = 1 - instrumentalness - speechiness`.\n"
    "- Similarity uses **hybrid metric** with **Adaptive Feature Weighting**."
)

# --- OPIS METRYKI: poprawne wzory (st.latex) ---
st.markdown("### How Similar Tracks Are Selected")

st.markdown(
    "- We analyze **8 normalized audio features** capturing musical style and mood.\n"
    "- For the selected track, we compute its similarity to all others using a **hybrid metric** with adaptive feature weights."
)

st.latex(r"\text{similarity} \;=\; \alpha\cdot \text{cosine}_w \;+\; (1-\alpha)\cdot \big(1 - \text{normalized Euclidean}_w\big)")

st.markdown("**Weighted cosine** and **weighted normalized Euclidean**:")
st.latex(r"\text{cosine}_w(a,b) \;=\; \frac{\sum_i w_i\, a_i b_i}{\sqrt{\sum_i w_i\, a_i^2}\;\sqrt{\sum_i w_i\, b_i^2}}")
st.latex(r"\text{normalized Euclidean}_w(a,b) \;=\; \frac{\sqrt{\sum_i w_i\, (a_i-b_i)^2}}{\sqrt{\sum_i w_i}}")

st.markdown("**Adaptive Feature Weighting** (większa waga dla cech wyjątkowych w utworze bazowym):")
st.latex(r"z_i \;=\; \frac{|x_i - \mu_i|}{\sigma_i + \varepsilon}")
st.latex(r"e_i \;=\; \min\!\left(\frac{z_i}{Z_{\text{cap}}},\,1\right)")
st.latex(r"w_i' \;=\; 1 + \beta \, e_i \qquad\quad \tilde{w}_i \;=\; \frac{w_i'}{\frac{1}{n}\sum_{j=1}^n w_j'}")
st.markdown(
    f"Default params: $\\alpha={ALPHA:.2f}$, $\\beta={ADAPTIVE_BETA:.2f}$, $Z_{{\\text{{cap}}}}={ADAPTIVE_ZCAP:.1f}$. "
    "Weights are renormalized to have mean 1 (so distances remain in [0,1])."
)

# --- ELBOW PLOT (dla metryki hybrydowej + adaptacji) ---
st.subheader("Elbow Plot for Nearest Neighbors (Hybrid + Adaptive)")

@st.cache_data
def compute_elbow_hybrid(df_norm, features, alpha=ALPHA, sample_size=800, k_max=15, seed=42):
    rng = np.random.default_rng(seed)
    X = df_norm[features].values.astype(float)
    n = X.shape[0]
    m = min(sample_size, n)
    idxs = rng.choice(n, size=m, replace=False)

    k_range = list(range(1, k_max + 1))
    sums = np.zeros(len(k_range), dtype=float)

    for idx in idxs:
        q = X[idx]
        w = compute_adaptive_weights(X, q, beta=ADAPTIVE_BETA, z_cap=ADAPTIVE_ZCAP)
        sims = _combined_similarity(q, X, alpha, weights=w)
        dis = 1.0 - sims
        dis[idx] = np.inf
        dis_sorted = np.sort(dis)
        for j, k in enumerate(k_range):
            kth = dis_sorted[k - 1] if (k - 1) < len(dis_sorted) else dis_sorted[-1]
            sums[j] += kth

    means = (sums / m).tolist()
    return k_range, means

k_values, avg_diss = compute_elbow_hybrid(df_norm, features_for_similarity, alpha=ALPHA, sample_size=800, k_max=15)
fig_elbow, ax_elbow = plt.subplots()
ax_elbow.plot(k_values, avg_diss, marker="o")
ax_elbow.set_xlabel("k")
ax_elbow.set_ylabel("Avg dissimilarity (1 - similarity) to k-th neighbor")
ax_elbow.set_title("Elbow Plot (Hybrid + Adaptive)")
st.pyplot(fig_elbow)

st.markdown("""
**Elbow analysis**: as with classic KNN, a small k around **5** is a sensible default — compact results without drifting into weak neighbors.
""")

# --- INTERFEJS WYSZUKIWANIA ---
st.subheader("Find Similar Tracks")
selected_combo = st.selectbox("Choose a track:", df['title_artist'].unique())
selected_index = df[df['title_artist'] == selected_combo].index[0]
use_popularity = st.checkbox("Sort by popularity within results", value=True)

if st.button("🔍 Find Similar"):
    results_df = find_similar_tracks(
        selected_index, df, df_norm, features_for_similarity, k=5, sort_by_popularity=use_popularity, alpha=ALPHA
    )
    st.write(f"Top 5 tracks similar to **{selected_combo}** (hybrid metric + adaptive weights, α={ALPHA:.2f}):")
    st.dataframe(results_df[['track_name', 'artists', 'similarity', 'popularity']].reset_index(drop=True), hide_index=True)

    # AI Evaluation
    ai_feedback = evaluate_similarity_with_ai(
        selected_combo, results_df[['track_name', 'artists', 'similarity']], alpha=ALPHA
    )
    st.markdown("### AI Evaluation of Recommendations")
    st.markdown("AI analyzes the musical similarity of the recommended tracks with each search. Here's the generated feedback:")
    st.write(ai_feedback)

# --- PODSUMOWANIE ---
st.markdown("""
### Thought Process & Ideas Worth Exploring

- **Adaptive Feature Weighting (implemented)**  
  Features that are unusually low/high for the base track carry more weight in similarity.

- **Two-Stage Filtering**  
  Use the hybrid-adaptive similarity to pick a candidate pool, then optionally apply semantic filters (same artist/album, unusual traits).

- **Popularity-Aware Re-Ranking**  
  Within the candidate pool, optional re-ranking by popularity may improve satisfaction.

- **Clustering & Diversity**  
  Cluster tracks by sonic profile to ensure at least one same-cluster suggestion and avoid numerically close but musically distant picks.
""")
