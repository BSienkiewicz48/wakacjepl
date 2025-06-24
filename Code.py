import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors
import openai

api_key = st.secrets["API_KEY"]

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


def evaluate_similarity_with_ai(selected_track, recommended_df):
    # Tworzymy prompt do oceny przez AI
    prompt = f"""
You are a music expert. Analyze the similarity between a selected song and its recommended songs based on the following data:

Selected track:
{selected_track}

Recommended tracks:
{recommended_df.to_string(index=False)}

Evaluate whether the recommendations make sense in terms of:
- genre
- mood
- instrumentation
- overall musical feel

Be concise, objective, and focus on musical similarity.
At the end of analysis of every track provide a straight opinion whether the recommendation is relevant or not and write it bolded.
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
st.markdown("""
This project assumes that:

- The goal is to recommend songs based on **audio similarity**, not user behavior.
- We focus on **intrinsic musical properties** rather than metadata like genre, album or release date.
- **Popularity is excluded from similarity scoring**, since it reflects external factors — but is still used to filter and prioritize results.
- **Track duration** is excluded as well, due to high variance and the presence of non-musical content (e.g. podcasts or live sets).
- Engineered features (`mood_score`, `vocals_strength`) are introduced to better reflect emotional and vocal characteristics of tracks.
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
- **`danceability`** shows a moderate positive correlation with both **`valence`** and **`energy`**, indicating that upbeat, energetic songs are often more danceable.
- **`acousticness`** is **negatively correlated** with most other features — especially `energy` and `loudness` — suggesting that acoustic tracks tend to be calmer and quieter.
- **`instrumentalness`** and **`speechiness`** are not strongly correlated with other features, making them useful for describing niche aspects of tracks (e.g. vocals vs. instrumental).

These patterns support the selection of features used in the similarity search — as they capture distinct and meaningful dimensions of a song's sound profile.
""")

# Wykresy rozkładu (oryginalne cechy)
st.markdown("### Feature Distributions")
st.markdown("""
Before building the recommendation logic, we explored the **raw distribution of available features** to understand their behavior and spot outliers, skews or irregularities.

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
- Several features (e.g. `liveness`, `acousticness`) are heavily concentrated near 0, indicating the dominance of studio-produced tracks.
""")

# Statystyki opisowe
st.subheader("Descriptive Statistics")
st.dataframe(stats.T.round(2))

st.markdown("### Feature Selection Strategy for Similarity Search")
st.markdown("""
- The dataset includes a wide range of audio and metadata features from Spotify — such as `popularity`, `duration`, `danceability`, `valence`, `loudness`, and more.
- For the purposes of recommending **similar-sounding tracks**, we focused on **intrinsic audio characteristics** — those that describe the sound and feel of the track, not its popularity or context.
- We excluded features like:
  - `popularity`: reflects user behavior, not audio similarity,
  - `duration_s`: varies heavily, with some tracks resembling **podcast-length content**, which could distort similarity scoring.
- Based on correlation analysis and musical intuition, we selected the following features:
  - `danceability`, `energy`, `valence`, `loudness`, `acousticness`, `tempo`, `mood_score` (*see below*), and `vocals_strength` (*see below*)
- These cover the essential dimensions of musical experience:
  - **rhythm** (`tempo`, `danceability`),
  - **intensity and production** (`energy`, `loudness`, `acousticness`),
  - **emotional tone** (`valence`, `mood_score`),
  - **vocal character** (`vocals_strength`)
- This combination of features forms a **compact yet expressive vector representation** of each track, well-suited for similarity search.
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


st.subheader("Elbow Plot for Nearest Neighbors")

@st.cache_data
def compute_elbow(df_norm, features):
    model = NearestNeighbors(metric='euclidean')
    model.fit(df_norm[features])
    k_range = range(1, 16)
    distances = []
    for k in k_range:
        dist, _ = model.kneighbors(df_norm[features], n_neighbors=k)
        mean_dist = np.mean(dist[:, -1])  # ostatni dystans
        distances.append(mean_dist)
    return list(k_range), distances

k_values, avg_dists = compute_elbow(df_norm, features_for_similarity)
fig_elbow, ax_elbow = plt.subplots()
ax_elbow.plot(k_values, avg_dists, marker="o")
ax_elbow.set_xlabel("k")
ax_elbow.set_ylabel("Avg distance to k-th neighbor")
ax_elbow.set_title("Elbow Plot")
st.pyplot(fig_elbow)


st.markdown("""
### Elbow Plot Analysis for Optimal `k`

The plot above illustrates how the average distance to the *k*-th nearest neighbor increases with larger values of `k`. Initially, the distance grows rapidly, but this growth slows down around **k = 5**, forming a noticeable **"elbow"** in the curve.

This "elbow point" indicates an optimal trade-off:
- ✅ Too small `k` might result in unstable or overly specific recommendations.
- 🚫 Too large `k` could dilute the quality by including less relevant, distant neighbors.

Based on this analysis, **`k = 5`** offers a balanced choice — providing relevant, compact recommendations without including unrelated outliers.
""")

# Interfejs wyszukiwania podobnych utworów
st.subheader("Find Similar Tracks")
selected_combo = st.selectbox("Choose a track:", df['title_artist'].unique())
selected_index = df[df['title_artist'] == selected_combo].index[0]

if st.button("🔍 Find Similar"):
    results_df = find_similar_tracks(selected_index, df, df_norm, features_for_similarity, k=5)
    st.write(f"Top 5 tracks similar to **{selected_combo}**:")
    st.dataframe(results_df.reset_index(drop=True), hide_index=True)
    
    # AI Evaluation
    ai_feedback = evaluate_similarity_with_ai(selected_combo, results_df[['track_name', 'artists']])
    st.markdown("### AI Evaluation of Recommendations")
    st.markdown("""
AI analyzes the musical similarity of the recommended tracks with each search. Here's the generated feedback:
""")
    st.write(ai_feedback)


# Opis działania
st.markdown("### How Similar Tracks Are Selected")
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

# Podsumowanie i potencjalne ulepszenia
st.markdown("### Summary and Potential Improvements")
st.markdown("""
- The system recommends similar tracks based solely on **normalized audio features**, intentionally ignoring metadata such as album or genre.
- The `popularity` feature was not used for measuring similarity, but **was used to filter out low-relevance tracks** and break ties when needed.
- Selected features were chosen to reflect **musical structure, emotion, and production style**, not user behavior.
- Outliers were capped at the 95th percentile, and engineered features like `mood_score` and `vocals_strength` were introduced for better expressiveness.

**What could improve user experience:**
- Ensure at least one recommendation is from the **same artist** — likely to increase perceived relevance and continuity.
- Separate long-duration tracks (e.g. podcasts) into a different comparison group.
- Suggest additional tracks from the **same album**, where applicable.
""")