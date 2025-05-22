import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_parquet(r'raw_data.parquet')

cols = [
    'popularity', 'duration_ms', 'danceability', 'energy', 'loudness',
    'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo'
]

corr_matrix = df[cols].corr()
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Matrix for Selected Features")