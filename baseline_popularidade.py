import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from tqdm import tqdm

# Carregando os dados
ratings = pd.read_csv("ratings.csv")
movies = pd.read_csv("movies.csv")

# Merge e amostragem de 1%
df = pd.merge(ratings, movies, on="movieId")
df = df.sample(frac=0.01, random_state=42).reset_index(drop=True)

# Função para calcular HR@K e NDCG@K para baseline
def calcular_metrics_baseline(df, k=10, num_folds=5):
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

    hr_scores = []
    ndcg_scores = []

    print("⌛ Validação Cruzada Baseline:")
    for train_index, test_index in tqdm(kf.split(df), total=num_folds):
        train_data = df.iloc[train_index]
        test_data = df.iloc[test_index]

        # Filmes mais populares no treino
        filmes_populares = train_data["title"].value_counts().head(k).index.tolist()

        grouped = test_data.groupby("userId")

        for user_id, group in grouped:
            filmes_reais = group["title"].tolist()

            hit = any(filme in filmes_reais for filme in filmes_populares)
            hr_scores.append(1 if hit else 0)

            # Cálculo do NDCG
            dcg = 0.0
            for i, filme in enumerate(filmes_populares):
                if filme in filmes_reais:
                    dcg = 1 / np.log2(i + 2)
                    break

            idcg = 1.0  # Ideal DCG = 1 se um relevante estiver no top-k
            ndcg_scores.append(dcg / idcg)

    print(f"\n📊 Resultados médios (Baseline Popularidade com {k} filmes no Top):")
    print(f"HR@{k}: {np.mean(hr_scores):.4f}")
    print(f"NDCG@{k}: {np.mean(ndcg_scores):.4f}")

# Rodar validação
calcular_metrics_baseline(df, k=10, num_folds=5)
