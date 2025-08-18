import pandas as pd
import tensorflow as tf
import tensorflow_recommenders as tfrs
import numpy as np
import argparse
import time
import re

# === 1. Argumentos CLI ===
parser = argparse.ArgumentParser(description="Testar modelo Two-Tower com parametros variaveis")
parser.add_argument("--epochs", type=int, default=5)
parser.add_argument("--embedding_size", type=int, default=64)
parser.add_argument("--optimizer", type=str, default="Adagrad")
parser.add_argument("--top_k", type=int, default=10)
parser.add_argument("--rodadas", type=int, default=3)
args = parser.parse_args()

# === 2. Carregar os dados ===
df = pd.read_csv("ratings.csv")
movies_df = pd.read_csv("movies.csv")

# Conversoes
df["userId"] = df["userId"].astype(str)
df["movieId"] = df["movieId"].astype(str)
movies_df["movieId"] = movies_df["movieId"].astype(str)

# Extrair ano e separar primeiro gênero
movies_df["year"] = movies_df["title"].str.extract(r"\((\d{4})\)")
movies_df["year"] = movies_df["year"].fillna("unknown")
movies_df["genre"] = movies_df["genres"].str.split("|").str[0].fillna("unknown")

# Juntar com nomes dos filmes
df = df.merge(movies_df[["movieId", "title", "year", "genre"]], on="movieId")
df = df[["userId", "title", "year", "genre"]].rename(columns={"userId": "user_id", "title": "movie_title"})
df = df.drop_duplicates()

# === 3. Loop de Avaliação com 3 rodadas de 500 usuários ===
metricas_finais = []

for rodada in range(args.rodadas):
    print(f"\n🔁 Rodada {rodada + 1}/{args.rodadas}...")
    
    # Amostragem de 500 usuários únicos
    unique_users = df["user_id"].unique()
    np.random.seed(rodada * 42)  # garantir variação entre rodadas
    np.random.shuffle(unique_users)
    selected_users = set(unique_users[:500])

    rodada_df = df[df["user_id"].isin(selected_users)]

    # Divisao por usuario
    unique_users_r = rodada_df["user_id"].unique()
    np.random.shuffle(unique_users_r)
    n_users = len(unique_users_r)
    train_users = set(unique_users_r[:int(0.8 * n_users)])
    val_users = set(unique_users_r[int(0.8 * n_users):int(0.9 * n_users)])
    test_users = set(unique_users_r[int(0.9 * n_users):])

    train_df = rodada_df[rodada_df["user_id"].isin(train_users)]
    val_df = rodada_df[rodada_df["user_id"].isin(val_users)]
    test_df = rodada_df[rodada_df["user_id"].isin(test_users)]

    def df_to_dataset(data):
        return tf.data.Dataset.from_tensor_slices({
            "user_id": tf.constant(data["user_id"].values),
            "movie_title": tf.constant(data["movie_title"].values),
            "genre": tf.constant(data["genre"].values),
            "year": tf.constant(data["year"].values)
        })

    ratings_train_ds = df_to_dataset(train_df)
    ratings_val_ds = df_to_dataset(val_df)
    ratings_test_ds = df_to_dataset(test_df)

    unique_movie_titles = rodada_df["movie_title"].unique().tolist()
    unique_user_ids = rodada_df["user_id"].unique().tolist()
    unique_genres = rodada_df["genre"].unique().tolist()
    unique_years = rodada_df["year"].unique().tolist()

    class MovieModel(tf.keras.Model):
        def __init__(self):
            super().__init__()
            self.title_embedding = tf.keras.Sequential([
                tf.keras.layers.StringLookup(vocabulary=unique_movie_titles, mask_token=None),
                tf.keras.layers.Embedding(len(unique_movie_titles) + 1, args.embedding_size)
            ])
            self.genre_embedding = tf.keras.Sequential([
                tf.keras.layers.StringLookup(vocabulary=unique_genres, mask_token=None),
                tf.keras.layers.Embedding(len(unique_genres) + 1, args.embedding_size // 2)
            ])
            self.year_embedding = tf.keras.Sequential([
                tf.keras.layers.StringLookup(vocabulary=unique_years, mask_token=None),
                tf.keras.layers.Embedding(len(unique_years) + 1, args.embedding_size // 2)
            ])
            self.concat = tf.keras.layers.Concatenate()

        def call(self, features):
            title_vec = self.title_embedding(features["movie_title"])
            genre_vec = self.genre_embedding(features["genre"])
            year_vec = self.year_embedding(features["year"])
            return self.concat([title_vec, genre_vec, year_vec])

    class UserModel(tf.keras.Model):
        def __init__(self):
            super().__init__()
            total_dim = args.embedding_size + 2 * (args.embedding_size // 2)
            self.embedding = tf.keras.Sequential([
                tf.keras.layers.StringLookup(vocabulary=unique_user_ids, mask_token=None),
                tf.keras.layers.Embedding(len(unique_user_ids) + 1, total_dim)
            ])
        def call(self, user_ids):
            return self.embedding(user_ids)

    class RetrievalModel(tfrs.models.Model):
        def __init__(self, user_model, movie_model):
            super().__init__()
            self.movie_model = movie_model
            self.user_model = user_model
            self.task = tfrs.tasks.Retrieval(
                metrics=tfrs.metrics.FactorizedTopK(
                    candidates=tf.data.Dataset.from_tensor_slices({
                        "movie_title": unique_movie_titles,
                        "genre": [rodada_df[rodada_df["movie_title"] == t]["genre"].values[0] for t in unique_movie_titles],
                        "year": [rodada_df[rodada_df["movie_title"] == t]["year"].values[0] for t in unique_movie_titles],
                    }).batch(128).map(lambda x: (x["movie_title"], movie_model(x)))
                )
            )

        def compute_loss(self, features, training=False):
            user_embeddings = self.user_model(features["user_id"])
            movie_embeddings = self.movie_model(features)
            return self.task(user_embeddings, movie_embeddings)

    model = RetrievalModel(UserModel(), MovieModel())
    optimizer = tf.keras.optimizers.Adagrad(0.01) if args.optimizer == "Adagrad" else tf.keras.optimizers.Adam(0.001)
    model.compile(optimizer=optimizer)

    cached_train = ratings_train_ds.shuffle(100_000).batch(2048).cache().prefetch(tf.data.AUTOTUNE)
    cached_val = ratings_val_ds.batch(4096).cache().prefetch(tf.data.AUTOTUNE)
    cached_test = ratings_test_ds.batch(4096).cache().prefetch(tf.data.AUTOTUNE)

    start = time.time()
    model.fit(cached_train, validation_data=cached_val, epochs=args.epochs, verbose=0)
    end = time.time()
    duration = round(end - start, 2)

    brute_force = tfrs.layers.factorized_top_k.BruteForce(model.user_model)
    brute_force.index_from_dataset(
        tf.data.Dataset.from_tensor_slices({
            "movie_title": unique_movie_titles,
            "genre": [rodada_df[rodada_df["movie_title"] == t]["genre"].values[0] for t in unique_movie_titles],
            "year": [rodada_df[rodada_df["movie_title"] == t]["year"].values[0] for t in unique_movie_titles],
        }).batch(128).map(lambda x: (x["movie_title"], model.movie_model(x)))
    )

    precisions, hrs, ndcgs = [], [], []
    for user_id in test_df["user_id"].unique():
        user_movies = [m.strip().lower() for m in test_df[test_df["user_id"] == user_id]["movie_title"].tolist()]
        if not user_movies:
            continue
        _, recommended_movies = brute_force(tf.constant([user_id]))
        recommended_movies = [m.decode('utf-8').strip().lower() for m in recommended_movies[0][:args.top_k].numpy()]
        hits = sum([1 for m in user_movies if m in recommended_movies])
        hr = 1.0 if hits > 0 else 0.0
        precision = hits / args.top_k
        ndcg = sum([1.0 / tf.math.log(tf.cast(i+2, tf.float32)).numpy()
                    for i, m in enumerate(recommended_movies) if m in user_movies])
        precisions.append(precision)
        hrs.append(hr)
        ndcgs.append(ndcg)

    precision_at_k = np.mean(precisions)
    hr_at_k = np.mean(hrs)
    ndcg_at_k = np.mean(ndcgs)

    print(f"Rodada {rodada + 1}: Tempo = {duration}s | Precision@{args.top_k} = {precision_at_k:.4f} | HR@{args.top_k} = {hr_at_k:.4f} | NDCG@{args.top_k} = {ndcg_at_k:.4f}")
    metricas_finais.append((precision_at_k, hr_at_k, ndcg_at_k))

# === MÉDIAS FINAIS ===
print("\n📊 MÉTRICAS FINAIS APÓS TODAS AS RODADAS:")
final_prec = np.mean([m[0] for m in metricas_finais])
final_hr = np.mean([m[1] for m in metricas_finais])
final_ndcg = np.mean([m[2] for m in metricas_finais])
print(f"Precision@{args.top_k}: {final_prec:.4f} | HR@{args.top_k}: {final_hr:.4f} | NDCG@{args.top_k}: {final_ndcg:.4f}")
