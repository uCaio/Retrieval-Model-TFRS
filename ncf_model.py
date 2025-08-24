import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, optimizers
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import argparse
import itertools

# ---------------------------
# Funções de métricas
# ---------------------------
def precision_at_k(recommended, relevant, k):
    rec_k = recommended[:k]
    hits = len(set(rec_k) & relevant)
    return hits / k

def hit_rate_at_k(recommended, relevant, k):
    return 1 if len(set(recommended[:k]) & relevant) > 0 else 0

def ndcg_at_k(recommended, relevant, k):
    dcg = 0.0
    for i, item in enumerate(recommended[:k]):
        if item in relevant:
            dcg = 1 / np.log2(i + 2)
            break
    return dcg

# ---------------------------
# Construção do modelo NCF
# ---------------------------
def build_ncf(num_users, num_items, embedding_size=64, num_layers=2, learning_rate=1e-3, optimizer_name="adam"):
    # Entradas
    user_input = layers.Input(shape=(1,), name="user_input")
    item_input = layers.Input(shape=(1,), name="item_input")

    # Embeddings
    user_embedding_gmf = layers.Embedding(num_users, embedding_size)(user_input)
    item_embedding_gmf = layers.Embedding(num_items, embedding_size)(item_input)

    user_embedding_mlp = layers.Embedding(num_users, embedding_size)(user_input)
    item_embedding_mlp = layers.Embedding(num_items, embedding_size)(item_input)

    # GMF
    gmf = layers.Multiply()([user_embedding_gmf, item_embedding_gmf])
    gmf = layers.Flatten()(gmf)

    # MLP
    mlp = layers.Concatenate()([user_embedding_mlp, item_embedding_mlp])
    mlp = layers.Flatten()(mlp)
    for i in range(num_layers):
        mlp = layers.Dense(64, activation="relu")(mlp)

    # Combinar GMF + MLP
    concat = layers.Concatenate()([gmf, mlp])
    output = layers.Dense(1, activation="sigmoid")(concat)

    # Escolher otimizador
    if optimizer_name == "adam":
        opt = optimizers.Adam(learning_rate=learning_rate)
    elif optimizer_name == "adagrad":
        opt = optimizers.Adagrad(learning_rate=learning_rate)
    else:
        opt = optimizers.SGD(learning_rate=learning_rate)

    model = Model(inputs=[user_input, item_input], outputs=output)
    model.compile(
        optimizer=opt,
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.AUC(name="auc")]
    )
    return model

# ---------------------------
# Função de avaliação Top-K
# ---------------------------
def evaluate_topk(model, train_df, test_df, item2idx, top_k=10):
    precisions, hrs, ndcgs = [], [], []
    unique_users = test_df["userId"].unique()

    for user_idx in tqdm(unique_users, desc="🔎 Avaliando usuários"):
        seen = set(train_df[train_df["userId"] == user_idx]["movieId"].tolist())
        candidates = [i for i in item2idx.values() if i not in seen]
        if not candidates:
            continue

        user_array = np.full(len(candidates), user_idx)
        scores = model.predict([user_array, np.array(candidates)], verbose=0)

        top_indices = np.argsort(scores.reshape(-1))[::-1][:top_k]
        top_items = [candidates[i] for i in top_indices]

        relevant = set(test_df[test_df["userId"] == user_idx]["movieId"].tolist())

        precisions.append(precision_at_k(top_items, relevant, top_k))
        hrs.append(hit_rate_at_k(top_items, relevant, top_k))
        ndcgs.append(ndcg_at_k(top_items, relevant, top_k))

    return (
        np.mean(precisions), np.std(precisions, ddof=1),
        np.mean(hrs), np.std(hrs, ddof=1),
        np.mean(ndcgs), np.std(ndcgs, ddof=1)
    )

# ---------------------------
# MAIN
# ---------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--top_k", type=int, default=10)
    args = parser.parse_args()

    # 1. Carregar dados
    ratings = pd.read_csv("ratings.csv")
    movies = pd.read_csv("movies.csv")
    df = pd.merge(ratings, movies, on="movieId")[["userId", "movieId"]]

    # Selecionar 500 usuários aleatórios
    usuarios_unicos = df["userId"].unique()
    usuarios_amostrados = np.random.choice(usuarios_unicos, size=500, replace=False)
    df = df[df["userId"].isin(usuarios_amostrados)].reset_index(drop=True)

    # Mapear para índices contínuos
    user2idx = {u: i for i, u in enumerate(df["userId"].unique())}
    item2idx = {m: i for i, m in enumerate(df["movieId"].unique())}
    df["userId"] = df["userId"].map(user2idx)
    df["movieId"] = df["movieId"].map(item2idx)
    df["rating"] = 1

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    X_train = [train_df["userId"].values, train_df["movieId"].values]
    y_train = train_df["rating"].values
    X_val = [test_df["userId"].values, test_df["movieId"].values]
    y_val = test_df["rating"].values

    num_users, num_items = len(user2idx), len(item2idx)

    # 2. Estratégia B: Grid Search manual
    optimizers_list = ["adam", "adagrad"]
    embedding_sizes = [32, 64, 128]
    num_layers_list = [1, 2]
    learning_rates = [1e-3, 1e-4]

    for optimizer_name, emb_size, num_layers, lr in itertools.product(
        optimizers_list, embedding_sizes, num_layers_list, learning_rates
    ):
        print(f"\n=== Testando: Opt={optimizer_name}, Embedding={emb_size}, Layers={num_layers}, LR={lr} ===")

        model = build_ncf(
            num_users, num_items,
            embedding_size=emb_size,
            num_layers=num_layers,
            learning_rate=lr,
            optimizer_name=optimizer_name
        )

        model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=args.epochs,
            batch_size=256,
            verbose=0
        )

        p_mean, p_std, hr_mean, hr_std, ndcg_mean, ndcg_std = evaluate_topk(
            model, train_df, test_df, item2idx, args.top_k
        )

        print(f"Precision@{args.top_k}: {p_mean:.4f} ± {p_std:.4f}")
        print(f"HR@{args.top_k}: {hr_mean:.4f} ± {hr_std:.4f}")
        print(f"NDCG@{args.top_k}: {ndcg_mean:.4f} ± {ndcg_std:.4f}")


if __name__ == "__main__":
    main()
