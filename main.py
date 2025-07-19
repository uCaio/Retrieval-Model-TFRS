import pandas as pd
import tensorflow as tf
import tensorflow_recommenders as tfrs
from sklearn.model_selection import train_test_split
import time

# === PARÂMETRO DE TESTE RÁPIDO ===
modo_teste_rapido = True  

# === 1. Carregamento dos dados do MovieLens 20M ===
df = pd.read_csv("ratings.csv")
movies_df = pd.read_csv("movies.csv")

# Conversão para string
df["userId"] = df["userId"].astype(str)
df["movieId"] = df["movieId"].astype(str)
movies_df["movieId"] = movies_df["movieId"].astype(str)

# Merge para obter os títulos
df = df.merge(movies_df[["movieId", "title"]], on="movieId")
df = df[["userId", "title"]].rename(columns={"userId": "user_id", "title": "movie_title"})

# Remover dados duplicados
df = df.drop_duplicates()

# Se estiver em modo teste, pega apenas uma amostra representativa
if modo_teste_rapido:
    df = df.sample(frac=0.01, random_state=42)  # 1% dos dados (~200 mil interações)

# Conversão para TensorFlow Dataset
ratings_ds = tf.data.Dataset.from_tensor_slices({
    "user_id": tf.constant(df["user_id"].values),
    "movie_title": tf.constant(df["movie_title"].values),
})

# Listas únicas
unique_movie_titles = df["movie_title"].unique().tolist()
unique_user_ids = df["user_id"].unique().tolist()

# === 2. Dataset de treino, validação e teste ===
shuffled = ratings_ds.shuffle(buffer_size=len(df), seed=42)

train_size = int(0.8 * len(df))
val_size = int(0.1 * len(df))

train = shuffled.take(train_size)
val = shuffled.skip(train_size).take(val_size)
test = shuffled.skip(train_size + val_size)

batch_size = 2048 if modo_teste_rapido else 8192

cached_train = train.shuffle(100_000).batch(batch_size).cache().prefetch(tf.data.AUTOTUNE)
cached_val = val.batch(4096).cache().prefetch(tf.data.AUTOTUNE)
cached_test = test.batch(4096).cache().prefetch(tf.data.AUTOTUNE)

# === 3. Definição dos modelos ===
class MovieModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.embedding = tf.keras.Sequential([
            tf.keras.layers.StringLookup(vocabulary=unique_movie_titles, mask_token=None),
            tf.keras.layers.Embedding(len(unique_movie_titles) + 1, 128)
        ])
    def call(self, titles):
        return self.embedding(titles)

class UserModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.embedding = tf.keras.Sequential([
            tf.keras.layers.StringLookup(vocabulary=unique_user_ids, mask_token=None),
            tf.keras.layers.Embedding(len(unique_user_ids) + 1, 128)
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
                candidates=tf.data.Dataset.from_tensor_slices(unique_movie_titles).batch(128).map(
                    lambda title: (title, movie_model(title))
                )
            )
        )
    def compute_loss(self, features, training=False):
        user_embeddings = self.user_model(features["user_id"])
        movie_embeddings = self.movie_model(features["movie_title"])
        return self.task(user_embeddings, movie_embeddings)

# === 4. Treinamento ===
model = RetrievalModel(UserModel(), MovieModel())
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))

print("\n⏳ Iniciando treinamento do modelo...\n")
inicio_treinamento = time.time()

model.fit(cached_train, validation_data=cached_val, epochs=10)

fim_treinamento = time.time()
duracao = fim_treinamento - inicio_treinamento
minutos = int(duracao // 60)
segundos = int(duracao % 60)

print(f"\n✅ Treinamento concluído em {minutos} minutos e {segundos} segundos.\n")

# === 5. MENU PARA NOVO USUÁRIO ===
print("\n📥 MENU DE SIMULAÇÃO PARA UM NOVO USUÁRIO")
print("Digite filmes que você gosta e o sistema recomendará outros parecidos.\n")

# === Criar índice com query_model=None (pois estamos usando embeddings já calculados)
brute_force_index = tfrs.layers.factorized_top_k.BruteForce(query_model=None)
brute_force_index.index_from_dataset(
    tf.data.Dataset.from_tensor_slices(unique_movie_titles).batch(128).map(
        lambda title: (title, model.movie_model(title))
    )
)

while True:
    entrada = input("Digite os filmes que você gostou (separados por vírgula), ou 'sair' para encerrar:\n> ")
    if entrada.lower() == "sair":
        print("👋 Encerrando sistema de recomendação.")
        break

    filmes_digitados = [f.strip() for f in entrada.split(",") if f.strip()]
    filmes_validos = []

    for query in filmes_digitados:
        matches = [t for t in unique_movie_titles if query.lower() in t.lower()]
        filmes_validos.extend(matches)

    filmes_validos = list(set(filmes_validos))

    if not filmes_validos:
        print("⚠️ Nenhum dos filmes foi encontrado.")
        continue

    print("\n✔️ Filmes encontrados:")
    for t in filmes_validos:
        print(" -", t)

    embeddings_filmes = model.movie_model(tf.constant(filmes_validos))
    embedding_usuario_novo = tf.reduce_mean(embeddings_filmes, axis=0)

    _, titulos_recomendados = brute_force_index(tf.expand_dims(embedding_usuario_novo, axis=0))

    print("\n🎬 Recomendações:\n")
    for i, titulo in enumerate(titulos_recomendados[0][:5].numpy(), start=1):
        print(f"{i}. {titulo.decode('utf-8')}")
    print("\n" + "-" * 50 + "\n")
