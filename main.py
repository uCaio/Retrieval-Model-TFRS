import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_recommenders as tfrs
import time

# 1. Carregar e pré-processar os dados
ratings = tfds.load("movielens/100k-ratings", split="train")
movies = tfds.load("movielens/100k-movies", split="train")

ratings = ratings.map(
    lambda x: {
        "movie_title": tf.reshape(x["movie_title"], []),
        "user_id": tf.reshape(x["user_id"], []),
    }
)

movies = movies.map(lambda x: {"movie_title": tf.reshape(x["movie_title"], [])})

# 2. Divisão dos dados (80% treino, 10% val, 10% teste)
total_size = 100_000
train_size = int(0.8 * total_size)
val_size = int(0.1 * total_size)

shuffled = ratings.shuffle(total_size, seed=42, reshuffle_each_iteration=False)
train = shuffled.take(train_size)
val = shuffled.skip(train_size).take(val_size)
test = shuffled.skip(train_size + val_size)

train = train.flat_map(lambda x: tf.data.Dataset.from_tensors(x))
val = val.flat_map(lambda x: tf.data.Dataset.from_tensors(x))
test = test.flat_map(lambda x: tf.data.Dataset.from_tensors(x))

cached_train = train.batch(8192).cache()
cached_val = val.batch(4096).cache()
cached_test = test.batch(4096).cache()

# 3. Vocabulários
movie_titles = movies.map(lambda x: x["movie_title"])
user_ids = ratings.map(lambda x: x["user_id"])

unique_movie_titles = list(set(title.numpy().decode("utf-8") for title in movie_titles))
unique_user_ids = list(set(uid.numpy().decode("utf-8") for uid in user_ids))

# 4. Modelos
class MovieModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.embedding = tf.keras.Sequential([
            tf.keras.layers.StringLookup(vocabulary=unique_movie_titles, mask_token=None),
            tf.keras.layers.Embedding(len(unique_movie_titles) + 1, 128),
        ])
    def call(self, titles):
        return self.embedding(titles)

class UserModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.embedding = tf.keras.Sequential([
            tf.keras.layers.StringLookup(vocabulary=unique_user_ids, mask_token=None),
            tf.keras.layers.Embedding(len(unique_user_ids) + 1, 128),
        ])
    def call(self, user_ids):
        return self.embedding(user_ids)

class MovieRetrievalModel(tfrs.models.Model):
    def __init__(self, user_model, movie_model):
        super().__init__()
        self.movie_model = movie_model
        self.user_model = user_model
        self.task = tfrs.tasks.Retrieval(
            metrics=tfrs.metrics.FactorizedTopK(
                candidates=movies.batch(128).map(
                    lambda x: (x["movie_title"], movie_model(x["movie_title"]))
                )
            )
        )
    def compute_loss(self, features, training=False):
        user_embeddings = self.user_model(features["user_id"])
        movie_embeddings = self.movie_model(features["movie_title"])
        return self.task(user_embeddings, movie_embeddings)
    
start_time = time.time()

# 5. Treinamento
model = MovieRetrievalModel(UserModel(), MovieModel())
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
model.fit(cached_train, validation_data=cached_val, epochs=10)

training_time = time.time() - start_time
print(f"⏱️ Tempo de treinamento: {training_time:.2f} segundos")

# 6. Menu Interativo para usuários novos
print("\n📥 MENU DE SIMULAÇÃO PARA UM NOVO USUÁRIO")
print("Digite filmes que você gosta e o sistema recomendará outros parecidos.")
print("Ex: 'Toy Story', 'Matrix', 'Aladdin'.\n")

while True:
    entrada = input("Digite os filmes que você gostou (separados por vírgula), ou 'sair' para encerrar:\n> ")

    if entrada.lower() == "sair":
        print("\nEncerrando o sistema de recomendações. 👋")
        break

    filmes_digitados = [f.strip() for f in entrada.split(",") if f.strip()]
    filmes_validos = []

    # Permitir busca parcial (como "toy", "matrix")
    for query in filmes_digitados:
        matches = [t for t in unique_movie_titles if query.lower() in t.lower()]
        filmes_validos.extend(matches)

    filmes_validos = list(set(filmes_validos))

    if not filmes_validos:
        print("⚠️ Nenhum dos filmes digitados foi reconhecido no vocabulário. Tente novamente.\n")
        continue

    print("\n✔️ Filmes encontrados:")
    for t in filmes_validos:
        print(" -", t)

    # Gerar embedding médio do novo usuário
    embeddings_filmes = model.movie_model(tf.constant(filmes_validos))
    embedding_usuario_novo = tf.reduce_mean(embeddings_filmes, axis=0)

    # ── CORREÇÃO AQUI ──
    # Usar modelo de identidade como query_model
    identity_query = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(128,)),
        tf.keras.layers.Lambda(lambda x: x)
    ])

    # Criar o índice
    brute_force_index = tfrs.layers.factorized_top_k.BruteForce(query_model=identity_query)
    brute_force_index.index_from_dataset(
        movies.batch(100).map(
            lambda x: (x["movie_title"], model.movie_model(x["movie_title"]))
        )
    )

    # Fazer a recomendação
    _, titulos_recomendados = brute_force_index(tf.expand_dims(embedding_usuario_novo, axis=0))

    print("\n🎬 Recomendações baseadas nos filmes informados:\n")
    for i, titulo in enumerate(titulos_recomendados[0, :6].numpy(), start=1):
        print(f"{i}. {titulo.decode('utf-8')}")
    print("\n" + "-" * 50 + "\n")
