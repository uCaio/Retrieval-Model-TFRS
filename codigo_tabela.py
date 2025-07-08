import pandas as pd
import matplotlib.pyplot as plt

# Ler o arquivo
df = pd.read_excel("Testes_Modelo_Recomendacao.xlsx")

# Converter colunas numéricas corretamente
df["Precisão Top-10"] = pd.to_numeric(df["Precisão Top-10"], errors='coerce')
df["Precisão Top-50"] = pd.to_numeric(df["Precisão Top-50"], errors='coerce')
df["Precisão Top-100"] = pd.to_numeric(df["Precisão Top-100"], errors='coerce')
df["Tempo de Treinamento (s)"] = df["Tempo de Treinamento (s)"].astype(str).str.replace(" segundos", "").astype(float)
df["Épocas"] = pd.to_numeric(df["Épocas"], errors='coerce')
df["Tamanho do Embedding"] = pd.to_numeric(df["Tamanho do Embedding"], errors='coerce')

# Gráfico: Precisão Top-10 por Épocas, separado por Tamanho do Embedding
plt.figure(figsize=(9,6))
for emb, grupo in df.groupby("Tamanho do Embedding"):
    plt.plot(grupo["Épocas"], grupo["Precisão Top-10"], marker='o', label=f"Embedding {emb}")

plt.title("Precisão Top-10 por Número de Épocas (por Tamanho de Embedding)")
plt.xlabel("Épocas")
plt.ylabel("Precisão Top-10")
plt.legend(title="Tamanho do Embedding")
plt.grid(True)
plt.tight_layout()
plt.show()

# Gráfico: Precisão Top-100 vs Tempo de Treinamento por Tamanho do Embedding
plt.figure(figsize=(9,6))
for emb, grupo in df.groupby("Tamanho do Embedding"):
    plt.plot(grupo["Tempo de Treinamento (s)"], grupo["Precisão Top-100"], marker='s', label=f"Embedding {emb}")

plt.title("Precisão Top-100 vs Tempo de Treinamento (por Tamanho de Embedding)")
plt.xlabel("Tempo de Treinamento (s)")
plt.ylabel("Precisão Top-100")
plt.legend(title="Tamanho do Embedding")
plt.grid(True)
plt.tight_layout()
plt.show()

# Gráfico: Precisão Top-50 por Épocas e Otimizador, colorido por Tamanho do Embedding
plt.figure(figsize=(9,6))
for (opt, emb), grupo in df.groupby(["Otimizador", "Tamanho do Embedding"]):
    label = f"{opt} - Embedding {emb}"
    plt.plot(grupo["Épocas"], grupo["Precisão Top-50"], marker='o', label=label)

plt.title("Precisão Top-50 por Épocas (Otimizador + Embedding)")
plt.xlabel("Épocas")
plt.ylabel("Precisão Top-50")
plt.legend(title="Configuração")
plt.grid(True)
plt.tight_layout()
plt.show()
