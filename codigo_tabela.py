import pandas as pd
import matplotlib.pyplot as plt

# Ler o arquivo
df = pd.read_excel("Testes_Modelo_Recomendacao.xlsx")

# Converter colunas numéricas corretamente
df["Precisão Top-10"] = pd.to_numeric(df["Precisão Top-10"], errors='coerce')
df["Precisão Top-50"] = pd.to_numeric(df["Precisão Top-50"], errors='coerce')
df["Precisão Top-100"] = pd.to_numeric(df["Precisão Top-100"], errors='coerce')
df["Tempo de Treinamento (s)"] = pd.to_numeric(
    df["Tempo de Treinamento (s)"].astype(str).str.replace(" Segundos", ""),
    errors='coerce'
)
df["Épocas"] = pd.to_numeric(df["Épocas"], errors='coerce')
df["Tamanho do Embedding"] = pd.to_numeric(df["Tamanho do Embedding"], errors='coerce')

# Métricas a iterar
tops = {
    "Precisão Top-10": "Top-10",
    "Precisão Top-50": "Top-50",
    "Precisão Top-100": "Top-100"
}

# 1. Precisão Top-x por Épocas (por Tamanho do Embedding)
for col, label in tops.items():
    plt.figure(figsize=(9, 6))
    for emb, grupo in df.groupby("Tamanho do Embedding"):
        plt.plot(grupo["Épocas"], grupo[col], marker='o', label=f"Embedding {emb}")
    plt.title(f"{label} por Número de Épocas (por Tamanho de Embedding)")
    plt.xlabel("Épocas")
    plt.ylabel(label)
    plt.legend(title="Tamanho do Embedding")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# 2. Precisão Top-x vs Tempo de Treinamento (por Tamanho do Embedding)
for col, label in tops.items():
    plt.figure(figsize=(9, 6))
    for emb, grupo in df.groupby("Tamanho do Embedding"):
        plt.plot(grupo["Tempo de Treinamento (s)"], grupo[col], marker='s', label=f"Embedding {emb}")
    plt.title(f"{label} vs Tempo de Treinamento (por Tamanho de Embedding)")
    plt.xlabel("Tempo de Treinamento (s)")
    plt.ylabel(label)
    plt.legend(title="Tamanho do Embedding")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# 3. Precisão Top-x por Épocas e Otimizador (colorido por Tamanho do Embedding)
for col, label in tops.items():
    plt.figure(figsize=(9, 6))
    for (opt, emb), grupo in df.groupby(["Otimizador", "Tamanho do Embedding"]):
        linha = f"{opt} - Embedding {emb}"
        plt.plot(grupo["Épocas"], grupo[col], marker='D', label=linha)
    plt.title(f"{label} por Épocas (Otimizador + Embedding)")
    plt.xlabel("Épocas")
    plt.ylabel(label)
    plt.legend(title="Configuração")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
