import numpy as np

# Substitua pelos seus valores coletados dos testes
precision_scores = [0.500, 0.0532, 0.0644, 0.0580, 0.0667, 0.0600]  # seus 6 valores de Precision@10
hr_scores = [0.2360, 0.2400, 0.2680, 0.2400, 0.2867, 0.2533]         # seus 6 valores de HR@10
ndcg_scores = [0.3490, 0.3516, 0.4562, 0.4239, 0.4716, 0.4008]       # seus 6 valores de NDCG@10

print("Precision@10 -> média:", np.mean(precision_scores), "desvio padrão:", np.std(precision_scores, ddof=1))
print("HR@10 -> média:", np.mean(hr_scores), "desvio padrão:", np.std(hr_scores, ddof=1))
print("NDCG@10 -> média:", np.mean(ndcg_scores), "desvio padrão:", np.std(ndcg_scores, ddof=1))
