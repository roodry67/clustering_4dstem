import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans , SpectralClustering
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from tqdm import tqdm

# Configuración
EMBEDDING_DIR = "RESULTS/MoS2/3_EMBED_AE/"
EMBEDDING_DIR = "RESULTS/SrTiO3/6_EMBED_AE_columns/"
N_CLUSTERS = 2
SAVE_PLOT = True
PLOT_NAME = "cluster_map.png"
CLUSTER ='SPECTRAL'
CLUSTER='KMEANS'

# 1. Cargar embeddings y coordenadas
embeddings = []
coords = []

print("\U0001F4E5 Cargando embeddings...")
for fname in tqdm(sorted(os.listdir(EMBEDDING_DIR))):
    if not fname.endswith(".npy"):
        continue
    path = os.path.join(EMBEDDING_DIR, fname)
    emb = np.load(path)
    embeddings.append(emb)

    i, j = map(int, fname.replace(".npy", "").split("_"))
    coords.append((i, j))

embeddings = np.array(embeddings)
coords = np.array(coords)

# 2. Normalizar embeddings
print("\U0001F9EA Estandarizando embeddings...")
scaler = StandardScaler()
embeddings_scaled = scaler.fit_transform(embeddings)
embeddings_scaled=embeddings
# 3. Clustering con KMeans
print("\U0001F50D Ejecutando KMeans...")
if CLUSTER == 'KMEANS':
    cluster_alg = KMeans(n_clusters=N_CLUSTERS, init='k-means++', random_state=42)
elif CLUSTER=='SPECTRAL':
    cluster_alg= SpectralClustering(n_clusters=N_CLUSTERS)
cluster_ids = cluster_alg.fit_predict(embeddings_scaled)

# 4. Crear mapa de clusters
max_i = coords[:, 0].max() + 1
max_j = coords[:, 1].max() + 1
cluster_map = np.full((max_i, max_j), -1)

for (i, j), cluster_id in zip(coords, cluster_ids):
    cluster_map[i, j] = cluster_id

plt.figure(figsize=(8, 8))
plt.imshow(cluster_map, cmap='tab10', interpolation='nearest')
plt.title(f"Mapa de clusters (K={N_CLUSTERS})")
plt.xlabel("j (columna)")
plt.ylabel("i (fila)")
plt.colorbar(label="ID de cluster")
plt.tight_layout()

if SAVE_PLOT:
    plt.savefig(PLOT_NAME, dpi=300)
    print(f"\U0001F5BC️ Mapa guardado en: {PLOT_NAME}")
else:
    plt.show()
# 5. t-SNE plot

exit(0)
print("\U0001F3A8 Generando t-SNE...")
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
embeddings_2d = tsne.fit_transform(embeddings_scaled)

plt.figure(figsize=(8, 6))
scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                      c=cluster_ids, cmap='tab10', s=10, alpha=0.8)
plt.title("t-SNE de embeddings escalados con colores de cluster")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.colorbar(scatter, label="ID de cluster")
plt.tight_layout()

plt.savefig("tsne_clusters.png", dpi=300)
print("\U0001F5BC️ t-SNE guardado como tsne_clusters.png")
