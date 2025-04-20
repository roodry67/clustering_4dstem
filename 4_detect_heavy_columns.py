import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from collections import Counter
from tqdm import tqdm

# ----------------------------
# Configuración
# ----------------------------
MAT="SrTiO3"
EMBEDDING_DIR = "RESULTS/"+MAT+"/3_EMBED_AE/"
N_CLUSTERS = 2
SAVE_MASK = True
SAVE_IMAGE = True
IMAGE_FILENAME_RAW = "RESULTS/"+MAT+"/heavy_columns_map.png"

# ----------------------------
# 1. Cargar embeddings y coord.
# ----------------------------
print("\U0001F4E5 Cargando embeddings...")
embeddings = []
coords = []

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

# ----------------------------
# 2. Escalar los embeddings
# ----------------------------
scaler = StandardScaler()
embeddings_scaled = scaler.fit_transform(embeddings)
embeddings_scaled = embeddings

# ----------------------------
# 3. Clustering (K=2)
# ----------------------------
print("\U0001F50D Ejecutando KMeans con K=2...")
kmeans = KMeans(n_clusters=N_CLUSTERS, init='k-means++', random_state=42)
cluster_ids = kmeans.fit_predict(embeddings_scaled)

# ----------------------------
# 4. Crear mapa de clusters
# ----------------------------
max_i = coords[:, 0].max() + 1
max_j = coords[:, 1].max() + 1
cluster_map = np.full((max_i, max_j), -1)

for (i, j), cluster_id in zip(coords, cluster_ids):
    cluster_map[i, j] = cluster_id

# ----------------------------
# 5. Detectar cluster de columnas (ignorando -1)
# ----------------------------
flattened = cluster_map.flatten()
valid_labels = flattened[flattened != -1]
counts = Counter(valid_labels)

print("Conteo de etiquetas (sin -1):", counts)

column_cluster = min(counts, key=counts.get)
print(f"🧱 Asignando cluster '{column_cluster}' como columnas pesadas (valor 1 en la máscara)")

# ----------------------------
# 6. Generar máscara binaria
# ----------------------------
column_mask = np.zeros_like(cluster_map, dtype=np.uint8)
column_mask[cluster_map == column_cluster] = 1

# Guardar máscara antes de limpiar
if SAVE_IMAGE:
    plt.imsave(IMAGE_FILENAME_RAW, column_mask, cmap="gray")
    print(f"🖼️ Imagen sin limpiar guardada como '{IMAGE_FILENAME_RAW}'")

