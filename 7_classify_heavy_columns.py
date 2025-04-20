import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# ----------------------------
# Configuración
# ----------------------------
EMBEDDING_DIR = "RESULTS/SrTiO3/6_EMBED_AE_columns/"
MASK_PATH = "RESULTS/SrTiO3/heavy_columns_map_filtered.png"

OUTPUT_IMAGE = "column_types_map.png"
N_CLUSTERS = 2
LAMBDA_DECAY = 1 # cuanto más alto, más énfasis en el centro

# ----------------------------
# 1. Cargar embeddings y coords
# ----------------------------
print("📦 Cargando embeddings...")
embedding_dict = {}
for fname in tqdm(os.listdir(EMBEDDING_DIR)):
    if not fname.endswith(".npy"):
        continue
    i, j = map(int, fname.replace(".npy", "").split("_"))
    embedding = np.load(os.path.join(EMBEDDING_DIR, fname))
    embedding_dict[(i, j)] = embedding

# ----------------------------
# 2. Cargar máscara binaria
# ----------------------------
mask = cv2.imread(MASK_PATH, cv2.IMREAD_GRAYSCALE)
_, binary_mask = cv2.threshold(mask, 127, 1, cv2.THRESH_BINARY)
#erode with circular pattern 4x4

# ----------------------------
# 3. Encontrar contornos (columnas)
# ----------------------------
print("🔍 Detectando contornos...")
contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

column_descriptors = []
column_indices = []  # Para dibujar luego

for idx, contour in enumerate(contours):
    # Crear máscara para el contorno individual
    column_mask = np.zeros_like(binary_mask)
    cv2.drawContours(column_mask, [contour], -1, color=1, thickness=-1)


    # Coordenadas dentro del contorno
    ys, xs = np.where(column_mask == 1)
    coords = list(zip(ys, xs))

    # Extraer embeddings y calcular centro de masas
    embeddings = []
    coords_valid = []

    for (i, j) in coords:
        key = (i, j)
        if key in embedding_dict:
            coord = np.array([i, j])
            embeddings.append(embedding_dict[key])
            coords_valid.append(coord)

    if not embeddings:
        continue

    embeddings = np.array(embeddings)
    coords_valid = np.array(coords_valid)
    center = coords_valid.mean(axis=0)
    

    # Calcular distancia y pesos exponenciales
    dists = np.linalg.norm(coords_valid - center, axis=1)
    weights = np.exp(-LAMBDA_DECAY * dists)
    weights /= weights.sum()

    # Media ponderada del embedding
    weighted_embedding = np.average(embeddings, axis=0, weights=weights)

    column_descriptors.append(weighted_embedding)
    column_indices.append(idx)

# ----------------------------
# 4. Clustering de columnas
# ----------------------------
print("🧠 Clustering de columnas pesadas (K=2)...")
column_descriptors = np.array(column_descriptors)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(column_descriptors)
X_scaled=column_descriptors

kmeans = KMeans(n_clusters=N_CLUSTERS, init='k-means++', random_state=42)
cluster_labels = kmeans.fit_predict(X_scaled)

# ----------------------------
# 5. Crear mapa final de tipos de columnas
# ----------------------------
final_map = np.full_like(binary_mask, fill_value=-1, dtype=np.int32)
try:
    for idx, contour in zip(column_indices, contours):
        label = cluster_labels[idx]
        cv2.drawContours(final_map, [contour], -1, color=int(label), thickness=-1)
except:
    print('what happened?')

# Normalizar a valores 0-1 para visualizar
final_map_vis = np.zeros_like(final_map, dtype=np.uint8)
final_map_vis[final_map == 0] = 100
final_map_vis[final_map == 1] = 200

# Guardar resultado
cv2.imwrite(OUTPUT_IMAGE, final_map_vis)
print(f"✅ Imagen guardada como: {OUTPUT_IMAGE}")

# Mostrar
plt.figure(figsize=(6, 6))
plt.imshow(final_map_vis, cmap="tab10")
plt.title(f"Tipos de columnas pesadas (K={N_CLUSTERS})")
plt.axis("off")
plt.tight_layout()
plt.show()
from sklearn.manifold import TSNE

# ----------------------------
# 6. Visualización t-SNE
# ----------------------------
print("🎨 Generando t-SNE de los descriptores de columna...")

tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_2d = tsne.fit_transform(X_scaled)

plt.figure(figsize=(6, 5))
scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1],
                      c=cluster_labels, cmap="tab10", s=30, alpha=0.9)
plt.title("t-SNE de columnas pesadas (colores = clusters)")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.colorbar(scatter, label="Cluster ID")
plt.tight_layout()
plt.savefig("tsne_column_types.png", dpi=300)
plt.show()

print("📌 t-SNE guardado como 'tsne_column_types.png'")
