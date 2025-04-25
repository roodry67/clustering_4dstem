# ==== Hiperparámetros ====
MAT="MoS2"
MAT="SrTiO3"
MAT="GaN"

folder_path = "RESULTS/"+MAT+"/2_DATA_pca/"
folder_path = "RESULTS/"+MAT+"/1_DATA/"
n_components = 3
n_clusters = 2
n_bins_radial_profile = 185
cut_radial_profile_at = 170

# ==== Carga de datos ====
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import center_of_mass
from sklearn.cluster import AgglomerativeClustering, KMeans
import umap
from sklearn.preprocessing import RobustScaler
from collections import Counter

def load_diffraction_patterns(folder_path):
    patterns = []
    coordinates = []
    for fname in sorted(os.listdir(folder_path)):
        if fname.endswith('.npy'):
            path = os.path.join(folder_path, fname)
            pattern = np.load(path)
            name = os.path.splitext(fname)[0]
            i, j = map(int, name.split('_'))

            patterns.append(pattern)
            coordinates.append((i, j))
    return patterns, np.array(coordinates)

# ==== Cálculo de descriptores ====
def radial_profile(image, center=None, n_bins=n_bins_radial_profile):
    y, x = np.indices(image.shape)
    if center is None:
        center = center_of_mass(image)
    r = np.sqrt((x - center[1])**2 + (y - center[0])**2)
    r = r.astype(int)
    tbin = np.bincount(r.ravel(), image.ravel(), minlength=n_bins)
    nr = np.bincount(r.ravel(), minlength=n_bins)
    radial_prof = tbin / np.maximum(nr, 1)
    return radial_prof[:cut_radial_profile_at]  # recortar perfil

def compute_descriptors(patterns):
    descriptors = []
    for pattern in patterns:
        profile = radial_profile(pattern)
        descriptors.append(profile)
    return np.stack(descriptors)

# ==== Selección y concatenación de descriptores ====
def prepare_embedding(descriptors):
    scaler = RobustScaler()
    descriptors_scaled = scaler.fit_transform(descriptors)
    reducer = umap.UMAP(n_components=n_components)
    embeddings = reducer.fit_transform(descriptors_scaled)
    return embeddings

# ==== Clustering ====
def perform_clustering(embeddings):
    clustering_model = AgglomerativeClustering(n_clusters=n_clusters, linkage='complete')
    #clustering_model = KMeans(n_clusters=n_clusters, random_state=42)
    labels = clustering_model.fit_predict(embeddings)
    return labels

def create_cluster_map(labels, coordinates):
    i_vals = coordinates[:, 0]
    j_vals = coordinates[:, 1]
    map_shape = (i_vals.max() + 1, j_vals.max() + 1)
    cluster_map = np.full(map_shape, -1, dtype=int)
    for (i, j), label in zip(coordinates, labels):
        cluster_map[i, j] = label
    return cluster_map

# ==== Pipeline Principal ====
patterns, coordinates = load_diffraction_patterns(folder_path)
descriptors = compute_descriptors(patterns)
embeddings = prepare_embedding(descriptors)
labels = perform_clustering(embeddings)
cluster_map = create_cluster_map(labels, coordinates)

# ==== Visualización ====
plt.figure(figsize=(6, 5))
plt.imshow(cluster_map, cmap="tab10")
plt.title("Mapa de Clusters de Difractogramas")
plt.xlabel("j")
plt.ylabel("i")
plt.colorbar(label="Cluster")
plt.tight_layout()
plt.show()
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
plt.imsave("RESULTS/"+MAT+"/heavy_cols_mask.png", column_mask, cmap="gray")
print(f"🖼️ Imagen sin limpiar guardada como 'heavy_cols_mask.png'")