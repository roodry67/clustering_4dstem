# classify_heavy_columns.py

# ==== Configuración ====
MAT = "SrTiO3"
MAT = "GaN"

folder_path = "RESULTS/" + MAT + "/1_DATA/"
mask_path = "RESULTS/" + MAT + "/heavy_columns_map_filtered.png"
n_components = 4
n_clusters = 3
n_bins_radial_profile = 185
cut_radial_profile_at = 170

# ==== Imports ====
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.ndimage import center_of_mass
from sklearn.cluster import AgglomerativeClustering, KMeans
import umap
from sklearn.preprocessing import RobustScaler
from collections import Counter

# ==== Funciones Auxiliares ====
def load_diffraction_patterns(folder_path):
    patterns = []
    coordinates = []
    for fname in sorted(os.listdir(folder_path)):
        if fname.endswith('.npy'):
            path = os.path.join(folder_path, fname)
            pattern = np.load(path)
            name = os.path.splitext(fname)[0]
            i, j = map(int, name.split('_'))
            if i == 0 or j == 0:
                continue
            if i > 80 or j > 80:
                continue
            patterns.append(pattern)
            coordinates.append((i, j))
    return patterns, np.array(coordinates)

def radial_profile(image, center=None, n_bins=n_bins_radial_profile):
    y, x = np.indices(image.shape)
    if center is None:
        center = center_of_mass(image)
    r = np.sqrt((x - center[1])**2 + (y - center[0])**2)
    r = r.astype(int)
    tbin = np.bincount(r.ravel(), image.ravel(), minlength=n_bins)
    nr = np.bincount(r.ravel(), minlength=n_bins)
    radial_prof = tbin / np.maximum(nr, 1)
    return radial_prof[:cut_radial_profile_at]

def compute_descriptors(patterns):
    descriptors = []
    for pattern in patterns:
        profile = radial_profile(pattern)
        descriptors.append(profile)
    return np.stack(descriptors)

def prepare_embedding(descriptors):
    scaler = RobustScaler()
    descriptors_scaled = scaler.fit_transform(descriptors)
    reducer = umap.UMAP(n_components=n_components)
    embeddings = reducer.fit_transform(descriptors_scaled)
    return embeddings

def perform_clustering(embeddings):
    clustering_model = AgglomerativeClustering(n_clusters=n_clusters, linkage='complete')
    labels = clustering_model.fit_predict(embeddings)
    return labels

def create_cluster_map(labels, coordinates, shape):
    cluster_map = np.full(shape, -1, dtype=int)
    for (i, j), label in zip(coordinates, labels):
        cluster_map[i, j] = label
    return cluster_map

# ==== Pipeline Principal ====
print("📥 Cargando patrones de difracción y máscara pesada...")
patterns, coordinates = load_diffraction_patterns(folder_path)
mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
mask = (mask < 127).astype(np.uint8)

selected_patterns = []
selected_coordinates = []
for pattern, (i, j) in zip(patterns, coordinates):
    if i < mask.shape[0] and j < mask.shape[1]:
        if mask[i, j] == 1:
            selected_patterns.append(pattern)
            selected_coordinates.append((i, j))

print(f"🔎 Total patrones seleccionados: {len(selected_patterns)}")

print("📈 Calculando descriptores y generando embedding...")
descriptors = compute_descriptors(selected_patterns)
embeddings = prepare_embedding(descriptors)

# Cargar intensidad integrada directamente desde archivo por cada coordenada
print("➕ Añadiendo intensidad integrada como cuarto descriptor...")
integrated_values = []
for (i, j) in selected_coordinates:
    fname = f"{i}_{j}.npy"
    path = os.path.join(folder_path, fname)
    data = np.load(path)
    integrated = np.sum(data)
    integrated_values.append(integrated)

# Normalizar las intensidades integradas
integrated_values = np.array(integrated_values).reshape(-1, 1)
integrated_values = (integrated_values - integrated_values.mean()) / integrated_values.std()

# Concatenar UMAP + intensidad integrada
full_descriptors = np.hstack([embeddings, integrated_values])
full_descriptors=embeddings
print("🔗 Realizando clustering...")
labels = perform_clustering(full_descriptors)

selected_coordinates = np.array(selected_coordinates)
full_shape = mask.shape
cluster_map = create_cluster_map(labels, selected_coordinates, full_shape)

plt.figure(figsize=(6, 5))
plt.imshow(cluster_map, cmap="tab10")
plt.title("Clustering dentro de columnas pesadas (UMAP + intensidad integrada)")
plt.xlabel("j")
plt.ylabel("i")
plt.colorbar(label="Cluster")
plt.tight_layout()
plt.show()

print("✅ Clasificación dentro de columnas pesadas completada.")