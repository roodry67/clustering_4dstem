# clustering_exploration.py

# ==== Configuración ====
MAT = "MoS2"
FOLDER_PATH = "RESULTS/" + MAT + "/2_DATA_pca/"
n_components_umap = 3
n_clusters = 6  # puedes cambiar este valor libremente
n_bins_radial_profile = 185
cut_radial_profile_at = 170

# ==== Imports ====
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import center_of_mass
from sklearn.cluster import AgglomerativeClustering, KMeans
import umap
from sklearn.preprocessing import RobustScaler

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
    reducer = umap.UMAP(n_components=n_components_umap)
    embeddings = reducer.fit_transform(descriptors_scaled)
    return embeddings

def perform_clustering(embeddings, n_clusters):
    clustering_model = AgglomerativeClustering(n_clusters=n_clusters, linkage='complete')
    labels = clustering_model.fit_predict(embeddings)
    return labels

# ==== Pipeline Principal ====
print("📥 Cargando patrones de difracción...")
patterns, coordinates = load_diffraction_patterns(FOLDER_PATH)

print("📈 Calculando descriptores y generando embedding...")
descriptors = compute_descriptors(patterns)
embeddings = prepare_embedding(descriptors)

print("➕ Añadiendo intensidad integrada como descriptor adicional...")
integrated_values = []
for (i, j) in coordinates:
    fname = f"{i}_{j}.npy"
    path = os.path.join(FOLDER_PATH, fname)
    data = np.load(path)
    integrated = np.sum(data)
    integrated_values.append(integrated)

integrated_values = np.array(integrated_values).reshape(-1, 1)
integrated_values = (integrated_values - integrated_values.mean()) / integrated_values.std()

# Concatenar embeddings + intensidad integrada
full_descriptors = np.hstack([embeddings, integrated_values])

print("🔗 Realizando clustering...")
labels = perform_clustering(full_descriptors, n_clusters)

# ==== Visualización ====
i_vals = coordinates[:, 0]
j_vals = coordinates[:, 1]
map_shape = (i_vals.max() + 1, j_vals.max() + 1)
cluster_map = np.full(map_shape, -1, dtype=int)

for (i, j), label in zip(coordinates, labels):
    cluster_map[i, j] = label

plt.figure(figsize=(6, 5))
plt.imshow(cluster_map, cmap="tab10")
plt.title(f"Clustering exploratorio (n_clusters={n_clusters})")
plt.xlabel("j")
plt.ylabel("i")
plt.colorbar(label="Cluster")
plt.tight_layout()
plt.show()

print("✅ Clustering exploratorio completado.")
