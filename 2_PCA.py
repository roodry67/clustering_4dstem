import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.decomposition import PCA
import cv2

# Parámetros
MAT = "MoS2"
INPUT_DIR = f"RESULTS/{MAT}/1_DATA/"
OUTPUT_DIR = f"RESULTS/{MAT}/2_DATA_pca/"
N_COMPONENTS = 64
threshold = 68

# Crear directorios de salida
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "images"), exist_ok=True)

# Cargar y aplanar
print("📥 Cargando datos y aplanando...")
files = sorted([f for f in os.listdir(INPUT_DIR) if f.endswith(".npy")])
all_flattened = []
for fname in tqdm(files):
    arr = np.load(os.path.join(INPUT_DIR, fname)).astype(np.float32)
    assert arr.shape == (256, 256), f"Tamaño inesperado en {fname}"
    all_flattened.append(arr.flatten())

X = np.stack(all_flattened)

# Normalización usando valores > threshold
mean = np.mean(X[X > threshold])
std = np.std(X[X > threshold])
X = (X - mean) / std
print(f"📊 Media: {mean:.2f}, Desviación estándar: {std:.2f} (solo valores > {threshold})")

# Aplicar PCA
print(f"🔍 Aplicando PCA con {N_COMPONENTS} componentes...")
pca = PCA(n_components=N_COMPONENTS)
X_pca = pca.fit_transform(X)
X_denoised = pca.inverse_transform(X_pca)

# Calcular valores para visualización
all_vals = X_denoised.flatten()
vmin = (threshold - mean) / std
vmax = np.percentile(all_vals, 99)
print(f"🎨 Normalización visual -> vmin (threshold): {vmin:.3f}, vmax (99%): {vmax:.3f}")

# Guardar arrays reconstruidos y generar imágenes
print("💾 Guardando arrays reconstruidos y generando imágenes...")
norm_values = []
for i, fname in enumerate(tqdm(files)):
    arr = X_denoised[i].reshape((256, 256))

    # Guardar .npy
    np.save(os.path.join(OUTPUT_DIR, fname), arr)

    # Imagen en colormap jet (con normalización global)
    arr_vis = (arr - vmin) / (vmax - vmin)
    arr_vis = np.clip(arr_vis, 0, 1)
    arr_vis = (arr_vis * 255).astype(np.uint8)
    arr_color = cv2.applyColorMap(arr_vis, cv2.COLORMAP_JET)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "images", fname.replace(".npy", ".png")), arr_color)

    norm_values.append(arr.flatten())

# Histograma de valores reconstruidos > threshold
x_h = np.concatenate(norm_values)
x_h = x_h[x_h > vmin]

plt.figure(figsize=(8, 5))
plt.hist(x_h, bins=200, color='teal', edgecolor='black', alpha=0.7)
plt.title(f'Histograma de valores tras PCA – {MAT}')
plt.xlabel('Valor reconstruido (> threshold)')
plt.ylabel('Frecuencia')
plt.tight_layout()
hist_path = os.path.join(OUTPUT_DIR, f"histograma_post_pca_{MAT}.png")
plt.savefig(hist_path)
plt.close()
print(f"🖼️ Histograma guardado en: {hist_path}")


print("✅ Proceso PCA + normalización + guardado completado.")
