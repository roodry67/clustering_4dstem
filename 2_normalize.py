import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2

MAT = "MoS2_MB"

# Configuración
INPUT_DIR = "RESULTS/" + MAT + "/1_DATA/"
OUTPUT_DIR = "RESULTS/" + MAT + "/2_DATA_normalized/"
PERCENTILE = 99
CENTRAL_CUT = 100  # opcional

os.makedirs(OUTPUT_DIR, exist_ok=True)
files = sorted([f for f in os.listdir(INPUT_DIR) if f.endswith(".npy")])

# Paso 1: Calcular valores globales
print("\U0001F4C8 Calculando valores globales...")
all_values = []
for fname in tqdm(files):
    arr = np.load(os.path.join(INPUT_DIR, fname)).astype(np.float32)
    #add gaussian blur
    # Central cut opcional:
    # h, w = arr.shape
    # arr = arr[h//2 - CENTRAL_CUT//2:h//2 + CENTRAL_CUT//2, w//2 - CENTRAL_CUT//2:w//2 + CENTRAL_CUT//2]
    all_values.append(arr.flatten())

all_values = np.concatenate(all_values)
#global_max = np.percentile(all_values, PERCENTILE)
global_max=np.max(all_values)


# Método de Otsu
hist, bin_edges = np.histogram(all_values, bins=512)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
total = all_values.size
sum_total = np.dot(hist, bin_centers)

weight_background = 0.0
sum_background = 0.0
max_var_between = 0.0
otsu_threshold = bin_centers[0]

for i in range(len(hist)):
    weight_background += hist[i]
    if weight_background == 0:
        continue
    weight_foreground = total - weight_background
    if weight_foreground == 0:
        break
    sum_background += bin_centers[i] * hist[i]
    mean_background = sum_background / weight_background
    mean_foreground = (sum_total - sum_background) / weight_foreground
    var_between = weight_background * weight_foreground * (mean_background - mean_foreground) ** 2
    if var_between > max_var_between:
        max_var_between = var_between
        otsu_threshold = bin_centers[i]

global_min = otsu_threshold

#global_min = 165


print(f"Global min (Otsu): {global_min:.4f}")
print(f"Global max (percentil {PERCENTILE}%): {global_max:.4f}")


# Crear carpeta de imágenes
os.makedirs(os.path.join(OUTPUT_DIR, "images"), exist_ok=True)

# Paso 2: Normalizar y guardar
print("\u2699\ufe0f Normalizando y guardando...")
norm_values = []
del all_values

for fname in tqdm(files):
    path_in = os.path.join(INPUT_DIR, fname)
    path_out = os.path.join(OUTPUT_DIR, fname)

    arr = np.load(path_in).astype(np.float32)
    arr_clipped = np.clip(arr, global_min, global_max)
    norm_arr = (arr_clipped - global_min) / (global_max - global_min)

    np.save(path_out, norm_arr)
    norm_values.append(norm_arr.flatten())

    # Guardar como imagen PNG
    norm_arr = (norm_arr * 255).astype(np.uint8)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "images", fname.replace(".npy", ".png")), norm_arr)

# Histograma después de normalizar
norm_values = np.concatenate(norm_values)
norm_values_filtered = norm_values[norm_values > 0]

plt.figure(figsize=(8, 5))
plt.hist(norm_values_filtered, bins=200, color='seagreen', edgecolor='black', alpha=0.7)
plt.title(f'Histograma de valores (después de normalizar) – {MAT}')
plt.xlabel('Valor normalizado (> 0)')
plt.ylabel('Frecuencia')
plt.tight_layout()
hist_path_post = os.path.join('.', f"histograma_post_normalizacion_{MAT}.png")
plt.savefig(hist_path_post)
plt.close()
print(f"\U0001F4CA Histograma (después de normalizar) guardado en: {hist_path_post}")

print("\u2705 Normalización global completada.")
# Paso 3: Calcular media y desviación típica
mean = float(np.mean(norm_values_filtered))
std = float(np.std(norm_values_filtered))

print(f"\n📏 Estadísticas finales para transform.Normalize:")
print(f"mean = {mean:.6f}")
print(f"std  = {std:.6f}")

# Guardar en archivo de texto
stats_path = os.path.join(OUTPUT_DIR, f"stats_normalizacion_{MAT}.txt")
with open(stats_path, "w") as f:
    f.write(f"mean = {mean:.6f}\n")
    f.write(f"std  = {std:.6f}\n")
    f.write(f"# Úsalo con transforms.Normalize(mean=[{mean:.6f}], std=[{std:.6f}])\n")

print(f"📝 Estadísticas guardadas en: {stats_path}")
