import cv2
import numpy as np
import matplotlib.pyplot as plt





# ----------------------------
# Configuración
# ----------------------------
MAT="SrTiO3"
IMAGE_FILENAME_RAW = "RESULTS/"+MAT+"/heavy_columns_map.png"
OUTPUT_FILENAME= "RESULTS/"+MAT+"/heavy_columns_map_filtered.png"
# Cargar imagen binaria
mask = cv2.imread(IMAGE_FILENAME_RAW, cv2.IMREAD_GRAYSCALE)

# Asegurar que sea binaria 0-1
_, binary = cv2.threshold(mask, 127, 1, cv2.THRESH_BINARY)

# Encontrar contornos
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Calcular áreas
areas = np.array([cv2.contourArea(cnt) for cnt in contours])

# Heurística: usar percentil 25 o IQR
q1 = np.percentile(areas, 25)
q3 = np.percentile(areas, 75)
iqr = q3 - q1
area_threshold = q1 - 1.5 * iqr
area_threshold = max(1, area_threshold)  # evitar valores negativos

print(f"Área mínima para conservar: {area_threshold:.2f}")

# Crear nueva máscara
filtered_mask = np.zeros_like(binary)

for cnt, area in zip(contours, areas):
    if area >= area_threshold:
        cv2.drawContours(filtered_mask, [cnt], -1, color=1, thickness=-1)
#last, perform a dilation to connect any gaps using a circle
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
filtered_mask = cv2.dilate(filtered_mask, kernel, iterations=1)
#now erode 3 iterations
kernel=np.ones((3,3),np.uint8)
filtered_mask = cv2.erode(filtered_mask, kernel, iterations=1)
eroding_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
filtered_mask = cv2.erode(filtered_mask, eroding_kernel, iterations=1)

dilation_kernel= np.ones((3,3))
filtered_mask = cv2.dilate(filtered_mask, dilation_kernel, iterations=1)
# Guardar resultado final
cv2.imwrite(OUTPUT_FILENAME, filtered_mask * 255)

