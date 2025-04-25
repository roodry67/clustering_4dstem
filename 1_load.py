import os
import numpy as np
import hyperspy.api as hs
from tqdm import tqdm
import dask

# ⚙️ Configuración
file_path = "./4DSTEM/GaN/Experimental/SI data (11)8C 5cm CL1-4 018A/Diffraction SI.dm4"
file_path = "/home/javier/workspace/clustering/4DSTEM/OLD/SI data (10)/Diffraction SI.dm4"
file_path = "./4DSTEM/MoS2/Atomico/Experimental/SI data (24)/Diffraction SI.dm4"
file_path = "./4DSTEM/SrTiO3/Experiment/Diffraction SI.dm4"

file_path='./4DSTEM/GaN/Experimental/SI data (11)8C 5cm CL1-4 018A/Diffraction SI.dm4'
output_folder = "RESULTS/GaN/1_DATA/"
BLOCK_SIZE = 80  # tamaño del bloque (en navegación)
MAX_WORKERS = 12   # ajusta según tu CPU
N=80
# Limita los hilos de Dask para evitar saturar la RAM
dask.config.set(scheduler='threads', num_workers=MAX_WORKERS)

os.makedirs(output_folder, exist_ok=True)

data = hs.load(file_path, lazy=True)
#print full size of data sample
print(f" Tamaño de la muestra: {data.data.shape}")
# Recorte (ajusta según tus necesidades)
data_cropped = data.inav[60:120,60:120]
nav_shape = data_cropped.axes_manager.navigation_shape
print(f" Tamaño recorte: {nav_shape}")

for i_start in tqdm(range(0, nav_shape[0], BLOCK_SIZE)):
    for j_start in  tqdm(range(0, nav_shape[1], BLOCK_SIZE)):
        i_end = min(i_start + BLOCK_SIZE, nav_shape[0])
        j_end = min(j_start + BLOCK_SIZE, nav_shape[1])

        block = data_cropped.inav[i_start:i_end, j_start:j_end]
        block_np = block.data.compute()  # Carga bloque a RAM

        for i in range(block_np.shape[0]):
            for j in (range(block_np.shape[1])):
                slice_array = block_np[i, j]
                filename = f"{i_start + i}_{j_start + j}.npy"
                np.save(os.path.join(output_folder, filename), slice_array)

        print(f"Bloque ({i_start}:{i_end}, {j_start}:{j_end}) guardado.")

print("Finalizado: todos los archivos .npy han sido exportados.")
