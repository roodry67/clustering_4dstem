import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
from models import Autoencoder
# Configuración
MAT = "SrTiO3"
DATA_DIR = "RESULTS/"+MAT+"/2_DATA_normalized/"
EMBEDDING_DIR = "RESULTS/"+MAT+"/3_EMBED_AE/"
BOTTLENECK_SIZE = 32
BATCH_SIZE = 256
EPOCHS = 3
LR=1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(EMBEDDING_DIR, exist_ok=True)

values ={}
values['SrTiO3']={'crop':200, 'mean':0.184759, 'std':0.132365}
values['GaN']={'crop':128, 'mean':0.232679, 'std':0.133979}
values['MoS2']={'crop':100, 'mean':0.270102, 'std':0.119688}
values['MoS2_MB']={'crop':100, 'mean':0.2652, 'std':0.1159}
#values['MoS2']={'crop':64, 'mean':0.325895, 'std':0.121174}


# Dataset personalizado
class NPYPatchDataset(Dataset):
    def __init__(self, folder):
        self.files = sorted([f for f in os.listdir(folder) if f.endswith(".npy")])
        self.folder = folder
        self.transform = transforms.Compose([
            transforms.ToTensor(),  # (1, 256, 256), normalizado entre 0 y 1
            transforms.CenterCrop(values[MAT]['crop']),
            transforms.Resize(128),
            transforms.Normalize(mean=[values[MAT]['mean']], std=[values[MAT]['std']]),

        ])

    

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]
        arr = np.load(os.path.join(self.folder, file)).astype(np.float32)
        arr = self.transform(arr)  # Añade dimensión de canal
        return arr, file



# Dataset y DataLoader
dataset = NPYPatchDataset(DATA_DIR)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

# Modelo
model = Autoencoder(bottleneck_dim=BOTTLENECK_SIZE).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.SmoothL1Loss()

#print nmumber of parameters in millions
num_params = sum(p.numel() for p in model.parameters()) / 1e6
print(f"Modelo con {num_params:.5f} millones de parámetros.")
# Entrenamiento
print("\U0001F680 Entrenando Autoencoder...")
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for x, _ in tqdm(dataloader):
        x = x.to(DEVICE)
        recon, _ = model(x)
        loss = criterion(recon, x)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {total_loss / len(dataset):.4f}")

# Guardar embeddings
print("\U0001F4BE Guardando embeddings en:", EMBEDDING_DIR)
model.eval()
embedding_dataset = DataLoader(dataset, batch_size=1, shuffle=False)

with torch.no_grad():
    for x, fname in tqdm(embedding_dataset):
        x = x.to(DEVICE)
        _, embedding = model(x)
        embedding = embedding.cpu().numpy().squeeze()  # (1,1,1024) → (1024,)
        #flatten
        embedding = embedding.flatten()
        out_path = os.path.join(EMBEDDING_DIR, fname[0].replace('.npy', '.npy'))
        np.save(out_path, embedding)

print("\u2705 Embeddings guardados correctamente.")
