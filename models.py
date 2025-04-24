import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Downsampling block ---
class DownsamplingBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_convs=3):
        super().__init__()
        layers = []
        for i in range(n_convs):
            layers.append(nn.Conv2d(in_channels if i == 0 else out_channels,
                                    out_channels, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.LeakyReLU(0.0001))
        self.convs = nn.Sequential(*layers)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.convs(x)
        x_pooled = self.pool(x)
        return x_pooled, x
        
# --- Encoder ---
class UNetEncoder(nn.Module):
    def __init__(self, input_channels=1, features=[8, 16, 32, 64,128,32,2], bottleneck_dim=128, input_size=256):
        super().__init__()
        self.down_blocks = nn.ModuleList()

        in_ch = input_channels
        for out_ch in features:
            self.down_blocks.append(DownsamplingBlock(in_ch, out_ch))
            in_ch = out_ch
        #add self attention layer 
        
        #final_size = input_size // (2 ** len(features))  
        #final_feat_dim = features[-1] * final_size * final_size
#
        #self.flatten = nn.Flatten()
        #self.mlp = nn.Sequential(
        #    nn.Linear(final_feat_dim, 512),
        #    nn.ReLU(),
        #    nn.Linear(512, bottleneck_dim),
        #    nn.Tanh()
        #)
        self.activation= nn.Tanh()

    def forward(self, x):
        for block in self.down_blocks:
            x, _ = block(x)
        #x = self.flatten(x)
        #z = self.mlp(x)
        x= self.activation(x)
        return x

# --- Upsampling block ---
class UpsamplingBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_convs=3):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        layers = []
        for i in range(n_convs):
            layers.append(nn.Conv2d(in_channels if i == 0 else out_channels,
                                    out_channels, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.LeakyReLU(0.0001))
        self.convs = nn.Sequential(*layers)

    def forward(self, x):
        x = self.upsample(x)
        x = self.convs(x)
        return x

# --- Decoder --- features=[8, 16, 32, 64,128,64,32]
class UNetDecoder(nn.Module):
    def __init__(self, output_channels=1, features=[2,32,128,64,32,16,8], bottleneck_dim=128, input_size=256):
        super().__init__()
        self.init_size =input_size // (2 ** len(features)) 
        #self.project = nn.Sequential(
        #    nn.Linear(bottleneck_dim, features[0] * self.init_size * self.init_size),
        #    nn.ReLU()
        #)
        #self.unflatten = nn.Unflatten(1, (features[0], self.init_size, self.init_size))

        self.up_blocks = nn.ModuleList()
        in_out_pairs = zip(features, features[1:] + [features[-1]])
        for in_ch, out_ch in in_out_pairs:
            self.up_blocks.append(UpsamplingBlock(in_ch, out_ch))

        self.final_conv = nn.Conv2d(features[-1], output_channels, kernel_size=1)

    def forward(self, x):
        #x = self.project(z)
        #x = self.unflatten(x)
        for block in self.up_blocks:
            x = block(x)
        x = self.final_conv(x)
        return x

# --- Autoencoder completo ---
class Autoencoder(nn.Module):
    def __init__(self, input_channels=1, output_channels=1, bottleneck_dim=128):
        super().__init__()
        self.encoder = UNetEncoder(input_channels=input_channels, bottleneck_dim=bottleneck_dim)
        self.decoder = UNetDecoder(output_channels=output_channels, bottleneck_dim=bottleneck_dim)

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, z
