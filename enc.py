import math
from typing import List, Tuple, Optional, NamedTuple
from torch.cuda.amp import autocast

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
import logging
import os
from torch.utils.data import Dataset,DataLoader, random_split
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau

torch.autograd.set_detect_anomaly(True)

# ------------ DATA ---------------
class FixedSegmentDataset(Dataset):
    def __init__(self, data_dir):
        self.data_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.pt')])
        self.data_dir = data_dir

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        return torch.load(os.path.join(self.data_dir, self.data_files[idx]))

# ------------ BFLOAT16 UTILS ---------------
class MixedPrecisionModule(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module
        self.to_bfloat16 = lambda x: x.bfloat16() if x.dtype == torch.float32 else x

    def forward(self, *args, **kwargs):
        args = [self.to_bfloat16(arg) for arg in args]
        kwargs = {k: self.to_bfloat16(v) for k, v in kwargs.items()}
        return self.module(*args, **kwargs)

def create_mixed_precision_model(model):
    for name, module in model.named_children():
        if isinstance(module, (nn.Conv1d, nn.Linear, nn.LSTM, nn.BatchNorm1d)):
            setattr(model, name, MixedPrecisionModule(module.to(torch.bfloat16)))
    return model

# ---------- MODELS ------------

class Discriminator(nn.Module):
    def __init__(self, channels=374, dimension=374):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(channels, dimension, kernel_size=7, stride=2, padding=3),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(dimension, dimension * 2, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(dimension * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(dimension * 2, dimension * 4, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(dimension * 4),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.fc = nn.Linear(dimension * 4, 1)

    def forward(self, x):
        x = x.to(torch.bfloat16) 
        x = self.conv_layers(x)
        x = torch.mean(x, dim=2)
        x = self.fc(x)
        return x
    
class QuantizedResult(NamedTuple):
    quantized: torch.Tensor
    codes: torch.Tensor
    penalty: torch.Tensor

class NeuralSEANetEncoder(nn.Module):
    def __init__(self, channels=374, dimension=374, depth=2, norm='time_group_norm'):
        super().__init__()
        self.channels = channels
        self.dimension = dimension
        self.conv1 = nn.Conv1d(channels, dimension, kernel_size=7, stride=1, padding=3)
        self.norm1 = nn.GroupNorm(1, dimension) if norm == 'time_group_norm' else nn.Identity()
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(dimension, dimension, kernel_size=8, stride=4, padding=2),
                nn.ReLU(),
                nn.GroupNorm(1, dimension) if norm == 'time_group_norm' else nn.Identity(),
                nn.LSTM(dimension, dimension, depth, batch_first=True)
            ) for _ in range(3)
        ])

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        for layer in self.layers:
            conv, relu, norm, lstm = layer
            x = conv(x)
            x = relu(x)
            x = norm(x)
            x = x.permute(0, 2, 1)  # (batch, time, channels)
            x, _ = lstm(x)
            x = x.permute(0, 2, 1)  # (batch, channels, time)
        return x

class NeuralSEANetDecoder(nn.Module):
    def __init__(self, channels=374, dimension=374, depth=2, norm='time_group_norm'):
        super().__init__()
        self.channels = channels
        self.dimension = dimension
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.LSTM(dimension, dimension, depth, batch_first=True),
                nn.GroupNorm(1, dimension) if norm == 'time_group_norm' else nn.Identity(),
                nn.ReLU(),
                nn.ConvTranspose1d(dimension, dimension, kernel_size=8, stride=4, padding=2)
            ) for _ in range(3)
        ])
        self.conv_out = nn.ConvTranspose1d(dimension, channels, kernel_size=7, stride=1, padding=3)

    def forward(self, x):
        for layer in self.layers:
            lstm, norm, relu, conv = layer
            x = x.permute(0, 2, 1)  # (batch, time, channels)
            x, _ = lstm(x)
            x = x.permute(0, 2, 1)  # (batch, channels, time)
            x = norm(x)
            x = relu(x)
            x = conv(x)
        x = self.conv_out(x)
        return x

class NeuralResidualVectorQuantizer(nn.Module):
    def __init__(self, dimension, n_q, bins):
        super().__init__()
        self.dimension = dimension
        self.n_q = n_q
        self.bins = bins
        self.codebook = nn.Parameter(torch.randn(n_q, bins, dimension))

    def forward(self, x, sample_rate, bandwidth):
        B, C, T = x.shape
        x_flat = x.permute(0, 2, 1).reshape(-1, C)
        
        # Reshape x_flat to match the dimensionality of self.codebook
        x_flat_expanded = x_flat.unsqueeze(1).unsqueeze(1).expand(-1, self.n_q, self.bins, -1)
        
        # Compute distances
        distances = torch.sum((x_flat_expanded - self.codebook) ** 2, dim=-1)
        
        # Find nearest neighbor
        indices = distances.argmin(dim=-1)
        
        quantized = torch.zeros_like(x_flat)
        for i in range(self.n_q):
            quantized += self.codebook[i][indices[:, i]]
        
        # Reshape quantized tensor
        quantized = quantized.reshape(B, T, C).permute(0, 2, 1)
        
        # Compute loss
        commitment_loss = F.mse_loss(quantized.detach(), x)
        codebook_loss = F.mse_loss(quantized, x.detach())
        loss = commitment_loss + codebook_loss
        
        # Straight-through estimator
        quantized = x + (quantized - x).detach()
        
        return QuantizedResult(quantized, indices, loss)
    
class NeuralEnCodecModel(nn.Module):
    def __init__(self, encoder, decoder, quantizer, target_bandwidths, sample_rate, discriminator, channels=374, normalize=True):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.quantizer = quantizer
        self.target_bandwidths = target_bandwidths
        self.sample_rate = sample_rate
        self.channels = channels
        self.normalize = normalize
        self.bandwidth = None
        self.criterion = NeuralLoss(discriminator)
    def _normalize_neural_data(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean = x.mean(dim=2, keepdim=True)
        std = x.std(dim=2, keepdim=True)
        return (x - mean) / (std + 1e-8), torch.cat([mean, std], dim=2)

    def _denormalize_neural_data(self, x: torch.Tensor, stats: torch.Tensor) -> torch.Tensor:
        mean, std = stats.chunk(2, dim=2)
        return x * std + mean

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.normalize:
            x, scale = self._normalize_neural_data(x)
        else:
            scale = None
        emb = self.encoder(x)
        return emb, scale

    def decode(self, quantized: torch.Tensor, scale: Optional[torch.Tensor]) -> torch.Tensor:
        out = self.decoder(quantized)
        if scale is not None:
            out = self._denormalize_neural_data(out, scale)
        return out
    
    def forward(self, x: torch.Tensor, real_signal: torch.Tensor):
        original_size = x.shape[2]
        emb, scale = self.encode(x)
        qv = self.quantizer(emb, self.sample_rate, self.bandwidth)
        quantized = qv.quantized
        decoded = self.decode(quantized, scale)
        
        if decoded.shape[2] != original_size:
            decoded = F.interpolate(decoded, size=original_size, mode='linear', align_corners=False)

        loss_rec, loss_components = self.criterion(decoded.to(torch.bfloat16), x.to(torch.bfloat16), real_signal.to(torch.bfloat16))
        total_loss = loss_rec + qv.penalty

        return decoded, total_loss, loss_components, (emb, scale)

def create_neural_encodec_model(discriminator=None, channels=374, dimension=374*4, target_bandwidths=[30], sample_rate=30000):
    encoder = NeuralSEANetEncoder(channels=channels, dimension=dimension)
    decoder = NeuralSEANetDecoder(channels=channels, dimension=dimension)
    n_q = int(sample_rate * target_bandwidths[-1] // (math.ceil(sample_rate / 4) * 10)) 
    quantizer = NeuralResidualVectorQuantizer(dimension=dimension, n_q=n_q, bins=1024)
    model = NeuralEnCodecModel(encoder, decoder, quantizer, target_bandwidths, sample_rate, discriminator, channels)
    return create_mixed_precision_model(model) 

# ---------- LOSSES ------------

class SSIMLoss(nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1

    def create_window(self, size, channel):
        def gaussian(window_size, sigma):
            gauss = torch.Tensor(
                [math.exp(-(x - window_size // 2)**2 / float(2 * sigma**2)) for x in range(window_size)]
            )
            return gauss / gauss.sum()

        # Create a 1D Gaussian window
        _1D_window = gaussian(size, 1.5).unsqueeze(1)  # Shape: [window_size, 1]
        
        # Expand to [1, 1, window_size] for conv1d
        window = _1D_window.t().unsqueeze(0)  # Shape: [1, 1, window_size]
        
        return window.expand(channel, 1, size).contiguous()

    def forward(self, pred, target):
        # Pred and target expected to be of shape (B, C, W)
        (_, channel, _) = pred.shape
        window = self.create_window(self.window_size, channel).to(pred.device).type(pred.dtype)

        mu1 = F.conv1d(pred, window, padding=self.window_size // 2, groups=channel)
        mu2 = F.conv1d(target, window, padding=self.window_size // 2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv1d(pred * pred, window, padding=self.window_size // 2, groups=channel) - mu1_sq
        sigma2_sq = F.conv1d(target * target, window, padding=self.window_size // 2, groups=channel) - mu2_sq
        sigma12 = F.conv1d(pred * target, window, padding=self.window_size // 2, groups=channel) - mu1_mu2

        C1 = 0.01**2
        C2 = 0.03**2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        return (1 - ssim_map).mean()
    

class NeuralLoss(nn.Module):
    def __init__(self, discriminator, mse_weight=0.1, corr_weight=0.1,
                 temporal_weight=0.1, ssim_weight=0.35, adv_weight=0.35):
        super().__init__()
        self.discriminator = discriminator
        self.mse_weight = mse_weight
        self.corr_weight = corr_weight
        self.temporal_weight = temporal_weight
        self.ssim_weight = ssim_weight
        self.adv_weight = adv_weight
        self.eps = 1e-8
        self.ssim_loss = SSIMLoss()
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='mean')

    def forward(self, pred, target, real_signal):
        predT, targetT = pred.transpose(1, 2), target.transpose(1, 2) # SSIM and Discriminator work channel wise, the others calculate losses temporally. 
        mse = self.mse_loss(predT, targetT)
        corr = self.correlation_loss(predT, targetT).mean()
        temporal = self.temporal_coherence_loss(predT, targetT).mean()
        ssim = self.ssim_loss(pred, target)

        # Discriminator Loss
        real_preds = self.discriminator(real_signal)
        fake_preds = self.discriminator(pred.detach())
        real_labels = torch.ones_like(real_preds)
        fake_labels = torch.zeros_like(fake_preds)
        
        adv_loss_real = self.bce_loss(real_preds, real_labels)
        adv_loss_fake = self.bce_loss(fake_preds, fake_labels)
        adv_loss = adv_loss_real + adv_loss_fake

        total_loss = (self.mse_weight * mse +
                      self.corr_weight * corr +
                      self.temporal_weight * temporal +
                      self.ssim_weight * ssim +
                      self.adv_weight * adv_loss)

        return total_loss, {
            'total_loss': total_loss.item(),
            'mse': mse.item(),
            'corr': corr.item(),
            'temporal': temporal.item(),
            'ssim': ssim.item(),
            'adv_loss': adv_loss.item()
        }

    def mse_loss(self, pred, target):
        pred, target = pred.float(), target.float()
        return F.mse_loss(pred, target, reduction='mean')
    
    def correlation_loss(self, pred, target):
        pred_centered = pred - pred.mean(dim=2, keepdim=True)
        target_centered = target - target.mean(dim=2, keepdim=True)
        
        pred_var = torch.sum(pred_centered**2, dim=2)
        target_var = torch.sum(target_centered**2, dim=2)
        
        pred_var = torch.clamp(pred_var, min=self.eps)
        target_var = torch.clamp(target_var, min=self.eps)
        
        cors = torch.sum(pred_centered * target_centered, dim=2) / (torch.sqrt(pred_var * target_var) + self.eps)
        cors = torch.clamp(cors, min=-1.0 + self.eps, max=1.0 - self.eps)
        
        return 1 - cors
    
    def temporal_coherence_loss(self, pred, target):
        pred_diff = pred[:, :, 1:] - pred[:, :, :-1]
        target_diff = target[:, :, 1:] - target[:, :, :-1]
        return F.mse_loss(pred_diff, target_diff, reduction='none').mean(dim=2)
    

# ---------- TRAINING LOOP ------------

def train_neural_encodec(model, discriminator, dataset, num_epochs, learning_rate, batch_size, device):
    train_size = int(0.95 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    model.to(device)
    discriminator.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    disc_optimizer = torch.optim.AdamW(discriminator.parameters(), lr=learning_rate )
    
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        discriminator.train()

        total_loss = 0.0
        num_batches = len(train_loader)
        
        with tqdm(total=num_batches, desc=f"Epoch {epoch+1}/{num_epochs}") as pbar:
            for batch_idx, data in enumerate(train_loader, 1):
                optimizer.zero_grad()
                disc_optimizer.zero_grad()

                x = data.to(device).bfloat16()
                real_signal = data.to(device).bfloat16()

                _, loss, _, _ = model(x, real_signal)

                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)

                optimizer.step()
                disc_optimizer.step()

                total_loss += loss.item()
                pbar.set_postfix({
                    'batch': f"{batch_idx}/{num_batches}",
                    'loss': f"{loss.item():.4f}",
                    'avg_loss': f"{total_loss / batch_idx:.4f}"
                })
                pbar.update(1)

            avg_train_loss = total_loss / len(train_loader)
            print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}")

        # Validation
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            for data in val_loader:
                x = data.to(device).bfloat16()
                real_signal = data.to(device).bfloat16()

                _, loss, _, _ = model(x, real_signal)
                val_loss += loss.item()

            avg_val_loss = val_loss / len(val_loader)
            print(f"Epoch [{epoch+1}/{num_epochs}], Val Loss: {avg_val_loss:.4f}")
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), 'best_model.pth')
                print(f"Saved best model with Val Loss: {best_val_loss:.4f}")

            scheduler.step(avg_val_loss)

    print("Training completed!")
    


if __name__ == "__main__":
    torch.manual_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")


    discriminator = Discriminator()
    discriminator = create_mixed_precision_model(discriminator).to(device)
    model = create_neural_encodec_model(discriminator=discriminator)
    model = model.to(device).to(torch.bfloat16)
    print("Model", sum(p.numel() for p in model.parameters() if p.requires_grad))
    print("Discriminator", sum(p.numel() for p in discriminator.parameters() if p.requires_grad))
    batch_size = 24
    
    # Train the model
    num_epochs = 1000
    learning_rate = 1e-4

    dataset = FixedSegmentDataset('/home/marisbasha/neural_encodec/fixed_dataset')

    train_neural_encodec(model, discriminator, dataset, num_epochs, learning_rate, batch_size, device)
    
