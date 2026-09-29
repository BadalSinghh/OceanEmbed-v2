"""
OceanEmbed - Model A: Dual-Branch FNO + U-Net Architecture (Main Novel Model)
SIH26066: Satellite Embedding-Based Deep Learning Framework

Mathematical & Architectural Design:
- Dual-Branch Parallel Processing:
  1. Local Spatial Branch (Multiscale Residual U-Net): Captures sharp mesoscale fronts, coastal boundaries, and local eddy gradients.
  2. Global Spectral Branch (Fourier Neural Operator): Captures basin-scale teleconnections (Arabian Sea <-> Bay of Bengal) and non-local operator regularizations.
- Cross-Branch Adaptive Gating:
  alpha = sigmoid(Conv(GELU(Conv([F_unet, F_fno]))))
  F_fused = alpha * F_unet + (1 - alpha) * F_fno
- Unified OceanEmbed Latent Space: [B, 64, 101, 241]
- Depth-Conditioned Decoder: Learned FiLM modulation for 15 standard depths [0-1000m].
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalAttentionEncoder(nn.Module):
    """
    Learns temporal interaction and aggregation across the 7-day observation window.
    Input: [B, C=7, T=7, H, W] -> Output: [B, out_channels=64, H, W]
    """
    def __init__(self, in_vars=7, num_days=7, out_channels=64):
        super().__init__()
        self.in_vars = in_vars
        self.num_days = num_days
        
        self.stem3d = nn.Sequential(
            nn.Conv3d(in_vars, 32, kernel_size=(3, 3, 3), padding=(1, 1, 1), bias=False),
            nn.GroupNorm(4, 32),
            nn.GELU(),
            nn.Conv3d(32, 32, kernel_size=(3, 3, 3), padding=(1, 1, 1), bias=False),
            nn.GroupNorm(4, 32),
            nn.GELU()
        )
        
        self.temporal_query = nn.Parameter(torch.randn(1, 32, 1, 1, 1))
        self.temporal_proj = nn.Conv3d(32, 32, kernel_size=1)
        
        self.proj2d = nn.Sequential(
            nn.Conv2d(32, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, out_channels),
            nn.GELU()
        )

    def forward(self, x):
        feat = self.stem3d(x) # [B, 32, T=7, H, W]
        q = self.temporal_query
        k = self.temporal_proj(feat)
        attn = (q * k).sum(dim=1, keepdim=True) / math.sqrt(32)
        attn_weights = F.softmax(attn, dim=2)
        pooled = (feat * attn_weights).sum(dim=2) # [B, 32, H, W]
        out = self.proj2d(pooled) # [B, 64, H, W]
        return out


class ResidualBlock2D(nn.Module):
    """
    Residual Convolutional Block with GroupNorm and GELU.
    """
    def __init__(self, channels):
        super().__init__()
        num_groups = 8 if channels >= 16 else 2
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups, channels)
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.block(x))


class SpectralConv2d(nn.Module):
    """
    2D Fourier Spectral Convolution Layer.
    Computes 2D FFT -> Complex matrix multiplication on low-frequency modes -> 2D IFFT.
    """
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        scale = 1.0 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat)
        )

    def compl_mul2d(self, input, weights):
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        orig_dtype = x.dtype
        x = x.float()
        B, C, H, W = x.shape
        
        x_ft = torch.fft.rfft2(x)
        out_ft = torch.zeros(B, self.out_channels, H, W // 2 + 1, dtype=torch.cfloat, device=x.device)
        
        out_ft[:, :, :self.modes1, :self.modes2] = self.compl_mul2d(
            x_ft[:, :, :self.modes1, :self.modes2], self.weights1
        )
        out_ft[:, :, -self.modes1:, :self.modes2] = self.compl_mul2d(
            x_ft[:, :, -self.modes1:, :self.modes2], self.weights2
        )

        x_out = torch.fft.irfft2(out_ft, s=(H, W))
        return x_out.to(orig_dtype)


class FNOBlock2D(nn.Module):
    """
    Spectral convolution operator + 1x1 bypass convolution + GroupNorm + GELU.
    """
    def __init__(self, channels, modes1, modes2):
        super().__init__()
        self.spectral_conv = SpectralConv2d(channels, channels, modes1, modes2)
        self.bypass = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.norm = nn.GroupNorm(8, channels)
        self.act = nn.GELU()

    def forward(self, x):
        x_spec = self.spectral_conv(x)
        x_by = self.bypass(x)
        return self.act(self.norm(x_spec + x_by))


class AdaptiveGateFusion(nn.Module):
    """
    Learns spatial-adaptive weighting between U-Net (local) and FNO (spectral) features.
    """
    def __init__(self, channels=64):
        super().__init__()
        self.gate_net = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.GroupNorm(8, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        self.refine = nn.Sequential(
            ResidualBlock2D(channels),
            ResidualBlock2D(channels)
        )

    def forward(self, feat_unet, feat_fno):
        cat = torch.cat([feat_unet, feat_fno], dim=1)
        alpha = self.gate_net(cat) # [B, 64, H, W] in [0, 1]
        fused = alpha * feat_unet + (1.0 - alpha) * feat_fno
        out = self.refine(fused)
        return out, alpha


class DepthConditionedDecoder(nn.Module):
    """
    Predicts 3D ocean temperature at 15 standard depths using learned depth conditioning.
    Input: OceanEmbed latent [B, embed_dim=64, H, W] -> Output: [B, 15, H, W]
    """
    def __init__(self, embed_dim=64, num_depths=15):
        super().__init__()
        self.num_depths = num_depths
        self.depth_embeddings = nn.Parameter(torch.randn(num_depths, embed_dim))
        
        self.film_gen = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU()
        )
        
        self.head = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, embed_dim),
            nn.GELU(),
            nn.Conv2d(embed_dim, 1, kernel_size=1)
        )

    def forward(self, ocean_embed):
        B, C, H, W = ocean_embed.shape
        outputs = []
        for d in range(self.num_depths):
            d_emb = self.depth_embeddings[d].unsqueeze(0)
            film = self.film_gen(d_emb)
            gamma, beta = film.chunk(2, dim=-1)
            gamma = gamma.view(1, C, 1, 1)
            beta = beta.view(1, C, 1, 1)
            
            modulated = ocean_embed * (1.0 + gamma) + beta
            temp_d = self.head(modulated)
            outputs.append(temp_d)
            
        return torch.cat(outputs, dim=1)


class OceanFNO_UNet(nn.Module):
    """
    Main Model A: Parallel Dual-Branch FNO + U-Net with Adaptive Gated Fusion.
    - Local Branch: Deep Multiscale Residual U-Net (local eddy, front, coastal features)
    - Spectral Branch: 4-Layer 2D Fourier Neural Operator (global basin-scale modes)
    - Fusion: Adaptive Spatial Gating -> Unified Latent OceanEmbed [B, 64, 101, 241]
    - Decoder: Depth-Conditioned FiLM Decoder for 15 Standard Depths [0-1000m]
    """
    def __init__(self, in_vars=7, num_days=7, num_depths=15, width=64, modes1=16, modes2=24, num_fno_layers=4):
        super().__init__()
        self.width = width
        self.modes1 = modes1
        self.modes2 = modes2
        
        # 1. Spatiotemporal Input Lifting (Shared)
        self.temporal_encoder = TemporalAttentionEncoder(in_vars=in_vars, num_days=num_days, out_channels=width)
        
        # -------------------------------------------------------------
        # BRANCH 1: Multiscale Residual U-Net
        # -------------------------------------------------------------
        self.enc1 = nn.Sequential(ResidualBlock2D(64), ResidualBlock2D(64))
        self.down1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, 128),
            nn.GELU()
        )
        self.enc2 = nn.Sequential(ResidualBlock2D(128), ResidualBlock2D(128))
        self.down2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(16, 256),
            nn.GELU()
        )
        self.enc3 = nn.Sequential(ResidualBlock2D(256), ResidualBlock2D(256))
        self.down3 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(32, 512),
            nn.GELU()
        )
        self.bottleneck = nn.Sequential(
            ResidualBlock2D(512), ResidualBlock2D(512), ResidualBlock2D(512)
        )
        
        self.up3_conv = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1, bias=False),
            nn.GroupNorm(16, 256),
            nn.GELU()
        )
        self.dec3 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(16, 256),
            nn.GELU(),
            ResidualBlock2D(256)
        )
        self.up2_conv = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1, bias=False),
            nn.GroupNorm(8, 128),
            nn.GELU()
        )
        self.dec2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 128),
            nn.GELU(),
            ResidualBlock2D(128)
        )
        self.up1_conv = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.GELU()
        )
        self.dec1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            ResidualBlock2D(64)
        )
        self.unet_out_proj = nn.Sequential(
            nn.Conv2d(64, width, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, width),
            nn.GELU()
        )

        # -------------------------------------------------------------
        # BRANCH 2: Fourier Neural Operator (FNO)
        # -------------------------------------------------------------
        self.coord_proj = nn.Conv2d(width + 2, width, kernel_size=1)
        self.fno_layers = nn.ModuleList([
            FNOBlock2D(channels=width, modes1=modes1, modes2=modes2)
            for _ in range(num_fno_layers)
        ])
        self.fno_out_proj = nn.Sequential(
            nn.Conv2d(width, width, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, width),
            nn.GELU()
        )

        # -------------------------------------------------------------
        # CROSS-BRANCH ADAPTIVE GATED FUSION
        # -------------------------------------------------------------
        self.fusion = AdaptiveGateFusion(channels=width)

        # -------------------------------------------------------------
        # DEPTH-CONDITIONED DECODER
        # -------------------------------------------------------------
        self.depth_decoder = DepthConditionedDecoder(embed_dim=width, num_depths=num_depths)

    def _get_grid(self, shape, device):
        B, H, W = shape
        grid_x = torch.linspace(-1, 1, steps=W, device=device).view(1, 1, 1, W).repeat(B, 1, H, 1)
        grid_y = torch.linspace(-1, 1, steps=H, device=device).view(1, 1, H, 1).repeat(B, 1, 1, W)
        return torch.cat((grid_y, grid_x), dim=1)

    def forward(self, x):
        # x: [B, 7, 7, 101, 241]
        B = x.shape[0]
        H, W = x.shape[-2], x.shape[-1]
        
        # 1. Spatiotemporal Lifting
        x0 = self.temporal_encoder(x) # [B, 64, 101, 241]
        
        # 2. Branch 1: U-Net forward pass
        e1 = self.enc1(x0) # [B, 64, 101, 241]
        d1 = self.down1(e1)
        e2 = self.enc2(d1) # [B, 128, 51, 121]
        d2 = self.down2(e2)
        e3 = self.enc3(d2) # [B, 256, 26, 61]
        d3 = self.down3(e3)
        b = self.bottleneck(d3) # [B, 512, 13, 31]
        
        u3 = F.interpolate(b, size=e3.shape[-2:], mode="bilinear", align_corners=False)
        u3 = self.up3_conv(u3)
        cat3 = torch.cat([u3, e3], dim=1)
        d_out3 = self.dec3(cat3)
        
        u2 = F.interpolate(d_out3, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        u2 = self.up2_conv(u2)
        cat2 = torch.cat([u2, e2], dim=1)
        d_out2 = self.dec2(cat2)
        
        u1 = F.interpolate(d_out2, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        u1 = self.up1_conv(u1)
        cat1 = torch.cat([u1, e1], dim=1)
        d_out1 = self.dec1(cat1)
        feat_unet = self.unet_out_proj(d_out1) # [B, 64, 101, 241]

        # 3. Branch 2: FNO forward pass
        grid = self._get_grid((B, H, W), x.device)
        feat_fno_in = self.coord_proj(torch.cat([x0, grid], dim=1))
        f_curr = feat_fno_in
        for fno_layer in self.fno_layers:
            f_curr = fno_layer(f_curr)
        feat_fno = self.fno_out_proj(f_curr) # [B, 64, 101, 241]

        # 4. Adaptive Gated Fusion
        ocean_embed, alpha_gate = self.fusion(feat_unet, feat_fno) # [B, 64, 101, 241]

        # 5. Depth-Conditioned Subsurface Reconstruction
        out = self.depth_decoder(ocean_embed) # [B, 15, 101, 241]

        return out, ocean_embed
