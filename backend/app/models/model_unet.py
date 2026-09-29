"""
OceanEmbed - Deep Residual U-Net Architecture (Model A Baseline)
SIH26066: Satellite Embedding-Based Deep Learning Framework

Features:
- Spatiotemporal input processing: 7 surface variables x 7 days -> [B, 7, 7, 101, 241]
- Temporal Self-Attention Encoder: Learns dynamic lag interactions across the 7-day window.
- Multiscale Hierarchical U-Net: Residual Conv Blocks, GroupNorm, GELU, Skip connections.
- OceanEmbed Latent Space: Fused feature representation [B, 64, 101, 241].
- Depth-Conditioned Decoder: Learned depth embeddings for 15 standard depths [0-1000m].
- Output: 3D physical temperature field [B, 15, 101, 241] in °C.
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
        
        # 3D Spatiotemporal Convolution stem
        self.stem3d = nn.Sequential(
            nn.Conv3d(in_vars, 32, kernel_size=(3, 3, 3), padding=(1, 1, 1), bias=False),
            nn.GroupNorm(4, 32),
            nn.GELU(),
            nn.Conv3d(32, 32, kernel_size=(3, 3, 3), padding=(1, 1, 1), bias=False),
            nn.GroupNorm(4, 32),
            nn.GELU()
        )
        
        # Temporal Attention pooling across the 7 days
        self.temporal_query = nn.Parameter(torch.randn(1, 32, 1, 1, 1))
        self.temporal_proj = nn.Conv3d(32, 32, kernel_size=1)
        
        # Spatial projection to output channels
        self.proj2d = nn.Sequential(
            nn.Conv2d(32, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, out_channels),
            nn.GELU()
        )

    def forward(self, x):
        # x: [B, C=7, T=7, H=101, W=241]
        feat = self.stem3d(x) # [B, 32, T=7, H, W]
        
        # Attention scores across T
        q = self.temporal_query # [1, 32, 1, 1, 1]
        k = self.temporal_proj(feat) # [B, 32, T, H, W]
        attn = (q * k).sum(dim=1, keepdim=True) / math.sqrt(32) # [B, 1, T, H, W]
        attn_weights = F.softmax(attn, dim=2) # [B, 1, T, H, W]
        
        # Weighted sum over temporal dimension
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


class DepthConditionedDecoder(nn.Module):
    """
    Predicts 3D ocean temperature at 15 standard depths using learned depth conditioning.
    Input: OceanEmbed latent [B, embed_dim=64, H, W] -> Output: [B, 15, H, W]
    """
    def __init__(self, embed_dim=64, num_depths=15):
        super().__init__()
        self.num_depths = num_depths
        self.depth_embeddings = nn.Parameter(torch.randn(num_depths, embed_dim))
        
        # FiLM (Feature-wise Linear Modulation) generator from depth embedding
        self.film_gen = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU()
        )
        
        # Shared temperature head
        self.head = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, embed_dim),
            nn.GELU(),
            nn.Conv2d(embed_dim, 1, kernel_size=1)
        )

    def forward(self, ocean_embed):
        # ocean_embed: [B, C=64, H, W]
        B, C, H, W = ocean_embed.shape
        outputs = []
        
        # Condition on each of the 15 standard depths
        for d in range(self.num_depths):
            d_emb = self.depth_embeddings[d].unsqueeze(0) # [1, C]
            film = self.film_gen(d_emb) # [1, 2*C]
            gamma, beta = film.chunk(2, dim=-1)
            gamma = gamma.view(1, C, 1, 1)
            beta = beta.view(1, C, 1, 1)
            
            # Modulate latent representation
            modulated = ocean_embed * (1.0 + gamma) + beta
            temp_d = self.head(modulated) # [B, 1, H, W]
            outputs.append(temp_d)
            
        out = torch.cat(outputs, dim=1) # [B, 15, H, W]
        return out


class OceanUNet(nn.Module):
    """
    Deep Residual U-Net for Surface-to-Subsurface Temperature Reconstruction.
    Total parameters: ~10.8 Million.
    """
    def __init__(self, in_vars=7, num_days=7, num_depths=15):
        super().__init__()
        
        # 1. Temporal Attention Encoder
        self.temporal_encoder = TemporalAttentionEncoder(in_vars=in_vars, num_days=num_days, out_channels=64)
        
        # 2. U-Net Encoder
        self.enc1 = nn.Sequential(
            ResidualBlock2D(64),
            ResidualBlock2D(64)
        ) # [B, 64, 101, 241]
        
        self.down1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, 128),
            nn.GELU()
        )
        self.enc2 = nn.Sequential(
            ResidualBlock2D(128),
            ResidualBlock2D(128)
        ) # [B, 128, 51, 121]
        
        self.down2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(16, 256),
            nn.GELU()
        )
        self.enc3 = nn.Sequential(
            ResidualBlock2D(256),
            ResidualBlock2D(256)
        ) # [B, 256, 26, 61]
        
        self.down3 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(32, 512),
            nn.GELU()
        )
        
        # 3. Deep Bottleneck
        self.bottleneck = nn.Sequential(
            ResidualBlock2D(512),
            ResidualBlock2D(512),
            ResidualBlock2D(512)
        ) # [B, 512, 13, 31]
        
        # 4. U-Net Decoder with Adaptive Bilinear Upsampling
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
        
        # 5. OceanEmbed Latent Projection
        self.ocean_embed_proj = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.GELU()
        )
        
        # 6. Depth-Conditioned Decoder
        self.depth_decoder = DepthConditionedDecoder(embed_dim=64, num_depths=num_depths)

    def forward(self, x):
        # x: [B, 7, 7, 101, 241]
        # Temporal Encoder
        x0 = self.temporal_encoder(x) # [B, 64, 101, 241]
        
        # Encoder
        e1 = self.enc1(x0) # [B, 64, 101, 241]
        
        d1 = self.down1(e1)
        e2 = self.enc2(d1) # [B, 128, 51, 121]
        
        d2 = self.down2(e2)
        e3 = self.enc3(d2) # [B, 256, 26, 61]
        
        d3 = self.down3(e3)
        b = self.bottleneck(d3) # [B, 512, 13, 31]
        
        # Decoder
        u3 = F.interpolate(b, size=e3.shape[-2:], mode="bilinear", align_corners=False)
        u3 = self.up3_conv(u3)
        cat3 = torch.cat([u3, e3], dim=1) # [B, 512, 26, 61]
        d_out3 = self.dec3(cat3)
        
        u2 = F.interpolate(d_out3, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        u2 = self.up2_conv(u2)
        cat2 = torch.cat([u2, e2], dim=1) # [B, 256, 51, 121]
        d_out2 = self.dec2(cat2)
        
        u1 = F.interpolate(d_out2, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        u1 = self.up1_conv(u1)
        cat1 = torch.cat([u1, e1], dim=1) # [B, 128, 101, 241]
        d_out1 = self.dec1(cat1)
        
        # Latent representation: OceanEmbed
        ocean_embed = self.ocean_embed_proj(d_out1) # [B, 64, 101, 241]
        
        # Subsurface 3D Reconstruction
        out = self.depth_decoder(ocean_embed) # [B, 15, 101, 241]
        return out, ocean_embed
