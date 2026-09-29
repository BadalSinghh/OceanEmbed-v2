"""
OceanEmbed - Vision Transformer 2D Architecture (Model Baseline 4)
SIH26066: Satellite Embedding-Based Deep Learning Framework

Features:
- Patch-based Spatial Tokenization: Patch size (4, 4) converts padded [104, 244] grid into 1,586 spatial tokens.
- Full Multi-Head Self-Attention: Learns basin-scale long-range interactions across distant ocean regions (e.g. Arabian Sea <-> Bay of Bengal teleconnections).
- 2D Learnable Positional Embeddings.
- Progressive Patch Decoder with residual spatial skip.
- OceanEmbed Latent Space: [B, 64, 101, 241].
- Depth-Conditioned Decoder: Learned depth embeddings for 15 standard depths [0-1000m].
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


class TransformerBlock(nn.Module):
    """
    Standard Transformer Encoder Layer with Pre-LayerNorm and MLP.
    """
    def __init__(self, embed_dim=256, num_heads=8, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        norm_x = self.norm1(x)
        attn_out, _ = self.attn(norm_x, norm_x, norm_x)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class OceanViT(nn.Module):
    """
    Vision Transformer (ViT-only Baseline 4) for Surface-to-Subsurface Reconstruction.
    Uses 6 Transformer Encoder Layers with 1,586 spatial tokens (patch size 4x4).
    Total parameters: ~8.6 Million.
    """
    def __init__(self, in_vars=7, num_days=7, num_depths=15, in_channels=64, embed_dim=256,
                 patch_size=4, num_layers=6, num_heads=8, mlp_ratio=4.0):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        
        # 1. Spatiotemporal Input Lifting
        self.temporal_encoder = TemporalAttentionEncoder(in_vars=in_vars, num_days=num_days, out_channels=in_channels)
        
        # Target spatial grid is 101 x 241
        # Padded grid: 104 x 244 (divisible by patch_size 4)
        # Patch grid: H_p = 104 / 4 = 26, W_p = 244 / 4 = 61
        # Total tokens: 26 * 61 = 1,586
        self.H_pad = 104
        self.W_pad = 244
        self.num_patches = (self.H_pad // patch_size) * (self.W_pad // patch_size) # 1586
        
        # Patch Projection
        self.patch_embed = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        
        # 2D Positional Embeddings
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, embed_dim) * 0.02)
        
        # 2. Transformer Encoder Layers
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        
        # 3. Patch Decoder / Reconstruction Head
        # Projects tokens back to 2D feature map and upsamples
        self.unpatch_conv = nn.Sequential(
            nn.Conv2d(embed_dim, 128, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 128),
            nn.GELU()
        )
        
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=4, bias=False),
            nn.GroupNorm(8, 64),
            nn.GELU()
        )
        
        # Refinement with residual spatial skip connection
        self.refine = nn.Sequential(
            nn.Conv2d(64 + in_channels, 64, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.GELU()
        )
        
        # 4. Depth-Conditioned Decoder
        self.depth_decoder = DepthConditionedDecoder(embed_dim=64, num_depths=num_depths)

    def forward(self, x):
        # x: [B, 7, 7, 101, 241]
        B = x.shape[0]
        
        # 1. Temporal Attention Encoder
        feat_spatial = self.temporal_encoder(x) # [B, 64, 101, 241]
        
        # 2. Pad to [104, 244] for 4x4 patch tokenization (pad 3 on bottom, 3 on right)
        feat_padded = F.pad(feat_spatial, (0, 3, 0, 3), mode="replicate") # [B, 64, 104, 244]
        
        # 3. Patch Embedding
        patches = self.patch_embed(feat_padded) # [B, 256, 26, 61]
        Hp, Wp = patches.shape[-2], patches.shape[-1]
        tokens = patches.flatten(2).transpose(1, 2) # [B, 1586, 256]
        
        # Add Positional Embeddings
        tokens = tokens + self.pos_embed
        
        # 4. Multi-Head Self-Attention Layers
        for layer in self.transformer_layers:
            tokens = layer(tokens)
        tokens = self.norm(tokens) # [B, 1586, 256]
        
        # 5. Reshape back to 2D patch grid
        feat_2d = tokens.transpose(1, 2).view(B, self.embed_dim, Hp, Wp) # [B, 256, 26, 61]
        feat_unpatched = self.unpatch_conv(feat_2d) # [B, 128, 26, 61]
        feat_up = self.upsample(feat_unpatched) # [B, 64, 104, 244]
        
        # Crop back to exact original grid [101, 241]
        feat_crop = feat_up[:, :, :101, :241] # [B, 64, 101, 241]
        
        # Concatenate residual spatial skip connection
        fused = self.refine(torch.cat([feat_crop, feat_spatial], dim=1)) # [B, 64, 101, 241]
        
        # Latent representation: OceanEmbed
        ocean_embed = fused
        
        # Subsurface 3D Reconstruction
        out = self.depth_decoder(ocean_embed) # [B, 15, 101, 241]
        return out, ocean_embed
