"""
OceanEmbed - Model B: Dual-Branch FNO + Vision Transformer Architecture
SIH26066: Satellite Embedding-Based Deep Learning Framework

Mathematical & Architectural Design:
- Dual Global Operator Processing:
  1. Attention Branch (Vision Transformer): Learns data-driven non-local spatial dependencies via multi-head self-attention across 1,586 patch tokens.
  2. Spectral Branch (Fourier Neural Operator): Learns continuous differential operator kernels in frequency domain via truncated Fourier modes (16x24).
- Adaptive Spatial Gating:
  alpha = sigmoid(Conv(GELU(Conv([F_vit, F_fno]))))
  F_fused = alpha * F_vit + (1 - alpha) * F_fno
- Latent Space: OceanEmbed [B, 64, 101, 241]
- Depth-Conditioned FiLM Decoder for 15 standard depths [0-1000m].
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
        feat = self.stem3d(x)
        q = self.temporal_query
        k = self.temporal_proj(feat)
        attn = (q * k).sum(dim=1, keepdim=True) / math.sqrt(32)
        attn_weights = F.softmax(attn, dim=2)
        pooled = (feat * attn_weights).sum(dim=2)
        out = self.proj2d(pooled)
        return out


class ResidualBlock2D(nn.Module):
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


class TransformerBlock(nn.Module):
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


class SpectralConv2d(nn.Module):
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

    def forward(self, feat_vit, feat_fno):
        cat = torch.cat([feat_vit, feat_fno], dim=1)
        alpha = self.gate_net(cat)
        fused = alpha * feat_vit + (1.0 - alpha) * feat_fno
        out = self.refine(fused)
        return out, alpha


class DepthConditionedDecoder(nn.Module):
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


class OceanFNO_ViT(nn.Module):
    """
    Main Model B: Parallel Dual-Branch FNO + Vision Transformer with Adaptive Gated Fusion.
    - Global Attention Branch: 6-Layer ViT with 1,586 spatial patch tokens (4x4 patches).
    - Global Spectral Branch: 4-Layer 2D Fourier Neural Operator (modes 16x24).
    - Fusion: Adaptive Spatial Gating -> Unified Latent OceanEmbed [B, 64, 101, 241].
    - Decoder: Depth-Conditioned FiLM Decoder for 15 Standard Depths [0-1000m].
    """
    def __init__(self, in_vars=7, num_days=7, num_depths=15, width=64, embed_dim=256,
                 patch_size=4, num_vit_layers=6, num_heads=8, modes1=16, modes2=24, num_fno_layers=4):
        super().__init__()
        self.width = width
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.modes1 = modes1
        self.modes2 = modes2
        
        # 1. Spatiotemporal Lifting (Shared)
        self.temporal_encoder = TemporalAttentionEncoder(in_vars=in_vars, num_days=num_days, out_channels=width)
        
        # -------------------------------------------------------------
        # BRANCH 1: Vision Transformer (ViT)
        # -------------------------------------------------------------
        self.H_pad = 104
        self.W_pad = 244
        self.num_patches = (self.H_pad // patch_size) * (self.W_pad // patch_size) # 1,586
        
        self.patch_embed = nn.Conv2d(width, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, embed_dim) * 0.02)
        
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=4.0)
            for _ in range(num_vit_layers)
        ])
        self.vit_norm = nn.LayerNorm(embed_dim)
        
        self.unpatch_conv = nn.Sequential(
            nn.Conv2d(embed_dim, 128, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 128),
            nn.GELU()
        )
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(128, width, kernel_size=4, stride=4, bias=False),
            nn.GroupNorm(8, width),
            nn.GELU()
        )
        self.vit_refine = nn.Sequential(
            nn.Conv2d(width + width, width, kernel_size=3, padding=1, bias=False),
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
        
        # 2. Branch 1: ViT forward pass
        feat_padded = F.pad(x0, (0, 3, 0, 3), mode="replicate") # [B, 64, 104, 244]
        patches = self.patch_embed(feat_padded) # [B, 256, 26, 61]
        Hp, Wp = patches.shape[-2], patches.shape[-1]
        tokens = patches.flatten(2).transpose(1, 2) + self.pos_embed # [B, 1586, 256]
        
        for layer in self.transformer_layers:
            tokens = layer(tokens)
        tokens = self.vit_norm(tokens)
        
        feat_2d = tokens.transpose(1, 2).view(B, self.embed_dim, Hp, Wp)
        feat_unpatched = self.unpatch_conv(feat_2d)
        feat_up = self.upsample(feat_unpatched)
        feat_crop = feat_up[:, :, :H, :W]
        feat_vit = self.vit_refine(torch.cat([feat_crop, x0], dim=1)) # [B, 64, 101, 241]

        # 3. Branch 2: FNO forward pass
        grid = self._get_grid((B, H, W), x.device)
        feat_fno_in = self.coord_proj(torch.cat([x0, grid], dim=1))
        f_curr = feat_fno_in
        for fno_layer in self.fno_layers:
            f_curr = fno_layer(f_curr)
        feat_fno = self.fno_out_proj(f_curr) # [B, 64, 101, 241]

        # 4. Adaptive Gated Fusion
        ocean_embed, alpha_gate = self.fusion(feat_vit, feat_fno) # [B, 64, 101, 241]

        # 5. Depth-Conditioned Subsurface Reconstruction
        out = self.depth_decoder(ocean_embed) # [B, 15, 101, 241]

        return out, ocean_embed
