"""
OceanEmbed - Fourier Neural Operator 2D Architecture (Model Baseline 3)
SIH26066: Satellite Embedding-Based Deep Learning Framework

Mathematical Formulation:
v_{l+1}(x) = sigma( K(v_l)(x) + W v_l(x) )
where K is the spectral integral kernel operator:
K(v)(x) = F^{-1}( R_{phi} * F(v) )(x)
- Global spectral convolution parameterizes long-range spatial dependencies.
- Truncated Fourier modes: modes1 = 16 (lat), modes2 = 24 (lon).
- Parameterized with complex weight tensors.
- Depth-conditioned decoder predicts physical temperature at 15 standard depths.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SpectralConv2d(nn.Module):
    """
    2D Fourier Spectral Convolution Layer.
    Computes FFT -> Complex matrix multiplication on low-frequency modes -> IFFT.
    """
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 # Number of Fourier modes in latitude (H)
        self.modes2 = modes2 # Number of Fourier modes in longitude (W)

        scale = 1.0 / (in_channels * out_channels)
        # Learnable complex weights for corner 1: top-left (low positive freqs)
        self.weights1 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat)
        )
        # Learnable complex weights for corner 2: bottom-left (low negative freqs)
        self.weights2 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat)
        )

    def compl_mul2d(self, input, weights):
        # input: [B, in_channels, H, W], weights: [in_channels, out_channels, H, W]
        # output: [B, out_channels, H, W]
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        orig_dtype = x.dtype
        x = x.float()
        B, C, H, W = x.shape
        
        # 1. 2D Real Fast Fourier Transform in float32
        x_ft = torch.fft.rfft2(x)

        # 2. Multiply relevant Fourier modes
        out_ft = torch.zeros(B, self.out_channels, H, W // 2 + 1, dtype=torch.cfloat, device=x.device)
        
        # Top-left corner (positive frequencies in y)
        out_ft[:, :, :self.modes1, :self.modes2] = self.compl_mul2d(
            x_ft[:, :, :self.modes1, :self.modes2], self.weights1
        )
        
        # Bottom-left corner (negative frequencies in y)
        out_ft[:, :, -self.modes1:, :self.modes2] = self.compl_mul2d(
            x_ft[:, :, -self.modes1:, :self.modes2], self.weights2
        )

        # 3. 2D Inverse Real Fast Fourier Transform back to spatial domain
        x_out = torch.fft.irfft2(out_ft, s=(H, W))
        return x_out.to(orig_dtype)


class FNOBlock2D(nn.Module):
    """
    Standard FNO Layer: SpectralConv2d + 1x1 Convolution bypass + GELU.
    """
    def __init__(self, channels, modes1, modes2):
        super().__init__()
        self.spectral_conv = SpectralConv2d(channels, channels, modes1, modes2)
        self.bypass = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.norm = nn.GroupNorm(8, channels)
        self.act = nn.GELU()

    def forward(self, x):
        # Spectral path + spatial bypass residual
        x_spec = self.spectral_conv(x)
        x_by = self.bypass(x)
        return self.act(self.norm(x_spec + x_by))


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


class OceanFNO(nn.Module):
    """
    Fourier Neural Operator (FNO-only Baseline 3) for Surface-to-Subsurface Temperature Reconstruction.
    Uses 4 deep FNO spectral blocks with modes1=16, modes2=24, channel_width=64.
    """
    def __init__(self, in_vars=7, num_days=7, num_depths=15, width=64, modes1=16, modes2=24, num_fno_layers=4):
        super().__init__()
        self.width = width
        self.modes1 = modes1
        self.modes2 = modes2
        
        # 1. Spatiotemporal Input Lifting
        self.temporal_encoder = TemporalAttentionEncoder(in_vars=in_vars, num_days=num_days, out_channels=width)
        
        # Grid positional encoding channels (lat, lon normalized to [-1, 1])
        self.coord_proj = nn.Conv2d(width + 2, width, kernel_size=1)
        
        # 2. Deep Spectral Operator Blocks
        self.fno_layers = nn.ModuleList([
            FNOBlock2D(channels=width, modes1=modes1, modes2=modes2)
            for _ in range(num_fno_layers)
        ])
        
        # 3. Latent OceanEmbed Projection
        self.ocean_embed_proj = nn.Sequential(
            nn.Conv2d(width, width, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, width),
            nn.GELU()
        )
        
        # 4. Depth-Conditioned Decoder
        self.depth_decoder = DepthConditionedDecoder(embed_dim=width, num_depths=num_depths)

    def _get_grid(self, shape, device):
        B, H, W = shape
        grid_x = torch.linspace(-1, 1, steps=W, device=device).view(1, 1, 1, W).repeat(B, 1, H, 1)
        grid_y = torch.linspace(-1, 1, steps=H, device=device).view(1, 1, H, 1).repeat(B, 1, 1, W)
        return torch.cat((grid_y, grid_x), dim=1) # [B, 2, H, W]

    def forward(self, x):
        # x: [B, 7, 7, 101, 241]
        B = x.shape[0]
        H, W = x.shape[-2], x.shape[-1]
        
        # Temporal Attention projection
        feat = self.temporal_encoder(x) # [B, 64, 101, 241]
        
        # Append continuous normalized spatial coordinate grid
        grid = self._get_grid((B, H, W), x.device) # [B, 2, 101, 241]
        feat = self.coord_proj(torch.cat([feat, grid], dim=1)) # [B, 64, 101, 241]
        
        # Pass through 4 Spectral Convolution Operator Blocks
        for fno_layer in self.fno_layers:
            feat = fno_layer(feat)
            
        # Latent representation: OceanEmbed
        ocean_embed = self.ocean_embed_proj(feat) # [B, 64, 101, 241]
        
        # 3D Subsurface Temperature Reconstruction
        out = self.depth_decoder(ocean_embed) # [B, 15, 101, 241]
        return out, ocean_embed
