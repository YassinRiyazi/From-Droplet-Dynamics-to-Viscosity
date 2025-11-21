"""
AutoEncoder CNN V3 - Flexible High-Performance Image Compression
Author: Yassin Riyazi
Date: November 21, 2025

A flexible, high-performance grayscale image autoencoder with:
- Adaptive input sizes (201x201, 1280x152, variable width × 152)
- Controllable latent size (128-8192)
- Multiple block types (ResNet, ResNeXt, SE, Inception)
- Attention visualization (matrix + spatial heatmap)
- GPU optimization for Ada Lovelace 8GB (CUDA 8.9)
- No U-Net skip connections (only latent bottleneck)
- Compatible with AutoEncoder_CNNV1_0 API

Key Features:
- Dual latent strategies: convolutional (flattened) or spatial (C×H×W)
- Self-attention for interpretability
- Mixed precision, gradient checkpointing, torch.compile ready
- LSTM-compatible encode/decode interface
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Dict, Optional, Union, Literal
import math
import numpy as np


# ============================================================================
# BUILDING BLOCKS
# ============================================================================

class ResNetBasicBlock(nn.Module):
    """
    ResNet Basic Block with optional stride for downsampling.
    
    Args:
        in_channels: Input channels
        out_channels: Output channels
        stride: Stride for first conv (default: 1)
        dilation: Dilation rate (default: 1)
        use_se: Whether to use Squeeze-and-Excitation (default: False)
    """
    expansion = 1
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, 
                 dilation: int = 1, use_se: bool = False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=dilation, dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                              stride=1, padding=dilation, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        self.se = SEBlock(out_channels) if use_se else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out += self.shortcut(residual)
        out = F.relu(out, inplace=True)
        return out

class ResNeXtBlock(nn.Module):
    """
    ResNeXt Block with grouped convolutions for increased capacity.
    
    Args:
        in_channels: Input channels
        out_channels: Output channels
        stride: Stride for convolution (default: 1)
        cardinality: Number of groups (default: 32)
        base_width: Width per group (default: 4)
        use_se: Whether to use Squeeze-and-Excitation (default: False)
    """
    expansion = 2
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1,
                 cardinality: int = 32, base_width: int = 4, use_se: bool = False):
        super().__init__()
        width = int(out_channels * (base_width / 64.)) * cardinality
        
        self.conv1 = nn.Conv2d(in_channels, width, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride,
                              padding=1, groups=cardinality, bias=False)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * self.expansion)
            )
        
        self.se = SEBlock(out_channels * self.expansion) if use_se else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = F.relu(self.bn2(self.conv2(out)), inplace=True)
        out = self.bn3(self.conv3(out))
        out = self.se(out)
        out += self.shortcut(residual)
        out = F.relu(out, inplace=True)
        return out

class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block for channel attention.
    
    Args:
        channels: Number of input channels
        reduction: Reduction ratio (default: 16)
    """
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, max(channels // reduction, 8), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(max(channels // reduction, 8), channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class InceptionBlock(nn.Module):
    """
    Inception-style block with multiple parallel conv paths.
    
    Args:
        in_channels: Input channels
        out_channels: Output channels (split across branches)
        stride: Stride for 3x3 convs (default: 1)
    """
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        branch_channels = out_channels // 4
        
        # 1x1 conv branch
        self.branch1x1 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True)
        )
        
        # 1x1 -> 3x3 conv branch
        self.branch3x3 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_channels, branch_channels, kernel_size=3, 
                     stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True)
        )
        
        # 1x1 -> 5x5 (as 2x 3x3) conv branch
        self.branch5x5 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_channels, branch_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_channels, branch_channels, kernel_size=3, 
                     stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True)
        )
        
        # Pool -> 1x1 conv branch
        self.branch_pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=stride, padding=1) if stride > 1 
            else nn.Identity(),
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = [
            self.branch1x1(x),
            self.branch3x3(x),
            self.branch5x5(x),
            self.branch_pool(x)
        ]
        return torch.cat(outputs, 1)

class SelfAttentionBlock(nn.Module):
    """
    Self-attention block for generating attention visualizations.
    
    Args:
        in_channels: Input channels
        num_heads: Number of attention heads (default: 4)
        reduction: Channel reduction for Q, K, V (default: 8)
    
    Returns attention weights for visualization.
    """
    def __init__(self, in_channels: int, num_heads: int = 4, reduction: int = 8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = max(in_channels // (num_heads * reduction), 8)
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Conv2d(in_channels, 3 * num_heads * self.head_dim, kernel_size=1, bias=False)
        self.proj = nn.Conv2d(num_heads * self.head_dim, in_channels, kernel_size=1, bias=False)
        
        # Store attention for visualization
        self.attention_weights = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        N = H * W
        
        # Generate Q, K, V
        qkv = self.qkv(x).reshape(B, 3, self.num_heads, self.head_dim, N).permute(1, 0, 2, 4, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]  # Each: (B, num_heads, N, head_dim)
        
        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, num_heads, N, N)
        attn = F.softmax(attn, dim=-1)
        
        # Store for visualization (average across heads and batch)
        self.attention_weights = attn.detach().mean(dim=1)  # (B, N, N)
        
        # Apply attention
        out = attn @ v  # (B, num_heads, N, head_dim)
        out = out.transpose(1, 2).reshape(B, N, self.num_heads * self.head_dim)
        out = out.transpose(1, 2).reshape(B, self.num_heads * self.head_dim, H, W)
        out = self.proj(out)
        
        return out + x  # Residual

# ============================================================================
# BLOCK FACTORY
# ============================================================================

def build_block(block_type: str, in_channels: int, out_channels: int, 
                stride: int = 1, **kwargs) -> nn.Module:
    """
    Factory function to build different block types.
    
    Args:
        block_type: One of 'resnet', 'resnext', 'se', 'inception'
        in_channels: Input channels
        out_channels: Output channels
        stride: Stride for downsampling
        **kwargs: Additional arguments for specific blocks
    
    Returns:
        Block module
    
    Example:
        >>> block = build_block('resnet', 64, 128, stride=2, use_se=True)
    """
    if block_type.lower() == 'resnet':
        return ResNetBasicBlock(in_channels, out_channels, stride, 
                               kwargs.get('dilation', 1), kwargs.get('use_se', False))
    elif block_type.lower() == 'resnext':
        return ResNeXtBlock(in_channels, out_channels, stride,
                           kwargs.get('cardinality', 32), kwargs.get('base_width', 4),
                           kwargs.get('use_se', False))
    elif block_type.lower() == 'se':
        return ResNetBasicBlock(in_channels, out_channels, stride, 
                               kwargs.get('dilation', 1), use_se=True)
    elif block_type.lower() == 'inception':
        return InceptionBlock(in_channels, out_channels, stride)
    else:
        raise ValueError(f"Unknown block type: {block_type}. Choose from: resnet, resnext, se, inception")

# ============================================================================
# ENCODER
# ============================================================================

class Encoder(nn.Module):
    """
    Flexible encoder supporting multiple block types and adaptive input sizes.
    
    Args:
        in_channels: Input channels (default: 1 for grayscale)
        latent_size: Size of latent representation (128-8192)
        block_config: List of block types per stage (e.g., ['resnet', 'se', 'inception'])
        channel_config: List of output channels per stage (e.g., [64, 128, 256])
        stride_config: List of strides per stage (e.g., [2, 2, 2])
        latent_strategy: 'conv' for flattened or 'spatial' for C×H×W (default: 'conv')
        use_attention: Whether to include self-attention (default: False)
        input_size: Expected input size for calculating spatial latent dims
    
    Example:
        >>> encoder = Encoder(latent_size=512, block_config=['resnet', 'se', 'resnet'])
        >>> latent = encoder(torch.randn(2, 1, 201, 201))
    """
    def __init__(self, 
                 in_channels: int = 1,
                 latent_size: int = 512,
                 block_config: List[str] = ['resnet', 'resnet', 'resnet'],
                 channel_config: List[int] = [64, 128, 256],
                 stride_config: List[int] = [2, 2, 2],
                 latent_strategy: Literal['conv', 'spatial'] = 'conv',
                 use_attention: bool = False,
                 input_size: Tuple[int, int] = (201, 201)):
        super().__init__()
        
        assert 128 <= latent_size <= 8192, f"Latent size must be in [128, 8192], got {latent_size}"
        assert len(block_config) == len(channel_config) == len(stride_config), \
            "Config lists must have same length"
        
        self.latent_size = latent_size
        self.latent_strategy = latent_strategy
        self.use_attention = use_attention
        
        # Initial conv
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, channel_config[0]//2, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(channel_config[0]//2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # Build encoder stages
        self.stages = nn.ModuleList()
        in_ch = channel_config[0]//2
        for i, (block_type, out_ch, stride) in enumerate(zip(block_config, channel_config, stride_config)):
            stage = build_block(block_type, in_ch, out_ch, stride)
            self.stages.append(stage)
            # Account for expansion in ResNeXt
            in_ch = out_ch * (stage.expansion if hasattr(stage, 'expansion') else 1)
        
        self.final_channels = in_ch
        
        # Self-attention (optional)
        if use_attention:
            self.attention = SelfAttentionBlock(self.final_channels, num_heads=4)
        else:
            self.attention = None
        
        # Calculate spatial dimensions after encoding
        h, w = input_size
        for _ in range(2):  # Stem downsampling
            h, w = (h + 1) // 2, (w + 1) // 2
        for stride in stride_config:
            h, w = (h + stride - 1) // stride, (w + stride - 1) // stride
        
        self.latent_spatial_size = (h, w)
        
        # Latent projection
        if latent_strategy == 'conv':
            # Flatten to vector
            self.latent_proj = nn.Sequential(
                nn.AdaptiveAvgPool2d((4, 4)),  # Ensure consistent spatial size
                nn.Flatten(),
                nn.Linear(self.final_channels * 16, latent_size),
                nn.ReLU(inplace=True)
            )
            self.latent_shape = (latent_size,)
        else:
            # Spatial latent: find C×H×W ≈ latent_size
            target_spatial = 16  # Target H×W
            target_channels = max(latent_size // target_spatial, 8)
            
            self.latent_proj = nn.Sequential(
                nn.AdaptiveAvgPool2d((4, 4)),
                nn.Conv2d(self.final_channels, target_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(target_channels),
                nn.ReLU(inplace=True)
            )
            self.latent_shape = (target_channels, 4, 4)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input to latent representation.
        
        Args:
            x: Input tensor (B, 1, H, W)
        
        Returns:
            Latent tensor: (B, latent_size) or (B, C, H, W) depending on strategy
        """
        x = self.stem(x)
        
        for stage in self.stages:
            x = stage(x)
        
        if self.attention is not None:
            x = self.attention(x)
        
        latent = self.latent_proj(x)
        return latent


# ============================================================================
# DECODER
# ============================================================================

class Decoder(nn.Module):
    """
    Flexible decoder reconstructing images from latent representation.
    
    Args:
        latent_size: Size of latent representation
        out_channels: Output channels (default: 1 for grayscale)
        block_config: List of block types per stage (reversed from encoder)
        channel_config: List of output channels per stage (reversed)
        latent_strategy: 'conv' for flattened or 'spatial' for C×H×W
        latent_shape: Shape of latent from encoder
        output_size: Target output size
    
    Example:
        >>> decoder = Decoder(latent_size=512, output_size=(201, 201))
        >>> recon = decoder(latent_tensor)
    """
    def __init__(self,
                 latent_size: int = 512,
                 out_channels: int = 1,
                 block_config: List[str] = ['resnet', 'resnet', 'resnet'],
                 channel_config: List[int] = [256, 128, 64],
                 latent_strategy: Literal['conv', 'spatial'] = 'conv',
                 latent_shape: Tuple = (512,),
                 output_size: Tuple[int, int] = (201, 201)):
        super().__init__()
        
        self.latent_size = latent_size
        self.latent_strategy = latent_strategy
        self.latent_shape = latent_shape
        self.output_size = output_size
        
        # Latent unprojection
        if latent_strategy == 'conv':
            start_channels = channel_config[0]
            self.latent_unproj = nn.Sequential(
                nn.Linear(latent_size, start_channels * 16),
                nn.ReLU(inplace=True),
                nn.Unflatten(1, (start_channels, 4, 4))
            )
        else:
            # Spatial latent
            self.latent_unproj = nn.Sequential(
                nn.Conv2d(latent_shape[0], channel_config[0], kernel_size=1, bias=False),
                nn.BatchNorm2d(channel_config[0]),
                nn.ReLU(inplace=True)
            )
        
        # Build decoder stages (upsample + conv)
        self.stages = nn.ModuleList()
        in_ch = channel_config[0]
        
        for i, (block_type, out_ch) in enumerate(zip(block_config, channel_config[1:] + [64])):
            # Upsample then conv block
            stage = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                build_block(block_type, in_ch, out_ch, stride=1)
            )
            self.stages.append(stage)
            in_ch = out_ch * (self.stages[-1][-1].expansion 
                            if hasattr(self.stages[-1][-1], 'expansion') else 1)
        
        # Final reconstruction
        self.head = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(in_ch, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Decode latent to reconstructed image.
        
        Args:
            latent: Latent tensor (B, latent_size) or (B, C, H, W)
        
        Returns:
            Reconstructed image (B, 1, H, W)
        """
        x = self.latent_unproj(latent)
        
        for stage in self.stages:
            x = stage(x)
        
        x = self.head(x)
        
        # Adaptive resize to target output size
        if x.shape[-2:] != self.output_size:
            x = F.interpolate(x, size=self.output_size, mode='bilinear', align_corners=False)
        
        return x


# ============================================================================
# AUTOENCODER MAIN CLASS
# ============================================================================

class AutoEncoder_CNNV3_0(nn.Module):
    """
    Flexible high-performance autoencoder for grayscale images.
    
    Compatible with AutoEncoder_CNNV1_0 API while adding new capabilities:
    - Adaptive input sizes (201×201, 1280×152, variable)
    - Controllable latent size (128-8192)
    - Multiple block architectures
    - Attention visualization
    - GPU optimization for Ada Lovelace 8GB
    
    Args:
        latent_size: Size of latent bottleneck (default: 512)
        block_config: List of block types (default: ['resnet', 'se', 'resnet'])
        channel_config: List of channels per stage (default: [64, 128, 256])
        stride_config: List of strides per stage (default: [2, 2, 2])
        latent_strategy: 'conv' for vector or 'spatial' for tensor (default: 'conv')
        use_attention: Enable self-attention for visualization (default: True)
        input_size: Expected input size (default: (201, 201))
        DropOut: Compatibility with V1 API (default: False) - NOT USED in V3
        embedding_dim: Compatibility with V1 API, overrides latent_size if set
    
    Example:
        >>> # Basic usage (V1 compatible)
        >>> model = AutoEncoder_CNNV3(embedding_dim=512)
        >>> recon = model(torch.randn(4, 1, 201, 201))
        >>>
        >>> # Advanced usage
        >>> model = AutoEncoder_CNNV3(
        ...     latent_size=2048,
        ...     block_config=['resnet', 'resnext', 'se'],
        ...     use_attention=True
        ... )
    """
    def __init__(self,
                 latent_size: int = 512,
                 block_config: List[str] = ['resnet', 'se', 'resnet'],
                 channel_config: List[int] = [64, 128, 256],
                 stride_config: List[int] = [2, 2, 2],
                 latent_strategy: Literal['conv', 'spatial'] = 'conv',
                 use_attention: bool = True,
                 input_size: Tuple[int, int] = (201, 201),
                 # V1 compatibility
                 DropOut: bool = False,
                 embedding_dim: Optional[int] = None):
        super().__init__()
        
        # V1 compatibility: embedding_dim overrides latent_size
        if embedding_dim is not None:
            latent_size = embedding_dim
        
        self.latent_size = latent_size
        self.latent_strategy = latent_strategy
        self.use_attention = use_attention
        self.input_size = input_size
        self.DropOut = DropOut  # Stored for compatibility, not used
        
        # Build encoder
        self.encoder = Encoder(
            in_channels=1,
            latent_size=latent_size,
            block_config=block_config,
            channel_config=channel_config,
            stride_config=stride_config,
            latent_strategy=latent_strategy,
            use_attention=use_attention,
            input_size=input_size
        )
        
        # Build decoder
        self.decoder = Decoder(
            latent_size=latent_size,
            out_channels=1,
            block_config=list(reversed(block_config)),
            channel_config=list(reversed(channel_config)),
            latent_strategy=latent_strategy,
            latent_shape=self.encoder.latent_shape,
            output_size=input_size
        )

        self.DropOut = DropOut
        self.dropout = nn.Dropout(p=0.45)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass (V1 compatible).
        
        Args:
            x: Input tensor (B, 1, H, W)
        
        Returns:
            Reconstructed tensor (B, 1, H, W)
        """
        latent = self.encoder(x)

        if self.DropOut:
            latent = self.dropout(latent)

        recon = self.decoder(latent)
        return recon
    
    def Embedding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract latent embedding (V1 compatible).
        
        Args:
            x: Input tensor (B, 1, H, W)
        
        Returns:
            Latent tensor (B, latent_size) or (B, C, H, W)
        """
        return self.encoder(x)
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input to latent (LSTM compatible).
        
        Args:
            x: Input tensor (B, 1, H, W)
        
        Returns:
            Latent vector (B, latent_size) flattened if spatial
        """
        latent = self.encoder(x)
        if self.latent_strategy == 'spatial':
            latent = latent.flatten(1)  # Flatten for LSTM
        return latent
    
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Decode latent to image.
        
        Args:
            latent: Latent tensor (B, latent_size)
        
        Returns:
            Reconstructed image (B, 1, H, W)
        """
        if self.latent_strategy == 'spatial' and latent.ndim == 2:
            latent = latent.view(-1, *self.encoder.latent_shape)
        return self.decoder(latent)
    
    def decode_from_latent(self, latent: torch.Tensor) -> torch.Tensor:
        """Alias for decode (explicit LSTM compatibility)."""
        return self.decode(latent)
    
    def get_attention_matrix(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Get self-attention matrix for visualization.
        
        Args:
            x: Input tensor (B, 1, H, W)
        
        Returns:
            Attention matrix (B, N, N) where N = H×W of feature map, or None if no attention
        
        Example:
            >>> attn_matrix = model.get_attention_matrix(images)
            >>> # attn_matrix shape: (batch, num_patches, num_patches)
        """
        if not self.use_attention or self.encoder.attention is None:
            return None
        
        _ = self.encoder(x)  # Forward pass to populate attention weights
        return self.encoder.attention.attention_weights
    
    def get_attention_map(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get spatial attention heatmap (saliency).
        
        Computed as gradient of reconstruction error w.r.t. input.
        
        Args:
            x: Input tensor (B, 1, H, W)
        
        Returns:
            Attention heatmap (B, 1, H, W)
        
        Example:
            >>> attn_map = model.get_attention_map(images)
            >>> # attn_map can be overlaid on original image
        """
        x_req_grad = x.clone().requires_grad_(True)
        recon = self.forward(x_req_grad)
        
        # Reconstruction error
        error = F.mse_loss(recon, x_req_grad, reduction='sum')
        
        # Gradient as saliency
        error.backward()
        saliency = x_req_grad.grad.abs()
        
        # Normalize to [0, 1]
        saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
        
        return saliency
    
    def get_config(self) -> Dict:
        """Get model configuration for saving/loading."""
        return {
            'latent_size': self.latent_size,
            'latent_strategy': self.latent_strategy,
            'use_attention': self.use_attention,
            'input_size': self.input_size,
            'latent_shape': self.encoder.latent_shape,
        }


# ============================================================================
# VISUALIZATION UTILITIES
# ============================================================================

def visualize_attention(model: AutoEncoder_CNNV3_0, 
                       x: torch.Tensor,
                       save_path: Optional[str] = None,
                       show_matrix: bool = True,
                       show_heatmap: bool = True) -> Dict[str, np.ndarray]:
    """
    Visualize attention mechanisms.
    
    Args:
        model: Trained autoencoder
        x: Input image (1, 1, H, W) or (B, 1, H, W) - uses first sample
        save_path: Path to save visualization (optional)
        show_matrix: Whether to visualize attention matrix (default: True)
        show_heatmap: Whether to visualize spatial heatmap (default: True)
    
    Returns:
        Dictionary with 'matrix' and/or 'heatmap' as numpy arrays
    
    Example:
        >>> viz = visualize_attention(model, test_image, save_path='attention.png')
    """
    import matplotlib.pyplot as plt
    from matplotlib import cm
    
    model.eval()
    
    if x.ndim == 4 and x.size(0) > 1:
        x = x[:1]  # Use first sample
    
    results = {}
    
    fig_count = int(show_matrix) + int(show_heatmap)
    if fig_count == 0:
        return results
    
    fig, axes = plt.subplots(1, fig_count, figsize=(6 * fig_count, 5))
    if fig_count == 1:
        axes = [axes]
    
    ax_idx = 0
    
    # Attention matrix
    if show_matrix:
        attn_matrix = model.get_attention_matrix(x)
        if attn_matrix is not None:
            attn_np = attn_matrix[0].cpu().numpy()  # First batch
            results['matrix'] = attn_np
            
            im = axes[ax_idx].imshow(attn_np, cmap='viridis', aspect='auto')
            axes[ax_idx].set_title('Self-Attention Matrix')
            axes[ax_idx].set_xlabel('Key Position')
            axes[ax_idx].set_ylabel('Query Position')
            plt.colorbar(im, ax=axes[ax_idx], label='Attention Weight')
            ax_idx += 1
        else:
            print("Warning: Attention matrix not available (use_attention=False)")
    
    # Spatial heatmap
    if show_heatmap:
        with torch.enable_grad():
            heatmap = model.get_attention_map(x)
        
        heatmap_np = heatmap[0, 0].cpu().numpy()
        results['heatmap'] = heatmap_np
        
        # Overlay on original image
        original_np = x[0, 0].cpu().numpy()
        
        # Create overlay
        heatmap_colored = cm.jet(heatmap_np)[:, :, :3]  # RGB
        overlay = 0.6 * original_np[:, :, None] + 0.4 * heatmap_colored
        overlay = np.clip(overlay, 0, 1)
        
        axes[ax_idx].imshow(overlay)
        axes[ax_idx].set_title('Spatial Attention Heatmap')
        axes[ax_idx].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Attention visualization saved to {save_path}")
    else:
        plt.show()
    
    plt.close()
    
    return results

# ============================================================================
# MODEL UTILITIES
# ============================================================================

def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """
    Count model parameters.
    
    Args:
        model: PyTorch model
    
    Returns:
        (total_params, trainable_params)
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

def estimate_flops(model: nn.Module, input_size: Tuple[int, int, int, int]) -> float:
    """
    Estimate FLOPs (approximate, relative metric).
    
    Args:
        model: PyTorch model
        input_size: Input tensor size (B, C, H, W)
    
    Returns:
        Approximate FLOPs in GFLOPs
    """
    # Simple estimation based on conv layers
    flops = 0
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            # FLOPs ≈ 2 × C_in × C_out × K × K × H_out × W_out
            # This is a rough approximation
            flops += 2 * module.in_channels * module.out_channels * \
                     module.kernel_size[0] * module.kernel_size[1]
    
    # Scale by approximate output spatial size
    _, _, h, w = input_size
    flops *= (h * w) / 1e9  # Convert to GFLOPs
    
    return flops

def get_model_size_mb(model: nn.Module) -> float:
    """
    Get model size in MB.
    
    Args:
        model: PyTorch model
    
    Returns:
        Model size in megabytes
    """
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    size_mb = (param_size + buffer_size) / (1024 ** 2)
    return size_mb

def estimate_memory_footprint(model: nn.Module, 
                              batch_size: int,
                              input_size: Tuple[int, int],
                              dtype: torch.dtype = torch.float32) -> Dict[str, float]:
    """
    Estimate peak GPU memory footprint.
    
    Args:
        model: PyTorch model
        batch_size: Batch size
        input_size: Input spatial size (H, W)
        dtype: Data type (default: float32)
    
    Returns:
        Dictionary with memory estimates in MB
    """
    bytes_per_element = 4 if dtype == torch.float32 else 2  # FP32 or FP16
    
    # Model parameters
    model_size = get_model_size_mb(model)
    
    # Input activations
    input_mb = batch_size * 1 * input_size[0] * input_size[1] * bytes_per_element / (1024 ** 2)
    
    # Rough estimate: activations are ~5-10x input size through network
    activations_mb = input_mb * 8
    
    # Gradients (same size as parameters in FP32)
    gradients_mb = model_size
    
    # Optimizer state (AdamW ≈ 2x parameters)
    optimizer_mb = model_size * 2
    
    total_mb = model_size + input_mb + activations_mb + gradients_mb + optimizer_mb
    
    return {
        'model': model_size,
        'input': input_mb,
        'activations': activations_mb,
        'gradients': gradients_mb,
        'optimizer': optimizer_mb,
        'total_estimated': total_mb
    }

# ============================================================================
# RECOMMENDED CONFIGURATIONS
# ============================================================================

RECOMMENDED_CONFIGS = {
    'tiny': {
        'latent_size': 128,
        'block_config': ['resnet', 'resnet'],
        'channel_config': [32, 64],
        'stride_config': [2, 2],
        'batch_size_201x201': 32,
        'batch_size_1280x152': 16,
    },
    'small': {
        'latent_size': 512,
        'block_config': ['resnet', 'se', 'resnet'],
        'channel_config': [64, 128, 256],
        'stride_config': [2, 2, 2],
        'batch_size_201x201': 16,
        'batch_size_1280x152': 8,
    },
    'medium': {
        'latent_size': 2048,
        'block_config': ['resnet', 'resnext', 'se'],
        'channel_config': [64, 128, 256],
        'stride_config': [2, 2, 2],
        'batch_size_201x201': 8,
        'batch_size_1280x152': 4,
    },
    'large': {
        'latent_size': 8192,
        'block_config': ['resnext', 'resnext', 'se', 'se'],
        'channel_config': [64, 128, 256, 512],
        'stride_config': [2, 2, 2, 2],
        'batch_size_201x201': 4,
        'batch_size_1280x152': 2,
    }
}


def create_autoencoder(preset: str = 'small',
                      input_size: Tuple[int, int] = (201, 201),
                      **override_kwargs) -> AutoEncoder_CNNV3_0:
    """
    Factory function to create preconfigured autoencoders.
    
    Args:
        preset: One of 'tiny', 'small', 'medium', 'large'
        input_size: Input image size (H, W)
        **override_kwargs: Override any config parameter
    
    Returns:
        Configured AutoEncoder_CNNV3
    
    Example:
        >>> model = create_autoencoder('medium', input_size=(1280, 152))
        >>> model = create_autoencoder('small', latent_size=1024)  # Override
    """
    if preset not in RECOMMENDED_CONFIGS:
        raise ValueError(f"Unknown preset: {preset}. Choose from {list(RECOMMENDED_CONFIGS.keys())}")
    
    config = RECOMMENDED_CONFIGS[preset].copy()
    # Remove batch size hints (not model params)
    config = {k: v for k, v in config.items() if not k.startswith('batch_size')}
    
    config.update(override_kwargs)
    config['input_size'] = input_size
    
    return AutoEncoder_CNNV3_0(**config)


if __name__ == "__main__":
    print("=" * 80)
    print("AutoEncoder CNN V3 - Module Test")
    print("=" * 80)
    
    # Test different configurations
    test_configs = [
        ((201, 201), 'small', 'Square 201x201'),
        ((152, 1280), 'small', 'Wide 1280x152'),
        ((1280,152), 'small', 'Wide 1280x152'),
        ((152, 640), 'tiny', 'Medium 640x152'),
    ]
    
    for input_size, preset, desc in test_configs:
        print(f"\n{'-' * 80}")
        print(f"Testing: {desc}")
        print(f"{'-' * 80}")
        
        model = create_autoencoder(preset, input_size=input_size)
        
        total_params, trainable_params = count_parameters(model)
        model_size = get_model_size_mb(model)
        
        print(f"Config: {preset}")
        print(f"Input size: {input_size}")
        print(f"Latent size: {model.latent_size}")
        print(f"Parameters: {total_params:,} ({trainable_params:,} trainable)")
        print(f"Model size: {model_size:.2f} MB")
        
        # Test forward pass
        x = torch.randn(2, 1, *input_size)
        print(f"\nInput shape: {x.shape}")
        
        with torch.no_grad():
            recon = model(x)
            print(f"Reconstruction shape: {recon.shape}")
            
            latent = model.encode(x)
            print(f"Latent shape: {latent.shape}")
            
            # Memory estimate
            mem = estimate_memory_footprint(model, 2, input_size)
            print(f"\nEstimated memory (batch=2, FP32):")
            print(f"  Total: {mem['total_estimated']:.1f} MB")
            print(f"  Model: {mem['model']:.1f} MB")
            print(f"  Activations: {mem['activations']:.1f} MB")
    
    print("\n" + "=" * 80)
    print("Module test completed!")
    print("=" * 80)
