"""
    Author: Yassin Riyazi (Modified by AI Assistant)
    Date: 20-11-2025

    Optimized Transformer-based Autoencoder for variable-sized grayscale images.
    
    Key Features:
    - Variable input sizes: 1280x152, 201x201, or any Xx152
    - Configurable latent dimension: 128-8192
    - Attention visualization capability
    - Optimized for Ada Lovelace architecture (RTX 40xx series)
    - Memory efficient with gradient checkpointing and Flash Attention 2
    - Compatible with LSTM downstream tasks
    
    Architecture:
    - Linear projection instead of patch embedding (assumed at dataset level)
    - Multi-head self-attention with Flash Attention 2 support
    - Efficient decoder with learned upsampling
    - Attention weight extraction for visualization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List
import math


import torch
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

class PositionalEncoding(nn.Module):
    """
    Learnable positional encoding that adapts to variable sequence lengths.
    More flexible than fixed sinusoidal encoding for variable input sizes.
    """
    def __init__(self, embed_dim: int, max_seq_len: int = 10000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        
        # Learnable positional embeddings (initialized with sinusoidal pattern)
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_seq_len, embed_dim))
        self._init_positional_encoding()
        
    def _init_positional_encoding(self):
        """Initialize with sinusoidal pattern for better convergence"""
        position = torch.arange(0, self.max_seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.embed_dim, 2) * 
                            (-math.log(10000.0) / self.embed_dim))
        
        pe = torch.zeros(1, self.max_seq_len, self.embed_dim)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        
        self.pos_embedding.data.copy_(pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch, seq_len, embed_dim)
        Returns:
            Tensor with positional encoding added
        """
        seq_len = x.size(1)
        x = x + self.pos_embedding[:, :seq_len, :]
        return self.dropout(x)


class EfficientMultiHeadAttention(nn.Module):
    """
    Optimized Multi-Head Attention with optional Flash Attention 2 support.
    Falls back to memory-efficient attention if Flash Attention is unavailable.
    """
    def __init__(self, 
                 embed_dim: int,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 use_flash_attention: bool = True):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.use_flash_attention = use_flash_attention
        
        # Single projection for Q, K, V for efficiency
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        # For attention visualization
        self.attention_weights = None
        
    def forward(self, x: torch.Tensor, return_attention: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: Input tensor of shape (batch, seq_len, embed_dim)
            return_attention: Whether to return attention weights
        Returns:
            output: Transformed tensor
            attention_weights: Optional attention matrix for visualization
        """
        B, N, C = x.shape
        
        # Generate Q, K, V in one shot
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # Each: (B, num_heads, N, head_dim)
        
        # Try Flash Attention 2 if available and enabled
        if self.use_flash_attention and hasattr(F, 'scaled_dot_product_attention'):
            # PyTorch 2.0+ has built-in Flash Attention support
            # Updated to use torch.nn.attention.sdpa_kernel (new API)
            with torch.nn.attention.sdpa_kernel(
                backends=[
                    torch.nn.attention.SDPBackend.FLASH_ATTENTION,
                    torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION,
                    torch.nn.attention.SDPBackend.MATH
                ]
            ):
                attn_output = F.scaled_dot_product_attention(
                    q, k, v,
                    dropout_p=self.dropout.p if self.training else 0.0,
                    is_causal=False
                )
                attention_weights = None  # Flash attention doesn't return weights
        else:
            # Standard attention with manual computation
            attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, num_heads, N, N)
            attn = attn.softmax(dim=-1)
            
            if return_attention:
                # Store for visualization (averaged across heads)
                attention_weights = attn.detach().mean(dim=1)  # (B, N, N)
            else:
                attention_weights = None
                
            attn = self.dropout(attn)
            attn_output = attn @ v  # (B, num_heads, N, head_dim)
        
        # Reshape and project
        attn_output = attn_output.transpose(1, 2).reshape(B, N, C)
        output = self.proj(attn_output)
        output = self.dropout(output)
        
        return output, attention_weights


class TransformerEncoderBlock(nn.Module):
    """
    Efficient Transformer encoder block with Pre-LN and optional gradient checkpointing.
    """
    def __init__(self,
                 embed_dim: int,
                 num_heads: int = 8,
                 mlp_ratio: float = 4.0,
                 dropout: float = 0.1,
                 use_flash_attention: bool = True):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.attn = EfficientMultiHeadAttention(
            embed_dim, num_heads, dropout, use_flash_attention
        )
        
        self.norm2 = nn.LayerNorm(embed_dim, eps=1e-6)
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x: torch.Tensor, return_attention: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Pre-LN architecture (more stable)
        attn_out, attn_weights = self.attn(self.norm1(x), return_attention)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x, attn_weights


class Encoder_Transformer(nn.Module):
    """
    Transformer encoder that compresses flattened images to latent representation.
    Designed for variable input sizes with efficient memory usage.
    """
    def __init__(self,
                 input_dim: int,
                 latent_dim: int = 512,
                 num_heads: int = 8,
                 num_layers: int = 6,
                 dropout: float = 0.1,
                 use_flash_attention: bool = True,
                 use_gradient_checkpointing: bool = False):
        """
        Args:
            input_dim: Flattened input dimension (e.g., 201*201 or 1280*152)
            latent_dim: Dimension of latent space (128-8192)
            num_heads: Number of attention heads
            num_layers: Number of transformer blocks
            dropout: Dropout rate
            use_flash_attention: Enable Flash Attention 2
            use_gradient_checkpointing: Enable gradient checkpointing for memory saving
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.use_gradient_checkpointing = use_gradient_checkpointing
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, latent_dim)
        
        # Positional encoding
        max_seq_len = max(10000, input_dim + 100)  # Buffer for safety
        self.pos_encoding = PositionalEncoding(latent_dim, max_seq_len, dropout)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(
                latent_dim, num_heads, mlp_ratio=4.0, 
                dropout=dropout, use_flash_attention=use_flash_attention
            )
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(latent_dim, eps=1e-6)
        
        # Global pooling to create fixed-size embedding
        self.pool = nn.AdaptiveAvgPool1d(1)
        
    def forward(self, x: torch.Tensor, return_attention: bool = False) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        """
        Args:
            x: Input tensor of shape (batch, input_dim) - flattened image
            return_attention: Whether to collect attention weights
        Returns:
            embedding: Latent representation (batch, latent_dim)
            attention_maps: List of attention matrices from each layer (if requested)
        """
        # Add sequence dimension: (B, input_dim) -> (B, 1, input_dim)
        # Then project: -> (B, 1, latent_dim)
        x = x.unsqueeze(1)
        x = self.input_proj(x)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Collect attention weights if requested
        attention_maps = [] if return_attention else None
        
        # Pass through transformer blocks
        for block in self.blocks:
            if self.use_gradient_checkpointing and self.training:
                x, attn = torch.utils.checkpoint.checkpoint(
                    block, x, return_attention, use_reentrant=False
                )
            else:
                x, attn = block(x, return_attention)
            
            if return_attention and attn is not None:
                attention_maps.append(attn)
        
        x = self.norm(x)
        
        # Global pooling: (B, 1, latent_dim) -> (B, latent_dim)
        x = x.squeeze(1)
        
        return x, attention_maps


class Decoder_Transformer(nn.Module):
    """
    Transformer decoder that reconstructs images from latent representation.
    Uses learned upsampling and transformer blocks.
    """
    def __init__(self,
                 output_dim: int,
                 latent_dim: int = 512,
                 num_heads: int = 8,
                 num_layers: int = 6,
                 dropout: float = 0.1,
                 use_flash_attention: bool = True,
                 use_gradient_checkpointing: bool = False):
        """
        Args:
            output_dim: Flattened output dimension (e.g., 201*201 or 1280*152)
            latent_dim: Dimension of latent space
            num_heads: Number of attention heads
            num_layers: Number of transformer blocks
            dropout: Dropout rate
            use_flash_attention: Enable Flash Attention 2
            use_gradient_checkpointing: Enable gradient checkpointing
        """
        super().__init__()
        
        self.output_dim = output_dim
        self.latent_dim = latent_dim
        self.use_gradient_checkpointing = use_gradient_checkpointing
        
        # Expand latent to sequence
        self.latent_expand = nn.Linear(latent_dim, latent_dim)
        
        # Positional encoding
        max_seq_len = max(10000, output_dim + 100)
        self.pos_encoding = PositionalEncoding(latent_dim, max_seq_len, dropout)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(  # Using same block architecture
                latent_dim, num_heads, mlp_ratio=4.0,
                dropout=dropout, use_flash_attention=use_flash_attention
            )
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(latent_dim, eps=1e-6)
        
        # Output projection
        self.output_proj = nn.Linear(latent_dim, output_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Latent tensor of shape (batch, latent_dim)
        Returns:
            reconstruction: Flattened image (batch, output_dim)
        """
        # Expand: (B, latent_dim) -> (B, 1, latent_dim)
        x = self.latent_expand(x).unsqueeze(1)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Pass through transformer blocks
        for block in self.blocks:
            if self.use_gradient_checkpointing and self.training:
                x, _ = torch.utils.checkpoint.checkpoint(
                    block, x, False, use_reentrant=False
                )
            else:
                x, _ = block(x, return_attention=False)
        
        x = self.norm(x)
        
        # Project to output: (B, 1, latent_dim) -> (B, 1, output_dim) -> (B, output_dim)
        x = self.output_proj(x).squeeze(1)
        
        return x


class Autoencoder_Transformer(nn.Module):
    """
    Complete Transformer-based Autoencoder with attention visualization.
    
    Features:
    - Variable input sizes (flattened images)
    - Configurable latent dimension
    - Attention weight extraction
    - Memory and speed optimized for Ada Lovelace GPUs
    - Compatible with LSTM downstream tasks
    """
    def __init__(self,
                 input_size: Tuple[int, int] = (201, 201),
                 latent_dim: int = 512,
                 num_heads: int = 8,
                 num_encoder_layers: int = 6,
                 num_decoder_layers: int = 6,
                 dropout: float = 0.1,
                 use_flash_attention: bool = True,
                 use_gradient_checkpointing: bool = False):
        """
        Args:
            input_size: Tuple of (height, width) for input images
            latent_dim: Dimension of latent space (128-8192 recommended)
            num_heads: Number of attention heads (must divide latent_dim)
            num_encoder_layers: Number of transformer blocks in encoder
            num_decoder_layers: Number of transformer blocks in decoder
            dropout: Dropout rate
            use_flash_attention: Enable Flash Attention 2 for speed
            use_gradient_checkpointing: Enable gradient checkpointing for memory
        """
        super().__init__()
        
        # Validate configuration
        assert latent_dim % num_heads == 0, f"latent_dim ({latent_dim}) must be divisible by num_heads ({num_heads})"
        assert 128 <= latent_dim <= 8192, f"latent_dim should be in range [128, 8192], got {latent_dim}"
        
        self.input_size = input_size
        self.input_dim = input_size[0] * input_size[1]
        self.latent_dim = latent_dim
        
        # Encoder and Decoder
        self.encoder = Encoder_Transformer(
            input_dim=self.input_dim,
            latent_dim=latent_dim,
            num_heads=num_heads,
            num_layers=num_encoder_layers,
            dropout=dropout,
            use_flash_attention=use_flash_attention,
            use_gradient_checkpointing=use_gradient_checkpointing
        )
        
        self.decoder = Decoder_Transformer(
            output_dim=self.input_dim,
            latent_dim=latent_dim,
            num_heads=num_heads,
            num_layers=num_decoder_layers,
            dropout=dropout,
            use_flash_attention=use_flash_attention,
            use_gradient_checkpointing=use_gradient_checkpointing
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        """Initialize weights following best practices for transformers"""
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward(self, x: torch.Tensor, return_attention: bool = False) -> Tuple[torch.Tensor, Optional[Dict[str, List[torch.Tensor]]]]:
        """
        Forward pass through the autoencoder.
        
        Args:
            x: Input tensor of shape (batch, 1, H, W) or (batch, H, W)
            return_attention: Whether to return attention weights for visualization
        
        Returns:
            reconstruction: Reconstructed image (batch, 1, H, W)
            attention_info: Dictionary containing attention maps from encoder (if requested)
        """
        # Handle both (B, 1, H, W) and (B, H, W) inputs
        if x.dim() == 4:
            B, C, H, W = x.shape
            assert C == 1, "Only grayscale images (1 channel) are supported"
            x = x.view(B, H * W)
        elif x.dim() == 3:
            B, H, W = x.shape
            x = x.view(B, H * W)
        else:
            B = x.size(0)
            H, W = self.input_size
        
        # Encode
        embedding, attention_maps = self.encoder(x, return_attention=return_attention)
        
        # Decode
        reconstruction = self.decoder(embedding)
        
        # Reshape to image format
        reconstruction = reconstruction.view(B, 1, H, W)
        
        return reconstruction

    def _forward(self, x: torch.Tensor, return_attention: bool = False) -> Tuple[torch.Tensor, Optional[Dict[str, List[torch.Tensor]]]]:
        """
        Forward pass through the autoencoder.
        
        Args:
            x: Input tensor of shape (batch, 1, H, W) or (batch, H, W)
            return_attention: Whether to return attention weights for visualization
        
        Returns:
            reconstruction: Reconstructed image (batch, 1, H, W)
            attention_info: Dictionary containing attention maps from encoder (if requested)
        """
        # Handle both (B, 1, H, W) and (B, H, W) inputs
        if x.dim() == 4:
            B, C, H, W = x.shape
            assert C == 1, "Only grayscale images (1 channel) are supported"
            x = x.view(B, H * W)
        elif x.dim() == 3:
            B, H, W = x.shape
            x = x.view(B, H * W)
        else:
            B = x.size(0)
            H, W = self.input_size
        
        # Encode
        embedding, attention_maps = self.encoder(x, return_attention=return_attention)
        
        # Decode
        reconstruction = self.decoder(embedding)
        
        # Reshape to image format
        reconstruction = reconstruction.view(B, 1, H, W)
        
        # Prepare attention info
        attention_info = None
        if return_attention and attention_maps:
            attention_info = {
                'encoder_attention': attention_maps,
                'num_layers': len(attention_maps),
                'shape': attention_maps[0].shape if attention_maps else None
            }
        
        return reconstruction, attention_info
    
    def Embedding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract embedding from input (compatible with LSTM usage).
        
        Args:
            x: Input tensor of shape (batch, 1, H, W) or (batch, H, W)
        
        Returns:
            embedding: Latent representation (batch, latent_dim)
        """
        # Handle input dimensions
        if x.dim() == 4:
            B, C, H, W = x.shape
            assert C == 1, "Only grayscale images (1 channel) are supported"
            x = x.view(B, H * W)
        elif x.dim() == 3:
            B, H, W = x.shape
            x = x.view(B, H * W)
        
        # Encode only
        embedding, _ = self.encoder(x, return_attention=False)
        return embedding
    
    def get_attention_maps(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract attention maps for visualization.
        
        Args:
            x: Input tensor of shape (batch, 1, H, W)
        
        Returns:
            Dictionary containing:
                - 'layer_X': Attention matrix for layer X (batch, seq_len, seq_len)
                - 'average': Average attention across all layers
        """
        with torch.no_grad():
            _, attention_info = self.forward(x, return_attention=True)
        
        if attention_info is None:
            return {}
        
        attention_maps = attention_info['encoder_attention']
        result = {}
        
        # Store individual layer attention
        for i, attn in enumerate(attention_maps):
            result[f'layer_{i}'] = attn
        
        # Compute average attention
        if attention_maps:
            result['average'] = torch.stack(attention_maps).mean(dim=0)
        
        return result
    
    def get_config(self) -> Dict:
        """Return model configuration for saving/loading"""
        return {
            'input_size': self.input_size,
            'latent_dim': self.latent_dim,
            'input_dim': self.input_dim,
            'num_encoder_layers': len(self.encoder.blocks),
            'num_decoder_layers': len(self.decoder.blocks),
        }


# Utility functions for attention visualization
def visualize_attention_map(attention_map: torch.Tensor, 
                           save_path: Optional[str] = None,
                           title: str = "Attention Map"):
    """
    Visualize attention matrix as a heatmap.
    
    Args:
        attention_map: Attention tensor (seq_len, seq_len) or (batch, seq_len, seq_len)
        save_path: Optional path to save the figure
        title: Plot title
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib required for visualization. Install with: pip install matplotlib")
        return
    
    # Handle batch dimension
    if attention_map.dim() == 3:
        attention_map = attention_map[0]  # Take first sample
    
    attn_np = attention_map.cpu().numpy()
    
    plt.figure(figsize=(10, 8))
    plt.imshow(attn_np, cmap='viridis', aspect='auto')
    plt.colorbar(label='Attention Weight')
    plt.title(title)
    plt.xlabel('Key Position')
    plt.ylabel('Query Position')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Attention map saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


# Configuration presets for different use cases
PRESET_CONFIGS = {
    'tiny': {
        'latent_dim': 128,
        'num_heads': 4,
        'num_encoder_layers': 3,
        'num_decoder_layers': 3,
        'dropout': 0.1,
    },
    'small': {
        'latent_dim': 256,
        'num_heads': 8,
        'num_encoder_layers': 4,
        'num_decoder_layers': 4,
        'dropout': 0.1,
    },
    'medium': {
        'latent_dim': 512,
        'num_heads': 8,
        'num_encoder_layers': 6,
        'num_decoder_layers': 6,
        'dropout': 0.1,
    },
    'large': {
        'latent_dim': 1024,
        'num_heads': 16,
        'num_encoder_layers': 8,
        'num_decoder_layers': 8,
        'dropout': 0.1,
    },
    'xlarge': {
        'latent_dim': 2048,
        'num_heads': 16,
        'num_encoder_layers': 12,
        'num_decoder_layers': 12,
        'dropout': 0.1,
    },
}


def create_autoencoder(input_size: Tuple[int, int],
                      preset: str = 'medium',
                      **kwargs) -> Autoencoder_Transformer:
    """
    Factory function to create autoencoder with preset configurations.
    
    Args:
        input_size: Tuple of (height, width)
        preset: One of 'tiny', 'small', 'medium', 'large', 'xlarge'
        **kwargs: Override any preset parameters
    
    Returns:
        Configured Autoencoder_Transformer instance
    """
    if preset not in PRESET_CONFIGS:
        raise ValueError(f"Unknown preset '{preset}'. Choose from {list(PRESET_CONFIGS.keys())}")
    
    config = PRESET_CONFIGS[preset].copy()
    config.update(kwargs)
    config['input_size'] = input_size
    
    return Autoencoder_Transformer(**config)


if __name__ == "__main__":
    



    """
    Example usage and testing
    """
    print("=" * 80)
    print("Transformer Autoencoder - Test Suite")
    print("=" * 80)
    
    # Test different input sizes
    test_configs = [
        ((201, 201), 'small', 'Square image (201x201)'),
        ((152, 1280), 'small', 'Wide image (1280x152)'),
        ((152, 640), 'tiny', 'Medium wide image (640x152)'),
    ]
    
    for input_size, preset, description in test_configs:
        print(f"\n{'-' * 80}")
        print(f"Testing: {description}")
        print(f"Input size: {input_size}, Preset: {preset}")
        print(f"{'-' * 80}")
        
        # Create model
        model = create_autoencoder(
            input_size=input_size,
            preset=preset,
            use_flash_attention=True,
            use_gradient_checkpointing=False
        )
        
        config = model.get_config()
        print(f"\nModel Configuration:")
        for key, value in config.items():
            print(f"  {key}: {value}")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\nParameters:")
        print(f"  Total: {total_params:,}")
        print(f"  Trainable: {trainable_params:,}")
        
        # Test forward pass
        batch_size = 2
        x = torch.randn(batch_size, 1, *input_size)
        
        print(f"\nInput shape: {x.shape}")
        
        # Standard forward
        with torch.no_grad():
            recon, _ = model(x, return_attention=False)
            print(f"Reconstruction shape: {recon.shape}")
            
            # Test embedding extraction
            embedding = model.Embedding(x)
            print(f"Embedding shape: {embedding.shape}")
            
            # Test attention extraction
            attention_maps = model.get_attention_maps(x[:1])  # Use single sample
            if attention_maps:
                print(f"\nAttention maps extracted:")
                for key, attn in attention_maps.items():
                    print(f"  {key}: {attn.shape}")
        
        print(f"\n✓ Test passed for {description}")
    
    print("\n" + "=" * 80)
    print("All tests completed successfully!")
    print("=" * 80)
    
    # Memory efficiency tips
    print("\n" + "=" * 80)
    print("GPU Optimization Tips for Ada Lovelace (RTX 40xx):")
    print("=" * 80)
    print("""
1. Enable Flash Attention 2 (requires PyTorch 2.0+):
   model = create_autoencoder(input_size, use_flash_attention=True)

2. Use mixed precision training (AMP):
   from torch.cuda.amp import autocast, GradScaler
   scaler = GradScaler()
   with autocast():
       loss = criterion(model(x), target)

3. Enable gradient checkpointing for large models:
   model = create_autoencoder(input_size, use_gradient_checkpointing=True)

4. Use torch.compile for up to 2x speedup (PyTorch 2.0+):
   model = torch.compile(model, mode='reduce-overhead')

5. Optimize batch size based on your 8GB VRAM:
   - For 201x201: batch_size = 32-64
   - For 1280x152: batch_size = 16-32
   - Monitor with: torch.cuda.max_memory_allocated()

6. Use DataLoader with pin_memory and num_workers:
   DataLoader(dataset, batch_size=32, pin_memory=True, num_workers=4)
    """)
