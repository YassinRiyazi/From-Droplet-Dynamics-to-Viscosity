import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchEmbedding(nn.Module):
    """
    Convert image into patches and embed them
    """
    def __init__(self,
                 patch_size: int = 10,
                 in_channels: int = 1,
                 embed_dim: int = 256)-> None:
        """
        Initializes the patch embedding layer.
        Args:
            patch_size (int): Size of the patches to be extracted from the image
            in_channels (int): Number of input channels
            embed_dim (int): Dimension of the embedding space
        Returns:
            None: Initializes the patch embedding layer with a convolutional layer.
        """
        super().__init__()
        self.patch_size = patch_size
        self.projection = nn.Conv2d(in_channels, embed_dim, 
                                  kernel_size=patch_size, stride=patch_size)
        
    def forward(self,
                x: torch.Tensor) -> torch.Tensor:
        x = self.projection(x)  # (B, embed_dim, H//patch_size, W//patch_size)
        x = x.flatten(2)  # (B, embed_dim, n_patches)
        x = x.transpose(1, 2)  # (B, n_patches, embed_dim)
        return x

class TransformerEncoder(nn.Module):
    """
    Transformer Encoder for image patches
    """
    def __init__(self, embed_dim: int = 256,
                 num_heads: int = 8,
                 num_layers: int = 6,
                 dropout: float = 0.1) -> None:
        """
        Initializes the transformer encoder.
        Args:
            embed_dim (int): Dimension of the embedding space
            num_heads (int): Number of attention heads
            num_layers (int): Number of transformer layers
            dropout (float): Dropout rate
        Returns:
            None
        """
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=4*embed_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor  :
        return self.transformer(x)

class TransformerDecoder(nn.Module):
    """
    Decode embedded features back to image
    """
    def __init__(self, embed_dim: int = 256,
                 patch_size: int = 10) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, patch_size * patch_size),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
    def forward(self, x, orig_size):
        B, L, E = x.shape
        H = W = int(L ** 0.5)
        
        x = self.decoder(x)  # (B, L, patch_size*patch_size)
        x = x.reshape(B, H, W, self.patch_size, self.patch_size)
        x = x.permute(0, 3, 1, 4, 2)  # Prepare for folding
        x = x.reshape(B, 1, H*self.patch_size, W*self.patch_size)
        return x

class TransformerAutoencoder(nn.Module):
    """
    Complete Transformer-based Autoencoder
    """
    def __init__(self, patch_size: int = 10, embed_dim: int = 256, num_heads: int = 8, num_layers: int = 6) -> None:
        """
        Initializes the transformer autoencoder.
        """
        super().__init__()
        self.patch_embed = PatchEmbedding(patch_size, in_channels=1, embed_dim=embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, (200//patch_size)**2, embed_dim))
        self.encoder = TransformerEncoder(embed_dim, num_heads, num_layers)
        self.decoder = TransformerDecoder(embed_dim, patch_size)
        
        # Initialize position embeddings
        nn.init.normal_(self.pos_embed, std=0.02)
        
    def forward(self, x):
        orig_size = x.size()
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.encoder(x)
        recon = self.decoder(x, orig_size)
        return recon, x  # Return reconstruction and embeddings 