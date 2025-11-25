"""
    Author: Yassin Riyazi
    Date: 11-24-2025
    Description: Transformer-based regressor for viscosity prediction from temporal sequences.

    Architecture:
        - CNN Autoencoder (frozen): Generates frame embeddings
        - Temporal Transformer: Processes sequence with multi-head self-attention
        - Regression head: Predicts viscosity from sequence representation
        
    Key Features:
        - Multi-head self-attention for temporal modeling
        - Positional encoding for sequence position awareness
        - Layer normalization and residual connections
        - Compatible with existing training pipeline
"""
import torch
import torch.nn as nn
import math
from typing import Union, Optional

if __name__ == "__main__":
    from AutoEncoder_CNNV1_0 import Autoencoder_CNN as Autoencoder_CNNV1
else:
    from .AutoEncoder_CNNV1_0 import Autoencoder_CNN as Autoencoder_CNNV1


class TemporalPositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for temporal sequences.
    """
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super(TemporalPositionalEncoding, self).__init__()  # type: ignore
        self.dropout = nn.Dropout(dropout)
        
        # Create positional encoding matrix
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            (batch, seq_len, d_model) with positional encoding
        """
        # Transpose for adding positional encoding
        x = x.transpose(0, 1)  # (seq_len, batch, d_model)
        x = x + self.pe[:x.size(0)]
        x = x.transpose(0, 1)  # (batch, seq_len, d_model)
        return self.dropout(x)


class TransformerRegressorBlock(nn.Module):
    """
    Single Transformer encoder block optimized for regression tasks.
    """
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1):
        super(TransformerRegressorBlock, self).__init__()  # type: ignore
        
        # Multi-head self-attention
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        # Feedforward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout for residual connections
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        self.activation = nn.GELU()
        
        # Store attention weights for inspection
        self.last_attn_weights = None
        
    def forward(self, x: torch.Tensor, src_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
            src_mask: Optional attention mask
        Returns:
            (batch, seq_len, d_model)
        """
        # Self-attention with residual
        attn_output, attn_weights = self.self_attn(x, x, x, attn_mask=src_mask, need_weights=True)
        self.last_attn_weights = attn_weights
        x = x + self.dropout1(attn_output)
        x = self.norm1(x)
        
        # Feedforward with residual
        ff_output = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = x + self.dropout2(ff_output)
        x = self.norm2(x)
        
        return x


class TransformerModel(nn.Module):
    """
    Transformer-based model for sequence regression.
    """
    def __init__(self,
                 input_dim: int,
                 d_model: int = 256,
                 nhead: int = 8,
                 num_layers: int = 4,
                 dim_feedforward: int = 1024,
                 dropout: float = 0.1,
                 device: torch.device | None = None):
        """
        Args:
            input_dim (int): Dimension of input features
            d_model (int): Transformer model dimension
            nhead (int): Number of attention heads
            num_layers (int): Number of transformer layers
            dim_feedforward (int): Feedforward network dimension
            dropout (float): Dropout rate
            device (torch.device): Device for computation
        """
        super(TransformerModel, self).__init__()  # type: ignore
        
        if device:
            self.device = device
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Project input to model dimension
        self.input_projection = nn.Linear(input_dim, d_model, device=self.device)
        
        # Positional encoding
        self.pos_encoder = TemporalPositionalEncoding(d_model, dropout=dropout)
        
        # Transformer encoder layers
        self.transformer_layers = nn.ModuleList([
            TransformerRegressorBlock(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        
        # Regression head
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model // 2, device=self.device),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1, device=self.device),
            nn.Sigmoid()
        )
        
        self.d_model = d_model
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_dim)
        Returns:
            (batch,) - predicted viscosity
        """
        # Project input
        x = self.input_projection(x)  # (batch, seq_len, d_model)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transformer layers
        for layer in self.transformer_layers:
            x = layer(x)
        
        # Global average pooling over sequence
        x = x.mean(dim=1)  # (batch, d_model)
        
        # Regression head
        output = self.fc(x).squeeze(-1)  # (batch,)
        
        return output
    
    def reset_states(self, x: torch.Tensor) -> None:
        """Dummy method for API compatibility with LSTMModel."""
        pass
    
    def get_attention_weights(self) -> list[torch.Tensor]:
        """
        Returns attention weights from all transformer layers.
        Returns:
            List of attention weight tensors from each layer
        """
        return [layer.last_attn_weights for layer in self.transformer_layers 
                if layer.last_attn_weights is not None]


class Encoder_Transformer(nn.Module):
    """
    Complete encoder with frozen CNN autoencoder and Transformer temporal model.
    """
    def __init__(self,
                 address_autoencoder: Union[str, None],
                 proj_dim: int,
                 input_dim: int = 2048,
                 d_model: int = 256,
                 nhead: int = 8,
                 num_layers: int = 4,
                 dim_feedforward: int = 1024,
                 dropout: float = 0.1,
                 Autoencoder_CNN: torch.nn.Module | None = None,
                 S4_size: int | None = 3) -> None:
        """
        Args:
            address_autoencoder (str): Path to pre-trained autoencoder
            proj_dim (int): Autoencoder embedding dimension
            input_dim (int): Total input dimension (embedding + S4_size features)
            d_model (int): Transformer model dimension
            nhead (int): Number of attention heads
            num_layers (int): Number of transformer layers
            dim_feedforward (int): Feedforward dimension
            dropout (float): Dropout rate
            Autoencoder_CNN (torch.nn.Module): Autoencoder class
            S4_size (int): Number of additional 4S-SROF features
        """
        super(Encoder_Transformer, self).__init__()  # type: ignore
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.S4_size = S4_size if S4_size is not None else 0
        
        # Load frozen autoencoder
        self.load_autoencoder(address_autoencoder, Autoencoder_CNN)
        
        # Feature normalization for S4-SROF features
        if self.S4_size > 0:
            self.feature_norm = nn.BatchNorm1d(self.S4_size, device=self.device)
            total_input_dim = proj_dim + self.S4_size
        else:
            self.feature_norm = None
            total_input_dim = proj_dim
        
        self.model = TransformerModel(
            input_dim=total_input_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            device=self.device
        )
        
        # Alias for compatibility
        self.transformer = self.model
        
    def _encoder(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract embeddings from images using frozen autoencoder.
        Args:
            x: (batch, seq_len, channels, height, width)
        Returns:
            (batch, seq_len, embedding_dim)
        """
        batch_size, seq_len = x.size(0), x.size(1)
        x = x.view(-1, *x.shape[2:])  # (batch*seq_len, channels, height, width)
        
        with torch.no_grad():
            embeddings = self.autoencoder.Embedding(x)  # (batch*seq_len, embedding_dim)
        
        embeddings = embeddings.view(batch_size, seq_len, -1)
        return embeddings
    
    def forward(self,
                x: torch.Tensor,
                x_additional: torch.Tensor | None = None) -> torch.Tensor:
        """
        Forward pass through encoder and transformer.
        Args:
            x: (batch, seq_len, channels, height, width)
            x_additional: (batch, seq_len, S4_size) - optional 4S-SROF features
        Returns:
            (batch,) - predicted viscosity
        """
        # Extract embeddings
        if hasattr(self, 'autoencoder') and self.autoencoder is not None:
            embeddings = self._encoder(x)  # (batch, seq_len, embedding_dim)
            embeddings = torch.cat([embeddings, x_additional], dim=-1)
        else:
            embeddings = x  # Assume input is already embeddings

        
        # Forward through model
        output = self.model(embeddings)
        
        return output
    
    def AttentionWeights(self) -> list[torch.Tensor] | None:
        """
        Get attention weights from all transformer layers.
        Returns:
            List of attention weight tensors
        """
        if hasattr(self.model, 'get_attention_weights'):
            return self.model.get_attention_weights()
        return None
    
    def load_autoencoder(self,
                        address_autoencoder: str | None,
                        Autoencoder_CNN: torch.nn.Module | None) -> None:
        """
        Load pre-trained autoencoder and freeze it.
        """
        if address_autoencoder is None or Autoencoder_CNN is None:
            return 
        

        self.autoencoder = Autoencoder_CNN(embedding_dim=1024).to(self.device)
        self.autoencoder.load_state_dict(torch.load(address_autoencoder, 
                                                    map_location=self.device))
        self.autoencoder.eval()
        
        # Freeze autoencoder
        for param in self.autoencoder.parameters():
            param.requires_grad = False


if __name__ == "__main__":
    from calflops import calculate_flops  # type: ignore
    
    batch_size = 4
    seq_length = 20
    embedding_dim = 1024
    
    # Test Transformer model
    print("=" * 50)
    print("Testing Transformer Model")
    print("=" * 50)
    
    model = TransformerModel(
        input_dim=embedding_dim,
        d_model=256,
        nhead=8,
        num_layers=4,
        dropout=0.1,
        device=torch.device("cpu")
    )
    
    # Test forward pass
    x = torch.randn(batch_size, seq_length, embedding_dim)
    output = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # Get attention weights
    attn_weights = model.get_attention_weights()
    if attn_weights:
        print(f"Number of attention layers: {len(attn_weights)}")
        print(f"Attention weights shape (layer 0): {attn_weights[0].shape}")
    
    # Calculate FLOPs
    input_shape = (batch_size, seq_length, embedding_dim)
    with torch.inference_mode():
        flops, macs, params = calculate_flops(
            model=model,
            input_shape=input_shape,
            output_as_string=True,
            output_precision=4
        )
        print(f"Transformer FLOPs: {flops}   MACs: {macs}   Params: {params}")
