# Transformer Autoencoder Usage Guide

## Overview

`AutoEncoder_TransformerV2_0.py` provides an optimized transformer-based image autoencoder designed for:
- **Variable input sizes**: 1280x152, 201x201, or any size x152
- **Configurable latent space**: 128 to 8192 dimensions
- **Attention visualization**: Extract and visualize attention matrices
- **GPU optimized**: Ada Lovelace architecture (RTX 40xx) with Flash Attention 2
- **LSTM compatibility**: Direct embedding extraction for sequential models

---

## Quick Start

### 1. Basic Usage

```python
from networks.AutoEncoder_TransformerV2_0 import create_autoencoder
import torch

# Create model for 201x201 images with medium preset
model = create_autoencoder(
    input_size=(201, 201),
    preset='medium',  # Options: 'tiny', 'small', 'medium', 'large', 'xlarge'
    use_flash_attention=True,
    use_gradient_checkpointing=False
)

# Move to GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Forward pass
x = torch.randn(32, 1, 201, 201).to(device)  # Batch of 32 images
reconstruction, _ = model(x)

print(f"Input shape: {x.shape}")
print(f"Reconstruction shape: {reconstruction.shape}")
```

### 2. Extract Embeddings for LSTM

```python
# Get latent embeddings (compatible with your LSTM network)
embeddings = model.Embedding(x)  # Shape: (batch, latent_dim)

print(f"Embedding shape: {embeddings.shape}")
# Output: torch.Size([32, 512]) for 'medium' preset

# Use with LSTM
import networks.AutoEncoder_CNN_LSTM as LSTM_net
lstm_model = LSTM_net.LSTMModel(
    LSTMEmbdSize=embeddings.size(1),  # Use latent_dim
    hidden_dim=256,
    num_layers=2
)
```

### 3. Visualize Attention Maps

```python
from networks.AutoEncoder_TransformerV2_0 import visualize_attention_map

# Get attention maps
attention_maps = model.get_attention_maps(x[:1])  # Single image

# Visualize average attention across all layers
visualize_attention_map(
    attention_maps['average'],
    save_path='attention_map.png',
    title='Average Attention Across All Layers'
)

# Visualize specific layer
for layer_idx in range(6):  # 6 layers in 'medium' preset
    visualize_attention_map(
        attention_maps[f'layer_{layer_idx}'],
        save_path=f'attention_layer_{layer_idx}.png',
        title=f'Layer {layer_idx} Attention'
    )
```

---

## Configuration Presets

| Preset | Latent Dim | Heads | Layers | Parameters | Memory (est.) | Speed |
|--------|-----------|-------|--------|------------|---------------|-------|
| `tiny` | 128 | 4 | 3+3 | ~500K | Low | Fast |
| `small` | 256 | 8 | 4+4 | ~2M | Low-Med | Fast |
| `medium` | 512 | 8 | 6+6 | ~8M | Medium | Balanced |
| `large` | 1024 | 16 | 8+8 | ~32M | High | Slow |
| `xlarge` | 2048 | 16 | 12+12 | ~128M | Very High | Slowest |

### Custom Configuration

```python
model = create_autoencoder(
    input_size=(1280, 152),
    preset='small',  # Start from a preset
    latent_dim=384,  # Override latent dimension
    num_heads=12,    # Override number of heads
    num_encoder_layers=5,  # Override encoder depth
    num_decoder_layers=5,  # Override decoder depth
    dropout=0.15,    # Override dropout rate
)
```

---

## Different Input Sizes

### Square Images (201x201)

```python
model_square = create_autoencoder(
    input_size=(201, 201),
    preset='medium',
    latent_dim=512
)

x_square = torch.randn(16, 1, 201, 201).to(device)
recon, _ = model_square(x_square)
```

### Wide Images (1280x152)

```python
model_wide = create_autoencoder(
    input_size=(1280, 152),  # Note: (H, W) format
    preset='small',  # Use smaller preset for larger images
    latent_dim=256
)

x_wide = torch.randn(8, 1, 1280, 152).to(device)
recon, _ = model_wide(x_wide)
```

### Variable Width (Xx152)

```python
# For 640x152 images
model_var = create_autoencoder(
    input_size=(640, 152),
    preset='small',
    latent_dim=256
)
```

---

## GPU Optimization for Ada Lovelace (RTX 40xx)

### 1. Enable Flash Attention 2 (Recommended)

Requires PyTorch 2.0+ with CUDA support.

```python
model = create_autoencoder(
    input_size=(201, 201),
    preset='medium',
    use_flash_attention=True  # 2-3x faster attention
)
```

### 2. Mixed Precision Training (AMP)

Reduces memory by ~50% and increases speed by ~30-40%.

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

for batch in dataloader:
    optimizer.zero_grad()
    
    with autocast():  # Enable mixed precision
        reconstruction, _ = model(batch)
        loss = criterion(reconstruction, batch)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

### 3. Gradient Checkpointing (Large Models)

Reduces memory by ~30-40% at cost of 10-15% speed.

```python
model = create_autoencoder(
    input_size=(201, 201),
    preset='large',  # Large model
    use_gradient_checkpointing=True  # Trade speed for memory
)
```

### 4. Torch Compile (PyTorch 2.0+)

Up to 2x speedup for inference and training.

```python
import torch

model = create_autoencoder(input_size=(201, 201), preset='medium')
model = torch.compile(model, mode='reduce-overhead')  # Compile for speed

# Use normally
reconstruction, _ = model(x)
```

### 5. Optimal Batch Sizes for 8GB VRAM

| Input Size | Preset | Batch Size (FP32) | Batch Size (AMP) | Gradient Checkpoint |
|------------|--------|-------------------|------------------|---------------------|
| 201x201 | small | 32-48 | 64-96 | 96-128 |
| 201x201 | medium | 16-24 | 32-48 | 48-64 |
| 201x201 | large | 8-12 | 16-24 | 24-32 |
| 1280x152 | small | 16-24 | 32-48 | 48-64 |
| 1280x152 | medium | 8-12 | 16-24 | 24-32 |

---

## Training Example

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

# Create model
model = create_autoencoder(
    input_size=(201, 201),
    preset='medium',
    use_flash_attention=True,
    use_gradient_checkpointing=False
).cuda()

# Optimizer and loss
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
criterion = nn.MSELoss()
scaler = GradScaler()

# Training loop
model.train()
for epoch in range(num_epochs):
    for batch_idx, images in enumerate(train_loader):
        images = images.cuda()
        
        optimizer.zero_grad()
        
        # Mixed precision forward pass
        with autocast():
            reconstruction, _ = model(images)
            loss = criterion(reconstruction, images)
        
        # Backward pass with gradient scaling
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        if batch_idx % 100 == 0:
            print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.6f}")

print("Training completed!")
```

---

## Attention Analysis Example

```python
import torch
import matplotlib.pyplot as plt
import numpy as np

# Load trained model
model.eval()

# Get a test image
with torch.no_grad():
    test_image = test_dataset[0].unsqueeze(0).cuda()
    
    # Get attention maps
    attention_maps = model.get_attention_maps(test_image)
    
    # Visualize all layers
    num_layers = len([k for k in attention_maps.keys() if k.startswith('layer_')])
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i in range(num_layers):
        attn = attention_maps[f'layer_{i}'][0].cpu().numpy()
        
        axes[i].imshow(attn, cmap='viridis', aspect='auto')
        axes[i].set_title(f'Layer {i}')
        axes[i].set_xlabel('Key Position')
        axes[i].set_ylabel('Query Position')
    
    plt.tight_layout()
    plt.savefig('all_attention_layers.png', dpi=300)
    plt.show()
    
    # Average attention
    avg_attn = attention_maps['average'][0].cpu().numpy()
    
    plt.figure(figsize=(10, 8))
    plt.imshow(avg_attn, cmap='viridis', aspect='auto')
    plt.colorbar(label='Attention Weight')
    plt.title('Average Attention Across All Layers')
    plt.xlabel('Key Position')
    plt.ylabel('Query Position')
    plt.savefig('average_attention.png', dpi=300)
    plt.show()
```

---

## Integration with Existing Workflow

### Replacing CNN Autoencoder

```python
# Old CNN autoencoder
from networks.AutoEncoder_CNNV1_0 import Autoencoder_CNN

old_model = Autoencoder_CNN(DropOut=False, embedding_dim=1024)

# New Transformer autoencoder (same interface)
from networks.AutoEncoder_TransformerV2_0 import create_autoencoder

new_model = create_autoencoder(
    input_size=(201, 201),
    preset='medium',
    latent_dim=1024  # Match CNN embedding dimension
)

# Both have the same interface:
reconstruction = new_model(x)  # Returns reconstruction
embedding = new_model.Embedding(x)  # Returns latent embedding
```

### Use with LSTM Network

```python
from networks.AutoEncoder_TransformerV2_0 import create_autoencoder
from networks.AutoEncoder_CNN_LSTM import LSTMModel

# Create autoencoder
autoencoder = create_autoencoder(
    input_size=(201, 201),
    preset='medium',
    latent_dim=512
).cuda()

# Create LSTM
lstm = LSTMModel(
    LSTMEmbdSize=512,  # Match autoencoder latent_dim
    hidden_dim=256,
    num_layers=2,
    dropout=0.2
).cuda()

# Process sequence of images
sequence_length = 10
batch_size = 16
image_sequence = torch.randn(batch_size, sequence_length, 1, 201, 201).cuda()

# Extract embeddings
embeddings_list = []
for t in range(sequence_length):
    frame = image_sequence[:, t]  # (batch, 1, 201, 201)
    embedding = autoencoder.Embedding(frame)  # (batch, 512)
    embeddings_list.append(embedding)

embeddings = torch.stack(embeddings_list, dim=1)  # (batch, seq_len, 512)

# Feed to LSTM
lstm_output = lstm(embeddings)
```

---

## Model Saving and Loading

```python
# Save model
checkpoint = {
    'model_state_dict': model.state_dict(),
    'config': model.get_config(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch': epoch,
}
torch.save(checkpoint, 'transformer_autoencoder.pth')

# Load model
checkpoint = torch.load('transformer_autoencoder.pth')

# Recreate model with same config
loaded_model = create_autoencoder(
    input_size=checkpoint['config']['input_size'],
    latent_dim=checkpoint['config']['latent_dim'],
    preset='medium'  # Adjust based on your saved config
)

loaded_model.load_state_dict(checkpoint['model_state_dict'])
loaded_model.eval()
```

---

## Performance Monitoring

```python
import torch

# Monitor memory usage
def print_memory_stats():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        max_allocated = torch.cuda.max_memory_allocated() / 1e9
        
        print(f"GPU Memory:")
        print(f"  Allocated: {allocated:.2f} GB")
        print(f"  Reserved: {reserved:.2f} GB")
        print(f"  Max Allocated: {max_allocated:.2f} GB")

# Benchmark speed
import time

model.eval()
with torch.no_grad():
    # Warmup
    for _ in range(10):
        _ = model(x)
    
    # Benchmark
    torch.cuda.synchronize()
    start = time.time()
    
    num_iterations = 100
    for _ in range(num_iterations):
        _ = model(x)
    
    torch.cuda.synchronize()
    end = time.time()
    
    elapsed = end - start
    throughput = (num_iterations * x.size(0)) / elapsed
    
    print(f"Throughput: {throughput:.2f} images/sec")
    print(f"Latency: {elapsed/num_iterations*1000:.2f} ms/batch")

print_memory_stats()
```

---

## Troubleshooting

### Out of Memory Error

1. **Reduce batch size**
2. **Enable gradient checkpointing**:
   ```python
   model = create_autoencoder(..., use_gradient_checkpointing=True)
   ```
3. **Use mixed precision**:
   ```python
   from torch.cuda.amp import autocast
   with autocast():
       output = model(input)
   ```
4. **Use smaller preset**: Switch from 'large' to 'medium' or 'small'

### Slow Training

1. **Enable Flash Attention**: `use_flash_attention=True`
2. **Use torch.compile**: `model = torch.compile(model)`
3. **Increase batch size** (if memory allows)
4. **Reduce number of layers**: Use custom config with fewer layers
5. **Use DataLoader optimization**:
   ```python
   DataLoader(dataset, batch_size=32, num_workers=4, pin_memory=True)
   ```

### Flash Attention Not Working

Check PyTorch version and CUDA availability:
```python
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")

# Flash Attention requires PyTorch >= 2.0
if hasattr(F, 'scaled_dot_product_attention'):
    print("Flash Attention available!")
else:
    print("Flash Attention not available. Update PyTorch.")
```

---

## Comparison: CNN vs Transformer

| Feature | CNN Autoencoder | Transformer Autoencoder |
|---------|----------------|-------------------------|
| **Latent Dim** | Fixed | Configurable (128-8192) |
| **Attention** | No | Yes (visualizable) |
| **Long-range deps** | Limited | Excellent |
| **Speed** | Fast | Medium (Fast with FA2) |
| **Memory** | Low | Medium-High |
| **Parameters** | ~1-5M | ~2-128M (preset dependent) |
| **Best for** | Small datasets | Large datasets, complex patterns |

---

## Citation

If you use this transformer autoencoder in your research, please cite:

```bibtex
@software{transformer_autoencoder_2025,
  author = {Yassin Riyazi},
  title = {Optimized Transformer Autoencoder for Image Compression},
  year = {2025},
  url = {https://github.com/YassinRiyazi/From-Droplet-Dynamics-to-Viscosity}
}
```

---

## Additional Resources

- [Flash Attention Paper](https://arxiv.org/abs/2205.14135)
- [Transformer Architecture](https://arxiv.org/abs/1706.03762)
- [PyTorch Mixed Precision Training](https://pytorch.org/docs/stable/amp.html)
- [Gradient Checkpointing](https://pytorch.org/docs/stable/checkpoint.html)
