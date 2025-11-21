# Networks Module

This directory contains neural network architectures for image compression and sequential modeling.

## Available Autoencoders

### 1. CNN-based Autoencoders

#### AutoEncoder_CNNV1_0.py
- **Type**: Convolutional Neural Network (CNN)
- **Input**: 200x200 grayscale images
- **Embedding**: Configurable (default: 100)
- **Features**:
  - Fast and memory-efficient
  - Fixed input size
  - Optional dropout
- **Best for**: Quick prototyping, limited resources

#### AutoEncoder_CNNV1_1.py
Similar to V1.0 with minor variations.

#### AutoEncoder_CNNV2_0.py
Enhanced version with improved architecture.

---

### 2. Transformer-based Autoencoder ⭐ NEW

#### AutoEncoder_TransformerV2_0.py

**Optimized transformer-based autoencoder with advanced features:**

##### Key Features
- ✅ **Variable input sizes**: 1280x152, 201x201, Xx152, or any size
- ✅ **Configurable latent space**: 128 to 8192 dimensions
- ✅ **Attention visualization**: Extract and visualize attention matrices
- ✅ **GPU optimized**: Ada Lovelace (RTX 40xx) with Flash Attention 2
- ✅ **Memory efficient**: Gradient checkpointing, mixed precision support
- ✅ **LSTM compatible**: Direct embedding extraction

##### Quick Start

```python
from networks.AutoEncoder_TransformerV2_0 import create_autoencoder

# Create model
model = create_autoencoder(
    input_size=(201, 201),
    preset='medium',  # tiny, small, medium, large, xlarge
    use_flash_attention=True,
    use_gradient_checkpointing=False
)

# Forward pass
x = torch.randn(32, 1, 201, 201).cuda()
reconstruction, attention_info = model(x, return_attention=True)

# Extract embeddings for LSTM
embeddings = model.Embedding(x)  # Shape: (batch, latent_dim)

# Visualize attention
attention_maps = model.get_attention_maps(x[:1])
```

##### Configuration Presets

| Preset | Latent Dim | Parameters | Memory | Speed | Use Case |
|--------|-----------|------------|--------|-------|----------|
| tiny | 128 | ~500K | Low | Fast | Quick tests |
| small | 256 | ~2M | Low-Med | Fast | Most tasks |
| medium | 512 | ~8M | Medium | Balanced | Production |
| large | 1024 | ~32M | High | Slow | Research |
| xlarge | 2048 | ~128M | Very High | Slowest | Large scale |

##### GPU Optimization

**For RTX 4060/4070 (8GB VRAM):**

```python
# Enable Flash Attention 2 (PyTorch 2.0+)
model = create_autoencoder(
    input_size=(201, 201),
    preset='medium',
    use_flash_attention=True  # 2-3x faster
)

# Mixed precision training (AMP)
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
with autocast():
    loss = criterion(model(x), target)

# Gradient checkpointing for large models
model = create_autoencoder(
    input_size=(201, 201),
    preset='large',
    use_gradient_checkpointing=True  # Save memory
)

# Torch compile (PyTorch 2.0+)
model = torch.compile(model, mode='reduce-overhead')
```

##### Documentation
- **Detailed guide**: [TRANSFORMER_USAGE_GUIDE.md](TRANSFORMER_USAGE_GUIDE.md)
- **Test script**: [test_transformer_autoencoder.py](test_transformer_autoencoder.py)
- **Comparison**: [compare_autoencoders.py](compare_autoencoders.py)

---

### 3. Legacy Transformer (Original)

#### AutoEncoder_Transformer.py
- Basic transformer implementation
- Fixed input size (200x200)
- Patch-based embedding
- **Note**: Consider using AutoEncoder_TransformerV2_0.py for new projects

---

## Sequential Models

### AutoEncoder_CNN_LSTM.py
LSTM network for processing sequences of embeddings from autoencoders.

**Integration Example:**
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
    num_layers=2
).cuda()

# Process sequence
sequence = torch.randn(16, 10, 1, 201, 201).cuda()  # (B, T, C, H, W)
embeddings = []
for t in range(10):
    emb = autoencoder.Embedding(sequence[:, t])
    embeddings.append(emb)
embeddings = torch.stack(embeddings, dim=1)  # (B, T, latent_dim)

output = lstm(embeddings)
```

---

## Utilities

### Flops.py
Calculate FLOPs and parameter counts for models.

---

## Comparison: CNN vs Transformer

| Feature | CNN | Transformer V2 |
|---------|-----|----------------|
| **Input Size** | Fixed (200x200) | Variable (any size) |
| **Latent Dim** | Fixed | 128-8192 |
| **Attention** | ❌ | ✅ Visualizable |
| **Long-range Deps** | Limited | Excellent |
| **Speed** | Very Fast | Fast (with FA2) |
| **Memory** | Low | Medium |
| **Parameters** | ~1-5M | ~0.5-128M |
| **GPU Optimization** | Basic | Advanced |
| **Best For** | Speed, efficiency | Flexibility, analysis |

### When to Use Each

**Use CNN Autoencoder if:**
- Maximum speed is critical
- Limited GPU memory (< 4GB)
- Small datasets (< 10k images)
- Fixed input size
- Simple feature extraction

**Use Transformer Autoencoder if:**
- Need attention visualization
- Large datasets (> 50k images)
- Variable input sizes
- Long-range dependencies matter
- Have sufficient GPU memory (≥ 6GB)
- Configurable latent space needed

---

## Testing

### Run Tests
```bash
# Test transformer autoencoder
python networks/test_transformer_autoencoder.py

# Compare CNN vs Transformer
python networks/compare_autoencoders.py
```

### Example Output
```
TRANSFORMER AUTOENCODER - COMPREHENSIVE TEST SUITE
================================================================================

TEST 1: Basic Functionality
  Input shape: torch.Size([4, 1, 201, 201])
  Reconstruction shape: torch.Size([4, 1, 201, 201])
  Embedding shape: torch.Size([4, 512])
  ✓ Basic functionality test passed!

...
```

---

## Installation Requirements

### Core Requirements
```bash
torch>=2.0.0  # For Flash Attention 2
torchvision
```

### Optional (for visualization)
```bash
matplotlib>=3.5.0
numpy
```

### GPU Requirements
- CUDA 11.7+ or 12.0+
- Recommended: Ada Lovelace (RTX 40xx) or Ampere (RTX 30xx)
- Minimum 4GB VRAM (6-8GB recommended)

---

## Recommended Workflow

### 1. For New Projects
Start with the Transformer autoencoder:
```python
from networks.AutoEncoder_TransformerV2_0 import create_autoencoder

model = create_autoencoder(
    input_size=(201, 201),
    preset='medium',
    use_flash_attention=True
)
```

### 2. For Speed-Critical Tasks
Use CNN autoencoder:
```python
from networks.AutoEncoder_CNNV1_0 import Autoencoder_CNN

model = Autoencoder_CNN(embedding_dim=512)
```

### 3. For Sequential Modeling
Combine autoencoder with LSTM:
```python
# Extract spatial features
autoencoder = create_autoencoder(...)
embeddings = autoencoder.Embedding(images)

# Model temporal dynamics
lstm = LSTMModel(LSTMEmbdSize=embeddings.size(1))
predictions = lstm(embeddings)
```

---

## Citation

```bibtex
@software{transformer_autoencoder_2025,
  author = {Yassin Riyazi},
  title = {Optimized Transformer Autoencoder for Image Compression},
  year = {2025},
  url = {https://github.com/YassinRiyazi/From-Droplet-Dynamics-to-Viscosity}
}
```

---

## Support

For detailed documentation:
- Transformer: [TRANSFORMER_USAGE_GUIDE.md](TRANSFORMER_USAGE_GUIDE.md)
- Examples: [test_transformer_autoencoder.py](test_transformer_autoencoder.py)
- Comparison: [compare_autoencoders.py](compare_autoencoders.py)

For issues or questions, please open an issue on the repository.
