# Transformer Autoencoder - Quick Reference

## One-Line Creation
```python
from networks.AutoEncoder_TransformerV2_0 import create_autoencoder
model = create_autoencoder(input_size=(201, 201), preset='medium')
```

## Common Use Cases

### 1. Basic Autoencoding
```python
x = torch.randn(32, 1, 201, 201).cuda()
reconstruction, _ = model(x)
```

### 2. Extract Embeddings for LSTM
```python
embeddings = model.Embedding(x)  # (batch, latent_dim)
```

### 3. Visualize Attention
```python
attention_maps = model.get_attention_maps(x[:1])
visualize_attention_map(attention_maps['average'], save_path='attention.png')
```

### 4. Different Input Sizes
```python
# Square images
model_square = create_autoencoder(input_size=(201, 201), preset='medium')

# Wide images
model_wide = create_autoencoder(input_size=(1280, 152), preset='small')

# Variable width
model_var = create_autoencoder(input_size=(640, 152), preset='small')
```

### 5. Custom Latent Dimension
```python
model = create_autoencoder(
    input_size=(201, 201),
    preset='medium',
    latent_dim=1024  # Override preset
)
```

## GPU Optimization Quick Setup

### RTX 4060/4070 (8GB VRAM) - Recommended
```python
from torch.cuda.amp import autocast, GradScaler

model = create_autoencoder(
    input_size=(201, 201),
    preset='medium',
    use_flash_attention=True,
    use_gradient_checkpointing=False
).cuda()

# Training with mixed precision
scaler = GradScaler()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

for images in dataloader:
    images = images.cuda()
    optimizer.zero_grad()
    
    with autocast():
        recon, _ = model(images)
        loss = criterion(recon, images)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

### Large Models (> 8GB VRAM)
```python
model = create_autoencoder(
    input_size=(201, 201),
    preset='large',
    use_flash_attention=True,
    use_gradient_checkpointing=True  # Save memory
).cuda()

# Optional: Compile for speed (PyTorch 2.0+)
model = torch.compile(model, mode='reduce-overhead')
```

## Preset Selection Guide

| Your Situation | Recommended Preset | Latent Dim | Batch Size |
|----------------|-------------------|------------|------------|
| Quick prototyping | `tiny` | 128 | 64 |
| Most tasks, 6-8GB VRAM | `small` | 256 | 32-48 |
| Production, good GPU | `medium` | 512 | 16-32 |
| Research, large dataset | `large` | 1024 | 8-16 |
| Maximum capacity | `xlarge` | 2048 | 4-8 |

## Optimal Batch Sizes (8GB VRAM)

| Input Size | Preset | FP32 | Mixed Precision | Grad Checkpoint |
|------------|--------|------|-----------------|-----------------|
| 201x201 | small | 32-48 | 64-96 | 96-128 |
| 201x201 | medium | 16-24 | 32-48 | 48-64 |
| 1280x152 | small | 16-24 | 32-48 | 48-64 |
| 1280x152 | medium | 8-12 | 16-24 | 24-32 |

## Integration with LSTM

```python
from networks.AutoEncoder_TransformerV2_0 import create_autoencoder
from networks.AutoEncoder_CNN_LSTM import LSTMModel

# Setup
autoencoder = create_autoencoder(input_size=(201, 201), latent_dim=512).cuda()
lstm = LSTMModel(LSTMEmbdSize=512, hidden_dim=256, num_layers=2).cuda()

# Process sequence
batch_size, seq_len = 16, 10
images = torch.randn(batch_size, seq_len, 1, 201, 201).cuda()

embeddings = torch.stack([
    autoencoder.Embedding(images[:, t]) for t in range(seq_len)
], dim=1)

output = lstm(embeddings)
```

## Attention Visualization

```python
import matplotlib.pyplot as plt

# Get attention
model.eval()
with torch.no_grad():
    attention_maps = model.get_attention_maps(test_image.unsqueeze(0).cuda())

# Plot
plt.figure(figsize=(10, 8))
plt.imshow(attention_maps['average'][0].cpu(), cmap='viridis')
plt.colorbar()
plt.title('Average Attention')
plt.savefig('attention.png')
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Out of memory | 1. Reduce batch size<br>2. Enable `use_gradient_checkpointing=True`<br>3. Use mixed precision<br>4. Use smaller preset |
| Too slow | 1. Enable `use_flash_attention=True`<br>2. Use `torch.compile(model)`<br>3. Increase batch size<br>4. Reduce num_layers |
| Flash Attention not working | Update PyTorch: `pip install torch>=2.0.0` |
| Poor reconstruction | 1. Increase latent_dim<br>2. Increase num_layers<br>3. Train longer<br>4. Check learning rate |

## Performance Tips

✅ **DO:**
- Use Flash Attention 2 (requires PyTorch 2.0+)
- Use mixed precision training (AMP)
- Use `torch.compile()` for PyTorch 2.0+
- Use `pin_memory=True` in DataLoader
- Start with smaller presets and scale up
- Monitor GPU memory with `torch.cuda.memory_allocated()`

❌ **DON'T:**
- Use gradient checkpointing unless necessary (slows training)
- Use batch size of 1 (inefficient)
- Load entire dataset to GPU at once
- Use return_attention=True during training (overhead)

## Save and Load

```python
# Save
checkpoint = {
    'model_state_dict': model.state_dict(),
    'config': model.get_config(),
    'optimizer_state_dict': optimizer.state_dict(),
}
torch.save(checkpoint, 'model.pth')

# Load
checkpoint = torch.load('model.pth')
model = create_autoencoder(
    input_size=checkpoint['config']['input_size'],
    latent_dim=checkpoint['config']['latent_dim'],
    preset='medium'
)
model.load_state_dict(checkpoint['model_state_dict'])
```

## Key Methods

```python
# Forward pass with reconstruction
recon, attention_info = model(x, return_attention=True)

# Get embedding only (for LSTM)
embedding = model.Embedding(x)

# Get attention maps
attention_maps = model.get_attention_maps(x)

# Get configuration
config = model.get_config()
```

## Documentation Links

- **Full guide**: `networks/TRANSFORMER_USAGE_GUIDE.md`
- **Test script**: `networks/test_transformer_autoencoder.py`
- **Comparison**: `networks/compare_autoencoders.py`
- **Networks README**: `networks/README.md`
