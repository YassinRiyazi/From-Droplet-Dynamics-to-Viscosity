# MotherFolderDataset Caching Guide

## Problem
Loading `MotherFolderDataset` is slow because it needs to:
- Scan directories for image files
- Read and parse CSV files (detections.csv, SROF files)
- Generate positional embeddings
- Build internal data structures

## Solution
Use the built-in **caching system** to save the dataset metadata once and reload it instantly.

## How It Works

### 1. Save Cache (First Time Only)
```python
from dataset.MotherFolderDataset import MotherFolderDataset, dicLoader

# Load dataset splits
dicAddressesTrain, _, _ = dicLoader(root="/media/d25u2/Dont/Viscosity")

# Create dataset (slow first time)
dataset = MotherFolderDataset(
    dicAddresses=dicAddressesTrain,
    stride=1,
    sequence_length=5
)

# Save cache for future use
dataset.save_cache("dataset_cache_train.pkl")
```

### 2. Load from Cache (Subsequent Times)
```python
from dataset.MotherFolderDataset import MotherFolderDataset

# Load dataset instantly from cache (very fast!)
dataset = MotherFolderDataset.load_cache("dataset_cache_train.pkl")

# Use dataset normally
print(f"Total samples: {len(dataset)}")
sample, label = dataset[0]
```

### 3. Smart Loading Pattern
```python
import os
from dataset.MotherFolderDataset import MotherFolderDataset, dicLoader

cache_path = "dataset_cache_train.pkl"

if os.path.exists(cache_path):
    # Load from cache (fast)
    dataset = MotherFolderDataset.load_cache(cache_path)
else:
    # Create from scratch and save cache
    dicAddressesTrain, _, _ = dicLoader(root="/media/d25u2/Dont/Viscosity")
    dataset = MotherFolderDataset(
        dicAddresses=dicAddressesTrain,
        stride=1,
        sequence_length=5
    )
    dataset.save_cache(cache_path)
```

## What Gets Cached?

The cache stores:
- ✓ Dataset configuration (stride, sequence_length, dicAddresses)
- ✓ Precomputed `DataAddress` lists (file paths, viscosity, drop positions, SROF data)
- ✓ Dataset length and maximum viscosity
- ✗ **NOT** the actual image data (images are loaded on-demand during `__getitem__`)

## Benefits

| Operation | Without Cache | With Cache |
|-----------|---------------|------------|
| First load | ~2-5 minutes | ~2-5 minutes |
| Subsequent loads | ~2-5 minutes | **~2-5 seconds** |
| Memory usage | Same | Same |
| Image loading | On-demand | On-demand |

## Important Notes

1. **Cache invalidation**: Regenerate cache if:
   - You add/remove images from directories
   - You change stride or sequence_length
   - CSV files (detections.csv, SROF) are updated

2. **Cache location**: Store caches in a dedicated folder:
   ```python
   cache_dir = "./Output/dataset_cache/"
   os.makedirs(cache_dir, exist_ok=True)
   ```

3. **Multiple configurations**: Create separate caches for different settings:
   ```python
   dataset.save_cache(f"cache_stride{stride}_seq{seq_len}.pkl")
   ```

## Example Script

See `dataset/example_cache_usage.py` for a complete working example that caches train/val/test datasets.

## API Reference

### `MotherFolderDataset.save_cache(filepath: str)`
Saves the dataset metadata to a pickle file.

**Parameters:**
- `filepath`: Path where cache will be saved (e.g., `"cache.pkl"`)

### `MotherFolderDataset.load_cache(filepath: str) -> MotherFolderDataset`
Class method that reconstructs a dataset from cache.

**Parameters:**
- `filepath`: Path to the cache file

**Returns:**
- Fully functional `MotherFolderDataset` instance
