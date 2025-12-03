# SROF Normalization Update

## Summary
Updated `MotherFolderDataset` and `DaughterFolderDataset` to support global normalization of SROF (4S-SROF) features across the entire dataset.

## Key Changes

### 1. **MotherFolderDataset.py**

#### Added `compute_normalization_stats()` method
- Collects all SROF data from all daughter datasets
- Computes global mean and std for each SROF feature column
- Stores statistics as `self.srof_mean` and `self.srof_std`
- Adds epsilon (1e-8) to prevent division by zero
- Called after loading all daughter datasets in `__init__`

#### Updated `save_cache()` and `load_cache()`
- Cache now includes `srof_mean` and `srof_std` arrays
- Fast cache loading restores normalization stats
- Backward compatible: uses default values if stats not in cache

#### Updated `DaughterSetLoader()`
- Passes normalization stats to each `DaughterFolderDataset` instance
- Stats are propagated back to daughters after computation

### 2. **DaughterFolderDataset.py**

#### Updated `__init__()`
- Added `srof_mean` and `srof_std` parameters (optional)
- Stores normalization statistics for use in `__getitem__`

#### Updated `__getitem__Normal()`
- **Removed 100-column duplication**: Previously concatenated `tilt` and `count` as 50-column blocks each
- **New structure**:
  - `tilt_col`: single column with tilt angle
  - `count_col`: single column with frame count
  - `SROF_combined`: drop_position (2 cols) + SROF features (8 cols) = 10 features
- **Normalization applied**: `(SROF_combined - mean) / std`
- **Final output**: `[tilt, count, normalized_drop_x, normalized_drop_y, normalized_SROF_features]`
  - Total: **12 features** instead of previous ~110

## Data Flow

### Before Normalization
```
Old SROF tensor shape: [seq_len, 110]
  - tilt duplicated 50 times
  - count duplicated 50 times  
  - drop_position (2 cols)
  - SROF raw values (8 cols)
```

### After Normalization
```
New SROF tensor shape: [seq_len, 12]
  - tilt (1 col, unnormalized)
  - count (1 col, unnormalized)
  - drop_position normalized (2 cols)
  - SROF normalized (8 cols)
```

## Benefits

1. **Fixed feature distribution**: SROF features now have zero mean and unit variance across the dataset
2. **Removed redundancy**: Eliminated 100 duplicate constant columns
3. **Improved gradient flow**: Normalized features prevent any single feature from dominating gradients
4. **Consistent splits**: Train/val/test use the same normalization statistics (computed from training data)
5. **Cached**: Normalization stats saved with dataset cache for fast reloading

## Important Notes

### Feature Count Change
- **Old**: `LSTMEmbdSize` was set to `proj_dim` (10 in `Train_LSTM_4SSROF.py`) but actual data had ~110 features
- **New**: Actual features = 12 (consistent with data structure)
- **Action Required**: Update `LSTMEmbdSize = 12` in training scripts

### Image Normalization
The SROF normalization complements but is separate from image normalization:
- Images: still `[0, 1]` from `ToTensor()` 
- **Recommended**: Re-enable `transforms.Normalize((0.5,), (0.5,))` to center images to `[-1, 1]`

### Cache Invalidation
- Old cache files do NOT contain normalization stats
- **First run after update**: Dataset will recompute stats and save new cache
- **Subsequent runs**: Stats loaded instantly from cache

## Usage Example

```python
from dataset import MotherFolderDataset, dicLoader

# Load addresses
dicTrain, dicVal, _ = dicLoader(root="/media/roboprocessing/Data/Viscosity")

# Create dataset (computes stats on first run)
train_ds = MotherFolderDataset(dicTrain, stride=2, sequence_length=10)
# Output: 
# Computed normalization stats: SROF shape (10,)
# SROF mean: [...]
# SROF std: [...]

# Cache for future runs
train_ds.save_cache("Output/dataset_cache_train_normalized.pkl")

# Next time: instant load with stats
train_ds = MotherFolderDataset.load_cache("Output/dataset_cache_train_normalized.pkl")
# Output:
# Normalization stats: SROF mean shape (10,), std shape (10,)
```

## Validation

After updating, verify normalization is working:

```python
# Check a sample
item = train_ds[0]
srof_features = item[2]  # [seq_len, 12]

# Features 2-11 should be approximately N(0,1)
print("Mean:", srof_features[:, 2:].mean().item())  # Should be ~0
print("Std:", srof_features[:, 2:].std().item())    # Should be ~1
```

## Migration Path

1. **Delete old cache files** (optional but recommended):
   ```bash
   rm Output/dataset_cache_train_*.pkl
   rm Output/dataset_cache_val_*.pkl
   ```

2. **Update training scripts**:
   - Change `LSTMEmbdSize` from `proj_dim` to `12`
   - Re-enable image normalization in `config.yaml` or dataset transform

3. **Rebuild caches**:
   - First run will be slow (computes stats)
   - Saves new cache with normalization stats
   - Future runs are fast

4. **Retrain models** with normalized features for best results
