"""
Example: How to use MotherFolderDataset caching for fast loading

The first time you run this, it will create the dataset from scratch (slow).
Subsequent runs will load from cache (very fast).
"""

import os
from MotherFolderDataset import MotherFolderDataset, dicLoader

def main():
    # Define cache file paths
    cache_dir = "/home/d25u2/Desktop/From-Droplet-Dynamics-to-Viscosity/Output"
    os.makedirs(cache_dir, exist_ok=True)
    
    cache_train = os.path.join(cache_dir, "dataset_cache_train.pkl")
    cache_val = os.path.join(cache_dir, "dataset_cache_val.pkl")
    cache_test = os.path.join(cache_dir, "dataset_cache_test.pkl")
    
    # Load dataset splits
    dicAddressesTrain, dicAddressesValidation, dicAddressesTest = dicLoader(
        root="/media/d25u2/Dont/Viscosity"
    )
    
    # ===== TRAINING DATASET =====
    if os.path.exists(cache_train):
        print("Loading TRAINING dataset from cache...")
        train_dataset = MotherFolderDataset.load_cache(cache_train)
    else:
        print("Creating TRAINING dataset from scratch (this will take time)...")
        train_dataset = MotherFolderDataset(
            dicAddresses=dicAddressesTrain,
            stride=1,
            sequence_length=5
        )
        print("Saving training dataset cache...")
        train_dataset.save_cache(cache_train)
    
    # ===== VALIDATION DATASET =====
    if os.path.exists(cache_val):
        print("Loading VALIDATION dataset from cache...")
        val_dataset = MotherFolderDataset.load_cache(cache_val)
    else:
        print("Creating VALIDATION dataset from scratch...")
        val_dataset = MotherFolderDataset(
            dicAddresses=dicAddressesValidation,
            stride=1,
            sequence_length=5
        )
        print("Saving validation dataset cache...")
        val_dataset.save_cache(cache_val)
    
    # ===== TEST DATASET =====
    if os.path.exists(cache_test):
        print("Loading TEST dataset from cache...")
        test_dataset = MotherFolderDataset.load_cache(cache_test)
    else:
        print("Creating TEST dataset from scratch...")
        test_dataset = MotherFolderDataset(
            dicAddresses=dicAddressesTest,
            stride=1,
            sequence_length=5
        )
        print("Saving test dataset cache...")
        test_dataset.save_cache(cache_test)
    
    print(f"\n✓ All datasets ready!")
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val:   {len(val_dataset)} samples")
    print(f"  Test:  {len(test_dataset)} samples")
    
    # Test loading a sample
    print("\nTesting sample loading...")
    sample, label = train_dataset[0]
    print(f"Sample shape: {sample.shape}, Normalized label: {label:.4f}")
    
    return train_dataset, val_dataset, test_dataset


if __name__ == "__main__":
    train_ds, val_ds, test_ds = main()
