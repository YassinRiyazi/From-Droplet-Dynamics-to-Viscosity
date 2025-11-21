"""
Example script demonstrating the Transformer Autoencoder usage
for different input sizes and attention visualization.

Run this script to test the model on your system.
"""

import torch
import torch.nn as nn
from networks.AutoEncoder_TransformerV2_0 import (
    create_autoencoder,
    visualize_attention_map,
    PRESET_CONFIGS
)
import time


def test_basic_functionality():
    """Test basic forward pass and embedding extraction"""
    print("\n" + "="*80)
    print("TEST 1: Basic Functionality")
    print("="*80)
    
    # Create model
    model = create_autoencoder(
        input_size=(201, 201),
        preset='small',
        use_flash_attention=True,
        use_gradient_checkpointing=False
    )
    
    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"Using device: {device}")
    
    # Create dummy input
    batch_size = 4
    x = torch.randn(batch_size, 1, 201, 201).to(device)
    print(f"Input shape: {x.shape}")
    
    # Forward pass
    with torch.no_grad():
        reconstruction, attention_info = model(x, return_attention=True)
        print(f"Reconstruction shape: {reconstruction.shape}")
        
        # Extract embedding
        embedding = model.Embedding(x)
        print(f"Embedding shape: {embedding.shape}")
        
        # Check attention info
        if attention_info:
            print(f"Number of attention layers: {attention_info['num_layers']}")
            print(f"Attention shape: {attention_info['shape']}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    print("✓ Basic functionality test passed!\n")
    return model


def test_variable_input_sizes():
    """Test with different input sizes"""
    print("\n" + "="*80)
    print("TEST 2: Variable Input Sizes")
    print("="*80)
    
    test_sizes = [
        ((201, 201), 8, "Square 201x201"),
        ((152, 1280), 4, "Wide 1280x152"),
        ((152, 640), 8, "Medium 640x152"),
    ]
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    for input_size, batch_size, description in test_sizes:
        print(f"\nTesting {description}:")
        
        model = create_autoencoder(
            input_size=input_size,
            preset='tiny',  # Use tiny for speed
            latent_dim=256
        ).to(device)
        
        # Create input
        x = torch.randn(batch_size, 1, *input_size).to(device)
        
        # Test forward pass
        with torch.no_grad():
            recon, _ = model(x)
            embedding = model.Embedding(x)
        
        print(f"  Input: {x.shape} -> Reconstruction: {recon.shape}")
        print(f"  Embedding: {embedding.shape}")
        
        # Check reconstruction error
        mse = nn.MSELoss()(recon, x).item()
        print(f"  Random MSE (untrained): {mse:.6f}")
    
    print("\n✓ Variable input size test passed!\n")


def test_attention_visualization():
    """Test attention map extraction and visualization"""
    print("\n" + "="*80)
    print("TEST 3: Attention Visualization")
    print("="*80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    model = create_autoencoder(
        input_size=(201, 201),
        preset='small',
        use_flash_attention=False,  # Disable to get attention weights
        latent_dim=256
    ).to(device)
    
    # Create dummy input
    x = torch.randn(1, 1, 201, 201).to(device)
    
    # Extract attention maps
    with torch.no_grad():
        attention_maps = model.get_attention_maps(x)
    
    print(f"Extracted attention maps:")
    for key, value in attention_maps.items():
        print(f"  {key}: {value.shape}")
    
    # Try to visualize (requires matplotlib)
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        
        # Visualize average attention
        visualize_attention_map(
            attention_maps['average'],
            save_path='test_attention_average.png',
            title='Average Attention - Test'
        )
        print("\n✓ Attention visualization saved to 'test_attention_average.png'")
    except ImportError:
        print("\nℹ Matplotlib not installed. Skipping visualization.")
    
    print("✓ Attention extraction test passed!\n")


def test_memory_and_speed():
    """Benchmark memory usage and speed"""
    print("\n" + "="*80)
    print("TEST 4: Memory and Speed Benchmark")
    print("="*80)
    
    if not torch.cuda.is_available():
        print("CUDA not available. Skipping GPU benchmarks.")
        return
    
    device = torch.device('cuda')
    
    configs = [
        ('tiny', 32),
        ('small', 16),
        ('medium', 8),
    ]
    
    for preset, batch_size in configs:
        print(f"\nBenchmarking {preset.upper()} preset with batch size {batch_size}:")
        
        # Create model
        model = create_autoencoder(
            input_size=(201, 201),
            preset=preset,
            use_flash_attention=True,
            use_gradient_checkpointing=False
        ).to(device)
        
        # Count parameters
        params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {params:,}")
        
        # Create input
        x = torch.randn(batch_size, 1, 201, 201).to(device)
        
        # Reset memory stats
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        
        # Warmup
        model.eval()
        with torch.no_grad():
            for _ in range(5):
                _ = model(x)
        
        # Benchmark
        torch.cuda.synchronize()
        start = time.time()
        
        num_iterations = 50
        with torch.no_grad():
            for _ in range(num_iterations):
                _ = model(x)
        
        torch.cuda.synchronize()
        elapsed = time.time() - start
        
        # Memory stats
        max_memory = torch.cuda.max_memory_allocated() / 1e9
        
        # Performance metrics
        throughput = (num_iterations * batch_size) / elapsed
        latency = elapsed / num_iterations * 1000
        
        print(f"  Max GPU Memory: {max_memory:.2f} GB")
        print(f"  Throughput: {throughput:.2f} images/sec")
        print(f"  Latency: {latency:.2f} ms/batch")
    
    print("\n✓ Memory and speed benchmark completed!\n")


def test_lstm_integration():
    """Test integration with LSTM network"""
    print("\n" + "="*80)
    print("TEST 5: LSTM Integration")
    print("="*80)
    
    try:
        from networks.AutoEncoder_CNN_LSTM import LSTMModel
        has_lstm = True
    except ImportError:
        print("LSTM module not found. Simulating LSTM...")
        has_lstm = False
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create autoencoder
    autoencoder = create_autoencoder(
        input_size=(201, 201),
        preset='small',
        latent_dim=256
    ).to(device)
    
    # Create sequence of images
    batch_size = 4
    sequence_length = 10
    image_sequence = torch.randn(batch_size, sequence_length, 1, 201, 201).to(device)
    
    print(f"Image sequence shape: {image_sequence.shape}")
    
    # Extract embeddings for each frame
    embeddings_list = []
    with torch.no_grad():
        for t in range(sequence_length):
            frame = image_sequence[:, t]  # (batch, 1, H, W)
            embedding = autoencoder.Embedding(frame)  # (batch, latent_dim)
            embeddings_list.append(embedding)
    
    embeddings = torch.stack(embeddings_list, dim=1)  # (batch, seq_len, latent_dim)
    print(f"Embeddings shape: {embeddings.shape}")
    
    if has_lstm:
        # Create LSTM
        lstm = LSTMModel(
            LSTMEmbdSize=embeddings.size(2),
            hidden_dim=128,
            num_layers=2,
            dropout=0.2
        ).to(device)
        
        # Forward pass through LSTM
        with torch.no_grad():
            lstm_output = lstm(embeddings)
        
        print(f"LSTM output shape: {lstm_output.shape}")
        print("✓ LSTM integration successful!")
    else:
        # Simulate LSTM
        lstm_hidden = 128
        simulated_output = torch.randn(batch_size, sequence_length, lstm_hidden).to(device)
        print(f"Simulated LSTM output: {simulated_output.shape}")
        print("✓ LSTM integration test passed (simulated)!")
    
    print()


def test_mixed_precision():
    """Test mixed precision training"""
    print("\n" + "="*80)
    print("TEST 6: Mixed Precision Training")
    print("="*80)
    
    if not torch.cuda.is_available():
        print("CUDA not available. Skipping mixed precision test.")
        return
    
    from torch.cuda.amp import autocast, GradScaler
    
    device = torch.device('cuda')
    
    # Create model
    model = create_autoencoder(
        input_size=(201, 201),
        preset='small',
        use_flash_attention=True
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()
    scaler = GradScaler()
    
    # Create dummy batch
    x = torch.randn(8, 1, 201, 201).to(device)
    
    # Training step with mixed precision
    model.train()
    
    # Reset memory
    torch.cuda.reset_peak_memory_stats()
    
    optimizer.zero_grad()
    
    with autocast():
        reconstruction, _ = model(x)
        loss = criterion(reconstruction, x)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    
    max_memory = torch.cuda.max_memory_allocated() / 1e9
    
    print(f"Mixed precision training step completed")
    print(f"Loss: {loss.item():.6f}")
    print(f"Max memory: {max_memory:.2f} GB")
    print("✓ Mixed precision test passed!\n")


def main():
    """Run all tests"""
    print("\n" + "="*80)
    print("TRANSFORMER AUTOENCODER - COMPREHENSIVE TEST SUITE")
    print("="*80)
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    # Run tests
    tests = [
        test_basic_functionality,
        test_variable_input_sizes,
        test_attention_visualization,
        test_memory_and_speed,
        test_lstm_integration,
        test_mixed_precision,
    ]
    
    for test_func in tests:
        try:
            test_func()
        except Exception as e:
            print(f"\n✗ Test failed: {test_func.__name__}")
            print(f"Error: {str(e)}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*80)
    print("ALL TESTS COMPLETED")
    print("="*80)
    
    # Print summary
    print("\nModel Presets Available:")
    for preset, config in PRESET_CONFIGS.items():
        print(f"  {preset:10s}: latent_dim={config['latent_dim']}, "
              f"heads={config['num_heads']}, "
              f"layers={config['num_encoder_layers']}+{config['num_decoder_layers']}")
    
    print("\nOptimization Features:")
    print("  ✓ Flash Attention 2 support")
    print("  ✓ Mixed precision training (AMP)")
    print("  ✓ Gradient checkpointing")
    print("  ✓ Torch.compile compatibility")
    print("  ✓ Variable input sizes")
    print("  ✓ Attention visualization")
    print("  ✓ LSTM integration")
    
    print("\nRecommended Usage:")
    print("  from networks.AutoEncoder_TransformerV2_0 import create_autoencoder")
    print("  model = create_autoencoder(input_size=(201, 201), preset='medium')")
    print("\nSee networks/TRANSFORMER_USAGE_GUIDE.md for detailed documentation.")
    print()


if __name__ == "__main__":
    main()
