"""Day 1 Verification Test: Preprocessing and Data Pipeline Integrity."""

import sys
import os
import time

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from PIL import Image
import torch
from src.data.preprocess import CLIPDataPreprocessor
from src.data.dataset import NewsDataset, generate_benchmark_samples, get_dataloaders


def test_day1_pipeline():
    print("=" * 60)
    print(" DAY 1 TEST: DATA PIPELINE & PREPROCESSING VALIDATION")
    print("=" * 60)

    # 1. Test Preprocessor
    print("\n[1/4] Testing CLIPDataPreprocessor...")
    preprocessor = CLIPDataPreprocessor()
    
    # Create test image
    dummy_img = Image.new("RGB", (300, 400), color=(100, 150, 200))
    dummy_text = "Breaking News: Major climate summit convenes in London to discuss global emissions."
    
    processed = preprocessor(dummy_img, dummy_text)
    
    print(f"  [OK] Image tensor shape: {processed['pixel_values'].shape} (Expected: torch.Size([3, 224, 224]))")
    print(f"  [OK] Input IDs shape: {processed['input_ids'].shape} (Expected: torch.Size([77]))")
    print(f"  [OK] Attention mask shape: {processed['attention_mask'].shape} (Expected: torch.Size([77]))")
    
    assert processed['pixel_values'].shape == torch.Size([3, 224, 224]), "Image shape mismatch!"
    assert processed['input_ids'].shape == torch.Size([77]), "Input IDs shape mismatch!"
    assert processed['attention_mask'].shape == torch.Size([77]), "Attention mask shape mismatch!"

    # 2. Test NewsDataset with Balanced Samples
    print("\n[2/4] Testing NewsDataset on balanced pairs...")
    test_samples = generate_benchmark_samples(num_samples=20, seed=42)
    dataset = NewsDataset(test_samples, preprocessor=preprocessor)
    print(f"  [OK] Dataset length: {len(dataset)}")
    
    first_item = dataset[0]
    print(f"  [OK] Sample 0 label: {first_item['label'].item()} (Authentic)")
    second_item = dataset[1]
    print(f"  [OK] Sample 1 label: {second_item['label'].item()} (Out-of-Context)")
    
    assert first_item['label'].item() == 1.0, "Label for authentic sample should be 1.0"
    assert second_item['label'].item() == 0.0, "Label for out-of-context sample should be 0.0"

    # 3. Test DataLoader Batching
    print("\n[3/4] Testing PyTorch DataLoader batch yield (batch_size=8)...")
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    start_time = time.time()
    batch = next(iter(loader))
    elapsed = time.time() - start_time
    
    print(f"  [OK] Batch pixel_values shape: {batch['pixel_values'].shape} (Expected: [8, 3, 224, 224])")
    print(f"  [OK] Batch input_ids shape:    {batch['input_ids'].shape} (Expected: [8, 77])")
    print(f"  [OK] Batch attention_mask:     {batch['attention_mask'].shape} (Expected: [8, 77])")
    print(f"  [OK] Batch label shape:        {batch['label'].shape} (Expected: [8, 1])")
    print(f"  [OK] Batch fetch time: {elapsed:.4f}s")
    
    assert batch['pixel_values'].shape == torch.Size([8, 3, 224, 224])
    assert batch['input_ids'].shape == torch.Size([8, 77])
    assert batch['label'].shape == torch.Size([8, 1])

    # 4. CUDA and Hardware Check
    print("\n[4/4] Checking Hardware & GPU Acceleration...")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"  [OK] CUDA Available: Yes ({gpu_name}, {vram_gb:.2f} GB VRAM)")
        # Test moving batch to CUDA
        pixel_cuda = batch['pixel_values'].cuda()
        print(f"  [OK] Successfully allocated batch to GPU memory: {pixel_cuda.device}")
    else:
        print("  [INFO] PyTorch CPU mode active for this local environment.")

    print("\n" + "=" * 60)
    print(" ALL DAY 1 PIPELINE TESTS PASSED SUCCESSFULLY! ")
    print("=" * 60)


if __name__ == "__main__":
    test_day1_pipeline()
