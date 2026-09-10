"""Day 2 Verification Test: Model Architecture, Parameter Counts, and Forward Pass."""

import sys
import os
import time

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from src.models.clip_backbone import FrozenCLIPBackbone
from src.models.cross_attention import (
    CrossAttentionClassifier,
    MultimodalInconsistencyDetector,
    count_parameters,
)


def test_day2_architecture():
    print("=" * 60)
    print(" DAY 2 TEST: ARCHITECTURE CONSTRUCTION & CROSS-ATTENTION HEAD")
    print("=" * 60)

    # 1. Test Frozen CLIP Backbone
    print("\n[1/4] Loading Frozen CLIP ViT-B/32 Backbone...")
    t0 = time.time()
    backbone = FrozenCLIPBackbone()
    load_time = time.time() - t0
    print(f"  [OK] Loaded CLIP backbone in {load_time:.2f}s")
    
    # Verify all backbone parameters are frozen
    all_frozen = all(not p.requires_grad for p in backbone.parameters())
    print(f"  [OK] Directive Check: All CLIP backbone weights strictly frozen? {all_frozen}")
    assert all_frozen, "ERROR: Backbone parameters must be strictly frozen!"

    # 2. Test Feature Extraction Shapes
    print("\n[2/4] Testing Patch & Word Token Feature Extraction...")
    batch_size = 4
    dummy_pixels = torch.randn(batch_size, 3, 224, 224)
    dummy_input_ids = torch.randint(0, 1000, (batch_size, 77))
    dummy_mask = torch.ones(batch_size, 77, dtype=torch.long)

    visual_patches, text_tokens = backbone(dummy_pixels, dummy_input_ids, dummy_mask)
    print(f"  [OK] Visual patches shape: {visual_patches.shape} (Expected: [{batch_size}, 49, 768])")
    print(f"  [OK] Text tokens shape:    {text_tokens.shape} (Expected: [{batch_size}, 77, 768])")

    assert visual_patches.shape == (batch_size, 49, 768), "Visual patches shape mismatch!"
    assert text_tokens.shape == (batch_size, 77, 768), "Text tokens shape mismatch!"

    # 3. Test CrossAttentionClassifier & Parameter Verification
    print("\n[3/4] Initializing CrossAttentionClassifier & Counting Parameters...")
    classifier = CrossAttentionClassifier(embed_dim=768, num_heads=8, mlp_hidden_dim=128, dropout=0.2)
    
    trainable_p, frozen_p = count_parameters(classifier)
    trainable_millions = trainable_p / 1e6
    print(f"  [OK] Cross-Attention Classifier Trainable Parameters: {trainable_p:,} (~{trainable_millions:.2f}M)")
    
    # Must be around 2.1M - 2.5M
    assert 2.0 <= trainable_millions <= 2.6, f"Parameter count {trainable_millions:.2f}M out of expected ~2.1M-2.5M range!"

    # Forward pass through classifier with attention weights
    out = classifier(text_tokens, visual_patches, attention_mask=dummy_mask, return_attention=True)
    logits = out["logits"]
    scores = out["consistency_score"]
    attn_weights = out["attention_weights"]

    print(f"  [OK] Output logits shape:            {logits.shape} (Expected: [{batch_size}, 1])")
    print(f"  [OK] Consistency score (S_c) shape: {scores.shape} (Expected: [{batch_size}, 1])")
    print(f"  [OK] Cross-attention map shape:      {attn_weights.shape} (Expected: [{batch_size}, 77, 49])")
    print(f"  [OK] Sample consistency scores S_c:  {[round(s.item(), 4) for s in scores]}")

    assert logits.shape == (batch_size, 1)
    assert scores.shape == (batch_size, 1)
    assert attn_weights.shape == (batch_size, 77, 49)
    assert (scores >= 0.0).all() and (scores <= 1.0).all(), "Consistency score must be bounded in [0.0, 1.0]!"

    # 4. Test End-to-End MultimodalInconsistencyDetector
    print("\n[4/4] Testing End-to-End MultimodalInconsistencyDetector...")
    full_model = MultimodalInconsistencyDetector()
    trainable_total, frozen_total = count_parameters(full_model)
    print(f"  [OK] Full Model Summary:")
    print(f"       - Frozen Backbone Parameters:    {frozen_total:,} (~{frozen_total/1e6:.1f}M)")
    print(f"       - Trainable Classifier Parameters: {trainable_total:,} (~{trainable_total/1e6:.2f}M)")
    print(f"       - Trainable Ratio:               {trainable_total / (trainable_total + frozen_total) * 100:.2f}%")

    t_start = time.time()
    e2e_out = full_model(dummy_pixels, dummy_input_ids, dummy_mask)
    latency_ms = (time.time() - t_start) * 1000
    print(f"  [OK] Forward Pass Latency (Batch Size {batch_size}, CPU): {latency_ms:.2f} ms ({latency_ms/batch_size:.2f} ms/sample)")

    print("\n" + "=" * 60)
    print(" ALL DAY 2 ARCHITECTURE TESTS PASSED SUCCESSFULLY! ")
    print("=" * 60)


if __name__ == "__main__":
    test_day2_architecture()
