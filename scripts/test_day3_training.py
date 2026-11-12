"""Day 3 Verification Test: Model Training, Validation Loop, Checkpointing, and Asset Generation."""

import sys
import os
import time

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from torch.utils.data import DataLoader

from src.models.cross_attention import MultimodalInconsistencyDetector
from src.data.dataset import NewsDataset, generate_benchmark_samples
from src.training.train import train_model
from src.training.evaluate import evaluate_model


def test_day3_pipeline():
    print("=" * 65)
    print(" DAY 3 TEST: TRAINING LOOP, CHECKPOINTING & EVALUATION ASSETS")
    print("=" * 65)

    # 1. Prepare mini-benchmark dataset for rapid local testing
    print("\n[1/4] Preparing balanced mini-benchmark splits (24 train, 8 val, 8 test)...")
    train_samples = generate_benchmark_samples(num_samples=24, seed=101)
    val_samples = generate_benchmark_samples(num_samples=8, seed=202)
    test_samples = generate_benchmark_samples(num_samples=8, seed=303)

    train_ds = NewsDataset(train_samples)
    val_ds = NewsDataset(val_samples)
    test_ds = NewsDataset(test_samples)

    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=4, shuffle=False)

    print(f"  [OK] Mini-datasets created successfully.")

    # 2. Initialize Model
    print("\n[2/4] Initializing MultimodalInconsistencyDetector...")
    model = MultimodalInconsistencyDetector()

    # Guardrail check
    backbone_frozen = all(not p.requires_grad for p in model.backbone.parameters())
    classifier_trainable = any(p.requires_grad for p in model.classifier.parameters())
    print(f"  [OK] Guardrail Check - Backbone frozen: {backbone_frozen} | Classifier trainable: {classifier_trainable}")
    assert backbone_frozen, "ERROR: Backbone must be strictly frozen!"
    assert classifier_trainable, "ERROR: Classifier head must have trainable parameters!"

    # 3. Test Training Loop for 2 Mini-Epochs with Checkpointing
    print("\n[3/4] Running 2-epoch training loop with validation checkpointing...")
    t0 = time.time()
    test_save_dir = "checkpoints"
    test_ckpt_file = "test_day3_checkpoint.pt"
    ckpt_full_path = os.path.join(test_save_dir, test_ckpt_file)

    train_result = train_model(
        config={
            "training": {
                "epochs": 2,
                "lr": 1e-3,
                "weight_decay": 1e-2,
                "save_dir": test_save_dir,
                "checkpoint_filename": test_ckpt_file,
            },
            "model": {},
            "data": {},
        },
        train_loader=train_loader,
        val_loader=val_loader,
        model=model,
        epochs=2,
    )

    elapsed = time.time() - t0
    print(f"  [OK] Training completed in {elapsed:.2f}s")
    assert os.path.exists(ckpt_full_path), f"ERROR: Expected checkpoint file at {ckpt_full_path} was not created!"
    print(f"  [OK] Checkpoint successfully verified at: {ckpt_full_path}")
    print(f"  [OK] Best Validation Macro-F1: {train_result['best_val_f1']:.4f}")

    # 4. Test Evaluation & Asset Generation
    print("\n[4/4] Running evaluation routine & verifying presentation assets...")
    test_assets_dir = "assets"
    eval_metrics = evaluate_model(
        checkpoint_path=ckpt_full_path,
        test_loader=test_loader,
        assets_dir=test_assets_dir,
    )

    cm_file = os.path.join(test_assets_dir, "confusion_matrix.png")
    roc_file = os.path.join(test_assets_dir, "roc_curve.png")
    report_file = os.path.join(test_assets_dir, "classification_report.json")

    assert os.path.exists(cm_file), f"Missing confusion matrix asset: {cm_file}"
    assert os.path.exists(roc_file), f"Missing ROC curve asset: {roc_file}"
    assert os.path.exists(report_file), f"Missing classification report asset: {report_file}"

    print(f"  [OK] Verified Confusion Matrix plot ({os.path.getsize(cm_file)} bytes)")
    print(f"  [OK] Verified ROC Curve plot ({os.path.getsize(roc_file)} bytes)")
    print(f"  [OK] Verified Classification Report JSON ({os.path.getsize(report_file)} bytes)")
    print(f"  [OK] Evaluation Metrics: Accuracy={eval_metrics['accuracy']:.4f}, Macro-F1={eval_metrics['macro_f1']:.4f}, ROC-AUC={eval_metrics['roc_auc']:.4f}")

    print("\n" + "=" * 65)
    print(" [PASSED] ALL DAY 3 TRAINING & EVALUATION TESTS SUCCESSFUL!")
    print("=" * 65)


if __name__ == "__main__":
    test_day3_pipeline()
