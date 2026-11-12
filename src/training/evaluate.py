"""Evaluation module: ROC-AUC, Confusion Matrix, and Classification Report generation."""

import os
import json
from typing import Dict, Any, Optional, List, Tuple
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server/Colab environments
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_curve,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

from src.models.cross_attention import MultimodalInconsistencyDetector
from src.data.dataset import NewsDataset, stream_news_clippings_subset


@torch.no_grad()
def run_inference_on_loader(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Runs inference and collects targets, consistency scores, and predicted labels.
    
    Returns:
        Tuple of (y_true, y_scores, y_pred).
    """
    model.eval()
    all_targets: List[float] = []
    all_scores: List[float] = []

    for batch in dataloader:
        pixel_values = batch["pixel_values"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        outputs = model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        scores = outputs["consistency_score"].cpu().flatten().tolist()
        targets = labels.cpu().flatten().tolist()

        all_scores.extend(scores)
        all_targets.extend(targets)

    y_true = np.array(all_targets, dtype=int)
    y_scores = np.array(all_scores, dtype=float)
    y_pred = (y_scores >= 0.5).astype(int)

    return y_true, y_scores, y_pred


def plot_confusion_matrix(
    cm: np.ndarray,
    save_path: str = "assets/confusion_matrix.png",
    class_names: Optional[List[str]] = None,
) -> None:
    """Plots and saves normalized confusion matrix heatmap."""
    if class_names is None:
        class_names = ["Out-of-Context", "Authentic"]

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
    )
    plt.title("Confusion Matrix — Multimodal Inconsistency", fontsize=13, pad=12)
    plt.xlabel("Predicted Label", fontsize=11)
    plt.ylabel("True Label", fontsize=11)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"  [Asset Saved] Confusion Matrix: {save_path}")


def plot_roc_curve(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    roc_auc: float,
    save_path: str = "assets/roc_curve.png",
) -> None:
    """Plots and saves the Receiver Operating Characteristic (ROC) curve."""
    fpr, tpr, _ = roc_curve(y_true, y_scores)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color="#2563eb", lw=2.5, label=f"Cross-Attention (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], color="#9ca3af", lw=1.5, linestyle="--", label="Random Chance (AUC = 0.500)")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=11)
    plt.ylabel("True Positive Rate (Recall)", fontsize=11)
    plt.title("ROC Curve — Out-of-Context Detection", fontsize=13, pad=12)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"  [Asset Saved] ROC-AUC Curve: {save_path}")


def evaluate_model(
    checkpoint_path: str = "checkpoints/best_cross_attention_model.pt",
    test_loader: Optional[DataLoader] = None,
    model: Optional[MultimodalInconsistencyDetector] = None,
    assets_dir: str = "assets",
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """Full evaluation routine computing metrics and generating seminar publication assets.
    
    Args:
        checkpoint_path: Path to model checkpoint .pt file.
        test_loader: Optional DataLoader for test set.
        model: Optional initialized model.
        assets_dir: Directory where figures and reports are saved.
        device: Device string ('cuda', 'cpu', etc.).
        
    Returns:
        Dictionary of computed metrics (accuracy, macro_f1, roc_auc, etc.).
    """
    target_device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
    os.makedirs(assets_dir, exist_ok=True)

    # 1. Load model & weights
    if model is None:
        model = MultimodalInconsistencyDetector()
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=target_device)
            if "classifier_state_dict" in checkpoint:
                model.classifier.load_state_dict(checkpoint["classifier_state_dict"])
            elif "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"], strict=False)
            print(f"[Evaluation] Loaded checkpoint weights from: {checkpoint_path}")
        else:
            print(f"[Evaluation] Warning: Checkpoint {checkpoint_path} not found. Evaluating initialized model.")

    model.to(target_device)

    # 2. Prepare test DataLoader if not passed
    if test_loader is None:
        print("[Evaluation] Loading test split (750 samples)...")
        test_samples = stream_news_clippings_subset(
            split="test",
            max_samples=750,
            fallback_benchmark=True,
        )
        test_dataset = NewsDataset(test_samples)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # 3. Collect inference predictions
    print("[Evaluation] Executing inference across test dataset...")
    y_true, y_scores, y_pred = run_inference_on_loader(model, test_loader, target_device)

    # 4. Compute metrics
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    
    # Handle single-class edge case for ROC-AUC
    if len(np.unique(y_true)) > 1:
        roc_auc = roc_auc_score(y_true, y_scores)
    else:
        roc_auc = 0.5

    cm = confusion_matrix(y_true, y_pred)
    report_dict = classification_report(
        y_true,
        y_pred,
        target_names=["Out-of-Context", "Authentic"],
        output_dict=True,
        zero_division=0,
    )
    report_text = classification_report(
        y_true,
        y_pred,
        target_names=["Out-of-Context", "Authentic"],
        zero_division=0,
    )

    print("\n" + "=" * 55)
    print(" CLASSIFICATION REPORT (CSC301 TEST BENCHMARK)")
    print("=" * 55)
    print(report_text)
    print(f"ROC-AUC Score: {roc_auc:.4f}")
    print("=" * 55 + "\n")

    # 5. Export graphical assets
    cm_path = os.path.join(assets_dir, "confusion_matrix.png")
    roc_path = os.path.join(assets_dir, "roc_curve.png")
    report_path = os.path.join(assets_dir, "classification_report.json")

    plot_confusion_matrix(cm, save_path=cm_path)
    plot_roc_curve(y_true, y_scores, roc_auc, save_path=roc_path)

    metrics_summary = {
        "accuracy": float(acc),
        "macro_f1": float(f1),
        "precision": float(prec),
        "recall": float(rec),
        "roc_auc": float(roc_auc),
        "confusion_matrix": cm.tolist(),
        "classification_report": report_dict,
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2)
    print(f"  [Asset Saved] Classification Metrics JSON: {report_path}")

    return metrics_summary


if __name__ == "__main__":
    evaluate_model()
